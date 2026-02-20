#!/usr/bin/env python3

import os
import json
import random
import sqlite3
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import re
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

try:
    from reprocess_url_datasets import derive_unified_from_url
except ImportError:
    derive_unified_from_url = None

logger = logging.getLogger(__name__)

# Default paths and config
_BASE_DIR = Path(os.getenv("PHISHING_MODEL_DIR", "/home/mark/Desktop/PhishingDetection"))
_DB_PATH = _BASE_DIR / "rlhf_phishing_detection.db"
_URL_MODEL_PATH = _BASE_DIR / "models" / "url_only_model.pkl"
_TRAINING_RESULTS_GLOB = [
    _BASE_DIR / "real_email_training_results_20250828_191221.json",
    _BASE_DIR / "real_email_training_results_20250828_171056.json",
]

DEFAULT_CONFIG_PATH = _BASE_DIR / "classifier_config.json"

# Default configuration – loaded from classifier_config.json when present
DEFAULT_CONFIG: Dict = {
    "base_metrics": {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.85,
        "f1_score": 0.82,
    },
    "model_weights": {
        "phishing_indicators": {
            "urgent": 0.7,
            "click": 0.8,
            "verify": 0.6,
        },
        "legitimate_indicators": {
            "meeting": 0.7,
            "newsletter": 0.6,
            "university": 0.8,
        },
        "domain_trust": {
            ".edu": 0.8,
            ".com": 0.3,
            "suspicious": -0.7,
        },
    },
    "exploration_rate": 0.05,
    "decision": {
        "margin_default": 0.20,
        "margin_high_recall": 0.10,
        "confidence_base": 65,
        "confidence_scale": 25,
        "confidence_min": 25,
        "confidence_max": 98,
        "confidence_uncertain_min": 30,
        "confidence_uncertain_max": 65,
        "accuracy_blend_base": 0.7,
        "accuracy_blend_recent": 0.3,
        "accuracy_confidence_offset": 0.05,
    },
    "url_ensemble": {
        "threshold": 0.6,
    },
    "url_features": [
        "email_length", "word_count", "char_count", "url_count",
        "has_url", "suspicious_url", "phishing_keywords", "financial_keywords",
        "urgency_words", "suspicious_words", "has_html", "has_script",
        "has_form", "has_iframe", "sender_suspicious", "domain_age",
        "path_depth", "digits_ratio", "hex_ratio", "query_count",
        "entropy", "punycode", "tld_risky", "ip_in_host", "brand_distance",
    ],
}


def _load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict:
    """Load classifier config from JSON; fall back to built-in defaults."""
    if path.exists():
        try:
            with open(path) as fh:
                user_cfg = json.load(fh)
            # Shallow-merge top-level keys so partial configs work fine
            merged = {**DEFAULT_CONFIG, **user_cfg}
            for key in ("base_metrics", "model_weights", "decision", "url_ensemble"):
                if key in DEFAULT_CONFIG and key in user_cfg:
                    merged[key] = {**DEFAULT_CONFIG[key], **user_cfg[key]}
            logger.debug("Loaded classifier config from %s", path)
            return merged
        except Exception as exc:
            logger.warning("Could not load config from %s (%s); using defaults.", path, exc)
    return dict(DEFAULT_CONFIG)

@dataclass
class PerformanceMetrics:
    accuracy_history: list = field(default_factory=list)
    precision_history: list = field(default_factory=list)
    recall_history: list = field(default_factory=list)
    f1_history: list = field(default_factory=list)
    feedback_count: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> "PerformanceMetrics":
        return cls(
            accuracy_history=data.get("accuracy_history", []),
            precision_history=data.get("precision_history", []),
            recall_history=data.get("recall_history", []),
            f1_history=data.get("f1_history", []),
            feedback_count=data.get("feedback_count", 0),
            correct_predictions=data.get("correct_predictions", 0),
            total_predictions=data.get("total_predictions", 0),
        )

    def to_dict(self) -> Dict:
        return {
            "accuracy_history": self.accuracy_history,
            "precision_history": self.precision_history,
            "recall_history": self.recall_history,
            "f1_history": self.f1_history,
            "feedback_count": self.feedback_count,
            "correct_predictions": self.correct_predictions,
            "total_predictions": self.total_predictions,
        }

class RLHFClassifierService:
    def __init__(
        self,
        db_path: Optional[str] = None,
        metric_calibration_pct: float = 0.0,
        config_path: Optional[str] = None,
    ):
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self.cfg = _load_config(Path(config_path) if config_path else DEFAULT_CONFIG_PATH)

        # Base metrics – overridden by training results / DB state below
        bm = self.cfg["base_metrics"]
        self.base_accuracy: float = bm["accuracy"]
        self.base_precision: float = bm["precision"]
        self.base_recall: float = bm["recall"]
        self.base_f1: float = bm["f1_score"]

        self.model_version: int = 1
        self.total_updates: int = 0
        self.model_weights: Dict = self.cfg["model_weights"]
        self.exploration_rate: float = self.cfg["exploration_rate"]

        self.performance_metrics = PerformanceMetrics(
            accuracy_history=[self.base_accuracy],
            precision_history=[self.base_precision],
            recall_history=[self.base_recall],
            f1_history=[self.base_f1],
        )

        self.metric_calibration_pct: float = max(0.0, min(0.2, float(metric_calibration_pct)))

        # Load persisted state (training results → DB, in that priority order)
        self._load_trained_model()
        self._load_rlhf_state()

        # URL ensemble
        self.enable_url_ensemble: bool = os.getenv("RLHF_ENSEMBLE_URL", "0") == "1"
        url_threshold_env = os.getenv("RLHF_URL_THRESHOLD")
        self.url_threshold: float = (
            float(url_threshold_env)
            if url_threshold_env is not None
            else self.cfg["url_ensemble"]["threshold"]
        )
        self.url_model = self._load_url_model()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_trained_model(self) -> None:
        """Populate base metrics from the most recent training-result file."""
        for filepath in _TRAINING_RESULTS_GLOB:
            if not filepath.exists():
                continue
            try:
                with open(filepath) as fh:
                    data = json.load(fh)
                metrics = data.get("metrics", {})
                self.base_accuracy = metrics.get("accuracy", self.base_accuracy)
                self.base_precision = metrics.get("precision", self.base_precision)
                self.base_recall = metrics.get("recall", self.base_recall)
                self.base_f1 = metrics.get("f1_score", self.base_f1)
                logger.debug("Loaded training metrics from %s", filepath)
                return
            except Exception as exc:
                logger.warning("Could not read training file %s: %s", filepath, exc)

    def _load_rlhf_state(self) -> None:
        """Restore model version, weights and performance metrics from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM rlhf_state WHERE id = 1")
                row = cursor.fetchone()
            if not row:
                return
            self.model_version = row[1]
            self.total_updates = row[2]
            try:
                self.model_weights = json.loads(row[3])
            except Exception:
                pass
            try:
                self.performance_metrics = PerformanceMetrics.from_dict(json.loads(row[4]))
            except Exception:
                pass
            try:
                base_perf = json.loads(row[6])
                self.base_accuracy = base_perf.get("accuracy", self.base_accuracy)
                self.base_precision = base_perf.get("precision", self.base_precision)
                self.base_recall = base_perf.get("recall", self.base_recall)
                self.base_f1 = base_perf.get("f1_score", self.base_f1)
            except Exception:
                pass
        except Exception as exc:
            logger.warning("Could not load RLHF state from DB: %s", exc)

    def _load_url_model(self):
        if not (self.enable_url_ensemble and joblib is not None):
            return None
        if not _URL_MODEL_PATH.exists():
            logger.warning("URL model not found at %s", _URL_MODEL_PATH)
            return None
        try:
            model = joblib.load(_URL_MODEL_PATH)
            logger.info("Loaded URL ensemble model from %s", _URL_MODEL_PATH)
            return model
        except Exception as exc:
            logger.warning("Could not load URL model: %s", exc)
            return None

    def estimate_current_accuracy(self) -> float:
        """Blend base accuracy with recent observed accuracy."""
        pm = self.performance_metrics
        blend = self.cfg["decision"]
        if pm.total_predictions > 0:
            recent = pm.correct_predictions / pm.total_predictions
            return (
                blend["accuracy_blend_base"] * self.base_accuracy
                + blend["accuracy_blend_recent"] * recent
            )
        return self.base_accuracy

    def classify_email_rlhf(self, email: Dict[str, str]) -> Tuple[str, float]:
        subject = email.get("subject", "").lower()
        sender = email.get("sender", "").lower()
        body = email.get("body", "").lower()
        text = f"{subject} {body}"

        weights = self.model_weights
        phishing_score = sum(
            w for kw, w in weights["phishing_indicators"].items() if kw in text
        )
        legitimate_score = sum(
            w for kw, w in weights["legitimate_indicators"].items() if kw in text
        )
        domain_adjustment = sum(
            w for domain, w in weights["domain_trust"].items() if domain in sender
        )

        final_phishing = phishing_score + max(0.0, -domain_adjustment)
        final_legitimate = legitimate_score + max(0.0, domain_adjustment)

        if random.random() < self.exploration_rate:
            final_phishing += random.uniform(-0.2, 0.2)
            final_legitimate += random.uniform(-0.2, 0.2)

        prediction, confidence = self._decide(final_phishing, final_legitimate, text)

        # Apply accuracy-based confidence scaling and optional calibration offset
        dec = self.cfg["decision"]
        confidence *= self.estimate_current_accuracy() + dec["accuracy_confidence_offset"]
        if self.metric_calibration_pct > 0:
            confidence -= self.metric_calibration_pct * 100.0
        confidence = max(dec["confidence_min"], min(dec["confidence_max"], confidence))

        self.performance_metrics.total_predictions += 1
        return prediction, confidence

    def _decide(
        self, phishing_score: float, legitimate_score: float, text: str
    ) -> Tuple[str, float]:
        """Translate raw scores into (label, raw_confidence) using config-driven thresholds."""
        dec = self.cfg["decision"]
        high_recall = os.getenv("RLHF_HIGH_RECALL", "0") == "1"

        margin_env = os.getenv("RLHF_DECISION_MARGIN")
        if margin_env is not None:
            try:
                margin = float(margin_env)
            except ValueError:
                margin = dec["margin_high_recall"] if high_recall else dec["margin_default"]
        else:
            margin = dec["margin_high_recall"] if high_recall else dec["margin_default"]

        score_diff = abs(phishing_score - legitimate_score)
        conf_base = dec["confidence_base"]
        conf_scale = dec["confidence_scale"]
        conf_max = dec["confidence_max"]

        if phishing_score > legitimate_score + margin:
            prediction = "phishing"
            confidence = min(conf_max, conf_base + score_diff * conf_scale)
        elif legitimate_score > phishing_score + margin:
            prediction = "legitimate"
            confidence = min(conf_max, conf_base + score_diff * conf_scale)
        else:
            prediction = "phishing" if high_recall else "uncertain"
            confidence = random.randint(
                dec["confidence_uncertain_min"], dec["confidence_uncertain_max"]
            )

        # URL ensemble override
        if (
            self.enable_url_ensemble
            and self.url_model is not None
            and derive_unified_from_url is not None
        ):
            first_url = self._extract_first_url(text)
            if first_url:
                url_proba = self._url_phishing_proba(first_url)
                if url_proba is not None and url_proba >= self.url_threshold:
                    prediction = "phishing"
                    confidence = max(confidence, url_proba * 100.0)

        return prediction, confidence

    @staticmethod
    def _extract_first_url(text: str) -> str:
        match = re.search(r"https?://[^\s'\"]+", text)
        return match.group(0) if match else ""

    def _url_phishing_proba(self, url: str) -> Optional[float]:
        try:
            feats = derive_unified_from_url(url)
            if not feats:
                return None
            url_features = self.cfg["url_features"]
            x = np.array([[float(feats.get(k, 0.0)) for k in url_features]], dtype=float)
            if hasattr(self.url_model, "predict_proba"):
                return float(self.url_model.predict_proba(x)[0][1])
            return 1.0 if int(self.url_model.predict(x)[0]) == 1 else 0.0
        except Exception as exc:
            logger.debug("URL proba failed for %s: %s", url, exc)
            return None
