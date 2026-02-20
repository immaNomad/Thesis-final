#!/usr/bin/env python3

import os
import json
import random
import sqlite3
from typing import Dict, Tuple
import re
import numpy as np
try:
    import joblib
except Exception:
    joblib = None
try:
    from reprocess_url_datasets import derive_unified_from_url
except Exception:
    derive_unified_from_url = None


class RLHFClassifierService:
    def __init__(self, db_path: str = "/home/mark/Desktop/PhishingDetection/rlhf_phishing_detection.db", metric_calibration_pct: float = 0.0):
        self.db_path = db_path
        # Defaults
        self.model_version = 1
        self.total_updates = 0
        self.base_accuracy = 0.85
        self.base_precision = 0.80
        self.base_recall = 0.85
        self.base_f1 = 0.82
        self.model_weights = {
            'phishing_indicators': {'urgent': 0.7, 'click': 0.8, 'verify': 0.6},
            'legitimate_indicators': {'meeting': 0.7, 'newsletter': 0.6, 'university': 0.8},
            'domain_trust': {'.edu': 0.8, '.com': 0.3, 'suspicious': -0.7}
        }
        self.exploration_rate = 0.05
        self.performance_metrics = {
            'accuracy_history': [self.base_accuracy],
            'precision_history': [self.base_precision],
            'recall_history': [self.base_recall],
            'f1_history': [self.base_f1],
            'feedback_count': 0,
            'correct_predictions': 0,
            'total_predictions': 0
        }
        try:
            self.metric_calibration_pct = max(0.0, min(0.2, float(metric_calibration_pct)))
        except Exception:
            self.metric_calibration_pct = 0.0
        # Load model and state
        self._load_trained_model()
        self._load_rlhf_state()
        # URL ensemble configuration
        self.enable_url_ensemble = os.getenv('RLHF_ENSEMBLE_URL', '0') == '1'
        try:
            self.url_threshold = float(os.getenv('RLHF_URL_THRESHOLD', '0.6'))
        except Exception:
            self.url_threshold = 0.6
        self.url_model = None
        if self.enable_url_ensemble and joblib is not None:
            try:
                model_path = "/home/mark/Desktop/PhishingDetection/models/url_only_model.pkl"
                if os.path.exists(model_path):
                    self.url_model = joblib.load(model_path)
            except Exception:
                self.url_model = None

    def _load_trained_model(self):
        try:
            training_files = [
                "/home/mark/Desktop/PhishingDetection/real_email_training_results_20250828_191221.json",
                "/home/mark/Desktop/PhishingDetection/real_email_training_results_20250828_171056.json"
            ]
            model_data = None
            for filepath in training_files:
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        model_data = json.load(f)
                    break
            if model_data:
                self.base_accuracy = model_data.get('metrics', {}).get('accuracy', self.base_accuracy)
                self.base_precision = model_data.get('metrics', {}).get('precision', self.base_precision)
                self.base_recall = model_data.get('metrics', {}).get('recall', self.base_recall)
                self.base_f1 = model_data.get('metrics', {}).get('f1_score', self.base_f1)
        except Exception:
            pass

    def _load_rlhf_state(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM rlhf_state WHERE id = 1')
            row = cursor.fetchone()
            conn.close()
            if row:
                self.model_version = row[1]
                self.total_updates = row[2]
                try:
                    self.model_weights = json.loads(row[3])
                except Exception:
                    pass
                try:
                    self.performance_metrics = json.loads(row[4])
                except Exception:
                    pass
                try:
                    base_perf = json.loads(row[6])
                    self.base_accuracy = base_perf.get('accuracy', self.base_accuracy)
                    self.base_precision = base_perf.get('precision', self.base_precision)
                    self.base_recall = base_perf.get('recall', self.base_recall)
                    self.base_f1 = base_perf.get('f1_score', self.base_f1)
                except Exception:
                    pass
        except Exception:
            pass

    def estimate_current_accuracy(self) -> float:
        if self.performance_metrics.get('total_predictions', 0) > 0:
            recent_accuracy = self.performance_metrics['correct_predictions'] / max(1, self.performance_metrics['total_predictions'])
            return 0.7 * self.base_accuracy + 0.3 * recent_accuracy
        return self.base_accuracy

    def classify_email_rlhf(self, email: Dict[str, str]) -> Tuple[str, float]:
        subject = email.get('subject', '').lower()
        sender = email.get('sender', '').lower()
        body = email.get('body', '').lower()
        text = subject + ' ' + body

        phishing_score = 0
        phishing_features = 0
        for indicator, weight in self.model_weights['phishing_indicators'].items():
            if indicator in text:
                phishing_score += weight
                phishing_features += 1

        legitimate_score = 0
        legitimate_features = 0
        for indicator, weight in self.model_weights['legitimate_indicators'].items():
            if indicator in text:
                legitimate_score += weight
                legitimate_features += 1

        domain_adjustment = 0
        for domain, weight in self.model_weights['domain_trust'].items():
            if domain in sender:
                domain_adjustment += weight

        final_phishing = phishing_score + max(0, -domain_adjustment)
        final_legitimate = legitimate_score + max(0, domain_adjustment)

        if random.random() < self.exploration_rate:
            final_phishing += random.uniform(-0.2, 0.2)
            final_legitimate += random.uniform(-0.2, 0.2)

        # Decision margin and recall bias controls
        high_recall = os.getenv('RLHF_HIGH_RECALL', '0') == '1'
        try:
            margin_env = float(os.getenv('RLHF_DECISION_MARGIN', '0.1' if high_recall else '0.2'))
        except Exception:
            margin_env = 0.2

        score_diff = abs(final_phishing - final_legitimate)
        if final_phishing > final_legitimate + margin_env:
            prediction = 'phishing'
            confidence = min(98, 65 + score_diff * 25)
        elif final_legitimate > final_phishing + margin_env:
            prediction = 'legitimate'
            confidence = min(98, 65 + score_diff * 25)
        else:
            # For high-recall operation, prefer classifying uncertain as phishing
            prediction = 'phishing' if high_recall else 'uncertain'
            confidence = random.randint(30, 65)

        # URL ensemble override if enabled
        if self.enable_url_ensemble and self.url_model is not None and derive_unified_from_url is not None:
            first_url = self._extract_first_url(text)
            if first_url:
                proba = self._url_phishing_proba(first_url)
                if proba is not None and proba >= self.url_threshold:
                    prediction = 'phishing'
                    confidence = max(confidence, float(proba) * 100.0)

        confidence = confidence * (self.estimate_current_accuracy() + 0.05)
        if self.metric_calibration_pct > 0:
            confidence = confidence - (self.metric_calibration_pct * 100.0)
        confidence = max(25, min(98, confidence))

        self.performance_metrics['total_predictions'] = self.performance_metrics.get('total_predictions', 0) + 1
        return prediction, confidence

    # --- URL ensemble helpers ---
    def _extract_first_url(self, text: str) -> str:
        try:
            m = re.search(r"https?://[^\s'\"]+", text)
            return m.group(0) if m else ''
        except Exception:
            return ''

    def _url_phishing_proba(self, url: str) -> float:
        try:
            feats = derive_unified_from_url(url) if derive_unified_from_url is not None else None
            if not feats:
                return None
            url_features = [
                'email_length','word_count','char_count','url_count','has_url','suspicious_url',
                'phishing_keywords','financial_keywords','urgency_words','suspicious_words',
                'has_html','has_script','has_form','has_iframe','sender_suspicious','domain_age',
                'path_depth','digits_ratio','hex_ratio','query_count','entropy','punycode','tld_risky','ip_in_host','brand_distance'
            ]
            x = np.array([[float(feats.get(k, 0.0)) for k in url_features]], dtype=float)
            if hasattr(self.url_model, 'predict_proba'):
                return float(self.url_model.predict_proba(x)[0][1])
            pred = int(self.url_model.predict(x)[0])
            return 1.0 if pred == 1 else 0.0
        except Exception:
            return None






