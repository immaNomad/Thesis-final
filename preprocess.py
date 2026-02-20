import os
import pandas as pd
import numpy as np
import mindspore.dataset as ds
import email
import re
import mailbox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import arff
from email.parser import Parser
from email.utils import parsedate_to_datetime
from email.header import decode_header, make_header
from html import unescape
import json
try:
    from reprocess_url_datasets import derive_unified_from_url
except Exception:
    derive_unified_from_url = None

# Set paths
DATASET_PATH = os.path.join(os.getcwd(), 'Datasets')
PROCESSED_PATH = os.path.join(os.getcwd(), 'Processed_Data')

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_PATH, exist_ok=True)

def preprocess_phishtank_dataset():
    """
    Preprocess PhishTank CSV dataset
    """
    print("Processing PhishTank dataset...")
    # Load the dataset
    phish_csv_path = os.path.join(DATASET_PATH, 'Phishtank Dataset.csv')
    
    # Try different encodings if one fails
    try:
        phish_df = pd.read_csv(phish_csv_path)
    except UnicodeDecodeError:
        try:
            phish_df = pd.read_csv(phish_csv_path, encoding='latin1')
        except:
            phish_df = pd.read_csv(phish_csv_path, encoding='ISO-8859-1')
    
    # Clean data
    # Drop any rows with missing values
    phish_df = phish_df.dropna()
    
    # Build unified URL features per row
    rows = []
    for _, r in phish_df.iterrows():
        url = str(r.get('url', ''))
        if derive_unified_from_url is not None:
            feats = derive_unified_from_url(url)
        else:
            tokens = re.split(r"[^a-z0-9]+", url.lower())
            shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly']
            suspicious_url = 1 if any(s in url for s in shorteners) else 0
            feats = {
                'email_length': len(url),
                'word_count': max(1, len([t for t in tokens if t])),
                'char_count': len(url),
                'url_count': 1 if url else 0,
                'has_url': 1 if url else 0,
                'suspicious_url': suspicious_url,
                'phishing_keywords': 0,
                'financial_keywords': 0,
                'urgency_words': 0,
                'suspicious_words': 0,
                'has_html': 0,
                'has_script': 0,
                'has_form': 0,
                'has_iframe': 0,
                'sender_suspicious': 0,
                'domain_age': 365,
            }
        feats['label'] = 1
        feats['__dataset'] = 'PhishTank'
        rows.append(feats)
    features_df = pd.DataFrame(rows)
    
    # Save processed dataset
    processed_file = os.path.join(PROCESSED_PATH, 'processed_phishtank.csv')
    features_df.to_csv(processed_file, index=False)
    
    print(f"PhishTank dataset processed and saved to {processed_file}")
    return features_df

def preprocess_enron_emails():
    """
    Preprocess Enron email dataset - IMPROVED VERSION
    Extracts real legitimate business emails for RLHF training
    """
    print("Processing Enron email dataset (improved extraction)...")
    enron_path = os.path.join(DATASET_PATH, "Enron's Email Dataset")
    
    if not os.path.exists(enron_path):
        print(f"Enron dataset not found at {enron_path}")
        return None
    
    # Lists to store email data
    email_data = []
    
    def extract_enron_email(file_path):
        """Extract email content, subject, and sender from Enron email file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split into header and body sections
            if '\n\n' in content:
                header_section, body_section = content.split('\n\n', 1)
            else:
                # If no clear separation, treat as all header
                header_section = content
                body_section = ""
            
            # Extract header information
            subject = ""
            sender = ""
            
            for line in header_section.split('\n'):
                line = line.strip()
                if line.startswith('Subject:'):
                    subject = line.replace('Subject:', '').strip()
                elif line.startswith('From:'):
                    sender = line.replace('From:', '').strip()
            
            # Clean the body content
            body = body_section.strip()
            
            # Remove common email artifacts
            body = re.sub(r'-----Original Message-----.*', '', body, flags=re.DOTALL)
            body = re.sub(r'<< File:.*?>>', '', body)
            body = re.sub(r'X-.*?:', '', body)
            body = re.sub(r'Message-ID:.*', '', body)
            
            # Clean up whitespace and formatting
            body = re.sub(r'\s+', ' ', body)
            body = body.strip()
            
            # Only return if we have meaningful content
            if len(body) > 20 and len(subject) > 0:
                return {
                    'email_text': body,
                    'subject': subject,
                    'sender': sender,
                    'label': 0  # Legitimate emails
                }
            
            return None
            
        except Exception as e:
            return None
    
    # Process emails from multiple employees for diversity
    email_count = 0
    max_emails = 1500  # Target number of legitimate emails
    employees_processed = 0
    max_employees = 20  # Process emails from 20 different employees for diversity
    
    print("Extracting real Enron email content...")
    
    # Get list of employee directories
    employee_dirs = [d for d in os.listdir(enron_path) 
                    if os.path.isdir(os.path.join(enron_path, d))][:max_employees]
    
    for employee in employee_dirs:
        if email_count >= max_emails:
            break
            
        employee_path = os.path.join(enron_path, employee)
        employees_processed += 1
        
        print(f"Processing employee {employees_processed}/{max_employees}: {employee}")
        
        # Check common email folders
        email_folders = ['inbox', 'sent', 'sent_items', '_sent_mail']
        
        for folder in email_folders:
            if email_count >= max_emails:
                break
                
            folder_path = os.path.join(employee_path, folder)
            if os.path.exists(folder_path):
                # Process email files in this folder
                email_files = os.listdir(folder_path)
                
                # Limit emails per folder to ensure diversity
                max_per_folder = min(50, len(email_files))
                
                for i, email_file in enumerate(email_files[:max_per_folder]):
                    if email_count >= max_emails:
                        break
                        
                    # Skip hidden files and directories
                    if email_file.startswith('.'):
                        continue
                        
                    email_path = os.path.join(folder_path, email_file)
                    
                    if os.path.isfile(email_path):
                        email_info = extract_enron_email(email_path)
                        
                        if email_info:
                            email_data.append(email_info)
                            email_count += 1
                            
                            if email_count % 100 == 0:
                                print(f"Extracted {email_count} legitimate emails...")
    
    print(f"Successfully extracted {len(email_data)} real Enron emails!")
    
    if len(email_data) < 10:
        print("ERROR: Still couldn't extract enough emails. Check directory structure.")
        return None
    
    # Create DataFrame with extracted emails
    enron_df = pd.DataFrame(email_data)
    
    # Extract comprehensive features using the same system as Jose Nazario
    features_list = []
    
    for _, row in enron_df.iterrows():
        features = extract_email_features(
            email_text=row['email_text'],
            subject=row['subject'],
            sender=row['sender']
        )
        features['label'] = row['label']  # All legitimate (0)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Ensure we have the same feature set as other datasets
    required_features = [
        'email_length', 'word_count', 'char_count', 'subject_length', 'subject_urgency',
        'url_count', 'has_url', 'suspicious_url', 'email_count', 'has_html',
        'has_script', 'has_form', 'has_iframe', 'phishing_keywords',
        'financial_keywords', 'sender_suspicious', 'urgency_words', 'suspicious_words',
        'link_count', 'image_count', 'attachment_indicators', 'domain_age', 'label', '__dataset'
    ]
    
    # Add missing features with default values if needed
    for feature in required_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
    
    # Reorder columns to match expected format
    features_df = features_df[required_features]
    
    # Save processed dataset
    processed_file = os.path.join(PROCESSED_PATH, 'processed_enron.csv')
    features_df['__dataset'] = 'Enron'
    features_df.to_csv(processed_file, index=False)
    
    print(f"Real Enron dataset processed and saved to {processed_file}")
    print(f"Total legitimate emails: {len(features_df)}")
    print(f"Features extracted: {list(features_df.columns)}")
    
    # Save sample emails for RLHF human feedback (similar to Jose Nazario)
    sample_emails = []
    for i, row in enron_df.head(200).iterrows():  # Save first 200 for human feedback
        sample_emails.append({
            'index': i,
            'subject': row['subject'],
            'sender': row['sender'],
            'content': row['email_text'][:500],  # First 500 characters
            'full_content': row['email_text'],
            'label': 0,  # Legitimate
            'source': 'enron'
        })
    
    # Save for RLHF human feedback
    feedback_file = os.path.join(PROCESSED_PATH, 'enron_emails_for_feedback.json')
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(sample_emails, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(sample_emails)} sample emails for RLHF feedback: {feedback_file}")
    
    return features_df

def preprocess_uci_dataset():
    """
    Preprocess UCI phishing dataset (ARFF format)
    """
    print("Processing UCI dataset...")
    
    # Look for ARFF file in main Datasets directory first
    arff_file = None
    for file in os.listdir(DATASET_PATH):
        if file.endswith('.arff'):
            arff_file = os.path.join(DATASET_PATH, file)
            break
    
    # If not found, try UCI subdirectory
    if not arff_file:
        uci_arff_path = os.path.join(DATASET_PATH, 'UCI')
        if os.path.exists(uci_arff_path):
            for file in os.listdir(uci_arff_path):
                if file.endswith('.arff'):
                    arff_file = os.path.join(uci_arff_path, file)
                    break
    
    if not arff_file:
        print("No ARFF file found in Datasets directory or UCI subdirectory")
        return None
    
    # Load ARFF file
    data, meta = arff.loadarff(arff_file)
    
    # Convert to DataFrame
    uci_df = pd.DataFrame(data)
    
    # Convert bytes to strings and handle labels
    for col in uci_df.columns:
        if uci_df[col].dtype == 'object':
            uci_df[col] = uci_df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    # Map labels to numeric values
    label_mapping = {'1': 1, '-1': 0}  # Assuming 1 = phishing, -1 = legitimate
    uci_df['label'] = uci_df['Result'].map(label_mapping)
    
    # Map UCI indicators into unified URL/email feature schema
    col = {c.lower(): c for c in uci_df.columns}
    def get_series(name, default=0):
        return uci_df[col[name]] if name in col else pd.Series([default]*len(uci_df))

    shortening = get_series('shortining_service', 0)
    https_token = get_series('https_token', 0)
    ip_addr = get_series('having_ip_address', 0)
    prefix_suffix = get_series('prefix_suffix', 0)
    suspicious_url = ((shortening == 1) | (https_token == 1) | (ip_addr == 1) | (prefix_suffix == 1)).astype(int)

    request_url = get_series('request_url', 0)
    url_of_anchor = get_series('url_of_anchor', 0)
    sfh = get_series('sfh', 0)
    iframe = get_series('iframe', 0)
    age = get_series('age_of_domain', 365)
    try:
        domain_age = age.astype(int)
    except Exception:
        domain_age = pd.Series([365]*len(uci_df))

    unified = pd.DataFrame({
        'email_length': 0,
        'word_count': 0,
        'char_count': 0,
        'url_count': 1,
        'has_url': 1,
        'suspicious_url': suspicious_url,
        'phishing_keywords': 0,
        'financial_keywords': 0,
        'urgency_words': 0,
        'suspicious_words': 0,
        'has_html': ((request_url != 0) | (url_of_anchor != 0)).astype(int),
        'has_script': 0,
        'has_form': (sfh != 0).astype(int),
        'has_iframe': iframe.astype(int),
        'sender_suspicious': 0,
        'domain_age': domain_age,
        'label': uci_df['label'].astype(int),
        '__dataset': 'UCI'
    })

    processed_file = os.path.join(PROCESSED_PATH, 'processed_uci.csv')
    unified.to_csv(processed_file, index=False)
    
    print(f"UCI dataset processed and saved to {processed_file}")
    return unified

def extract_email_features(email_text, subject="", sender=""):
    """
    Extract comprehensive features from email content for RLHF system
    """
    features = {}
    
    # Basic text features
    clean_text = email_text or ""
    features['email_length'] = len(clean_text)
    features['word_count'] = len(clean_text.split()) if clean_text else 0
    features['char_count'] = len(clean_text)
    
    # Subject features
    features['subject_length'] = len(subject) if subject else 0
    features['subject_urgency'] = 1 if any(word in (subject or '').lower() for word in 
                                          ['urgent', 'immediate', 'asap', 'deadline', 'expire']) else 0
    
    # URL features
    url_pattern = r'https?://[^\s\"]+'
    lower_text = (clean_text or '').lower()
    urls = re.findall(url_pattern, lower_text)
    href_urls = re.findall(r'href=["\'](.*?)["\']', lower_text)
    src_urls = re.findall(r'src=["\'](.*?)["\']', lower_text)
    all_urls = urls + href_urls + src_urls
    features['url_count'] = len(all_urls)
    features['has_url'] = 1 if all_urls else 0
    shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'buff.ly']
    redirect_tokens = ['redirect', 'r=', 'u=']
    features['suspicious_url'] = 1 if any(any(s in u for s in shorteners) or any(t in u for t in redirect_tokens) for u in all_urls) else 0
    
    # Email addresses
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    emails = re.findall(email_pattern, lower_text)
    features['email_count'] = len(emails)
    
    # HTML features
    features['has_html'] = 1 if ('<html' in lower_text or '<body' in lower_text) else 0
    features['has_script'] = 1 if '<script' in lower_text else 0
    features['has_form'] = 1 if '<form' in lower_text else 0
    features['has_iframe'] = 1 if '<iframe' in lower_text else 0
    features['link_count'] = len(re.findall(r'<a\s', lower_text)) + len(urls)
    features['image_count'] = len(re.findall(r'<img\s', lower_text))
    
    # Suspicious content
    phishing_keywords = [
        'verify', 'suspend', 'update', 'confirm', 'click here', 'download', 'login', 'reset',
        'restricted', 'unusual activity', 'security alert', 'account locked', 'reactivate'
    ]
    features['phishing_keywords'] = sum(1 for word in phishing_keywords if word in lower_text)
    
    # Financial indicators
    financial_keywords = ['bank', 'account', 'credit card', 'payment', 'transaction', 'refund', 'billing', 'invoice']
    features['financial_keywords'] = sum(1 for word in financial_keywords if word in lower_text)
    urgency_terms = ['urgent', 'immediately', 'asap', 'expire', 'last chance']
    suspicious_terms = ['free', 'win', 'prize', 'bonus', 'limited offer']
    features['urgency_words'] = sum(1 for w in urgency_terms if w in lower_text or w in (subject or '').lower())
    features['suspicious_words'] = sum(1 for w in suspicious_terms if w in lower_text)
    features['attachment_indicators'] = features.get('attachment_indicators', 0)
    features['domain_age'] = features.get('domain_age', 365)
    
    # Sender features
    features['sender_suspicious'] = 1 if sender and any(word in sender.lower() for word in 
                                                       ['noreply', 'no-reply', 'admin', 'support']) else 0
    
    return features

def preprocess_jose_nazario_corpus():
    """
    Preprocess Jose Nazario phishing corpus with RLHF integration
    """
    print("Processing Jose Nazario phishing corpus...")
    nazario_path = os.path.join(DATASET_PATH, 'jose_nazario_corpus')
    
    if not os.path.exists(nazario_path):
        print(f"Jose Nazario corpus not found at {nazario_path}")
        return None
    
    emails_data = []
    email_contents = []  # Store full email content for human feedback
    
    # Get all files in the corpus
    corpus_files = [f for f in os.listdir(nazario_path) if f != 'README.txt']
    
    print(f"Found {len(corpus_files)} corpus files")
    
    for file_name in corpus_files:
        file_path = os.path.join(nazario_path, file_name)
        print(f"Processing {file_name}...")
        
        try:
            if file_name.endswith('.mbox'):
                # Process mbox files
                mbox = mailbox.mbox(file_path)
                for message in mbox:
                    email_data = process_email_message(message, file_name)
                    if email_data:
                        emails_data.append(email_data)
                        # Store full email for human feedback
                        try:
                            subj = str(make_header(decode_header(message.get('Subject', '')))) if message.get('Subject') else ''
                        except Exception:
                            subj = message.get('Subject', '')
                        try:
                            sndr = str(make_header(decode_header(message.get('From', '')))) if message.get('From') else ''
                        except Exception:
                            sndr = message.get('From', '')
                        email_contents.append({
                            'id': len(email_contents),
                            'subject': subj,
                            'sender': sndr,
                            'content': str(message.get_payload()),
                            'source': file_name
                        })
            else:
                # Process other formats (assuming they're also email files)
                with open(file_path, 'r', encoding='latin1', errors='ignore') as f:
                    content = f.read()
                    # Try to parse as email
                    try:
                        parser = Parser()
                        message = parser.parsestr(content)
                        email_data = process_email_message(message, file_name)
                        if email_data:
                            emails_data.append(email_data)
                            try:
                                subj = str(make_header(decode_header(message.get('Subject', '')))) if message.get('Subject') else ''
                            except Exception:
                                subj = message.get('Subject', '')
                            try:
                                sndr = str(make_header(decode_header(message.get('From', '')))) if message.get('From') else ''
                            except Exception:
                                sndr = message.get('From', '')
                            email_contents.append({
                                'id': len(email_contents),
                                'subject': subj,
                                'sender': sndr,
                                'content': str(message.get_payload()),
                                'source': file_name
                            })
                    except Exception as e:
                        print(f"Could not parse {file_name}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    if not emails_data:
        print("No emails could be processed from Jose Nazario corpus")
        return None
    
    print(f"Successfully processed {len(emails_data)} emails")
    
    # Create DataFrame
    nazario_df = pd.DataFrame(emails_data)
    
    # Save full email content for RLHF human feedback
    human_feedback_file = os.path.join(PROCESSED_PATH, 'nazario_emails_for_feedback.json')
    with open(human_feedback_file, 'w', encoding='utf-8') as f:
        json.dump(email_contents, f, indent=2, ensure_ascii=False)
    
    print(f"Email content for human feedback saved to {human_feedback_file}")
    
    # Save processed features
    processed_file = os.path.join(PROCESSED_PATH, 'processed_nazario.csv')
    nazario_df.to_csv(processed_file, index=False)
    
    print(f"Jose Nazario corpus processed and saved to {processed_file}")
    return nazario_df

def process_email_message(message, source_file):
    """
    Process a single email message and extract features
    """
    try:
        # Get email components and decode MIME headers safely
        raw_subject = message.get('Subject', '')
        raw_sender = message.get('From', '')
        try:
            subject = str(make_header(decode_header(raw_subject))) if raw_subject else ''
        except Exception:
            subject = str(raw_subject) if raw_subject else ''
        try:
            sender = str(make_header(decode_header(raw_sender))) if raw_sender else ''
        except Exception:
            sender = str(raw_sender) if raw_sender else ''
        
        # Get email body
        attachment_count = 0
        if message.is_multipart():
            body = ""
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            body += payload.decode('utf-8', errors='ignore')
                        except:
                            body += str(payload)
                elif part.get_content_type() == "text/html" and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            body = payload.decode('utf-8', errors='ignore')
                        except:
                            body = str(payload)
                cd = part.get('Content-Disposition', '') or ''
                if 'attachment' in cd.lower():
                    attachment_count += 1
        else:
            payload = message.get_payload(decode=True)
            if payload:
                try:
                    body = payload.decode('utf-8', errors='ignore')
                except:
                    body = str(payload)
            else:
                body = str(message.get_payload())
        
        # Clean body text
        body = re.sub(r'\s+', ' ', body).strip()
        
        if len(body) < 10:  # Skip very short emails
            return None
        
        # Extract features
        features = extract_email_features(body, subject, sender)
        features['attachment_indicators'] = attachment_count
        
        # Add metadata
        features['source_file'] = source_file
        features['subject'] = subject[:100]  # Truncate for storage
        features['sender'] = sender[:100]    # Truncate for storage
        
        # All emails in Jose Nazario corpus are phishing
        features['label'] = 1
        features['__dataset'] = 'Nazario'
        
        return features
        
    except Exception as e:
        print(f"Error processing email: {e}")
        return None

def create_mindspore_dataset(features, labels, batch_size=32):
    """
    Create MindSpore dataset from features and labels
    """
    # Convert to numpy arrays with proper data types
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    # Create dataset
    dataset = ds.NumpySlicesDataset({"features": features, "labels": labels}, shuffle=True)
    dataset = dataset.batch(batch_size)
    
    return dataset

def combine_datasets():
    """
    Combine all processed datasets for training
    """
    datasets_dict = {}
    scaler = StandardScaler()
    
    # PhishTank dataset
    if os.path.exists(os.path.join(PROCESSED_PATH, 'processed_phishtank.csv')):
        phish_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'processed_phishtank.csv'))
        
        # Split features and labels
        phish_X = phish_df.drop('label', axis=1)
        phish_y = phish_df['label']
        phish_X_train, phish_X_test, phish_y_train, phish_y_test = train_test_split(
            phish_X, phish_y, test_size=0.2, random_state=42
        )
        
        # Normalize features
        phish_X_train = scaler.fit_transform(phish_X_train)
        phish_X_test = scaler.transform(phish_X_test)
        
        # Create MindSpore datasets
        phish_train_dataset = create_mindspore_dataset(phish_X_train, phish_y_train)
        phish_test_dataset = create_mindspore_dataset(phish_X_test, phish_y_test)
        
        datasets_dict['phishtank'] = (phish_train_dataset, phish_test_dataset)
    
    # Enron dataset
    if os.path.exists(os.path.join(PROCESSED_PATH, 'processed_enron.csv')):
        enron_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'processed_enron.csv'))
        
        enron_X = enron_df.drop('label', axis=1)
        enron_y = enron_df['label']
        enron_X_train, enron_X_test, enron_y_train, enron_y_test = train_test_split(
            enron_X, enron_y, test_size=0.2, random_state=42
        )
        
        # Normalize features
        enron_X_train = scaler.fit_transform(enron_X_train)
        enron_X_test = scaler.transform(enron_X_test)
        
        # Create MindSpore datasets
        enron_train_dataset = create_mindspore_dataset(enron_X_train, enron_y_train)
        enron_test_dataset = create_mindspore_dataset(enron_X_test, enron_y_test)
        
        datasets_dict['enron'] = (enron_train_dataset, enron_test_dataset)
    
    # UCI dataset
    if os.path.exists(os.path.join(PROCESSED_PATH, 'processed_uci.csv')):
        uci_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'processed_uci.csv'))
        
        uci_X = uci_df.drop('label', axis=1)
        uci_y = uci_df['label']
        uci_X_train, uci_X_test, uci_y_train, uci_y_test = train_test_split(
            uci_X, uci_y, test_size=0.2, random_state=42
        )
        
        # Normalize features
        uci_X_train = scaler.fit_transform(uci_X_train)
        uci_X_test = scaler.transform(uci_X_test)
        
        # Create MindSpore datasets
        uci_train_dataset = create_mindspore_dataset(uci_X_train, uci_y_train)
        uci_test_dataset = create_mindspore_dataset(uci_X_test, uci_y_test)
        
        datasets_dict['uci'] = (uci_train_dataset, uci_test_dataset)
    
    # Jose Nazario corpus (NEW)
    if os.path.exists(os.path.join(PROCESSED_PATH, 'processed_nazario.csv')):
        nazario_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'processed_nazario.csv'))
        
        # Remove non-numeric columns for training
        numeric_columns = nazario_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in nazario_df.columns:
            numeric_columns = [col for col in numeric_columns if col != 'label']
        
        nazario_X = nazario_df[numeric_columns]
        nazario_y = nazario_df['label']
        nazario_X_train, nazario_X_test, nazario_y_train, nazario_y_test = train_test_split(
            nazario_X, nazario_y, test_size=0.2, random_state=42
        )
        
        # Normalize features
        nazario_X_train = scaler.fit_transform(nazario_X_train)
        nazario_X_test = scaler.transform(nazario_X_test)
        
        # Create MindSpore datasets
        nazario_train_dataset = create_mindspore_dataset(nazario_X_train, nazario_y_train)
        nazario_test_dataset = create_mindspore_dataset(nazario_X_test, nazario_y_test)
        
        datasets_dict['nazario'] = (nazario_train_dataset, nazario_test_dataset)
    
    # SpamAssassin Ham corpus (NEW)
    if os.path.exists(os.path.join(PROCESSED_PATH, 'processed_spamassassin.csv')):
        spamassassin_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'processed_spamassassin.csv'))
        
        # Remove non-numeric columns for training
        numeric_columns = spamassassin_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in spamassassin_df.columns:
            numeric_columns = [col for col in numeric_columns if col != 'label']
        
        spamassassin_X = spamassassin_df[numeric_columns]
        spamassassin_y = spamassassin_df['label']
        spamassassin_X_train, spamassassin_X_test, spamassassin_y_train, spamassassin_y_test = train_test_split(
            spamassassin_X, spamassassin_y, test_size=0.2, random_state=42
        )
        
        # Normalize features
        spamassassin_X_train = scaler.fit_transform(spamassassin_X_train)
        spamassassin_X_test = scaler.transform(spamassassin_X_test)
        
        # Create MindSpore datasets
        spamassassin_train_dataset = create_mindspore_dataset(spamassassin_X_train, spamassassin_y_train)
        spamassassin_test_dataset = create_mindspore_dataset(spamassassin_X_test, spamassassin_y_test)
        
        datasets_dict['spamassassin'] = (spamassassin_train_dataset, spamassassin_test_dataset)
    
    # Ling-Spam corpus (NEW)
    if os.path.exists(os.path.join(PROCESSED_PATH, 'processed_lingspam.csv')):
        lingspam_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'processed_lingspam.csv'))
        
        # Remove non-numeric columns for training
        numeric_columns = lingspam_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in lingspam_df.columns:
            numeric_columns = [col for col in numeric_columns if col != 'label']
        
        lingspam_X = lingspam_df[numeric_columns]
        lingspam_y = lingspam_df['label']
        lingspam_X_train, lingspam_X_test, lingspam_y_train, lingspam_y_test = train_test_split(
            lingspam_X, lingspam_y, test_size=0.2, random_state=42
        )
        
        # Normalize features
        lingspam_X_train = scaler.fit_transform(lingspam_X_train)
        lingspam_X_test = scaler.transform(lingspam_X_test)
        
        # Create MindSpore datasets
        lingspam_train_dataset = create_mindspore_dataset(lingspam_X_train, lingspam_y_train)
        lingspam_test_dataset = create_mindspore_dataset(lingspam_X_test, lingspam_y_test)
        
        datasets_dict['lingspam'] = (lingspam_train_dataset, lingspam_test_dataset)
    
    return datasets_dict

def preprocess_spamassassin_ham():
    """
    Preprocess SpamAssassin Ham Corpus - legitimate emails
    Adds more legitimate email diversity to balance the dataset
    """
    print("Processing SpamAssassin Ham Corpus...")
    ham_paths = [
        os.path.join(DATASET_PATH, 'spamassassin_ham', 'easy_ham'),
        os.path.join(DATASET_PATH, 'spamassassin_ham', 'easy_ham_2')
    ]
    
    email_data = []
    
    def extract_spamassassin_email(file_path):
        """Extract email content from SpamAssassin ham file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse email using email module
            msg = email.message_from_string(content)
            
            # Extract headers
            subject = msg.get('Subject', '').strip()
            sender = msg.get('From', '').strip()
            
            # Extract body content
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            try:
                                body += payload.decode('utf-8', errors='ignore')
                            except:
                                body += str(payload)
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    try:
                        body = payload.decode('utf-8', errors='ignore')
                    except:
                        body = str(payload)
            
            # Clean the body content
            body = re.sub(r'\n+', ' ', body)  # Replace newlines with spaces
            body = re.sub(r'\s+', ' ', body)  # Normalize whitespace
            body = body.strip()
            
            # Only return if we have meaningful content
            if len(body) > 20 and len(subject) > 0:
                return {
                    'email_text': body,
                    'subject': subject,
                    'sender': sender,
                    'label': 0  # Legitimate emails
                }
            
            return None
            
        except Exception as e:
            return None
    
    # Process all email files from both directories
    email_count = 0
    max_emails = 4000  # Increased limit for more legitimate emails
    
    print("Extracting SpamAssassin legitimate emails from multiple directories...")
    
    for ham_path in ham_paths:
        if not os.path.exists(ham_path):
            print(f"Ham corpus not found at {ham_path}, skipping...")
            continue
            
        print(f"Processing directory: {os.path.basename(ham_path)}")
        
        # Get all email files in the directory
        email_files = [f for f in os.listdir(ham_path) 
                       if os.path.isfile(os.path.join(ham_path, f)) and not f.startswith('.') and f != 'cmds']
        
        for email_file in email_files:
            if email_count >= max_emails:
                break
                
            email_path = os.path.join(ham_path, email_file)
            email_info = extract_spamassassin_email(email_path)
            
            if email_info:
                email_data.append(email_info)
                email_count += 1
                
                if email_count % 500 == 0:
                    print(f"Extracted {email_count} SpamAssassin legitimate emails...")
        
        if email_count >= max_emails:
            break
    
    print(f"Successfully extracted {len(email_data)} SpamAssassin legitimate emails!")
    
    if len(email_data) < 10:
        print("ERROR: Not enough SpamAssassin emails extracted.")
        return None
    
    # Create DataFrame with extracted emails
    spam_df = pd.DataFrame(email_data)
    
    # Extract comprehensive features using the same system as other datasets
    features_list = []
    
    for _, row in spam_df.iterrows():
        features = extract_email_features(
            email_text=row['email_text'],
            subject=row['subject'],
            sender=row['sender']
        )
        features['label'] = row['label']  # All legitimate (0)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Ensure we have the same feature set as other datasets
    required_features = [
        'email_length', 'word_count', 'char_count', 'subject_length', 'subject_urgency',
        'url_count', 'has_url', 'suspicious_url', 'email_count', 'has_html',
        'has_script', 'has_form', 'has_iframe', 'phishing_keywords',
        'financial_keywords', 'sender_suspicious', 'label'
    ]
    
    # Add missing features with default values if needed
    for feature in required_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
    
    # Reorder columns to match expected format
    features_df = features_df[required_features]
    
    # Save processed dataset
    processed_file = os.path.join(PROCESSED_PATH, 'processed_spamassassin.csv')
    features_df['__dataset'] = 'SpamAssassin'
    features_df.to_csv(processed_file, index=False)
    
    print(f"SpamAssassin Ham dataset processed and saved to {processed_file}")
    print(f"Total legitimate emails: {len(features_df)}")
    
    # Save sample emails for RLHF human feedback
    sample_emails = []
    for i, row in spam_df.head(300).iterrows():  # Save first 300 for human feedback
        sample_emails.append({
            'index': i,
            'subject': row['subject'],
            'sender': row['sender'],
            'content': row['email_text'][:500],  # First 500 characters
            'full_content': row['email_text'],
            'label': 0,  # Legitimate
            'source': 'spamassassin'
        })
    
    # Save for RLHF human feedback
    feedback_file = os.path.join(PROCESSED_PATH, 'spamassassin_emails_for_feedback.json')
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(sample_emails, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(sample_emails)} sample emails for RLHF feedback: {feedback_file}")
    
    return features_df

def preprocess_lingspam_corpus():
    """
    Preprocess Ling-Spam Corpus - academic legitimate emails
    From Athens University of Economics and Business
    """
    print("Processing Ling-Spam Corpus...")
    corpus_path = os.path.join(DATASET_PATH, 'ling_spam', 'lingspam_public', 'bare')
    
    if not os.path.exists(corpus_path):
        print(f"Ling-Spam corpus not found at {corpus_path}")
        return None
    
    email_data = []
    
    def extract_lingspam_email(file_path):
        """Extract email content from Ling-Spam file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None
                
            # First line is usually "Subject: ..."
            subject = lines[0].strip()
            if subject.startswith('Subject: '):
                subject = subject[9:]  # Remove "Subject: " prefix
            
            # Rest is the email body
            body = ' '.join(lines[1:]).strip()
            
            # Clean the content
            body = re.sub(r'\s+', ' ', body)  # Normalize whitespace
            body = body.strip()
            
            # Extract sender info (usually not available in this format)
            sender = "linguist-list@example.com"  # Default for academic list
            
            # Only return if we have meaningful content
            if len(body) > 30 and len(subject) > 0:
                return {
                    'email_text': body,
                    'subject': subject,
                    'sender': sender,
                    'label': 0  # Legitimate emails
                }
            
            return None
            
        except Exception as e:
            return None
    
    # Process all parts (part1 to part10)
    email_count = 0
    max_emails = 2000  # Limit for balance
    
    print("Extracting Ling-Spam legitimate emails...")
    
    # Iterate through all parts
    for part_num in range(1, 11):  # part1 to part10
        part_dir = os.path.join(corpus_path, f'part{part_num}')
        
        if not os.path.exists(part_dir):
            continue
            
        print(f"Processing part{part_num}...")
        
        # Get all files in this part
        email_files = [f for f in os.listdir(part_dir) 
                       if f.endswith('.txt') and not f.startswith('spmsga')]  # Skip spam files
        
        for email_file in email_files:
            if email_count >= max_emails:
                break
                
            email_path = os.path.join(part_dir, email_file)
            email_info = extract_lingspam_email(email_path)
            
            if email_info:
                email_data.append(email_info)
                email_count += 1
                
                if email_count % 200 == 0:
                    print(f"Extracted {email_count} Ling-Spam legitimate emails...")
        
        if email_count >= max_emails:
            break
    
    print(f"Successfully extracted {len(email_data)} Ling-Spam legitimate emails!")
    
    if len(email_data) < 10:
        print("ERROR: Not enough Ling-Spam emails extracted.")
        return None
    
    # Create DataFrame with extracted emails
    lingspam_df = pd.DataFrame(email_data)
    
    # Extract comprehensive features using the same system as other datasets
    features_list = []
    
    for _, row in lingspam_df.iterrows():
        features = extract_email_features(
            email_text=row['email_text'],
            subject=row['subject'],
            sender=row['sender']
        )
        features['label'] = row['label']  # All legitimate (0)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Ensure we have the same feature set as other datasets
    required_features = [
        'email_length', 'word_count', 'char_count', 'subject_length', 'subject_urgency',
        'url_count', 'has_url', 'suspicious_url', 'email_count', 'has_html',
        'has_script', 'has_form', 'has_iframe', 'phishing_keywords',
        'financial_keywords', 'sender_suspicious', 'urgency_words', 'suspicious_words',
        'link_count', 'image_count', 'attachment_indicators', 'domain_age', 'label', '__dataset'
    ]
    
    # Add missing features with default values if needed
    for feature in required_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
    
    # Reorder columns to match expected format
    features_df = features_df[required_features]
    
    # Save processed dataset
    processed_file = os.path.join(PROCESSED_PATH, 'processed_lingspam.csv')
    features_df['__dataset'] = 'LingSpam'
    features_df.to_csv(processed_file, index=False)
    
    print(f"Ling-Spam dataset processed and saved to {processed_file}")
    print(f"Total legitimate emails: {len(features_df)}")
    
    # Save sample emails for RLHF human feedback
    sample_emails = []
    for i, row in lingspam_df.head(200).iterrows():  # Save first 200 for human feedback
        sample_emails.append({
            'index': i,
            'subject': row['subject'],
            'sender': row['sender'],
            'content': row['email_text'][:500],  # First 500 characters
            'full_content': row['email_text'],
            'label': 0,  # Legitimate
            'source': 'lingspam'
        })
    
    # Save for RLHF human feedback
    feedback_file = os.path.join(PROCESSED_PATH, 'lingspam_emails_for_feedback.json')
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(sample_emails, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(sample_emails)} sample emails for RLHF feedback: {feedback_file}")
    
    return features_df

def main():
    """
    Main preprocessing function
    """
    print("Starting preprocessing pipeline...")
    
    # Process each dataset
    preprocess_phishtank_dataset()
    preprocess_enron_emails()
    preprocess_uci_dataset()
    preprocess_jose_nazario_corpus()  # NEW: Process Jose Nazario corpus
    preprocess_spamassassin_ham()  # NEW: Process SpamAssassin Ham Corpus
    preprocess_lingspam_corpus()  # NEW: Process Ling-Spam Corpus
    
    print("\nAll datasets processed successfully!")
    print("Files created:")
    for file in os.listdir(PROCESSED_PATH):
        print(f"  - {file}")

if __name__ == "__main__":
    main() 