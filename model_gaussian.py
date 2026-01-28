import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())
from Spam_Email_Detection.preprocessing import load_data, preprocess_text

def train_gaussian():
    print("Loading data...")
    dataset_path = 'Spam_Email_Detection/dataset/sms.tsv'
    df = load_data(dataset_path)
    
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label_num']
    
    # Convert to dense for GaussianNB
    print("Converting to dense array...")
    X_dense = X.toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size=0.2, random_state=42)
    
    print("Training Gaussian NB...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Gaussian NB Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, 'Spam_Email_Detection/model_gaussian.pkl')
    joblib.dump(vectorizer, 'Spam_Email_Detection/vectorizer_tfidf.pkl')
    print("Model and vectorizer saved.")

if __name__ == "__main__":
    train_gaussian()
