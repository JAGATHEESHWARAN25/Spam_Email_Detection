import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import os

# Add current directory to path to allow importing preprocessing
sys.path.append(os.getcwd())
from Spam_Email_Detection.preprocessing import load_data, preprocess_text

def train_multinomial():
    print("Loading data...")
    dataset_path = 'Spam_Email_Detection/dataset/sms.tsv'
    df = load_data(dataset_path)
    
    print("Preprocessing text (this may take a moment)...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Feature Extraction (BoW)
    from sklearn.feature_extraction.text import CountVectorizer
    print("Vectorizing (BoW)...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label_num']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training Multinomial NB...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Multinomial NB Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save
    joblib.dump(model, 'Spam_Email_Detection/model_multinomial.pkl')
    joblib.dump(vectorizer, 'Spam_Email_Detection/vectorizer_bow.pkl')
    print("Model and vectorizer saved.")

if __name__ == "__main__":
    train_multinomial()
