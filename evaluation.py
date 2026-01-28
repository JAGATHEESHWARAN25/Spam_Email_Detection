import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import sys
import os

sys.path.append(os.getcwd())
from Spam_Email_Detection.preprocessing import load_data, preprocess_text
from Spam_Email_Detection.visualization import plot_confusion_matrix, plot_roc_curve

def evaluate_models():
    print("Loading data for evaluation...")
    dataset_path = 'Spam_Email_Detection/dataset/sms.tsv'
    df = load_data(dataset_path)
    print("Preprocessing...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    y = df['label_num']

    print("Loading models...")
    mnb = joblib.load('Spam_Email_Detection/model_multinomial.pkl')
    vec_bow = joblib.load('Spam_Email_Detection/vectorizer_bow.pkl')
    
    gnb = joblib.load('Spam_Email_Detection/model_gaussian.pkl')
    vec_tfidf = joblib.load('Spam_Email_Detection/vectorizer_tfidf.pkl')

    print("Preparing test sets...")
    X_bow = vec_bow.transform(df['processed_text'])
    X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

    X_tfidf = vec_tfidf.transform(df['processed_text'])
    X_dense = X_tfidf.toarray()
    X_train_dense, X_test_dense, y_train_dense, y_test_dense = train_test_split(X_dense, y, test_size=0.2, random_state=42)

    results = {}

    print("\n--- Evaluating Multinomial NB ---")
    y_pred_mnb = mnb.predict(X_test_bow)
    y_prob_mnb = mnb.predict_proba(X_test_bow)[:, 1]
    
    results['Multinomial NB'] = {
        'Accuracy': accuracy_score(y_test, y_pred_mnb),
        'Precision': precision_score(y_test, y_pred_mnb),
        'Recall': recall_score(y_test, y_pred_mnb),
        'F1 Score': f1_score(y_test, y_pred_mnb)
    }

    plot_confusion_matrix(y_test, y_pred_mnb, 'Multinomial NB Confusion Matrix', 'Spam_Email_Detection/reports/mnb_confusion_matrix.png')
    plot_roc_curve(y_test, y_prob_mnb, 'Multinomial NB ROC Curve', 'Spam_Email_Detection/reports/mnb_roc_curve.png')

    print("Running k-Fold CV for Multinomial NB...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores_mnb = cross_val_score(mnb, X_bow, y, cv=kf, scoring='accuracy')
    results['Multinomial NB']['CV Mean Accuracy'] = cv_scores_mnb.mean()
    results['Multinomial NB']['CV Std Dev'] = cv_scores_mnb.std()

    print("\n--- Evaluating Gaussian NB ---")
    y_pred_gnb = gnb.predict(X_test_dense)
    y_prob_gnb = gnb.predict_proba(X_test_dense)[:, 1]

    results['Gaussian NB'] = {
        'Accuracy': accuracy_score(y_test_dense, y_pred_gnb),
        'Precision': precision_score(y_test_dense, y_pred_gnb),
        'Recall': recall_score(y_test_dense, y_pred_gnb),
        'F1 Score': f1_score(y_test_dense, y_pred_gnb)
    }

    plot_confusion_matrix(y_test_dense, y_pred_gnb, 'Gaussian NB Confusion Matrix', 'Spam_Email_Detection/reports/gnb_confusion_matrix.png')
    plot_roc_curve(y_test_dense, y_prob_gnb, 'Gaussian NB ROC Curve', 'Spam_Email_Detection/reports/gnb_roc_curve.png')

    print("Running k-Fold CV for Gaussian NB...")
    cv_scores_gnb = cross_val_score(gnb, X_dense, y, cv=kf, scoring='accuracy')
    results['Gaussian NB']['CV Mean Accuracy'] = cv_scores_gnb.mean()
    results['Gaussian NB']['CV Std Dev'] = cv_scores_gnb.std()
    
    print("\n" + "="*80)
    print(f"{'Metric':<20} | {'Multinomial NB':<20} | {'Gaussian NB':<20}")
    print("-" * 80)
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Mean Accuracy', 'CV Std Dev']:
        print(f"{metric:<20} | {results['Multinomial NB'][metric]:.4f}               | {results['Gaussian NB'][metric]:.4f}")
    print("="*80)
    
    best_model = "Multinomial NB" if results['Multinomial NB']['F1 Score'] > results['Gaussian NB']['F1 Score'] else "Gaussian NB"
    print(f"\nConclusion: {best_model} performs better based on F1 Score.")

if __name__ == "__main__":
    evaluate_models()
