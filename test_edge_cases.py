import joblib
import sys
import os
import pandas as pd

# Add path
sys.path.append(os.getcwd())
try:
    from Spam_Email_Detection.preprocessing import preprocess_text
except ImportError:
    from preprocessing import preprocess_text

def test_edge_cases():
    print("--- Edge Case Testing ---")
    
    # Load Model
    try:
        model_path = 'Spam_Email_Detection/model_multinomial.pkl'
        vec_path = 'Spam_Email_Detection/vectorizer_bow.pkl'
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
    except FileNotFoundError:
        print("Models not found. Please train first.")
        return

    edge_cases = [
        {"text": "FREE FREE FREE", "type": "Short Spam (All Caps)"},
        {"text": "Hi, click this link", "type": "Short Spam (Action)"},
        {"text": "Discount for meeting tomorrow", "type": "Mixed Ham/Spam Keywords"},
        {"text": "Win money", "type": "Very Short Spam"},
        {"text": "Hello, how are you?", "type": "Short Ham"},
        {"text": "Buy one get one free", "type": "Known False Negative (Fixed)"}
    ]
    
    print(f"{'Type':<30} | {'Text':<30} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 90)
    
    for case in edge_cases:
        text = case['text']
        # Predict
        processed = preprocess_text(text)
        features = vectorizer.transform([processed])
        pred_num = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        
        label = "SPAM" if pred_num == 1 else "HAM"
        confidence = prob[pred_num] * 100
        
        print(f"{case['type']:<30} | {text[:30]:<30} | {label:<10} | {confidence:.2f}%")

if __name__ == "__main__":
    test_edge_cases()
