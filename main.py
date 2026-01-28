import pandas as pd
import joblib
import sys
import os

sys.path.append(os.getcwd())
try:
    from Spam_Email_Detection.preprocessing import preprocess_text
except ImportError:
    from preprocessing import preprocess_text

def predict_spam(text, model, vectorizer):
    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    return prediction, probability
def main():
    print("--- Spam Email Detection Demo ---")
    model_path = 'Spam_Email_Detection/model_multinomial.pkl'
    vec_path = 'Spam_Email_Detection/vectorizer_bow.pkl'
    if not os.path.exists(model_path):
        if os.path.exists('model_multinomial.pkl'):
             model_path = 'model_multinomial.pkl'
             vec_path = 'vectorizer_bow.pkl'
        else:
            print("Model files not found. Retraining model...")
            try:
                from Spam_Email_Detection.model_multinomial import train_multinomial
                train_multinomial()
            except ImportError:
                 from model_multinomial import train_multinomial
                 train_multinomial()
            print("Training complete. Reloading...")

    try:     
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
    except FileNotFoundError:
        print("Error: Could not load or train models.")
        return

    test_emails = [
        "Congratulations! You won a free prize. Call now to claim your award!",
        "Hey, are we still meeting for lunch today?",
        "URGENT: Your account has been compromised. Click here to reset password.",
        "Attached is the report you requested yesterday."
    ]
    
    print("\nProcessing test emails...\n")
    
    for email in test_emails:
        label_num, probs = predict_spam(email, model, vectorizer)
        label = "SPAM" if label_num == 1 else "HAM"
        confidence = probs[label_num] * 100
        
        print(f"Email: \"{email}\"")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
        print("-" * 50)
    print("\nInteractive Mode (Type 'exit' to quit)")
    while True:
        user_input = input("\nEnter email text: ")
        if  user_input.lower() == 'exit':
            break
        
        label_num, probs = predict_spam(user_input, model, vectorizer)
        label = "SPAM" if label_num == 1 else "HAM"
        confidence = probs[label_num] * 100
        
        print(f"\nPrediction:  [{label}]")
        print(f"Confidence:  {confidence:.2f}%")
        if label == "SPAM":
            print("Action:      Moved to Spam Folder")
        else:
            print("Action:      Delivered to Inbox")
        print("-" * 30)

if __name__ == "__main__":
    main()
