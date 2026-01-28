import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os

# Ensure nltk resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_data(filepath):
    """Loads the dataset from TSV file."""
    # SMS Spam Collection is tab-separated, no header, columns: label, message
    # Convert label to binary: spam=1, ham=0
    df = pd.read_csv(filepath, sep='\t', names=['label', 'text'])
    df['label_num'] = df.label.map({'ham':0, 'spam':1})
    return df

def clean_text(text):
    """
    1. Lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Remove numbers
    text = "".join([char for char in text if not char.isdigit()])
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

def preprocess_text(text):
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> stem
    """
    ps = PorterStemmer()
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    cleaned = clean_text(text)
    tokens = cleaned.split() # Simple whitespace tokenization
    
    # Stop word removal and stemming
    processed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(processed_tokens)

if __name__ == "__main__":
    # Test run
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset', 'sms.tsv')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
    else:
        print("Loading data...")
        df = load_data(dataset_path)
        print(f"Data loaded. Shape: {df.shape}")
        
        print("Preprocessing text...")
        df['processed_text'] = df['text'].apply(preprocess_text)
        print("Preprocessing complete.")
        
        # Save processed version
        output_path = 'dataset/spam_processed.csv'
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

        # Show sample
        print(df[['text', 'processed_text', 'label']].head())
