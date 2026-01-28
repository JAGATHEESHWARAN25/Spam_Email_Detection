# Spam Email Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Type](https://img.shields.io/badge/ML-NLP-green.svg)

A production-ready Machine Learning project that detects Spam emails with **>98.5% Accuracy**. The system uses Natural Language Processing (NLP) and Naive Bayes algorithms to classify messages as either **SPAM** or **HAM** (legitimate).

---

## 🚀 Key Features

*   **Robust Preprocessing**: Includes lowercasing, stop-word removal, and Porter Stemming to handle real-world text variations.
*   **Dual-Model Architecture**: Implements and compares **Multinomial Naive Bayes** (for discrete BoW counts) and **Gaussian Naive Bayes** (for continuous TF-IDF scores).
*   **Edge-Case Handling**: Successfully detects tricky spam patterns (e.g., "FREE FREE", "Buy one get one") via data augmentation.
*   **Real-time Prediction**: Interactive CLI tool to classify user input instantly with confidence scores.
*   **Comprehensive Evaluation**: Verified using **10-Fold Cross-Validation** to ensure stability and reliability.

---

## 📂 Project Structure

```
Spam_Email_Detection/
├── dataset/
│   ├── sms.tsv                 # Original labeled dataset (augmented)
│   └── spam_processed.csv      # Processed data after cleaning
├── reports/
│   ├── preprocessing_report.md # Detail on data cleaning steps
│   ├── performance_evaluation.md # Deep dive into model metrics
│   └── *.png                   # Confusion matrices and ROC curves
├── preprocessing.py            # Core NLP pipeline
├── model_multinomial.py        # MNB Training Script (The Champion Model)
├── evaluation.py               # Metrics, Plots, and Cross-Validation
├── main.py                     # Real-world Application / Demo
└── README.md                   # This file
```

---

## 📊 Dataset & Results

We used the **SMS Spam Collection Dataset**, augmented with specific edge cases to improve sensitivity.

| Metric | Multinomial NB | Gaussian NB |
| :--- | :--- | :--- |
| **Accuracy** | **~98,6%** | ~87.8% |
| **Precision** | **>0.95** | Moderate |
| **F1-Score** | **>0.95** | Low |

**Why Multinomial NB?**
Multinomial Naive Bayes assumes a multinomial distribution of data, which aligns perfectly with text classification where features are Word Counts (Bag of Words). Gaussian NB assumes a normal distribution, which is less accurate for sparse text data.

---

## 🛠️ How to Run

### 1. Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas scikit-learn nltk matplotlib seaborn
```

### 2. Auto-Train & Run Demo
Simply run `main.py`. The system is smart enough to detect if models are missing and will automatically retrain them for you.
```bash
python Spam_Email_Detection/main.py
```
*   *Type your email text when prompted to get a prediction.*

### 3. Generate Reports (Optional)
To regenerate the evaluation metrics and plots:
```bash
python Spam_Email_Detection/evaluation.py
```

---

## 🔍 Example Usage

**Input:**
> "Congratulations! You won a free prize. Call now!"

**Output:**
```
Prediction:  [SPAM]
Confidence:  100.00%
Action:      Moved to Spam Folder
```

**Input:**
> "Hey, are we still meeting for lunch today?"

**Output:**
```
Prediction:  [HAM]
Confidence:  100.00%
Action:      Delivered to Inbox
```

---

## 🧪 Robustness Validation

We didn't just split the data once. We ran **10-Fold Cross-Validation** to prove the model works on any random subset of data. The Multinomial NB model achieved a mean accuracy of **98.6%** with strictly low variance, proving it is production-ready.
