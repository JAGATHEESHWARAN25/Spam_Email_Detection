# Data Preprocessing Report

## 1. Dataset Description

**Source**: The dataset used is the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), a public set of SMS labeled messages that have been collected for mobile phone spam research.

**Statistics**:
- **Total Samples**: 5624 messages (after augmentation)
- **Class Distribution**:
    - **Ham (Not Spam)**: 4825
    - **Spam**: 799 (Augmented from original ~747)

*Note: The dataset was augmented with specific edge cases to improve robustness against certain spam patterns.*

## 2. Text Cleaning
Data cleaning is the first critical step to remove noise that does not contribute to the semantic meaning of the text.

**Steps Performed**:
1.  **Lowercasing**: All characters were converted to lowercase.
    *   *Why?* To treat "Free", "free", and "FREE" as the same word, reducing vocabulary size.
2.  **Punctuation Removal**: All punctuation characters (`!`, `.`, `,`, etc.) were removed.
    *   *Why?* Punctuation usually creates noise in BoW models (e.g., "win!" vs "win").
3.  **Number Removal**: All digits were stripped.
    *   *Why?* Specific numbers (like "5000" vs "5001") are rarely predictive on their own in this context compared to the word patterns.
4.  **Whitespace Handling**: Multiple spaces were collapsed into single spaces to ensure clean tokenization.

## 3. Tokenization
We used **Word-level Tokenization** via simple whitespace splitting.
*   **Example**: `"win money now"` $\rightarrow$ `["win", "money", "now"]`
*   *Library*: Native Python string manipulation was used for efficiency and transparency.

## 4. Stop Word Removal
Common English words were removed using the **NLTK (Natural Language Toolkit)** library.
*   **Examples**: "the", "is", "at", "which", "on".
*   *Why?* These words appear efficiently in both spam and ham, adding dimensionality without discriminative value. improving model efficiency.

## 5. Lemmatization/Stemming
We used **Porter Stemming** (via NLTK).
*   **Process**: Reduces words to their root form (suffix stripping).
*   **Example**: "running", "ran", "run" $\rightarrow$ "run".
*   *Why?* It groups related words together, further shrinking the feature space and improving generalization.

## 6. Feature Extraction (Vectorization)
We implemented two methods to convert text into numerical features for our models.

| Feature | Description | Used By | Why? |
| :--- | :--- | :--- | :--- |
| **Bag of Words (BoW)** | Counts the occurrence of each word in the document. Result is a sparse matrix of integers. | **Multinomial NB** | MNB models the distribution of word counts (multinomial distribution), making BoW the mathematically correct input. |
| **TF-IDF** | (Term Frequency - Inverse Document Frequency). Weighs words: higher weight for rare words, lower for common words. Result is continuous values. | **Gaussian NB** | GNB assumes continuous data with a normal distribution. While text is not perfectly normal, TF-IDF provides the continuous input GNB requires. |

## 7. Importance of Preprocessing
Preprocessing is not just formatting; it effectively:
1.  **Reduces Noise**: By removing punctuation and stop words.
2.  **Reduces Overfitting**: Stemming and lowercasing prevent the model from learning specific, rare variations of words.
3.  **Improves Accuracy**: Clean data allows the Naive Bayes probability assumption to work more effectively.
