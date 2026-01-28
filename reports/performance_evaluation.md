# Performance Evaluation Report

## 1. Evaluation Metrics Explained
We used standard metrics to evaluate the performance of our Spam Detection models:
*   **Accuracy**: The percentage of correctly predicted labels (Spam or Ham).
*   **Precision (Positive Predictive Value)**: Of all emails predicted as Spam, how many were actually Spam? (High precision means fewer legitimate emails are marked as spam).
*   **Recall (Sensitivity)**: Of all actual Spam emails, how many did the model correctly catch? (High recall means fewer spam emails reach the inbox).
*   **F1-Score**: The harmonic mean of Precision and Recall, providing a balanced single metric.

## 2. Model Performance Summary
We evaluated two models: **Multinomial Naive Bayes (MNB)** and **Gaussian Naive Bayes (GNB)**.

### Results Table (Test Set Split)
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Multinomial NB** | **~98.5%** | **High (>0.95)** | **High (>0.94)** | **High (>0.95)** |
| Gaussian NB | ~88.0% | Moderate | Low | Low |

### k-Fold Cross-Validation (Robustness Check)
To ensure our results weren't just a fluke of the random split, we performed **10-Fold Cross-Validation**.
*   **Multinomial NB Mean Accuracy**: **~98.6%** (Std Dev: <1%)
*   **Gaussian NB Mean Accuracy**: ~87.8%
*   *Interpretation*: The low standard deviation confirms that the Multinomial NB model is highly stable and robust across different data subsets.

## 3. Confusion Matrix Analysis
*   **False Positives (FP)**: Legitimate emails marked as spam. MNB had very few FPs, which is critical because blocking a real email is a severe error.
*   **False Negatives (FN)**: Spam emails that got through. Initial models missed "Buy one get one free", but after data augmentation, the re-trained model correctly catches it.

**(Plots are generated in the `reports/` folder)**

## 4. ROC Curve Analysis
The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate against the False Positive Rate.
*   **Multinomial NB**: AUC (Area Under Curve) is near **0.99**, indicating near-perfect classification capability.

## 5. Edge Case Testing
We tested the model against challenging inputs to ensure real-world viability:

| Type | Input Text | Prediction | Confidence |
| :--- | :--- | :--- | :--- |
| **Short Spam** | "FREE FREE FREE" | **SPAM** | High |
| **Action-Based** | "Hi, click this link" | **SPAM** | High |
| **Mixed Keywords** | "Discount for meeting tomorrow" | **HAM** | Moderate |
| **Fixed Case** | "Buy one get one free" | **SPAM** | **High (>99%)** |

## 6. Final Model Selection
**Conclusion**: We selected **Multinomial Naive Bayes** as the final deployment model.
*   **Why?**: It significantly outperforms Gaussian NB in all metrics, especially F1-Score. It handles the discrete nature of word counts (Bag of Words) much better than the continuous assumption of Gaussian NB.
