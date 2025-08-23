# Machine Learning Based Fraud Detection System

## Dataset

We used [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) dataset. This is a highly unbalanced dataset with 492 frauds out of 284,807 transactions (0.172%). The dataset is given as a set of principle components after dimensional reduction (PCA). The last two columns are 'Amount' and 'Class' where the latter being the fraud/not fraud ground truth.

Find citations in `citations.md`

## Models

| Model | Total Frauds | True Positives (TP) | False Negatives (FN) | False Positives (FP) | Recall (Catch Rate) |
|-------|--------------|---------------------|----------------------|----------------------|---------------------|
| **Logistic Regression** | 98 | 90 | 8 | 1,416 | 0.918 (91.8%) |
| **LightGBM** | 98 | 86 | 12 | 40 | 0.878 (87.8%) |
| **XGBoost** | 98 | 60 | 38 | 2 | 0.612 (61.2%) |

### Key Insights

#### Best Recall (Fraud Detection Rate)
- **Logistic Regression**: 91.8% - Catches most frauds but with many false alarms
- **LightGBM**: 87.8% - Good balance of fraud detection and false alarms
- **XGBoost**: 61.2% - Misses many frauds but very few false alarms

#### False Positive Analysis
- **Logistic Regression**: 1,416 false alarms - Very high false positive rate
- **LightGBM**: 40 false alarms - Reasonable false positive rate
- **XGBoost**: 2 false alarms - Excellent precision, minimal false alarms

#### Trade-offs
| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Logistic** | Highest fraud detection rate | Too many false alarms (unusable in practice) |
| **LightGBM** | Good balance of detection and precision | Moderate false alarm rate |
| **XGBoost** | Extremely low false alarms | Misses too many actual frauds |

#### Conclusion
**LightGBM** appears to offer the best balance for fraud detection, catching 87.8% of frauds while keeping false alarms manageable at 40 cases.