# Machine Learning Based Fraud Detection System

## Dataset

We used [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) dataset. This is a highly unbalanced dataset with 492 frauds out of 284,807 transactions (0.172%). The dataset is given as a set of principle components after dimensional reduction (PCA). The last two columns are 'Amount' and 'Class' where the latter being the fraud/not fraud ground truth.

Find citations in `citations.md`

## Models


| Model  | ROC AUC | Average Precision |
| ------ | ------- | ----------------- |
| Logistic Regression  | 0.971113  | 0.713542 |
| LightGBM  | 0.963068  |  0.838613 | 
| **XGboost** | **0.976652** | **0.870020** |

We used XGboost for the structured data as it was able to predict 90/98 frauds correctly during testing. The Precison-Recall curve is given below,

<img width="803" height="701" alt="image" src="https://github.com/user-attachments/assets/5503121c-c4b5-4653-b618-f975628fc536" />

