# CODETECHTASK2
Name:ANNEPU SURYA
Company:CODETECH IT SOLUTIONS
Intern ID:COD41084
Domain:Machine Learning

Credit Card Fraud Detection
This project implements a machine learning solution to detect fraudulent credit card transactions. Using a dataset of transactions, the project addresses the challenge of highly imbalanced data by employing techniques like SMOTE (Synthetic Minority Over-sampling Technique) and evaluates the model's performance through various metrics.

Introduction
Credit card fraud is a critical issue affecting financial institutions and individuals globally. With an extremely low occurrence rate, fraudulent transactions are difficult to detect but crucial to identify. This project applies Random Forest Classifier, a supervised learning algorithm, to detect fraudulent transactions. It handles imbalanced data effectively using SMOTE and provides insights into feature importance for fraud detection.

Dataset Description
The dataset used in this project is the Credit Card Fraud Detection Dataset from Kaggle, containing 284,807 transactions with the following characteristics:

Features: 28 PCA-transformed features (V1 to V28), along with Time and Amount.
Target Variable:
0: Legitimate transaction.
1: Fraudulent transaction.
Class Imbalance:
Legitimate transactions (~99.83%).
Fraudulent transactions (~0.17%).
Dataset Source: Kaggle Credit Card Fraud Detection.

Project Workflow
Data Exploration:

Analyze the dataset's structure and class distribution.
Visualize class imbalance using a count plot.
Data Preprocessing:

Separate features (X) and target (y).
Split the dataset into training and testing subsets.
Imbalance Handling:

Apply SMOTE to balance the training data by oversampling the minority class.
Model Training:

Train a Random Forest Classifier on the balanced training set.
Evaluation:

Evaluate the model's performance using:
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
ROC-AUC Curve
Feature Importance:

Identify and visualize the top features influencing the predictions.
Features
Data Handling:

Loading and exploring the credit card fraud dataset.
Addressing the class imbalance using SMOTE.
Visualization:

Class distribution visualization using a count plot.
Feature importance bar chart.
Modeling:

Training and evaluating a Random Forest Classifier.
Performance Metrics:

Confusion Matrix for error analysis.
ROC-AUC Curve for classification performance.
Classification Report for detailed metrics.
