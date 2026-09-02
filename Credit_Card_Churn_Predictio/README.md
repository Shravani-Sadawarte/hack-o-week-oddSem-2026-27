# 💳 Credit Card Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer churn is an important challenge for credit-card companies. Identifying customers who are likely to leave can help businesses take proactive retention measures.

This project develops a **Machine Learning classification system** to predict whether a credit-card customer is likely to:

- 🟢 Remain an **Existing Customer**
- 🔴 Become an **Attrited Customer**

The project uses customer demographic, account, and transaction-related information to train and evaluate multiple classification models.

Three Machine Learning models were implemented and compared:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Among the three models, **Gradient Boosting achieved the best overall performance**.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze customer churn data.
- Perform data cleaning and preprocessing.
- Handle categorical and numerical features.
- Engineer meaningful transaction-related features.
- Split the dataset into training and testing sets.
- Apply feature scaling and categorical encoding.
- Train multiple Machine Learning classification models.
- Perform 5-fold Stratified Cross-Validation.
- Evaluate models using multiple performance metrics.
- Analyze confusion matrices and ROC curves.
- Identify influential features using feature importance.
- Demonstrate customer-level churn prediction.
- Compare low-risk and high-risk customers.

---

## 📊 Dataset

### Dataset Used

**Credit Card Customers / BankChurners Dataset**

**Source:** Kaggle

The dataset contains information about **10,127 credit-card customers** and originally contains **23 columns**.

### Target Variable

The target variable is:

```text
Attrition_Flag
