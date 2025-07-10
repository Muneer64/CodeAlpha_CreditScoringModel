# Credit Scoring Prediction Model

This project is part of my **CodeAlpha Internship**. The goal of this project is to predict whether a person is creditworthy based on their financial data.

---

## 📊 Objective:
Predict creditworthiness (good or bad credit) based on various financial features.

---

## 📁 Dataset:
The dataset used contains details like:
- Income  
- Debts  
- Payment History  
- Credit Utilization  
- Age  
- Loan Amount  
- Number of Credit Cards  

---

## 🧠 Machine Learning Models Used:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

---

## 🔎 Process:
- Loaded and cleaned the data.
- Scaled the features for better model performance.
- Trained the above models on the dataset.
- Evaluated models using:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score

---

## 📈 Results:
- **Logistic Regression** & **Decision Tree** gave good accuracy.
- **Random Forest** also performed well but had slightly lower recall.

---

## 📂 Files Included:
- `credit_scoring_model.py` → Source code for the ML model.
- `credit_data.csv` → Sample credit dataset.

---

## 🚀 How to Run:
```bash
python credit_scoring_model.py
