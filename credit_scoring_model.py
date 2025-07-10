import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
data = {
    'Income': [40000, 50000, 30000, 60000, 35000, 80000, 25000, 75000],
    'Debt': [10000, 15000, 5000, 20000, 7000, 10000, 3000, 12000],
    'Payment_History': [0, 2, 1, 0, 3, 0, 2, 1],
    'Credit_Utilization': [30, 45, 25, 50, 35, 20, 15, 40],
    'Age': [25, 35, 23, 45, 30, 50, 22, 40],
    'Creditworthy': [1, 0, 1, 0, 0, 1, 1, 0]
}

df = pd.read_csv('credit_data.csv')

X = df.drop('Creditworthy', axis=1)
y = df['Creditworthy']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred)
    }

for name, metrics in results.items():
    print(f"\n{name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
