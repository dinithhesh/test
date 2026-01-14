import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Load model
model = joblib.load('svm_model.pkl')

# Load data
data = pd.read_csv('data/dataset.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Predictions
y_pred = model.predict(X)
print(classification_report(y, y_pred))
