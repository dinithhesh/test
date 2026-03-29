import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/colombo_flood_with_probabilities.csv")

# Example: Assuming last column is target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'svm_model.pkl')

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
