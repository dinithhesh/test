import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/colombo_flood_with_probabilities.csv")

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ SVM MODEL ------------------
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# Save SVM model
joblib.dump(svm_model, "svm_model.pkl")

# ------------------ LOGISTIC REGRESSION MODEL ------------------
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

logreg_pred = logreg_model.predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_pred)

# Save Logistic Regression model
joblib.dump(logreg_model, "logistic_regression_model.pkl")

# ------------------ RESULTS ------------------
print("Training completed ✅")
print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"Logistic Regression Accuracy: {logreg_acc:.4f}")

# Optional: save metrics for CI/CD tracking
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/metrics.txt", "w") as f:
    f.write(f"SVM Accuracy: {svm_acc:.4f}\n")
    f.write(f"Logistic Regression Accuracy: {logreg_acc:.4f}\n")

