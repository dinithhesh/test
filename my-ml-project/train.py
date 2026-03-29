import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

def run_training():
    data = pd.read_csv('data/dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc}")

    joblib.dump(model, 'model.pkl')
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

if __name__ == "__main__":
    run_training()
