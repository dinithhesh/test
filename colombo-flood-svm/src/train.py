import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_data(path, target_col="label"):
    df = pd.read_csv(path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def train_and_save(X, y, out_path="model.joblib"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    artifact = {"model": model, "scaler": scaler}
    joblib.dump(artifact, out_path)
    print("Model saved at:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="model.joblib")
    args = parser.parse_args()

    X, y = load_data(args.data)
    train_and_save(X, y, out_path=args.out)

if __name__ == "__main__":
    main()
