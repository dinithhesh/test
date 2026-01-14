import joblib
import pandas as pd

# Load trained model
model = joblib.load('svm_model.pkl')

# Example new data (replace with your features)
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(new_data)
print("Prediction:", prediction)
