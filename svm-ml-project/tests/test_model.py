import joblib
import pandas as pd

def test_model_predict_shape():
    model = joblib.load('svm_model.pkl')
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]])
    pred = model.predict(sample)
    assert pred.shape[0] == 1
