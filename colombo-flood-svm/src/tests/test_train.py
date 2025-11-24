import pandas as pd
from src.train import train_and_save

def test_training_runs(tmp_path):
    df = pd.DataFrame({
        "feat1": [0, 1, 0, 1],
        "feat2": [1, 0, 1, 0],
        "label": [0, 1, 0, 1]
    })

    out_path = tmp_path / "model.joblib"
    train_and_save(df.drop("label", axis=1), df["label"], out_path=str(out_path))

    assert out_path.exists()
