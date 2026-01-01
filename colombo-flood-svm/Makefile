.PHONY: install lint test train

install:
    python -m pip install -r requirements.txt

lint:
    flake8 src

test:
    pytest -q

train:
    python src/train.py --data data/colombo_flood_balanced_1000.csv --out artifacts/model.joblib
