# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/ data/
CMD ["python", "src/train.py", "--data", "data/colombo_flood_balanced_1000.csv", "--out", "artifacts/model.joblib"]
