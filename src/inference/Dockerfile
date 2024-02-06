FROM python:3.8-slim

WORKDIR /app

COPY ../../requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ../../src/ ./src/
COPY ../../data/ ./data/
COPY ../../outputs/models/ ./outputs/models/

CMD ["python", "./src/inference/run_inference.py"]