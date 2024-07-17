
FROM python:3.10.7-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cach-dir pandas scikit-learn

CMD ["python", "spam_detector.py"]