import requests
import random
import time

# API endpoint
URL = "http://localhost:8000/predict"

# Send N predictions
NUM_REQUESTS = 100
DELAY_SECONDS = 1  # delay between requests

# Generate a random Iris-like measurement
def random_iris_features():
    # Typical ranges for Iris dataset:
    # sepal length, sepal width, petal length, petal width
    return [
        round(random.uniform(4.3, 7.9), 1),  # sepal length
        round(random.uniform(2.0, 4.4), 1),  # sepal width
        round(random.uniform(1.0, 6.9), 1),  # petal length
        round(random.uniform(0.1, 2.5), 1)   # petal width
    ]

for i in range(NUM_REQUESTS):
    features = random_iris_features()
    payload = {"features": features}

    try:
        r = requests.post(URL, json=payload)
        if r.status_code == 200:
            print(f"[{i+1}/{NUM_REQUESTS}] Sent: {features} → Prediction: {r.json()}")
        else:
            print(f"[{i+1}/{NUM_REQUESTS}] Error: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"[{i+1}/{NUM_REQUESTS}] Failed to connect: {e}")

    time.sleep(DELAY_SECONDS)
