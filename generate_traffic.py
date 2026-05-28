#!/usr/bin/env python3
"""
generate_traffic.py — Generate test traffic for monitoring screenshots
Sends requests to inference endpoint to populate metrics.
"""

import requests
import random
import time
import sys

INFERENCE_URL = "http://localhost:5001/predict"

# Sample mushroom features (preprocessed)
# These should match the columns in mushrooms_preprocessed.csv
SAMPLE_FEATURES = {
    "cap-shape": random.randint(0, 5),
    "cap-surface": random.randint(0, 3),
    "cap-color": random.randint(0, 9),
    "bruises": random.randint(0, 1),
    "odor": random.randint(0, 8),
    "gill-attachment": random.randint(0, 1),
    "gill-spacing": random.randint(0, 1),
    "gill-size": random.randint(0, 1),
    "gill-color": random.randint(0, 11),
    "stalk-shape": random.randint(0, 1),
    "stalk-root": random.randint(0, 4),
    "stalk-surface-above-ring": random.randint(0, 3),
    "stalk-surface-below-ring": random.randint(0, 3),
    "stalk-color-above-ring": random.randint(0, 8),
    "stalk-color-below-ring": random.randint(0, 8),
    "veil-type": 0,
    "veil-color": random.randint(0, 2),
    "ring-number": random.randint(0, 2),
    "ring-type": random.randint(0, 7),
    "spore-print-color": random.randint(0, 8),
    "population": random.randint(0, 5),
    "habitat": random.randint(0, 6),
}


def generate_request():
    """Send a single prediction request."""
    features = {k: [v] for k, v in SAMPLE_FEATURES.items()}
    # Randomize some features
    for key in features:
        if random.random() > 0.7:
            features[key] = [random.randint(0, 11)]

    try:
        resp = requests.post(
            INFERENCE_URL,
            json={"features": features},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    print(f"Generating {count} requests to {INFERENCE_URL}...")

    success = 0
    errors = 0

    for i in range(count):
        ok = generate_request()
        if ok:
            success += 1
        else:
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{count}] success={success} errors={errors}")

        time.sleep(delay)

    print(f"\nDone! success={success} errors={errors}")


if __name__ == "__main__":
    main()
