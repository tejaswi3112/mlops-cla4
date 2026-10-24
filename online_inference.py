"""
Task 2: Online Inference
------------------------
Demonstrates Online (Real-time / Request-Response) Inference:
- Loads the serialized pipeline 'model.pkl'.
- Selects a single unseen passenger record from 'unseen_test_data.csv'.
- Generates immediate real-time prediction and probability score.
- Displays the passenger attributes, prediction output, and latency.
"""

import time
import pandas as pd
import joblib

MODEL_FILE = "model.pkl"
UNSEEN_DATA_FILE = "unseen_test_data.csv"
FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


def run_online_inference(record_index: int = 0):
    print("=" * 65)
    print("  TASK 2: ONLINE INFERENCE (SINGLE REAL-TIME PREDICTION)")
    print("=" * 65)

    # 1. Load trained model pipeline
    print(f"[*] Loading model from '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    print("[+] Model pipeline loaded successfully.")

    # 2. Load unseen test dataset
    df_unseen = pd.read_csv(UNSEEN_DATA_FILE)
    if record_index >= len(df_unseen):
        record_index = 0

    single_passenger = df_unseen.iloc[[record_index]]
    features = single_passenger[FEATURE_COLS]

    # 3. Perform Online Inference & measure latency
    start_time = time.perf_counter()
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000

    # 4. Display Results
    survived_label = "Survived (1)" if prediction == 1 else "Did Not Survive (0)"
    confidence = probabilities[prediction] * 100

    print("\n--- Selected Unseen Passenger Details ---")
    if 'PassengerId' in single_passenger.columns:
        print(f"Passenger ID    : {single_passenger['PassengerId'].values[0]}")
    if 'Name' in single_passenger.columns:
        print(f"Name            : {single_passenger['Name'].values[0]}")
    print(f"Class (Pclass)  : {single_passenger['Pclass'].values[0]}")
    print(f"Sex             : {single_passenger['Sex'].values[0]}")
    print(f"Age             : {single_passenger['Age'].values[0]}")
    print(f"SibSp / Parch   : {single_passenger['SibSp'].values[0]} / {single_passenger['Parch'].values[0]}")
    print(f"Fare            : ${single_passenger['Fare'].values[0]:.2f}")
    print(f"Embarked        : {single_passenger['Embarked'].values[0]}")
    if 'Survived' in single_passenger.columns:
        gt_label = "Survived (1)" if single_passenger['Survived'].values[0] == 1 else "Did Not Survive (0)"
        print(f"Ground Truth    : {gt_label}")

    print("\n--- Online Inference Result ---")
    print(f"Predicted Class : {prediction} -> {survived_label}")
    print(f"Confidence      : {confidence:.2f}%")
    print(f"Probabilities   : P(Did Not Survive)={probabilities[0]:.4f}, P(Survived)={probabilities[1]:.4f}")
    print(f"Inference Latency: {latency_ms:.2f} ms")
    print("=" * 65)

    return {
        "prediction": int(prediction),
        "label": survived_label,
        "probabilities": probabilities.tolist(),
        "latency_ms": latency_ms
    }


if __name__ == "__main__":
    run_online_inference(record_index=0)
