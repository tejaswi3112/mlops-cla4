"""
Task 4: Streaming Inference
---------------------------
Demonstrates Streaming (Event-Driven / Near-Real-Time) Inference:
- Loads the trained pipeline 'model.pkl'.
- Selects 10 unseen passenger records from 'unseen_test_data.csv'.
- Simulates an event stream (e.g., Kafka / Kinesis / IoT message queue)
  by processing one record at a time with a 1-second arrival delay.
- Displays the real-time inference prediction for each incoming event.
"""

import time
from datetime import datetime
import pandas as pd
import joblib

MODEL_FILE = "model.pkl"
UNSEEN_DATA_FILE = "unseen_test_data.csv"
FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
STREAM_COUNT = 10
DELAY_SECONDS = 1.0


def run_streaming_inference():
    print("=" * 80)
    print("  TASK 4: STREAMING INFERENCE (SIMULATED EVENT-DRIVEN STREAM)")
    print("=" * 80)

    # 1. Load trained model pipeline
    print(f"[*] Loading model from '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    print("[+] Model pipeline loaded successfully.")

    # 2. Select 10 unseen passenger records
    df_unseen = pd.read_csv(UNSEEN_DATA_FILE)
    streaming_records = df_unseen.iloc[:STREAM_COUNT].copy()
    print(f"[*] Initializing stream simulation for {STREAM_COUNT} events with {DELAY_SECONDS}s delay...\n")
    print(f"{'EVENT':<7} | {'TIMESTAMP':<12} | {'PASSENGER ID':<13} | {'FEATURES (Cls/Sex/Age/Fare)':<30} | {'PREDICTION':<18} | {'CONF':<7} | {'LATENCY'}")
    print("-" * 105)

    stream_latencies = []

    # 3. Simulate Event Stream Processing
    for idx, (_, row) in enumerate(streaming_records.iterrows(), start=1):
        # Extract features for single record
        single_df = pd.DataFrame([row[FEATURE_COLS]])

        # Measure model processing time
        t0 = time.perf_counter()
        pred = model.predict(single_df)[0]
        probs = model.predict_proba(single_df)[0]
        infer_latency_ms = (time.perf_counter() - t0) * 1000
        stream_latencies.append(infer_latency_ms)

        # Formatting event information
        event_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        pass_id = str(row.get('PassengerId', f"P-{idx}"))
        pclass = row['Pclass']
        sex = row['Sex']
        age = f"{row['Age']:.0f}" if pd.notnull(row['Age']) else "NA"
        fare = f"${row['Fare']:.1f}" if pd.notnull(row['Fare']) else "NA"
        feat_str = f"Class {pclass} | {sex:<6} | Age:{age:<3} | {fare}"

        label = "Survived (1)" if pred == 1 else "Did Not Survive (0)"
        conf = f"{probs[pred]*100:.1f}%"

        print(f"#{idx:<6} | {event_time:<12} | {pass_id:<13} | {feat_str:<30} | {label:<18} | {conf:<7} | {infer_latency_ms:.2f} ms")

        # Simulate 1-second stream arrival interval (except after the final event)
        if idx < STREAM_COUNT:
            time.sleep(DELAY_SECONDS)

    print("-" * 105)
    print("\n--- Streaming Inference Summary ---")
    print(f"Events Ingested & Processed : {STREAM_COUNT}")
    print(f"Average Event Model Latency : {sum(stream_latencies)/len(stream_latencies):.2f} ms")
    print(f"Simulation Inter-arrival Gap: {DELAY_SECONDS} second")
    print(f"Stream Status               : COMPLETED (All events consumed)")
    print("=" * 80)


if __name__ == "__main__":
    run_streaming_inference()
