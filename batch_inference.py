"""
Task 3: Batch Inference
-----------------------
Demonstrates Batch (Vectorized / High-Throughput) Inference:
- Loads the trained pipeline 'model.pkl'.
- Selects 20 unseen passenger records from 'unseen_test_data.csv'.
- Generates predictions and probability distributions for all 20 records in a single vectorized batch pass.
- Saves the predictions and metadata into 'batch_predictions.csv'.
- Displays a formatted batch summary table and throughput metrics.
"""

import time
import pandas as pd
import joblib

MODEL_FILE = "model.pkl"
UNSEEN_DATA_FILE = "unseen_test_data.csv"
OUTPUT_FILE = "batch_predictions.csv"
FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
BATCH_SIZE = 20


def run_batch_inference():
    print("=" * 75)
    print("  TASK 3: BATCH INFERENCE (HIGH-THROUGHPUT BULK PROCESSING)")
    print("=" * 75)

    # 1. Load trained model pipeline
    print(f"[*] Loading model from '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    print("[+] Model pipeline loaded successfully.")

    # 2. Select 20 unseen passenger records
    df_unseen = pd.read_csv(UNSEEN_DATA_FILE)
    batch_records = df_unseen.iloc[:BATCH_SIZE].copy()
    print(f"[*] Selected {len(batch_records)} unseen records for batch inference.")

    features_batch = batch_records[FEATURE_COLS]

    # 3. Vectorized Batch Prediction
    print("[*] Generating batch predictions for all 20 records simultaneously...")
    start_time = time.perf_counter()
    predictions = model.predict(features_batch)
    probabilities = model.predict_proba(features_batch)
    batch_time_ms = (time.perf_counter() - start_time) * 1000
    avg_latency_ms = batch_time_ms / BATCH_SIZE

    # 4. Prepare Output DataFrame
    results_df = batch_records.copy()
    results_df['Predicted_Survived'] = predictions
    results_df['Prediction_Label'] = [
        "Survived" if p == 1 else "Did Not Survive" for p in predictions
    ]
    results_df['Prob_Did_Not_Survive'] = probabilities[:, 0].round(4)
    results_df['Prob_Survived'] = probabilities[:, 1].round(4)

    if 'Survived' in results_df.columns:
        results_df['Is_Correct'] = (results_df['Predicted_Survived'] == results_df['Survived'])
        batch_accuracy = results_df['Is_Correct'].mean() * 100
    else:
        batch_accuracy = None

    # 5. Save to batch_predictions.csv
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[+] Successfully saved batch predictions to '{OUTPUT_FILE}'.")

    # 6. Display Formatted Results Summary
    display_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 'Predicted_Survived', 'Prediction_Label', 'Prob_Survived']
    if 'Survived' in results_df.columns:
        display_cols.insert(6, 'Survived')

    print("\n--- Batch Prediction Results (20 Unseen Records) ---")
    print(results_df[display_cols].to_string(index=False))

    print("\n--- Batch Inference Performance Metrics ---")
    print(f"Total Records Processed : {BATCH_SIZE}")
    print(f"Total Batch Time        : {batch_time_ms:.2f} ms")
    print(f"Average Latency / Record: {avg_latency_ms:.2f} ms")
    print(f"Throughput              : {BATCH_SIZE / (batch_time_ms / 1000):.1f} records/sec")
    if batch_accuracy is not None:
        print(f"Batch Accuracy          : {batch_accuracy:.1f}%")
    print("=" * 75)

    return results_df


if __name__ == "__main__":
    run_batch_inference()
