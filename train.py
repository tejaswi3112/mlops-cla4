"""
Task 1: Model Training and Serialization
----------------------------------------
This script loads the Titanic dataset, preprocesses features using an
end-to-end Scikit-Learn Pipeline, splits the data into train and test sets,
trains a Random Forest Classifier, evaluates the model, saves the complete
trained pipeline into 'model.pkl', and exports the held-out unseen records
to 'unseen_test_data.csv' for downstream inference strategies.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_FILE = "train.csv"
MODEL_FILE = "model.pkl"
UNSEEN_DATA_FILE = "unseen_test_data.csv"


def download_dataset_if_needed():
    if not os.path.exists(DATA_FILE):
        print(f"[*] Downloading Titanic dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, DATA_FILE)
        print(f"[+] Download complete: '{DATA_FILE}'")
    else:
        print(f"[*] Found existing dataset: '{DATA_FILE}'")


def train_and_evaluate():
    download_dataset_if_needed()
    
    # 1. Load dataset
    print("\n" + "="*60)
    print("STEP 1: LOADING DATASET")
    print("="*60)
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # 2. Define Features and Target
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target_col = 'Survived'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 3. Train-Test Split (80% Train, 20% Unseen Test)
    # Stratified split ensures balanced target distribution in both subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Unseen test set: {X_test.shape[0]} samples")

    # Save unseen records with their identifiers & ground truth for Tasks 2, 3, 4
    test_records = df.loc[X_test.index].copy()
    test_records.to_csv(UNSEEN_DATA_FILE, index=False)
    print(f"[+] Saved held-out unseen records to '{UNSEEN_DATA_FILE}' ({len(test_records)} rows)")

    # 4. Build Preprocessing and Modeling Pipeline
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42))
    ])

    # 5. Train Model
    print("\n" + "="*60)
    print("STEP 2: TRAINING MODEL PIPELINE")
    print("="*60)
    pipeline.fit(X_train, y_train)
    print("[+] Model pipeline training completed successfully.")

    # 6. Evaluate Model on Unseen Test Set
    print("\n" + "="*60)
    print("STEP 3: EVALUATING MODEL ON UNSEEN TEST DATA")
    print("="*60)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Did Not Survive (0)", "Survived (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save the Complete Pipeline
    print("\n" + "="*60)
    print("STEP 4: SERIALIZING PIPELINE USING JOBLIB")
    print("="*60)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"[+] Successfully saved complete pipeline to '{MODEL_FILE}'")

    # Verification of Saved Model
    loaded_pipeline = joblib.load(MODEL_FILE)
    sample_test = X_test.iloc[:1]
    sample_pred = loaded_pipeline.predict(sample_test)[0]
    sample_prob = loaded_pipeline.predict_proba(sample_test)[0]
    print(f"[+] Verified loaded model prediction: {sample_pred} with probabilities: {sample_prob}")
    print("="*60)


if __name__ == "__main__":
    train_and_evaluate()
