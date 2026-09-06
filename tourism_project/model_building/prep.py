"""
Data Preparation Script
- Loads dataset from repo
- Cleans data (handles missing values, fixes typos, drops unnecessary columns)
- Splits into train/test sets (80/20, stratified)
- Saves train.csv and test.csv for the next pipeline job
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

DATA_PATH  = "tourism.csv"
OUTPUT_DIR = "tourism_project/model_building"

def clean_data(df):
    """Clean and preprocess the raw dataset."""

    # Drop unnecessary columns
    drop_cols = ["Unnamed: 0", "CustomerID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fix Gender typo: 'Fe Male' → 'Female'
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")

    # Fix MaritalStatus: 'Unmarried' → 'Single' (same meaning)
    df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

    # Handle missing values
    # Numerical: fill with median
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "ProdTaken"]
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Categorical: fill with mode
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    return df

def prepare_data():
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    df = clean_data(df)

    # Separate features and target
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save splits
    train_df = X_train.copy()
    train_df["ProdTaken"] = y_train.values
    test_df  = X_test.copy()
    test_df["ProdTaken"]  = y_test.values

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path  = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"\nTrain set: {train_df.shape} → saved to {train_path}")
    print(f"Test set : {test_df.shape}  → saved to {test_path}")
    print(f"\nTarget distribution (train):")
    print(y_train.value_counts().to_string())
    print("\nData Preparation Complete ✅")

if __name__ == "__main__":
    prepare_data()
