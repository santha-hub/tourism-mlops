"""
Data Registration Script
- Validates that the dataset has all expected columns
- Prints a summary of the dataset
"""

import pandas as pd
import sys
import os

# Path to the dataset inside the repo
DATA_PATH = "tourism.csv"

# Expected columns based on the data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

def register_data(path):
    print("=" * 60)
    print("DATA REGISTRATION")
    print("=" * 60)

    # Load dataset
    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"\nDataset loaded successfully from: {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Validate columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing_cols:
        print(f"\nERROR: Missing expected columns: {missing_cols}")
        sys.exit(1)
    else:
        print("\nColumn validation PASSED ✅ — All expected columns present")

    if extra_cols:
        print(f"Extra columns found (will be dropped): {extra_cols}")

    # Print summary
    print("\n--- Dataset Summary ---")
    print(f"Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts().to_string())
    print(f"\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\nNumerical statistics:")
    print(df.describe().round(2).to_string())
    print("\nData Registration Complete ✅")

if __name__ == "__main__":
    register_data(DATA_PATH)
