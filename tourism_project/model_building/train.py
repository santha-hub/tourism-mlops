"""
Model Training Script with MLflow Experiment Tracking
- Loads train/test splits
- Builds preprocessing + XGBoost pipeline
- Tunes hyperparameters with GridSearchCV
- Logs params/metrics to MLflow
- Saves best model to deployment folder
"""

import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, f1_score,
                              accuracy_score, roc_auc_score)
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "tourism_project/model_building/train.csv"
TEST_PATH   = "tourism_project/model_building/test.csv"
MODEL_PATH  = "tourism_project/deployment/best_model.joblib"
MLFLOW_DIR  = "tourism_project/model_building/mlruns"

# ── Feature definitions ────────────────────────────────────────────────────────
CAT_FEATURES = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]
NUM_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]
TARGET = "ProdTaken"

def build_pipeline():
    """Build the preprocessing + model pipeline."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)
    ])
    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=4   # handles class imbalance (80:20 ratio)
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

def train():
    print("=" * 60)
    print("MODEL TRAINING WITH MLFLOW TRACKING")
    print("=" * 60)

    # Load splits
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///tourism_project/model_building/mlflow.db")
    mlflow.set_experiment("wellness_tourism_package_prediction")

    # Hyperparameter grid
    param_grid = {
        "model__n_estimators":  [100, 200],
        "model__max_depth":     [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample":     [0.8, 1.0]
    }

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="xgboost_gridsearch"):

        print("\nRunning GridSearchCV (this may take a few minutes)...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model  = grid_search.best_estimator_

        print(f"\nBest parameters: {best_params}")
        print(f"Best CV F1     : {grid_search.best_score_:.4f}")

        # Log all tuned parameters
        for param, value in best_params.items():
            mlflow.log_param(param.replace("model__", ""), value)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring_metric", "f1")

        # Evaluate on test set
        y_pred      = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]

        test_f1       = f1_score(y_test, y_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_roc_auc  = roc_auc_score(y_test, y_pred_prob)

        # Log metrics
        mlflow.log_metric("test_f1",       test_f1)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_roc_auc",  test_roc_auc)
        mlflow.log_metric("best_cv_f1",    grid_search.best_score_)

        # Log the model
        mlflow.sklearn.log_model(
    best_model,
    "best_model",
    skops_trusted_types=[
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]
)

        print(f"\n── Test Set Performance ──")
        print(f"F1 Score : {test_f1:.4f}")
        print(f"Accuracy : {test_accuracy:.4f}")
        print(f"ROC-AUC  : {test_roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
              target_names=["Not Purchased", "Purchased"]))

    # Save best model for deployment
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model saved to: {MODEL_PATH} ✅")
    print("\nModel Training Complete ✅")

if __name__ == "__main__":
    train()
