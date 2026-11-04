
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# To track ML Flow
import mlflow
import mlflow.sklearn

DATA_PATH = "data/raw/students.csv"
MODEL_DIR = "models"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score"
]

TARGET = "final_score"


def train_models():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    algorithms = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = []
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("Student Performance Prediction")

    best_mae = float("inf")
    best_model = None
    best_name = None

    for name, algorithm in algorithms.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", algorithm)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print("MAE:", round(mae, 2))
        print("RMSE:", round(rmse, 2))
        print("R2:", round(r2, 4))

        # To Track ML Flow
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_metric("mae", float(mae))
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("r2", float(r2))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="student-performance-model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree"
                ]
            )
        results.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        if mae < best_mae:
            best_mae = mae
            best_model = pipeline
            best_name = name

    joblib.dump(best_model, f"{MODEL_DIR}/student_model.pkl")

    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "mae": float(best_mae)
    }

    with open(f"{MODEL_DIR}/metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nBest model:", best_name)
    print("Model saved successfully!")


if __name__ == "__main__":
    train_models()
