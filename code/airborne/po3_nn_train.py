#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models

FEATURE_COLUMNS = [
    "smoke_age",
    "normalized_distance_to_center",
    "CH2O",
    "NO2",
]
TARGET_COLUMN = "pO3"
FIRE_ID_COLUMN = "fire_id"
TEST_FIRE_ID = 1.0

HIDDEN_LAYERS = [1, 2]
HIDDEN_UNITS = [16, 32, 64]
LEARNING_RATES = [1e-4, 1e-3]
WEIGHT_DECAY = [0, 1e-4, 1e-3]
DROPOUT = [0, 0.1]

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_PATIENCE = 10
HIGH_PO3_THRESHOLD = 50.0
HIGH_PO3_WEIGHT = 10.0

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = FEATURE_COLUMNS + [TARGET_COLUMN, FIRE_ID_COLUMN]
    data = df[cols].dropna()
    data = data[data[TARGET_COLUMN] > 0].copy()
    return data

def split_by_fire(
    data: pd.DataFrame,
    test_fire_id: float = TEST_FIRE_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[data[FIRE_ID_COLUMN] != test_fire_id]
    test = data[data[FIRE_ID_COLUMN] == test_fire_id]
    if len(train) == 0:
        raise ValueError("Training set is empty after fire split.")
    if len(test) == 0:
        raise ValueError(f"No rows found for test fire_id={test_fire_id}.")
    return train, test

def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def sample_weights(
    y: np.ndarray,
    threshold: float = HIGH_PO3_THRESHOLD,
    high_weight: float = HIGH_PO3_WEIGHT,
) -> np.ndarray:
    return np.where(y > threshold, high_weight, 1.0).astype(float)

def build_tuned_model(
    n_features: int,
    n_layers: int,
    units: int,
    dropout: float,
) -> models.Sequential:
    model = models.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    for _ in range(n_layers):
        model.add(layers.Dense(units, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    return model

def build_baseline_model(n_features: int) -> models.Sequential:
    return models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])

def predict_po3(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return np.exp(model.predict(X, verbose=0).flatten()) - 1

def make_early_stopper(patience: int) -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

def fit_model(
    model: keras.Model,
    X_train_scaled: np.ndarray,
    y_train_log: np.ndarray,
    *,
    optimizer: keras.optimizers.Optimizer,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
    sample_weight: np.ndarray | None = None,
) -> keras.callbacks.History:
    model.compile(optimizer=optimizer, loss="mse")
    return model.fit(
        X_train_scaled,
        y_train_log,
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[make_early_stopper(patience)],
        verbose=0,
        shuffle=True,
    )

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.maximum(y_pred, 0))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
    }

def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label} — "
        f"RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}, "
        f"log RMSE: {metrics['rmse_log']:.3f}, log R²: {metrics['r2_log']:.3f}"
    )

def train_baseline(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
) -> tuple[keras.Model, dict[str, float]]:
    K.clear_session()
    model = build_baseline_model(X_train_scaled.shape[1])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    fit_model(
        model,
        X_train_scaled,
        np.log1p(y_train),
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        patience=patience,
        sample_weight=sample_weights(y_train),
    )
    y_pred = predict_po3(model, X_test_scaled)
    metrics = evaluate_predictions(y_test, y_pred)
    return model, metrics

def run_grid_search(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
    progress_every: int = 10,
) -> pd.DataFrame:
    y_train_log = np.log1p(y_train)
    grid = list(product(HIDDEN_LAYERS, HIDDEN_UNITS, LEARNING_RATES, WEIGHT_DECAY, DROPOUT))
    results: list[dict] = []

    print(f"Running grid search over {len(grid)} configurations...")
    for i, (n_layers, units, lr, weight_decay, dropout) in enumerate(grid, start=1):
        K.clear_session()
        candidate = build_tuned_model(
            X_train_scaled.shape[1],
            n_layers,
            units,
            dropout,
        )
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        history = fit_model(
            candidate,
            X_train_scaled,
            y_train_log,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            patience=patience,
            sample_weight=sample_weights(y_train),
        )

        val_loss = float(min(history.history["val_loss"]))
        y_pred = predict_po3(candidate, X_test_scaled)
        test_metrics = evaluate_predictions(y_test, y_pred)

        results.append({
            "hidden_layers": n_layers,
            "hidden_units": units,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "val_loss": val_loss,
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_rmse_log": test_metrics["rmse_log"],
            "test_r2_log": test_metrics["r2_log"],
        })

        if i % progress_every == 0 or i == len(grid):
            print(f"  {i}/{len(grid)} done")

    return pd.DataFrame(results).sort_values("val_loss").reset_index(drop=True)

def train_best_config(
    best: pd.Series,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
) -> tuple[keras.Model, dict[str, float]]:
    K.clear_session()
    model = build_tuned_model(
        X_train_scaled.shape[1],
        int(best["hidden_layers"]),
        int(best["hidden_units"]),
        float(best["dropout"]),
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=float(best["learning_rate"]),
        weight_decay=float(best["weight_decay"]),
    )
    fit_model(
        model,
        X_train_scaled,
        np.log1p(y_train),
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        patience=patience,
        sample_weight=sample_weights(y_train),
    )
    y_pred = predict_po3(model, X_test_scaled)
    metrics = evaluate_predictions(y_test, y_pred)
    return model, metrics

def save_artifacts(
    model: keras.Model,
    scaler: StandardScaler,
    output_dir: Path,
    *,
    test_fire_id: float = TEST_FIRE_ID,
    results_df: pd.DataFrame | None = None,
    best_config: dict | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "po3_nn_model.keras"
    scaler_path = output_dir / "po3_scaler.pkl"
    model.save(model_path)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    if results_df is not None:
        results_df.to_csv(output_dir / "po3_nn_grid_search_results.csv", index=False)

    summary = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "sample_weight": {
            "threshold": HIGH_PO3_THRESHOLD,
            "weight_above": HIGH_PO3_WEIGHT,
            "weight_at_or_below": 1.0,
        },
        "test_fire_id": test_fire_id,
        "best_config": best_config,
        "test_metrics": metrics,
    }
    with (output_dir / "po3_nn_training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "training_data_po3.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Directory for saved model, scaler, and results",
    )
    parser.add_argument(
        "--export-training",
        type=Path,
        default=None,
        help="Results CSV from steady_state_po3.jl. Write the training table and exit.",
    )
    parser.add_argument(
        "--test-fire-id",
        type=float,
        default=TEST_FIRE_ID,
        help="Fire ID held out for testing",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip Step 4 baseline model training",
    )
    parser.add_argument(
        "--skip-grid-search",
        action="store_true",
        help="Skip hyperparameter grid search",
    )
    parser.add_argument(
        "--baseline-lr",
        type=float,
        default=1e-3,
        help="Learning rate for baseline Adam optimizer",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TensorFlow and NumPy",
    )
    return parser.parse_args()

def export_training_table(results_csv: Path, out_csv: Path) -> None:
    columns = [
        "smoke_age",
        "normalized_distance_to_center",
        "CH2O",
        "NO2",
        "pO3",
        "fire_id",
    ]
    df = pd.read_csv(results_csv, usecols=columns)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

def main() -> None:
    args = parse_args()
    if args.export_training is not None:
        export_training_table(args.export_training, args.data)
        return

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(f"Loading data from {args.data}")
    data = load_data(args.data)
    train_df, test_df = split_by_fire(data, test_fire_id=args.test_fire_id)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    print(
        f"Train samples: {len(train_df)} "
        f"({train_df[FIRE_ID_COLUMN].nunique()} fires), "
        f"test samples: {len(test_df)} (fire {args.test_fire_id})"
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    baseline_metrics = None
    if not args.skip_baseline:
        print("\nTraining baseline model (2x64 Dense, Adam)...")
        _, baseline_metrics = train_baseline(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            learning_rate=args.baseline_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            patience=args.patience,
        )
        print_metrics("Baseline", baseline_metrics)

    results_df = None
    best_config = None
    best_metrics = None
    final_model = None

    if not args.skip_grid_search:
        print("\nStarting hyperparameter grid search...")
        results_df = run_grid_search(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            patience=args.patience,
        )

        best = results_df.iloc[0]
        best_config = best.to_dict()
        print("\nBest configuration (lowest val_loss):")
        for key, value in best_config.items():
            print(f"  {key}: {value}")

        if baseline_metrics is not None:
            print(
                f"\nBaseline test RMSE: {baseline_metrics['rmse']:.2f}, "
                f"R²: {baseline_metrics['r2']:.3f}"
            )
        print(
            f"Best grid config test RMSE: {best['test_rmse']:.2f}, "
            f"R²: {best['test_r2']:.3f}"
        )

        print("\nRetraining best configuration...")
        final_model, best_metrics = train_best_config(
            best,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            patience=args.patience,
        )
        print_metrics("Retrained best", best_metrics)

    if final_model is None and not args.skip_baseline:
        print("\nSaving baseline model (grid search skipped)...")
        final_model, best_metrics = train_baseline(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            learning_rate=args.baseline_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            patience=args.patience,
        )
        best_config = {
            "hidden_layers": 2,
            "hidden_units": 64,
            "learning_rate": args.baseline_lr,
            "weight_decay": 0.0,
            "dropout": 0.0,
        }

    if final_model is None:
        raise RuntimeError("No model was trained. Remove --skip-baseline and/or --skip-grid-search.")

    save_artifacts(
        final_model,
        scaler,
        args.output_dir,
        test_fire_id=args.test_fire_id,
        results_df=results_df,
        best_config=best_config,
        metrics=best_metrics,
    )

    if results_df is not None:
        print(f"Saved grid search results to {args.output_dir / 'po3_nn_grid_search_results.csv'}")

if __name__ == "__main__":
    main()
