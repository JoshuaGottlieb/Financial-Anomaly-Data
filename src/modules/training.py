import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import ParameterGrid

# Custom functions for loading and saving
from modules.io_utils import save_object

def fit_models(
    base_model,
    model_type: str,
    dataset_type: str,
    grid_dict: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: str,
    models_dir: str
) -> None:
    """
    Fit a model for every parameter combination in a grid and saves the fitted model to disk.
    Computes anomaly scores and binary anomaly predictions, evaluates clustering metrics, and
    saves the results to CSV files for downstream analysis.

    Args:
        base_model: The sklearn-compatible estimator (e.g., GaussianMixture).
        model_type (str): Identifier used for naming output files.
        dataset_type (str): Tag appended to filenames (e.g., 'raw', 'reduced').
        grid_dict (Dict[str, Any]): Parameter grid defining model variations.
        train_df (pd.DataFrame): Training data; last column is ignored.
        test_df (pd.DataFrame): Test data; last column must contain anomaly labels.
        results_dir (str): Directory to which output CSV files will be written.
    """

    # Convert the parameter grid into a list of unique configurations
    param_grid = list(ParameterGrid(grid_dict))
    total_models = len(param_grid)

    # Split the input data: all columns except the last are used as features
    X_train = train_df.iloc[:, :-1]
    X_test = test_df.iloc[:, :-1]

    # Gaussian Mixture Models do not directly have predictions for anomalies
    if model_type == "gmm":
        # The last column in the train set contains suspected anomaly labels for thresholding
        y_train = train_df.iloc[:, -1]
    
        # Estimate the expected anomaly rate so we can threshold score_samples appropriately for GMM
        expected_anomaly_rate = y_train.sum() / len(y_train)
    
    # The last column in the test set contains suspected anomaly labels for evaluation
    y_test = test_df.iloc[:, -1]

    # Dictionaries for storing all model predictions and evaluation metrics
    all_predictions = {}
    all_metrics = {}

    print(f"Beginning model fitting over {total_models} parameter combinations...\n")

    for i, params in enumerate(param_grid, start = 1):
        print(f"[{i}/{total_models}] Fitting model with parameters: {params}")

        # Create a fresh model instance for the given parameter configuration
        model = clone(base_model).set_params(**params)

        # Train the model on the full feature set
        model.fit(X_train)

        # Instantiate dictionary for holding model metrics
        metrics = {}
        
        # Add information criteria and thresholding for Gaussian Mixture Models
        if model_type == "gmm":
            metrics["aic"] = model.aic(X_test)
            metrics["bic"] = model.bic(X_test)

            # Compute sample-level anomaly scores for the test set
            scores = model.score_samples(X_test)
            
            # Convert scores into anomaly predictions using a percentile threshold
            threshold = np.quantile(scores, expected_anomaly_rate)
            predictions = (scores < threshold).astype(int)

        # Use built-in methods for predictions for Isolation Forest
        elif model_type == "iforest":
            # Use decision function to calculate sample-level amonaly scores for the test set
            scores = model.decision_function(X_test)

            # Calculate predictions directly using predict: -1 for anomaly and 1 for normal
            predictions = model.predict(X_test)

            # Convert predictions to labels: 0 for normal, 1 for anomaly
            predictions = (predictions == -1).astype(int)

        # Compute clustering-based evaluation metrics on predicted anomaly groups
        metrics.update({
            "silhouette_score": silhouette_score(X_test, predictions),
            "calinski_harabasz_score": calinski_harabasz_score(X_test, predictions),
            "davies_bouldin_score": davies_bouldin_score(X_test, predictions)
        })

        # Create a readable parameter identifier for storing and saving outputs
        params_str = "-".join(f"{k}_{v}" for k, v in params.items())

        # Store metrics and predictions for this parameter configuration
        all_metrics[params_str] = metrics
        all_predictions[f"{params_str}-scores"] = scores
        all_predictions[f"{params_str}-predictions"] = predictions

        # Save trained model
        model_filename = f"{model_type}-{dataset_type}-{params_str}"
        model_path = os.path.join(models_dir, model_filename)

        save_object(model, model_path)

        print(f"[{i}/{total_models}] Model saved to: {model_path}\n")

    # Store the ground-truth anomaly labels alongside predictions
    all_predictions["suspected_anomaly"] = y_test.values

    # Convert results into DataFrames for exporting
    metrics_frame = pd.DataFrame(all_metrics).reset_index(names = "score")
    predictions_frame = pd.DataFrame(all_predictions)

    # Build output file paths
    metrics_path = os.path.join(results_dir, f"{model_type}-{dataset_type}-metrics.csv")
    predictions_path = os.path.join(results_dir, f"{model_type}-{dataset_type}-predictions.csv")

    # Write results to CSV
    metrics_frame.to_csv(metrics_path, index = False)
    predictions_frame.to_csv(predictions_path, index = False)

    print("All models completed.")
    print(f"Metrics written to: {metrics_path}")
    print(f"Predictions written to: {predictions_path}")

    return