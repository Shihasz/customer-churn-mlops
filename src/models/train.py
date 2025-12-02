"""
Model training module for customer churn prediction.
Trains XGBoost model with MLflow tracking.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_dir, save_json
from config import config

logger = get_logger(__name__)


class ChurnModelTrainer:
    """Handles model training for churn prediction."""

    def __init__(self):
        """Initialize model trainer with configuration."""
        self.target_column = config.get("model.target_column")
        self.test_size = config.get("data.test_size")
        self.random_state = config.get("data.random_state")
        self.model_type = config.get("model.type")
        self.hyperparameters = config.get("model.hyperparameters")

        # MLflow configuration
        self.mlflow_tracking_uri = config.get("mlflow.tracking_uri")
        self.experiment_name = config.get("mlflow.experiment_name")

        self.model = None

        logger.info("ChurnModelTrainer initialized")

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training.

        Args:
            df: Input dataframe with features and target

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")

        # Separate features and target
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataframe"
            )

        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        # Remove non-numeric columns (keep only encoded versions)
        non_numeric_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if non_numeric_cols:
            logger.info(f"Dropping non-numeric columns: {non_numeric_cols}")
            X = X.drop(non_numeric_cols, axis=1)

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained model
        """
        logger.info("=" * 60)
        logger.info("Training XGBoost model...")
        logger.info("=" * 60)

        # Log hyperparameters
        logger.info(f"Hyperparameters: {self.hyperparameters}")

        # Initialize model
        self.model = xgb.XGBClassifier(**self.hyperparameters)

        # Train model
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

        logger.info("Model training completed")

        return self.model

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("Evaluating model...")
        logger.info("=" * 60)

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Log metrics
        logger.info("Model Performance:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        logger.info("\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
        logger.info(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")

        # Classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))

        logger.info("=" * 60)

        return metrics

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importance = self.model.feature_importances_

        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        logger.info("\nTop 10 Important Features:")
        logger.info(feature_importance.head(10).to_string(index=False))

        return feature_importance

    def save_model(self, output_dir: str, model_name: str = "churn_model") -> str:
        """
        Save trained model to disk.

        Args:
            output_dir: Directory to save model
            model_name: Name of the model file

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        output_path = Path(output_dir)
        ensure_dir(output_path)

        model_path = output_path / f"{model_name}.pkl"

        import joblib

        joblib.dump(self.model, model_path)

        logger.info(f"Model saved to {model_path}")

        return str(model_path)

    def train_and_evaluate(self, df: pd.DataFrame) -> dict:
        """
        Complete training and evaluation pipeline.

        Args:
            df: Input dataframe with features and target

        Returns:
            Dictionary with model, metrics, and other results
        """
        logger.info("=" * 60)
        logger.info("Starting training and evaluation pipeline")
        logger.info("=" * 60)

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Train model
        model = self.train_model(X_train, y_train)

        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)

        # Feature importance
        feature_importance = self.get_feature_importance(X_train.columns.tolist())

        # Save model
        model_path = self.save_model("models")

        results = {
            "model": model,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model_path": model_path,
            "feature_names": X_train.columns.tolist(),
            "n_features": X_train.shape[1],
        }

        logger.info("=" * 60)
        logger.info("Training and evaluation completed successfully")
        logger.info("=" * 60)

        return results


def main():
    """Main function to run model training."""
    # Load engineered data
    data_path = Path(config.get("data.processed_data_path")) / "churn_data_features.csv"

    if not data_path.exists():
        logger.error(f"Engineered data not found at {data_path}")
        logger.info("Please run feature engineering first:")
        logger.info("  python3 src/data/feature_engineering.py")
        return

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Initialize trainer
    trainer = ChurnModelTrainer()

    # Train and evaluate
    results = trainer.train_and_evaluate(df)

    # Save metrics
    metrics_path = Path("models") / "metrics.json"
    save_json(results["metrics"], str(metrics_path))
    logger.info(f"Metrics saved to {metrics_path}")

    # Save feature importance
    fi_path = Path("models") / "feature_importance.csv"
    results["feature_importance"].to_csv(fi_path, index=False)
    logger.info(f"Feature importance saved to {fi_path}")

    return results


if __name__ == "__main__":
    main()
