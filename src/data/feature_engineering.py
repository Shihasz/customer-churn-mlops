"""
Feature engineering module for customer churn prediction.
Creates new features and prepares data for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_dir, save_json
from config import config

logger = get_logger(__name__)


class FeatureEngineer:
    """Handles feature engineering for churn prediction."""

    def __init__(self):
        """Initialize feature engineer with configuration."""
        self.target_column = config.get("model.target_column")
        self.numerical_features = config.get("model.features.numerical", [])
        self.categorical_features = config.get("model.features.categorical", [])
        self.scaler = StandardScaler()
        self.label_encoders = {}
        logger.info("FeatureEngineer initialized")

    def create_tenure_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure groups from tenure months.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with tenure groups
        """
        if "tenure" in df.columns:
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 72],
                labels=["0-1 year", "1-2 years", "2-4 years", "4+ years"],
            )
            logger.info("Created tenure_group feature")

        return df

    def create_charge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to charges.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with charge features
        """
        if "TotalCharges" in df.columns and "tenure" in df.columns:
            # Average monthly charges (total/tenure)
            df["avg_monthly_charges"] = df["TotalCharges"] / (
                df["tenure"] + 1
            )  # +1 to avoid division by zero

            # Charge ratio (monthly vs average)
            if "MonthlyCharges" in df.columns:
                df["charge_ratio"] = df["MonthlyCharges"] / df["avg_monthly_charges"]

            logger.info("Created charge-related features")

        return df

    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from service subscriptions.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with service features
        """
        # Count of services
        service_columns = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        available_services = [col for col in service_columns if col in df.columns]

        if available_services:
            # Count "Yes" across service columns
            df["total_services"] = df[available_services].apply(
                lambda row: sum(1 for val in row if val == "Yes"), axis=1
            )
            logger.info(
                f"Created total_services feature from {len(available_services)} service columns"
            )

        # Internet service binary
        if "InternetService" in df.columns:
            df["has_internet"] = (df["InternetService"] != "No").astype(int)
            logger.info("Created has_internet feature")

        # Phone service binary
        if "PhoneService" in df.columns:
            df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)
            logger.info("Created has_phone feature")

        return df

    def encode_categorical_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.

        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)

        Returns:
            Dataframe with encoded features
        """
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # Remove target column if present
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)

        logger.info(f"Encoding {len(categorical_cols)} categorical features")

        for col in categorical_cols:
            if fit:
                # Fit and transform
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
            else:
                # Transform only (use fitted encoder)
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels
                    df[f"{col}_encoded"] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    logger.warning(f"No encoder found for {col}, skipping")

        return df

    def scale_numerical_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            df: Input dataframe
            fit: Whether to fit scaler (True for training, False for inference)

        Returns:
            Dataframe with scaled features
        """
        # Get numerical columns (excluding target)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)

        # Remove encoded columns (we'll scale originals only)
        numerical_cols = [col for col in numerical_cols if not col.endswith("_encoded")]

        logger.info(f"Scaling {len(numerical_cols)} numerical features")

        if numerical_cols:
            if fit:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
                logger.info("Fitted and transformed numerical features")
            else:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
                logger.info("Transformed numerical features using fitted scaler")

        return df

    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Main feature engineering pipeline.

        Args:
            df: Input dataframe
            fit: Whether to fit transformers

        Returns:
            Dataframe with engineered features
        """
        logger.info("=" * 60)
        logger.info("Starting feature engineering pipeline")
        logger.info("=" * 60)

        # Create new features
        df = self.create_tenure_groups(df)
        df = self.create_charge_features(df)
        df = self.create_service_features(df)

        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)

        # Scale numerical features
        df = self.scale_numerical_features(df, fit=fit)

        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        logger.info("=" * 60)

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature names (excluding target).

        Args:
            df: Input dataframe

        Returns:
            List of feature names
        """
        features = [col for col in df.columns if col != self.target_column]
        return features

    def save_artifacts(self, output_dir: str) -> None:
        """
        Save feature engineering artifacts (scalers, encoders).

        Args:
            output_dir: Directory to save artifacts
        """
        import joblib

        output_path = Path(output_dir)
        ensure_dir(output_path)

        # Save scaler
        scaler_path = output_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        # Save label encoders
        encoders_path = output_path / "label_encoders.pkl"
        joblib.dump(self.label_encoders, encoders_path)
        logger.info(f"Saved label encoders to {encoders_path}")

        # Save feature metadata
        metadata = {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "n_encoders": len(self.label_encoders),
        }
        save_json(metadata, str(output_path / "feature_metadata.json"))
        logger.info(
            f"Saved feature metadata to {output_path / 'feature_metadata.json'}"
        )


def main():
    """Main function to run feature engineering."""
    from src.data.preprocessing import DataPreprocessor

    # Load preprocessed data
    preprocessor = DataPreprocessor()
    input_path = Path(config.get("data.processed_data_path")) / "churn_data_cleaned.csv"

    if not input_path.exists():
        logger.info("Preprocessed data not found. Running preprocessing first...")
        df = preprocessor.preprocess(config.get("data.raw_data_path"), str(input_path))
    else:
        df = preprocessor.load_data(str(input_path))

    # Engineer features
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df, fit=True)

    # Save engineered data
    output_path = (
        Path(config.get("data.processed_data_path")) / "churn_data_features.csv"
    )
    df_engineered.to_csv(output_path, index=False)
    logger.info(f"Engineered data saved to {output_path}")

    # Save artifacts
    engineer.save_artifacts(config.get("data.processed_data_path"))

    return df_engineered


if __name__ == "__main__":
    main()
