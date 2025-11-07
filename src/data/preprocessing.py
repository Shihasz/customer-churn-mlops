"""
Data preprocessing module for customer churn prediction.
Handles data cleaning, missing values, and basic transformations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_dir
from config import config

logger = get_logger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for churn prediction."""

    def __init__(self):
        """Initialize the preprocessor with configuration."""
        self.target_column = config.get("model.target_column")
        self.random_state = config.get("data.random_state")
        logger.info("DataPreprocessor initialized")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with handled missing values.
        """
        logger.info("Handling missing values...")

        # Log missing values before handling
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        else:
            logger.info("No missing values found")

        # Convert 'TotalCharges' to numeric (might have spaces)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

            # Fill missing TotalCharges with median
            if df["TotalCharges"].isnull().sum() > 0:
                median_charges = df["TotalCharges"].median()
                df["TotalCharges"].fillna(median_charges, inplace=True)
                logger.info(
                    f"Filled {df['TotalCharges'].isnull().sum()} missing TotalCharges with median: {median_charges:.2f}"
                )

        # Drop rows with missing target variable
        if self.target_column in df.columns:
            initial_rows = len(df)
            df = df.dropna(subset=[self.target_column])
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing target variable")

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe without duplicates.
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            logger.info(f"Removed {dropped_rows} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing irrelevant columns and fixing data types.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Cleaning data...")

        # Remove customerID column (not useful for modeling)
        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)
            logger.info("Removed customerID column")

        # Convert target variable to binary (0/1)
        if self.target_column in df.columns:
            df[self.target_column] = df[self.target_column].map({"Yes": 1, "No": 0})
            logger.info(f"Converted {self.target_column} to binary (0/1)")
            logger.info(f"Churn distribution:\n{df[self.target_column].value_counts()}")

        # Convert SeniorCitizen to Yes/No for consistency
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
            logger.info("Converted SeniorCitizen to Yes/No format")

        return df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the dataset.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_duplicates": df.duplicated().sum(),
            "n_missing": df.isnull().sum().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(
                include=["object"]
            ).columns.tolist(),
        }

        if self.target_column in df.columns:
            # Check if target is numeric (0/1) before calculating mean (churn rate)
            if df[self.target_column].dtype in [np.int64, np.float64]:
                summary["target_distribution"] = (
                    df[self.target_column].value_counts().to_dict()
                )
                # Churn rate is the mean of the binary target column
                summary["churn_rate"] = df[self.target_column].mean()
            else:
                # If target is still categorical, only provide distribution
                summary["target_distribution"] = (
                    df[self.target_column].value_counts().to_dict()
                )
                logger.warning(
                    f"Target column '{self.target_column}' is not numeric. Skipping churn rate calculation."
                )
        return summary

    def preprocess(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Args:
            input_path (str): Path to the input CSV file.
            output_path (str, optional): Path to save the preprocessed CSV file. Defaults to None.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        logger.info("=" * 60)
        logger.info("Starting data preprocessing pipeline")
        logger.info("=" * 60)

        # Load data
        df = self.load_data(input_path)

        # Get initial summary
        initial_summary = self.get_data_summary(df)
        logger.info(
            f"Initial data summary: {initial_summary['n_rows']} rows, {initial_summary['n_columns']} columns"
        )

        # Preprocessing steps
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.clean_data(df)

        # Get final summary
        final_summary = self.get_data_summary(df)
        logger.info(
            f"Final data summary: {final_summary['n_rows']} rows, {final_summary['n_columns']} columns"
        )
        logger.info(f"Churn rate: {final_summary.get('churn_rate', 0):.2%}")

        # Save processed data if output path is provided
        if output_path:
            ensure_dir(Path(output_path).parent)
            df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")

        logger.info("=" * 60)
        logger.info("Data preprocessing completed successfully")
        logger.info("=" * 60)

        return df


def main():
    """Main function to run preprocessing."""
    preprocessor = DataPreprocessor()

    # Get paths from config
    input_path = config.get("data.raw_data_path")
    output_path = (
        Path(config.get("data.processed_data_path")) / "churn_data_cleaned.csv"
    )

    # Run preprocessing
    df = preprocessor.preprocess(input_path, str(output_path))

    return df


if __name__ == "__main__":
    main()
