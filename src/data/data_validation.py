"""
Data validation module using Great Expectations.
Validates data quality and schema.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class DataValidator:
    """Validates data quality using predefined rules."""

    def __init__(self):
        """Initialize data validator."""
        self.target_column = config.get("model.target_column")
        logger.info("DataValidator initialized")

    def validate_schema(self, df: pd.DataFrame) -> dict:
        """
        Validate dataframe schema.

        Args:
            df: Input dataframe

        Returns:
            Validation results dictionary
        """
        logger.info("Validating data schema...")

        results = {"schema_valid": True, "errors": [], "warnings": []}

        # Check if target column exists
        if self.target_column not in df.columns:
            results["schema_valid"] = False
            results["errors"].append(f"Target column '{self.target_column}' not found")

        # Check for empty dataframe
        if len(df) == 0:
            results["schema_valid"] = False
            results["errors"].append("Dataframe is empty")

        # Check for columns with all null values
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            results["warnings"].append(f"Columns with all null values: {null_columns}")

        logger.info(
            f"Schema validation: {'PASSED' if results['schema_valid'] else 'FAILED'}"
        )

        return results

    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality.

        Args:
            df: Input dataframe

        Returns:
            Validation results dictionary
        """
        logger.info("Validating data quality...")

        results = {"quality_valid": True, "checks": {}}

        # Check missing value percentage
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        results["checks"]["missing_percentage"] = missing_pct

        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 50}
        if high_missing:
            results["quality_valid"] = False
            logger.warning(f"Columns with >50% missing values: {high_missing}")

        # Check duplicate percentage
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        results["checks"]["duplicate_percentage"] = dup_pct

        if dup_pct > 10:
            results["quality_valid"] = False
            logger.warning(f"High duplicate percentage: {dup_pct:.2f}%")

        # Check target distribution
        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts(normalize=True).to_dict()
            results["checks"]["target_distribution"] = target_dist

            # Check for severe class imbalance
            if len(target_dist) > 0:
                min_class_pct = min(target_dist.values()) * 100
                if min_class_pct < 5:
                    logger.warning(
                        f"Severe class imbalance detected: {min_class_pct:.2f}%"
                    )

        # Check numerical columns for outliers (using IQR method)
        numerical_cols = df.select_dtypes(include=["number"]).columns
        outlier_counts = {}

        for col in numerical_cols:
            if col != self.target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))
                ).sum()
                outlier_counts[col] = int(outliers)

        results["checks"]["outlier_counts"] = outlier_counts

        logger.info(
            f"Data quality validation: {'PASSED' if results['quality_valid'] else 'FAILED'}"
        )

        return results

    def validate_business_rules(self, df: pd.DataFrame) -> dict:
        """
        Validate business-specific rules.

        Args:
            df: Input dataframe

        Returns:
            Validation results dictionary
        """
        logger.info("Validating business rules...")

        results = {"rules_valid": True, "violations": []}

        # Rule 1: tenure should be non-negative
        if "tenure" in df.columns:
            negative_tenure = (df["tenure"] < 0).sum()
            if negative_tenure > 0:
                results["rules_valid"] = False
                results["violations"].append(
                    f"Found {negative_tenure} records with negative tenure"
                )

        # Rule 2: MonthlyCharges should be positive
        if "MonthlyCharges" in df.columns:
            invalid_charges = (df["MonthlyCharges"] <= 0).sum()
            if invalid_charges > 0:
                results["rules_valid"] = False
                results["violations"].append(
                    f"Found {invalid_charges} records with invalid monthly charges"
                )

        # Rule 3: TotalCharges should be >= MonthlyCharges for customers with tenure > 0
        if all(
            col in df.columns for col in ["TotalCharges", "MonthlyCharges", "tenure"]
        ):
            invalid_total = (
                (df["tenure"] > 0) & (df["TotalCharges"] < df["MonthlyCharges"])
            ).sum()
            if invalid_total > 0:
                results["violations"].append(
                    f"Found {invalid_total} records where TotalCharges < MonthlyCharges despite tenure > 0"
                )

        logger.info(
            f"Business rules validation: {'PASSED' if results['rules_valid'] else 'FAILED'}"
        )

        return results

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Run all validations.

        Args:
            df: Input dataframe

        Returns:
            Combined validation results
        """
        logger.info("=" * 60)
        logger.info("Starting data validation")
        logger.info("=" * 60)

        results = {
            "overall_valid": True,
            "schema": self.validate_schema(df),
            "quality": self.validate_data_quality(df),
            "business_rules": self.validate_business_rules(df),
        }

        # Overall validation status
        results["overall_valid"] = (
            results["schema"]["schema_valid"]
            and results["quality"]["quality_valid"]
            and results["business_rules"]["rules_valid"]
        )

        logger.info("=" * 60)
        logger.info(
            f"Validation result: {'PASSED ✓' if results['overall_valid'] else 'FAILED ✗'}"
        )
        logger.info("=" * 60)

        return results


def main():
    """Main function to run validation."""
    validator = DataValidator()

    # Validate raw data
    raw_data_path = config.get("data.raw_data_path")
    logger.info(f"\nValidating raw data: {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)
    results_raw = validator.validate(df_raw)

    # Validate processed data if it exists
    processed_data_path = (
        Path(config.get("data.processed_data_path")) / "churn_data_cleaned.csv"
    )
    if processed_data_path.exists():
        logger.info(f"\nValidating processed data: {processed_data_path}")
        df_processed = pd.read_csv(processed_data_path)
        results_processed = validator.validate(df_processed)

    return results_raw


if __name__ == "__main__":
    main()
