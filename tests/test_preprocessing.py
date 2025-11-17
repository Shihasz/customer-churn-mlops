"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        "customerID": ["CUST001", "CUST002", "CUST003", "CUST004", "CUST005"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "tenure": [12, 24, 6, 48, 3],
        "MonthlyCharges": [50.0, 80.0, 30.0, 100.0, 25.0],
        "TotalCharges": [600.0, 1920.0, 180.0, 4800.0, ""],
        "Contract": [
            "Month-to-month",
            "One year",
            "Month-to-month",
            "Two year",
            "Month-to-month",
        ],
        "Churn": ["No", "No", "Yes", "No", "Yes"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance."""
    return DataPreprocessor()


def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization."""
    assert preprocessor.target_column == "Churn"
    assert preprocessor.random_state == 42


def test_handle_missing_values(preprocessor, sample_data):
    """Test missing value handling."""
    df = preprocessor.handle_missing_values(sample_data)

    # Check TotalCharges was converted to numeric and filled
    assert df["TotalCharges"].dtype in [np.float64, np.int64]
    assert df["TotalCharges"].isnull().sum() == 0


def test_remove_duplicates(preprocessor, sample_data):
    """Test duplicate removal."""
    # Add a duplicate row
    df_with_dup = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)

    df = preprocessor.remove_duplicates(df_with_dup)

    # Check duplicate was removed
    assert len(df) == len(sample_data)


def test_clean_data(preprocessor, sample_data):
    """Test data cleaning."""
    df = preprocessor.clean_data(sample_data)

    # Check customerID was removed
    assert "customerID" not in df.columns

    # Check Churn was converted to binary
    assert df["Churn"].dtype in [np.int64, np.float64]
    assert set(df["Churn"].unique()).issubset({0, 1})

    # Check SeniorCitizen was converted to Yes/No
    assert df["SeniorCitizen"].dtype == object
    assert set(df["SeniorCitizen"].unique()).issubset({"Yes", "No"})


def test_get_data_summary(preprocessor, sample_data):
    """Test data summary generation."""
    summary = preprocessor.get_data_summary(sample_data)

    # Check summary contains expected keys
    assert "n_rows" in summary
    assert "n_columns" in summary
    assert "numerical_columns" in summary
    assert "categorical_columns" in summary

    # Check values
    assert summary["n_rows"] == len(sample_data)
    assert summary["n_columns"] == len(sample_data.columns)


def test_preprocess_pipeline(preprocessor, sample_data, tmp_path):
    """Test complete preprocessing pipeline."""
    # Save sample data to temporary file
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sample_data.to_csv(input_path, index=False)

    # Run preprocessing
    df = preprocessor.preprocess(str(input_path), str(output_path))

    # Check output file was created
    assert output_path.exists()

    # Check data was processed
    assert "customerID" not in df.columns
    assert df["Churn"].dtype in [np.int64, np.float64]
    assert df.isnull().sum().sum() == 0  # No missing values


def test_handle_empty_dataframe(preprocessor):
    """Test handling of empty dataframe."""
    df_empty = pd.DataFrame()
    summary = preprocessor.get_data_summary(df_empty)

    assert summary["n_rows"] == 0
    assert summary["n_columns"] == 0


def test_churn_rate_calculation(preprocessor, sample_data):
    """Test churn rate calculation."""
    df = preprocessor.clean_data(sample_data)
    summary = preprocessor.get_data_summary(df)

    # Check churn rate is calculated correctly
    expected_churn_rate = (sample_data["Churn"] == "Yes").mean()
    assert "churn_rate" in summary
    assert summary["churn_rate"] == expected_churn_rate
