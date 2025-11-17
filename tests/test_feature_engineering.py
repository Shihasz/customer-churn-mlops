"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": ["No", "Yes", "No", "No", "Yes"],
        "tenure": [12, 24, 6, 48, 3],
        "MonthlyCharges": [50.0, 80.0, 30.0, 100.0, 25.0],
        "TotalCharges": [600.0, 1920.0, 180.0, 4800.0, 75.0],
        "Contract": [
            "Month-to-month",
            "One year",
            "Month-to-month",
            "Two year",
            "Month-to-month",
        ],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No", "Fiber optic", "DSL"],
        "OnlineSecurity": ["Yes", "No", "No internet service", "Yes", "No"],
        "Churn": [0, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer()


def test_engineer_initialization(engineer):
    """Test feature engineer initialization."""
    assert engineer.target_column == "Churn"
    assert isinstance(engineer.numerical_features, list)
    assert isinstance(engineer.categorical_features, list)


def test_create_tenure_groups(engineer, sample_data):
    """Test tenure group creation."""
    df = engineer.create_tenure_groups(sample_data)

    # Check tenure_group column was created
    assert "tenure_group" in df.columns

    # Check categories are correct
    expected_categories = ["0-1 year", "1-2 years", "2-4 years", "4+ years"]
    actual_categories = df["tenure_group"].dropna().astype(str).unique()

    for cat in actual_categories:
        assert cat in expected_categories


def test_create_charge_features(engineer, sample_data):
    """Test charge feature creation."""
    df = engineer.create_charge_features(sample_data)

    # Check new features were created
    assert "avg_monthly_charges" in df.columns
    assert "charge_ratio" in df.columns

    # Check values are reasonable
    assert (df["avg_monthly_charges"] > 0).all()


def test_create_service_features(engineer, sample_data):
    """Test service feature creation."""
    df = engineer.create_service_features(sample_data)

    # Check new features were created
    assert "total_services" in df.columns
    assert "has_internet" in df.columns
    assert "has_phone" in df.columns

    # Check data types
    assert df["has_internet"].dtype in [np.int64, np.float64]
    assert df["has_phone"].dtype in [np.int64, np.float64]


def test_encode_categorical_features(engineer, sample_data):
    """Test categorical encoding."""
    df = engineer.encode_categorical_features(sample_data.copy(), fit=True)

    # Check encoded columns were created
    categorical_cols = sample_data.select_dtypes(include=["object"]).columns
    categorical_cols = [col for col in categorical_cols if col != "Churn"]

    for col in categorical_cols:
        assert f"{col}_encoded" in df.columns
        assert df[f"{col}_encoded"].dtype in [np.int64, np.float64]


def test_scale_numerical_features(engineer, sample_data):
    """Test numerical feature scaling."""
    df = engineer.scale_numerical_features(sample_data.copy(), fit=True)

    # Check numerical columns were scaled
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    for col in numerical_cols:
        if col in df.columns:
            # Scaled values should have mean ~0 and std ~1
            assert abs(df[col].mean()) < 1.0
            assert abs(df[col].std() - 1.0) < 0.5


def test_engineer_features_pipeline(engineer, sample_data):
    """Test complete feature engineering pipeline."""
    df = engineer.engineer_features(sample_data.copy(), fit=True)

    # Check shape increased (new features added)
    assert df.shape[1] > sample_data.shape[1]

    # Check no missing values in engineered features
    assert df.isnull().sum().sum() == 0


def test_get_feature_names(engineer, sample_data):
    """Test feature name extraction."""
    df = engineer.engineer_features(sample_data.copy(), fit=True)
    features = engineer.get_feature_names(df)

    # Check target column is excluded
    assert "Churn" not in features

    # Check features is a list
    assert isinstance(features, list)
    assert len(features) > 0


def test_save_artifacts(engineer, sample_data, tmp_path):
    """Test saving of artifacts."""
    # Engineer features first to fit transformers
    df = engineer.engineer_features(sample_data.copy(), fit=True)

    # Save artifacts
    engineer.save_artifacts(str(tmp_path))

    # Check files were created
    assert (tmp_path / "scaler.pkl").exists()
    assert (tmp_path / "label_encoders.pkl").exists()
    assert (tmp_path / "feature_metadata.json").exists()


def test_transform_mode(engineer, sample_data):
    """Test feature engineering in transform mode (fit=False)."""
    # First fit on data
    df_train = engineer.engineer_features(sample_data.copy(), fit=True)

    # Then transform new data
    df_test = engineer.engineer_features(sample_data.copy(), fit=False)

    # Check shapes match
    assert df_train.shape[1] == df_test.shape[1]
