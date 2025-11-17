"""
Unit tests for data validation module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_validation import DataValidator


@pytest.fixture
def valid_data():
    """Create valid sample data."""
    data = {
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
        "Churn": [0, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def validator():
    """Create DataValidator instance."""
    return DataValidator()


def test_validator_initialization(validator):
    """Test validator initialization."""
    assert validator.target_column == "Churn"


def test_validate_schema_valid(validator, valid_data):
    """Test schema validation with valid data."""
    results = validator.validate_schema(valid_data)

    assert results["schema_valid"] is True
    assert len(results["errors"]) == 0


def test_validate_schema_missing_target(validator, valid_data):
    """Test schema validation with missing target column."""
    df = valid_data.drop("Churn", axis=1)
    results = validator.validate_schema(df)

    assert results["schema_valid"] is False
    assert len(results["errors"]) > 0


def test_validate_schema_empty_dataframe(validator):
    """Test schema validation with empty dataframe."""
    df = pd.DataFrame()
    results = validator.validate_schema(df)

    assert results["schema_valid"] is False


def test_validate_data_quality(validator, valid_data):
    """Test data quality validation."""
    results = validator.validate_data_quality(valid_data)

    assert "checks" in results
    assert "missing_percentage" in results["checks"]
    assert "duplicate_percentage" in results["checks"]


def test_validate_business_rules_valid(validator, valid_data):
    """Test business rules with valid data."""
    results = validator.validate_business_rules(valid_data)

    assert results["rules_valid"] is True
    assert len(results["violations"]) == 0


def test_validate_business_rules_negative_tenure(validator, valid_data):
    """Test business rules with negative tenure."""
    df = valid_data.copy()
    df.loc[0, "tenure"] = -5

    results = validator.validate_business_rules(df)

    assert results["rules_valid"] is False
    assert len(results["violations"]) > 0


def test_validate_business_rules_invalid_charges(validator, valid_data):
    """Test business rules with invalid charges."""
    df = valid_data.copy()
    df.loc[0, "MonthlyCharges"] = -10

    results = validator.validate_business_rules(df)

    assert results["rules_valid"] is False


def test_validate_complete(validator, valid_data):
    """Test complete validation pipeline."""
    results = validator.validate(valid_data)

    assert "overall_valid" in results
    assert "schema" in results
    assert "quality" in results
    assert "business_rules" in results


def test_high_missing_values(validator, valid_data):
    """Test detection of high missing values."""
    df = valid_data.copy()
    df.loc[:3, "MonthlyCharges"] = np.nan

    results = validator.validate_data_quality(df)

    # Should detect high missing percentage
    missing_pct = results["checks"]["missing_percentage"]["MonthlyCharges"]
    assert missing_pct > 50
