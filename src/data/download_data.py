"""
Script to download sample customer churn dataset.
Using Telco Customer Churn dataset for demonstration.
"""

import pandas
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_dir

logger = get_logger(__name__)


def download_sample_data():
    """Download and save sample churn dataset."""

    logger.info("Creating sample customer churn dataset...")

    # Create a realistic sample dataset.
    # In production, this would be replaced with actual data download logic.
    import numpy as np

    np.random.seed(42)
    n_samples = 5000

    data = {
        "customerID": [f"CUST{i:05d}" for i in range(n_samples)],
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples),
        "tenure": np.random.randint(1, 73, n_samples),
        "PhoneService": np.random.choice(["Yes", "No"], n_samples, p=[0.9, 0.1]),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_samples),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "OnlineSecurity": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples
        ),
        "OnlineBackup": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples
        ),
        "DeviceProtection": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples
        ),
        "TechSupport": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples
        ),
        "StreamingTV": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples
        ),
        "StreamingMovies": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples
        ),
        "Contract": np.random.choice(
            ["Month-to-month", "One year", "Two year"], n_samples
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n_samples,
        ),
        "MonthlyCharges": np.random.uniform(18.0, 120.0, n_samples).round(2),
        "TotalCharges": np.random.uniform(18.0, 8500.0, n_samples).round(2),
    }

    # Create target variable with some logic
    churn_prob = (
        0.1  # base rate
        + (data["Contract"] == "Month-to-month") * 0.3
        + (data["tenure"] < 12) * 0.2
        + (data["MonthlyCharges"] > 80) * 0.15
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    data["Churn"] = np.random.binomial(1, churn_prob).astype("str")
    data["Churn"] = ["Yes" if x == "1" else "No" for x in data["Churn"]]

    df = pandas.DataFrame(data)

    # Save to data/raw
    output_dir = ensure_dir("data/raw")
    output_path = output_dir / "churn_data.csv"

    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")

    return df


if __name__ == "__main__":
    download_sample_data()
