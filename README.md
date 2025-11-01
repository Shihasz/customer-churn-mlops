# Customer Churn Prediction MLOps Pipeline

A production-ready MLOps pipeline for predicting customer churn with automated training, deployment, and monitoring.

## Features

- Automated ML pipelines with MLflow tracking
- Model versioning and registry
- REST API for predictions
- Docker containerization
- CI/CD with GitHub Actions
- Model monitoring and drift detection

## Tech Stack

- Python 3.12
- MLflow for experiment tracking
- FastAPI for model serving
- Docker ofr containerization
- DVC for data versioning
- GitHub Actions for CI/CD

## Project-Structure

```text
customer-churn-mlops/
├── src/        # Source code
│   ├── data/       # Data processing
│   ├── models/     # Model training
│   ├── api/        # FastAPI application
│   └── utils/      # Utility functions
├── notebooks/  # Jupyter notebooks for EDA
├── tests/      # Unit and integration tests
├── config/     # Configuration files
├── scripts/    # Utility scripts
├── data/       # Data directory (gitignored)
├── models/     # Saved models (gitignored)
├── logs/       # Application logs
└── logs/       # Documentation
```

## Setup (WSL2/Linux)

### Prerequisites

- Python 3.11+
- Git
- Virtual environment support

### Installation

1. Clone the repository (or create project):

```bash
mkdir customer-churn-mlops
cd customer-churn-mlops
```

2. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Getting Started

// TODO

## Development

// TODO

## License

MIT License

## Author

Mohammed Shihas PK
