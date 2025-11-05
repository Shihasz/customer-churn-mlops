import mlflow
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def init_mlflow():
    """Initialize MLflow tracking and create experiment."""

    # Set tracking URI
    tracking_uri = config.get("mlflow_tracking_uri")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Create experiment if it doesn't exist
    experiment_name = config.get("mlflow.experiment_name")

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                tags={
                    "project": config.get("project.name"),
                    "version": config.get("project.version"),
                },
            )
            logger.info(
                f"Created new experiment {experiment_name} (ID: {experiment_id})"
            )
        else:
            logger.info(
                f"Using existing experiment {experiment_name} (ID: {experiment.experiment_id})"
            )
    except Exception as e:
        logger.error(f"Error initializing MLflow: {e}")
        raise

    logger.info("MLflow initialization complete!")


if __name__ == "__main__":
    init_mlflow()
