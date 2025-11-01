"""File utility functions."""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file to dictionary.

    Args:
        filepath: Input file path

    Returns:
        Loaded dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)
