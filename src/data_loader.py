"""
Data loader module for Resume Screening and Candidate Ranking System.
Provides helper functions to locate, validate, and load the resume dataset safely.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

# Default dataset path within the project
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "archive" / "Resume" / "Resume.csv"


def get_dataset_path(custom_path: Optional[str] = None) -> Path:
    """Return the resolved Path object for the dataset."""
    if custom_path:
        return Path(custom_path).resolve()
    return DEFAULT_DATASET_PATH


def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the resume dataset from the specified or default CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset containing resume records.
    
    Raises:
        FileNotFoundError: If the dataset file does not exist at the target path.
    """
    path = get_dataset_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {path}. "
            "Please ensure 'Resume.csv' exists in 'archive/Resume/'."
        )
    return pd.read_csv(path)
