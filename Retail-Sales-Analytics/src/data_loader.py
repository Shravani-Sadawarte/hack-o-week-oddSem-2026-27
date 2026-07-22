"""
data_loader.py
----------------
Handles reading the raw dataset from disk and performing an initial,
non-destructive inspection of it (shape, dtypes, sample rows, missing
values). Keeping this separate from cleaning/analysis keeps the
pipeline modular and easy to unit test.
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """Responsible solely for reading data in and reporting on its raw shape."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.raw_df: pd.DataFrame | None = None

    def load_data(self, verbose: bool = True) -> pd.DataFrame:
        """Read the CSV file into a DataFrame and print a short inspection report."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.filepath}")

        self.raw_df = pd.read_csv(self.filepath)

        if verbose:
            self._inspect()

        return self.raw_df

    def _inspect(self) -> None:
        df = self.raw_df
        print("=" * 60)
        print("DATA LOADING SUMMARY")
        print("=" * 60)
        print(f"File          : {self.filepath.name}")
        print(f"Shape         : {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Missing cells : {int(df.isnull().sum().sum())}")
        print(f"Duplicate rows: {int(df.duplicated().sum())}")
        print("-" * 60)
        print("Column dtypes:")
        print(df.dtypes)
        print("-" * 60)
        print("Sample rows:")
        print(df.head(3))
        print("=" * 60, "\n")
