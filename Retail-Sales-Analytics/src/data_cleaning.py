"""
data_cleaning.py
------------------
Cleans the raw supermarket sales data and engineers new features used
throughout the rest of the analysis (Month, Weekday, Hour, Sales
Category, Revenue Class, etc.).
"""

import numpy as np
import pandas as pd


class DataCleaner:
    """Cleans a raw DataFrame and engineers analysis-ready features."""

    # Standardised column names -> friendlier, consistent snake-ish labels
    RENAME_MAP = {
        "Invoice ID": "invoice_id",
        "Branch": "branch",
        "City": "city",
        "Customer type": "customer_type",
        "Gender": "gender",
        "Product line": "product_line",
        "Unit price": "unit_price",
        "Quantity": "quantity",
        "Tax 5%": "tax",
        "Total": "total",
        "Date": "date",
        "Time": "time",
        "Payment": "payment",
        "cogs": "cogs",
        "gross margin percentage": "gross_margin_pct",
        "gross income": "gross_income",
        "Rating": "rating",
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ------------------------------------------------------------------ #
    # Public pipeline entry point
    # ------------------------------------------------------------------ #
    def clean(self, verbose: bool = True) -> pd.DataFrame:
        """Run the full cleaning + feature engineering pipeline."""
        rows_before = len(self.df)

        (
            self._rename_columns()
            ._remove_duplicates()
            ._handle_missing_values()
            ._fix_dtypes()
            ._remove_invalid_entries()
            ._engineer_features()
        )

        if verbose:
            rows_after = len(self.df)
            print("=" * 60)
            print("DATA CLEANING SUMMARY")
            print("=" * 60)
            print(f"Rows before cleaning : {rows_before}")
            print(f"Rows after cleaning  : {rows_after}")
            print(f"Rows removed         : {rows_before - rows_after}")
            print(f"Columns (final)      : {list(self.df.columns)}")
            print("=" * 60, "\n")

        return self.df

    # ------------------------------------------------------------------ #
    # Individual cleaning steps (each returns self so calls can chain)
    # ------------------------------------------------------------------ #
    def _rename_columns(self) -> "DataCleaner":
        self.df = self.df.rename(columns=self.RENAME_MAP)
        return self

    def _remove_duplicates(self) -> "DataCleaner":
        self.df = self.df.drop_duplicates()
        return self

    def _handle_missing_values(self) -> "DataCleaner":
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        categorical_cols = self.df.select_dtypes(include="object").columns

        # Numeric gaps -> median (robust to outliers), categorical gaps -> mode
        for col in numeric_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())

        for col in categorical_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])

        return self

    def _fix_dtypes(self) -> "DataCleaner":
        self.df["date"] = pd.to_datetime(self.df["date"], format="mixed", dayfirst=False)
        # Keep a proper time-of-day column (as time objects) plus an hour int column
        self.df["time"] = pd.to_datetime(self.df["time"], format="%H:%M").dt.time

        for col in ["unit_price", "quantity", "tax", "total", "cogs", "gross_income", "rating"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df["quantity"] = self.df["quantity"].astype(int)

        for col in ["branch", "city", "customer_type", "gender", "product_line", "payment"]:
            self.df[col] = self.df[col].astype("category")

        return self

    def _remove_invalid_entries(self) -> "DataCleaner":
        # Business-rule sanity checks: negative/zero prices, quantities or totals
        # are invalid transactions and would distort every downstream KPI.
        mask_valid = (
            (self.df["unit_price"] > 0)
            & (self.df["quantity"] > 0)
            & (self.df["total"] > 0)
            & (self.df["rating"].between(0, 10))
        )
        self.df = self.df[mask_valid].reset_index(drop=True)
        return self

    def _engineer_features(self) -> "DataCleaner":
        df = self.df

        df["month"] = df["date"].dt.month_name()
        df["weekday"] = df["date"].dt.day_name()
        df["day"] = df["date"].dt.day
        df["hour"] = df["time"].apply(lambda t: t.hour)

        # Sales Category: bucket transaction size by quantity purchased
        df["sales_category"] = [
            "Bulk" if q >= 8 else "Medium" if q >= 4 else "Small" for q in df["quantity"]
        ]

        # Revenue Class: bucket transaction value using quartiles (data-driven, not
        # arbitrary cut-offs) via NumPy percentile
        q1, q2, q3 = np.percentile(df["total"], [25, 50, 75])
        df["revenue_class"] = [
            "Low" if t <= q1 else "Medium" if t <= q2 else "High" if t <= q3 else "Premium"
            for t in df["total"]
        ]

        self.df = df
        return self
