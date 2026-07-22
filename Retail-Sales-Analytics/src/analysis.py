"""
analysis.py
-------------
All the number-crunching lives here. Every method returns plain Python /
NumPy / Pandas objects (dicts, Series, DataFrames) so that visualization.py
and insights.py can consume them without recomputing anything.
"""

import numpy as np
import pandas as pd


class SalesAnalysis:
    """Pure analysis layer: takes a cleaned DataFrame, answers business questions."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ------------------------------------------------------------------ #
    # Overall KPIs
    # ------------------------------------------------------------------ #
    def calculate_kpis(self) -> dict:
        df = self.df
        total_arr = df["total"].to_numpy()  # NumPy array -> vectorised stats

        return {
            "total_revenue": float(np.sum(total_arr)),
            "total_transactions": int(len(df)),
            "total_quantity_sold": int(df["quantity"].sum()),
            "average_order_value": float(np.mean(total_arr)),
            "median_order_value": float(np.median(total_arr)),
            "std_order_value": float(np.std(total_arr)),
            "average_rating": float(np.mean(df["rating"].to_numpy())),
            "total_gross_income": float(df["gross_income"].sum()),
        }

    # ------------------------------------------------------------------ #
    # Branch analysis
    # ------------------------------------------------------------------ #
    def branch_analysis(self) -> pd.DataFrame:
        summary = (
            self.df.groupby("branch", observed=True)
            .agg(
                total_revenue=("total", "sum"),
                total_gross_income=("gross_income", "sum"),
                avg_rating=("rating", "mean"),
                transactions=("invoice_id", "count"),
                avg_basket_size=("quantity", "mean"),
            )
            .round(2)
            .sort_values("total_revenue", ascending=False)
        )
        return summary

    # ------------------------------------------------------------------ #
    # Customer analysis
    # ------------------------------------------------------------------ #
    def customer_analysis(self) -> dict:
        df = self.df

        gender_split = df["gender"].value_counts().to_dict()
        membership_split = df["customer_type"].value_counts().to_dict()

        spend_by_gender = df.groupby("gender", observed=True)["total"].mean().round(2).to_dict()
        rating_by_membership = (
            df.groupby("customer_type", observed=True)["rating"].mean().round(2).to_dict()
        )
        spend_by_membership = (
            df.groupby("customer_type", observed=True)["total"].mean().round(2).to_dict()
        )

        # Dictionary comprehension: cross-tab of membership x gender counts
        segment_counts = {
            f"{ctype}-{gender}": len(df[(df["customer_type"] == ctype) & (df["gender"] == gender)])
            for ctype in df["customer_type"].cat.categories
            for gender in df["gender"].cat.categories
        }

        return {
            "gender_split": gender_split,
            "membership_split": membership_split,
            "avg_spend_by_gender": spend_by_gender,
            "avg_rating_by_membership": rating_by_membership,
            "avg_spend_by_membership": spend_by_membership,
            "segment_counts": segment_counts,
        }

    # ------------------------------------------------------------------ #
    # Product analysis
    # ------------------------------------------------------------------ #
    def product_analysis(self) -> pd.DataFrame:
        summary = (
            self.df.groupby("product_line", observed=True)
            .agg(
                total_revenue=("total", "sum"),
                total_quantity=("quantity", "sum"),
                avg_rating=("rating", "mean"),
                avg_unit_price=("unit_price", "mean"),
                transactions=("invoice_id", "count"),
            )
            .round(2)
            .sort_values("total_revenue", ascending=False)
        )
        return summary

    def product_branch_pivot(self) -> pd.DataFrame:
        """Pivot table: revenue of each product line, broken down by branch."""
        pivot = pd.pivot_table(
            self.df,
            values="total",
            index="product_line",
            columns="branch",
            aggfunc="sum",
            observed=True,
        ).round(2)
        return pivot

    # ------------------------------------------------------------------ #
    # Payment analysis
    # ------------------------------------------------------------------ #
    def payment_analysis(self) -> dict:
        df = self.df
        counts = df["payment"].value_counts()
        # Dictionary comprehension: payment method -> total revenue generated
        revenue_by_payment = {
            method: round(float(df.loc[df["payment"] == method, "total"].sum()), 2)
            for method in counts.index
        }
        return {
            "counts": counts.to_dict(),
            "revenue_by_payment": revenue_by_payment,
            "most_preferred": counts.idxmax(),
        }

    # ------------------------------------------------------------------ #
    # City analysis
    # ------------------------------------------------------------------ #
    def city_analysis(self) -> pd.DataFrame:
        summary = (
            self.df.groupby("city", observed=True)
            .agg(
                total_revenue=("total", "sum"),
                total_quantity=("quantity", "sum"),
                avg_rating=("rating", "mean"),
                transactions=("invoice_id", "count"),
            )
            .round(2)
            .sort_values("total_revenue", ascending=False)
        )
        return summary

    # ------------------------------------------------------------------ #
    # Time analysis
    # ------------------------------------------------------------------ #
    def time_analysis(self) -> dict:
        df = self.df

        sales_by_hour = df.groupby("hour", observed=True)["total"].sum().round(2)
        sales_by_weekday = (
            df.groupby("weekday", observed=True)["total"]
            .sum()
            .reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            .round(2)
        )
        sales_by_month = df.groupby("month", observed=True)["total"].sum().round(2)

        return {
            "sales_by_hour": sales_by_hour,
            "sales_by_weekday": sales_by_weekday,
            "sales_by_month": sales_by_month,
            "peak_hour": int(sales_by_hour.idxmax()),
            "best_weekday": sales_by_weekday.idxmax(),
        }

    # ------------------------------------------------------------------ #
    # Rating analysis
    # ------------------------------------------------------------------ #
    def rating_analysis(self) -> dict:
        df = self.df
        rating_by_branch = df.groupby("branch", observed=True)["rating"].mean().round(2)

        numeric_cols = ["quantity", "total", "gross_income", "rating"]
        correlation_matrix = df[numeric_cols].corr().round(3)

        return {
            "rating_by_branch": rating_by_branch,
            "highest_rated_branch": rating_by_branch.idxmax(),
            "rating_distribution": df["rating"].describe().round(2).to_dict(),
            "correlation_matrix": correlation_matrix,
        }

    # ------------------------------------------------------------------ #
    # NumPy showcase: normalization / vectorised broadcasting example
    # ------------------------------------------------------------------ #
    def normalized_revenue_by_branch(self) -> dict:
        """Min-max normalise branch revenue to a 0-1 scale using NumPy broadcasting."""
        branch_revenue = self.df.groupby("branch", observed=True)["total"].sum()
        arr = branch_revenue.to_numpy()
        normalized = (arr - arr.min()) / (arr.max() - arr.min())  # broadcasting in action
        return dict(zip(branch_revenue.index, np.round(normalized, 3)))
