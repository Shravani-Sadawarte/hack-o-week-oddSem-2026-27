"""
visualization.py
-------------------
All chart-drawing logic. Every method saves a PNG into outputs/charts/
and returns the file path, so sales_analyzer.py / insights.py can log
what was produced.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering, safe for scripts & servers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")


class Visualizer:
    """Generates and saves every chart used in the analysis."""

    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs/charts"):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, filename: str) -> str:
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # ------------------------------------------------------------------ #
    # Branch charts
    # ------------------------------------------------------------------ #
    def plot_branch_sales(self, branch_summary: pd.DataFrame) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        branch_summary["total_revenue"].plot(kind="bar", ax=axes[0], color=sns.color_palette("Set2"))
        axes[0].set_title("Total Revenue by Branch")
        axes[0].set_ylabel("Revenue")
        axes[0].set_xlabel("Branch")

        axes[1].pie(
            branch_summary["total_revenue"],
            labels=branch_summary.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("Set2"),
        )
        axes[1].set_title("Revenue Share by Branch")

        fig.suptitle("Branch Performance", fontsize=14, fontweight="bold")
        return self._save(fig, "branch_sales.png")

    # ------------------------------------------------------------------ #
    # Customer charts
    # ------------------------------------------------------------------ #
    def plot_customer_analysis(self) -> str:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        sns.countplot(data=self.df, x="gender", hue="customer_type", ax=axes[0])
        axes[0].set_title("Customer Count: Gender x Membership")

        membership_counts = self.df["customer_type"].value_counts()
        axes[1].pie(
            membership_counts, labels=membership_counts.index, autopct="%1.1f%%",
            startangle=90, colors=sns.color_palette("pastel"),
        )
        axes[1].set_title("Member vs Normal Split")

        sns.boxplot(data=self.df, x="customer_type", y="total", ax=axes[2])
        axes[2].set_title("Spending Distribution by Membership")
        axes[2].set_ylabel("Total Spend")

        fig.suptitle("Customer Analysis", fontsize=14, fontweight="bold")
        return self._save(fig, "customer_analysis.png")

    # ------------------------------------------------------------------ #
    # Product charts
    # ------------------------------------------------------------------ #
    def plot_product_analysis(self, product_summary: pd.DataFrame, pivot: pd.DataFrame) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        product_summary["total_revenue"].sort_values().plot(
            kind="barh", ax=axes[0], color=sns.color_palette("crest", len(product_summary))
        )
        axes[0].set_title("Revenue by Product Line")
        axes[0].set_xlabel("Revenue")

        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[1])
        axes[1].set_title("Revenue: Product Line x Branch")

        fig.suptitle("Product Performance", fontsize=14, fontweight="bold")
        return self._save(fig, "product_analysis.png")

    # ------------------------------------------------------------------ #
    # Payment charts
    # ------------------------------------------------------------------ #
    def plot_payment_distribution(self, payment_counts: dict) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = list(payment_counts.keys())
        values = list(payment_counts.values())

        axes[0].pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Set3"))
        axes[0].set_title("Payment Method Share")

        sns.barplot(x=labels, y=values, ax=axes[1], hue=labels, palette="Set3", legend=False)
        axes[1].set_title("Transactions by Payment Method")
        axes[1].set_ylabel("Number of Transactions")

        fig.suptitle("Payment Method Analysis", fontsize=14, fontweight="bold")
        return self._save(fig, "payment_distribution.png")

    # ------------------------------------------------------------------ #
    # City charts
    # ------------------------------------------------------------------ #
    def plot_city_sales(self, city_summary: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(8, 5))
        city_summary["total_revenue"].plot(kind="bar", ax=ax, color=sns.color_palette("flare", len(city_summary)))
        ax.set_title("Total Revenue by City", fontsize=14, fontweight="bold")
        ax.set_ylabel("Revenue")
        ax.set_xlabel("City")
        return self._save(fig, "city_sales.png")

    # ------------------------------------------------------------------ #
    # Time charts
    # ------------------------------------------------------------------ #
    def plot_time_analysis(self, sales_by_hour: pd.Series, sales_by_month: pd.Series) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sales_by_hour.plot(kind="line", marker="o", ax=axes[0], color="darkorange")
        axes[0].set_title("Sales by Hour of Day")
        axes[0].set_xlabel("Hour")
        axes[0].set_ylabel("Revenue")

        sales_by_month.plot(kind="bar", ax=axes[1], color=sns.color_palette("mako", len(sales_by_month)))
        axes[1].set_title("Sales by Month")
        axes[1].set_ylabel("Revenue")

        fig.suptitle("Time-Based Sales Trends", fontsize=14, fontweight="bold")
        return self._save(fig, "monthly_sales.png")

    def plot_rating_histogram(self) -> str:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.df["rating"], bins=15, kde=True, ax=ax, color="mediumpurple")
        ax.set_title("Customer Rating Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Rating")
        return self._save(fig, "rating_histogram.png")

    # ------------------------------------------------------------------ #
    # Correlation heatmap
    # ------------------------------------------------------------------ #
    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, fmt=".2f")
        ax.set_title("Correlation Matrix: Quantity, Total, Gross Income, Rating", fontsize=12, fontweight="bold")
        return self._save(fig, "correlation_matrix.png")

    def plot_weekday_violin(self, sales_by_weekday_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sns.violinplot(data=self.df, x="weekday", y="total", order=order, ax=ax, hue="weekday",
                        palette="Set2", legend=False)
        ax.set_title("Spend Distribution by Weekday", fontsize=14, fontweight="bold")
        ax.set_ylabel("Transaction Total")
        return self._save(fig, "weekday_spend_violin.png")
