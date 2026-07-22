"""
sales_analyzer.py
--------------------
The public-facing orchestrator class. This is the "front door" of the
project: it wires DataLoader -> DataCleaner -> SalesAnalysis ->
Visualizer -> insights together behind one clean OOP interface.

    analyzer = SalesAnalyzer("data/supermarket_sales.csv")
    analyzer.load_data()
    analyzer.clean_data()
    analyzer.analyze_sales()
    analyzer.generate_report()
"""

from datetime import datetime
from pathlib import Path

from .data_loader import DataLoader
from .data_cleaning import DataCleaner
from .analysis import SalesAnalysis
from .visualization import Visualizer
from . import insights


class SalesAnalyzer:
    """End-to-end retail sales analytics pipeline."""

    def __init__(
        self,
        data_path: str = "data/supermarket_sales.csv",
        charts_dir: str = "outputs/charts",
        reports_dir: str = "outputs/reports",
        cleaned_data_path: str = "outputs/cleaned_dataset.csv",
    ):
        self.data_path = data_path
        self.charts_dir = charts_dir
        self.reports_dir = Path(reports_dir)
        self.cleaned_data_path = cleaned_data_path

        self.raw_df = None
        self.df = None
        self.analysis: SalesAnalysis | None = None
        self.viz: Visualizer | None = None

        # Result caches, populated as each *_analysis() method runs
        self.kpis = {}
        self.results = {}
        self.chart_paths = []
        self.insight_texts = {}

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #
    def load_data(self):
        loader = DataLoader(self.data_path)
        self.raw_df = loader.load_data()
        return self.raw_df

    def clean_data(self):
        if self.raw_df is None:
            self.load_data()
        cleaner = DataCleaner(self.raw_df)
        self.df = cleaner.clean()

        Path(self.cleaned_data_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.cleaned_data_path, index=False)

        self.analysis = SalesAnalysis(self.df)
        self.viz = Visualizer(self.df, output_dir=self.charts_dir)
        return self.df

    def analyze_sales(self) -> dict:
        """Top-level KPIs for the whole dataset."""
        if self.analysis is None:
            self.clean_data()
        self.kpis = self.analysis.calculate_kpis()
        return self.kpis

    def branch_analysis(self):
        summary = self.analysis.branch_analysis()
        self.results["branch"] = summary
        self.insight_texts["branch"] = insights.branch_insight(summary)
        return summary

    def customer_analysis(self):
        stats = self.analysis.customer_analysis()
        self.results["customer"] = stats
        self.insight_texts["customer"] = insights.customer_insight(stats)
        return stats

    def product_analysis(self):
        summary = self.analysis.product_analysis()
        pivot = self.analysis.product_branch_pivot()
        self.results["product"] = summary
        self.results["product_pivot"] = pivot
        self.insight_texts["product"] = insights.product_insight(summary)
        return summary

    def payment_analysis(self):
        stats = self.analysis.payment_analysis()
        self.results["payment"] = stats
        self.insight_texts["payment"] = insights.payment_insight(stats)
        return stats

    def city_analysis(self):
        summary = self.analysis.city_analysis()
        self.results["city"] = summary
        self.insight_texts["city"] = insights.city_insight(summary)
        return summary

    def time_analysis(self):
        stats = self.analysis.time_analysis()
        self.results["time"] = stats
        self.insight_texts["time"] = insights.time_insight(stats)
        return stats

    def rating_analysis(self):
        stats = self.analysis.rating_analysis()
        self.results["rating"] = stats
        self.insight_texts["rating"] = insights.rating_insight(stats)
        return stats

    def run_full_analysis(self):
        """Convenience method: run every analysis step in sequence."""
        self.analyze_sales()
        self.branch_analysis()
        self.customer_analysis()
        self.product_analysis()
        self.payment_analysis()
        self.city_analysis()
        self.time_analysis()
        self.rating_analysis()
        return self.results

    # ------------------------------------------------------------------ #
    # Charts
    # ------------------------------------------------------------------ #
    def save_charts(self) -> list:
        if not self.results:
            self.run_full_analysis()

        self.chart_paths = [
            self.viz.plot_branch_sales(self.results["branch"]),
            self.viz.plot_customer_analysis(),
            self.viz.plot_product_analysis(self.results["product"], self.results["product_pivot"]),
            self.viz.plot_payment_distribution(self.results["payment"]["counts"]),
            self.viz.plot_city_sales(self.results["city"]),
            self.viz.plot_time_analysis(
                self.results["time"]["sales_by_hour"], self.results["time"]["sales_by_month"]
            ),
            self.viz.plot_rating_histogram(),
            self.viz.plot_correlation_matrix(self.results["rating"]["correlation_matrix"]),
            self.viz.plot_weekday_violin(self.results["time"]["sales_by_weekday"]),
        ]
        return self.chart_paths

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def generate_report(self, filename: str = "sales_report.txt") -> str:
        if not self.results:
            self.run_full_analysis()

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.reports_dir / filename

        branch_summary = self.results["branch"]
        product_summary = self.results["product"]
        city_summary = self.results["city"]
        payment_stats = self.results["payment"]
        time_stats = self.results["time"]
        rating_stats = self.results["rating"]

        lines = [
            "=" * 60,
            "RETAIL SALES ANALYTICS - SUMMARY REPORT",
            "=" * 60,
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Records analyzed : {len(self.df)}",
            "",
            "--- KEY PERFORMANCE INDICATORS ---",
            f"Total Revenue          : Rs. {self.kpis['total_revenue']:,.2f}",
            f"Total Transactions     : {self.kpis['total_transactions']}",
            f"Total Quantity Sold    : {self.kpis['total_quantity_sold']}",
            f"Average Order Value    : Rs. {self.kpis['average_order_value']:,.2f}",
            f"Total Gross Income     : Rs. {self.kpis['total_gross_income']:,.2f}",
            f"Average Customer Rating: {self.kpis['average_rating']:.2f}",
            "",
            "--- BRANCH PERFORMANCE ---",
            f"Highest Revenue Branch : {branch_summary.index[0]}",
            f"Highest Rated Branch   : {rating_stats['highest_rated_branch']}",
            self.insight_texts.get("branch", ""),
            "",
            "--- PRODUCT PERFORMANCE ---",
            f"Highest Revenue Product Line: {product_summary.index[0]}",
            self.insight_texts.get("product", ""),
            "",
            "--- PAYMENT PREFERENCES ---",
            f"Most Preferred Payment Method: {payment_stats['most_preferred']}",
            self.insight_texts.get("payment", ""),
            "",
            "--- CITY PERFORMANCE ---",
            f"Highest Revenue City: {city_summary.index[0]}",
            self.insight_texts.get("city", ""),
            "",
            "--- TIME TRENDS ---",
            f"Peak Sales Hour: {time_stats['peak_hour']}:00",
            f"Best Weekday   : {time_stats['best_weekday']}",
            self.insight_texts.get("time", ""),
            "",
            "--- CUSTOMER SATISFACTION ---",
            self.insight_texts.get("rating", ""),
            "",
            "=" * 60,
            "End of Report",
            "=" * 60,
        ]

        report_text = "\n".join(str(line) for line in lines)
        report_path.write_text(report_text, encoding="utf-8")
        return str(report_path)

    # ------------------------------------------------------------------ #
    # Console dashboard
    # ------------------------------------------------------------------ #
    def print_dashboard(self):
        if not self.results:
            self.run_full_analysis()

        branch_summary = self.results["branch"]
        product_summary = self.results["product"]
        payment_stats = self.results["payment"]
        rating_stats = self.results["rating"]

        print("=" * 45)
        print(" RETAIL SALES DASHBOARD".center(45))
        print("=" * 45)
        print(f"Total Revenue          : Rs. {self.kpis['total_revenue']:,.2f}")
        print(f"Total Orders           : {self.kpis['total_transactions']}")
        print(f"Average Order Value    : Rs. {self.kpis['average_order_value']:,.2f}")
        print(f"Best Branch            : Branch {branch_summary.index[0]}")
        print(f"Best Product Line      : {product_summary.index[0]}")
        print(f"Top Payment Method     : {payment_stats['most_preferred']}")
        print(f"Highest Rated Branch   : Branch {rating_stats['highest_rated_branch']}")
        print()
        print(f"Charts saved to:\n{self.charts_dir}/")
        print()
        print(f"Report generated:\n{self.reports_dir}/sales_report.txt")
        print("=" * 45)
