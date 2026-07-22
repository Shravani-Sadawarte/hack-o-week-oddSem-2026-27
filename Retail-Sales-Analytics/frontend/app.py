"""
Retail Sales Analytics -- Streamlit Dashboard
------------------------------------------------
Interactive frontend on top of the same SalesAnalyzer pipeline used by
main.py and the Jupyter notebook. Run from the project root with:

    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.sales_analyzer import SalesAnalyzer

sns.set_theme(style="whitegrid", palette="Set2")

st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------- #
# Cached pipeline run -- this is the expensive part, so cache it.
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Running analytics pipeline...")
def get_analyzer() -> SalesAnalyzer:
    analyzer = SalesAnalyzer(
        data_path=str(PROJECT_ROOT / "data" / "supermarket_sales.csv"),
        charts_dir=str(PROJECT_ROOT / "outputs" / "charts"),
        reports_dir=str(PROJECT_ROOT / "outputs" / "reports"),
        cleaned_data_path=str(PROJECT_ROOT / "outputs" / "cleaned_dataset.csv"),
    )
    analyzer.load_data()
    analyzer.clean_data()
    analyzer.run_full_analysis()
    return analyzer


analyzer = get_analyzer()
df = analyzer.df
kpis = analyzer.kpis
results = analyzer.results
insight_texts = analyzer.insight_texts

# ---------------------------------------------------------------------- #
# Sidebar filters
# ---------------------------------------------------------------------- #
st.sidebar.title("🛒 Filters")
branches = st.sidebar.multiselect("Branch", options=sorted(df["branch"].unique()), default=sorted(df["branch"].unique()))
cities = st.sidebar.multiselect("City", options=sorted(df["city"].unique()), default=sorted(df["city"].unique()))
customer_types = st.sidebar.multiselect(
    "Customer Type", options=sorted(df["customer_type"].unique()), default=sorted(df["customer_type"].unique())
)

filtered = df[
    df["branch"].isin(branches) & df["city"].isin(cities) & df["customer_type"].isin(customer_types)
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered):,}** of {len(df):,} transactions")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with the same `SalesAnalyzer` pipeline used by `main.py` and the Jupyter notebook.")

# ---------------------------------------------------------------------- #
# Header + KPI row
# ---------------------------------------------------------------------- #
st.title("🛒 Retail Sales Analytics & Business Insights")
st.caption("Supermarket Sales Dataset · 1,000 transactions · 3 branches")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"₹{filtered['total'].sum():,.0f}")
k2.metric("Transactions", f"{len(filtered):,}")
k3.metric("Avg Order Value", f"₹{filtered['total'].mean():,.2f}" if len(filtered) else "—")
k4.metric("Avg Rating", f"{filtered['rating'].mean():.2f}" if len(filtered) else "—")
k5.metric("Total Qty Sold", f"{int(filtered['quantity'].sum()):,}")

st.markdown("---")

tabs = st.tabs(
    ["🏬 Branch", "👥 Customer", "📦 Product", "💳 Payment", "🏙️ City", "⏱️ Time", "⭐ Rating"]
)

# ---------------------------------------------------------------------- #
# Branch tab
# ---------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Branch Performance")
    branch_summary = filtered.groupby("branch", observed=True).agg(
        total_revenue=("total", "sum"),
        total_gross_income=("gross_income", "sum"),
        avg_rating=("rating", "mean"),
        transactions=("invoice_id", "count"),
    ).round(2).sort_values("total_revenue", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(branch_summary["total_revenue"])
    with c2:
        fig, ax = plt.subplots()
        ax.pie(branch_summary["total_revenue"], labels=branch_summary.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
        ax.set_title("Revenue Share by Branch")
        st.pyplot(fig)

    st.dataframe(branch_summary, use_container_width=True)
    st.info(insight_texts.get("branch", ""))

# ---------------------------------------------------------------------- #
# Customer tab
# ---------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Customer Analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.bar_chart(filtered["gender"].value_counts())
        st.caption("Transactions by Gender")
    with c2:
        st.bar_chart(filtered["customer_type"].value_counts())
        st.caption("Member vs Normal")
    with c3:
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered, x="customer_type", y="total", ax=ax)
        ax.set_title("Spend by Membership")
        st.pyplot(fig)

    st.info(insight_texts.get("customer", ""))

# ---------------------------------------------------------------------- #
# Product tab
# ---------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Product Line Performance")
    product_summary = filtered.groupby("product_line", observed=True).agg(
        total_revenue=("total", "sum"),
        total_quantity=("quantity", "sum"),
        avg_rating=("rating", "mean"),
    ).round(2).sort_values("total_revenue", ascending=False)

    st.bar_chart(product_summary["total_revenue"])
    st.dataframe(product_summary, use_container_width=True)

    pivot = pd.pivot_table(filtered, values="total", index="product_line", columns="branch", aggfunc="sum", observed=True).round(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.info(insight_texts.get("product", ""))

# ---------------------------------------------------------------------- #
# Payment tab
# ---------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Payment Method Analysis")
    counts = filtered["payment"].value_counts()
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set3"))
        st.pyplot(fig)
    with c2:
        st.bar_chart(counts)

    st.info(insight_texts.get("payment", ""))

# ---------------------------------------------------------------------- #
# City tab
# ---------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("City Performance")
    city_summary = filtered.groupby("city", observed=True).agg(
        total_revenue=("total", "sum"),
        total_quantity=("quantity", "sum"),
        avg_rating=("rating", "mean"),
    ).round(2).sort_values("total_revenue", ascending=False)

    st.bar_chart(city_summary["total_revenue"])
    st.dataframe(city_summary, use_container_width=True)
    st.info(insight_texts.get("city", ""))

# ---------------------------------------------------------------------- #
# Time tab
# ---------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Time-Based Trends")
    sales_by_hour = filtered.groupby("hour", observed=True)["total"].sum()
    sales_by_month = filtered.groupby("month", observed=True)["total"].sum()

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(sales_by_hour)
        st.caption("Revenue by Hour of Day")
    with c2:
        st.bar_chart(sales_by_month)
        st.caption("Revenue by Month")

    st.info(insight_texts.get("time", ""))

# ---------------------------------------------------------------------- #
# Rating tab
# ---------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Ratings & Correlation")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.histplot(filtered["rating"], bins=15, kde=True, ax=ax, color="mediumpurple")
        st.pyplot(fig)
    with c2:
        corr = filtered[["quantity", "total", "gross_income", "rating"]].corr().round(3)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)

    st.info(insight_texts.get("rating", ""))

st.markdown("---")
st.caption("Retail Sales Analytics & Business Insights Dashboard · Built with Python, Pandas, NumPy, Matplotlib, Seaborn & Streamlit")
