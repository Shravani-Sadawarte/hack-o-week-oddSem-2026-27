"""
insights.py
-------------
Turns raw analysis numbers into short, plain-English business
interpretations -- the thing that separates "a bunch of charts" from
an actual analytics deliverable.
"""

import pandas as pd


def branch_insight(branch_summary: pd.DataFrame) -> str:
    top = branch_summary.index[0]
    revenue = branch_summary.iloc[0]["total_revenue"]
    top_rated = branch_summary["avg_rating"].idxmax()
    return (
        f"Branch {top} generated the highest revenue (₹{revenue:,.0f}), suggesting stronger "
        f"footfall or operational performance at that location. Branch {top_rated} earned the "
        f"highest average customer rating, which may indicate service quality worth replicating "
        f"across other branches."
    )


def customer_insight(customer_stats: dict) -> str:
    gender_split = customer_stats["gender_split"]
    top_gender = max(gender_split, key=gender_split.get)
    membership_split = customer_stats["membership_split"]
    top_membership = max(membership_split, key=membership_split.get)
    spend_by_membership = customer_stats["avg_spend_by_membership"]
    higher_spender = max(spend_by_membership, key=spend_by_membership.get)

    return (
        f"{top_gender} customers made up the largest share of transactions, and '{top_membership}' "
        f"is the more common customer type. On average, '{higher_spender}' customers spend more per "
        f"transaction (₹{spend_by_membership[higher_spender]:,.2f}), making membership conversion "
        f"a potentially valuable growth lever."
    )


def product_insight(product_summary: pd.DataFrame) -> str:
    top_revenue_product = product_summary.index[0]
    top_qty_product = product_summary["total_quantity"].idxmax()
    return (
        f"'{top_revenue_product}' is the top revenue-generating product line, "
        f"while '{top_qty_product}' sells the highest volume. If these differ, it signals "
        f"an opportunity to review pricing or bundle strategy between high-volume and "
        f"high-value categories."
    )


def payment_insight(payment_stats: dict) -> str:
    top_method = payment_stats["most_preferred"]
    revenue_by_payment = payment_stats["revenue_by_payment"]
    top_revenue_method = max(revenue_by_payment, key=revenue_by_payment.get)
    return (
        f"'{top_method}' is the most frequently used payment method. "
        f"'{top_revenue_method}' brings in the most total revenue, which is worth noting when "
        f"prioritising checkout experience or promotional payment partnerships."
    )


def city_insight(city_summary: pd.DataFrame) -> str:
    top_city = city_summary.index[0]
    revenue = city_summary.iloc[0]["total_revenue"]
    return (
        f"{top_city} is the highest-performing city by revenue (₹{revenue:,.0f}). "
        f"This location could be prioritised for inventory allocation, staffing, or "
        f"targeted local marketing."
    )


def time_insight(time_stats: dict) -> str:
    peak_hour = time_stats["peak_hour"]
    best_day = time_stats["best_weekday"]
    return (
        f"Sales peak around {peak_hour}:00, indicating this is the busiest window for "
        f"staffing and promotions. {best_day} is the strongest day of the week for revenue, "
        f"making it a good candidate for flash sales or extended hours."
    )


def rating_insight(rating_stats: dict) -> str:
    best_branch = rating_stats["highest_rated_branch"]
    mean_rating = rating_stats["rating_distribution"]["mean"]
    corr = rating_stats["correlation_matrix"]
    rating_total_corr = corr.loc["rating", "total"]
    relationship = "a weak" if abs(rating_total_corr) < 0.2 else "a moderate" if abs(rating_total_corr) < 0.5 else "a strong"
    return (
        f"Branch {best_branch} has the highest average customer rating. "
        f"Overall average satisfaction sits at {mean_rating:.2f}/10. Correlation analysis shows "
        f"{relationship} relationship (r={rating_total_corr:.2f}) between rating and transaction value, "
        f"meaning satisfaction alone {'is not a strong predictor' if abs(rating_total_corr) < 0.2 else 'does relate'} "
        f"of how much a customer spends."
    )
