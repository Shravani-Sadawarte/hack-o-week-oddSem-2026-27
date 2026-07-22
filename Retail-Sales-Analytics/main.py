"""
main.py
---------
Menu-driven entry point for the Retail Sales Analytics project.

Run with:
    python main.py
"""

import sys

from src.sales_analyzer import SalesAnalyzer

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False


def c(text, color=None):
    """Wrap text in a colorama color if available, else return plain text."""
    if COLOR and color:
        return f"{color}{text}{Style.RESET_ALL}"
    return text


MENU = """
==================================================
   RETAIL SALES ANALYTICS & BUSINESS INSIGHTS
==================================================
 1. Load Data
 2. Clean Data
 3. Sales Analysis (KPIs)
 4. Customer Analysis
 5. Branch Analysis
 6. Product Analysis
 7. Payment Analysis
 8. City Analysis
 9. Time Analysis
10. Rating Analysis
11. Save All Charts
12. Generate Report
13. Run FULL Pipeline (recommended)
14. Show Dashboard
 0. Exit
==================================================
"""


def run_menu():
    analyzer = SalesAnalyzer()

    while True:
        print(c(MENU, Fore.CYAN if COLOR else None))
        choice = input("Select an option: ").strip()

        try:
            if choice == "1":
                analyzer.load_data()
            elif choice == "2":
                analyzer.clean_data()
                print(c("Data cleaned and saved to outputs/cleaned_dataset.csv", Fore.GREEN if COLOR else None))
            elif choice == "3":
                kpis = analyzer.analyze_sales()
                for k, v in kpis.items():
                    print(f"  {k:25s}: {v:,.2f}" if isinstance(v, float) else f"  {k:25s}: {v}")
            elif choice == "4":
                print(analyzer.customer_analysis())
            elif choice == "5":
                print(analyzer.branch_analysis())
            elif choice == "6":
                print(analyzer.product_analysis())
            elif choice == "7":
                print(analyzer.payment_analysis())
            elif choice == "8":
                print(analyzer.city_analysis())
            elif choice == "9":
                stats = analyzer.time_analysis()
                print(stats["sales_by_hour"])
                print(stats["sales_by_weekday"])
            elif choice == "10":
                print(analyzer.rating_analysis())
            elif choice == "11":
                paths = analyzer.save_charts()
                print(c(f"Saved {len(paths)} charts to outputs/charts/", Fore.GREEN if COLOR else None))
            elif choice == "12":
                path = analyzer.generate_report()
                print(c(f"Report saved to {path}", Fore.GREEN if COLOR else None))
            elif choice == "13":
                run_full_pipeline(analyzer)
            elif choice == "14":
                analyzer.print_dashboard()
            elif choice == "0":
                print(c("Goodbye!", Fore.YELLOW if COLOR else None))
                sys.exit(0)
            else:
                print(c("Invalid option, please try again.", Fore.RED if COLOR else None))
        except Exception as exc:
            print(c(f"Error: {exc}", Fore.RED if COLOR else None))


def run_full_pipeline(analyzer: SalesAnalyzer = None) -> SalesAnalyzer:
    """Run every stage of the pipeline end-to-end (used by menu option 13 and imports)."""
    analyzer = analyzer or SalesAnalyzer()
    print(c("\n[1/5] Loading data...", Fore.CYAN if COLOR else None))
    analyzer.load_data()

    print(c("[2/5] Cleaning data & engineering features...", Fore.CYAN if COLOR else None))
    analyzer.clean_data()

    print(c("[3/5] Running full business analysis...", Fore.CYAN if COLOR else None))
    analyzer.run_full_analysis()

    print(c("[4/5] Saving charts...", Fore.CYAN if COLOR else None))
    analyzer.save_charts()

    print(c("[5/5] Generating report...", Fore.CYAN if COLOR else None))
    analyzer.generate_report()

    print()
    analyzer.print_dashboard()
    return analyzer


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_pipeline()
    else:
        run_menu()
