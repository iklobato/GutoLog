import streamlit as st

from freight_budget_management.web import dashboard as freight_budget_dashboard


def main() -> None:
    """Run the Freight Budget Management Streamlit app."""
    st.set_page_config(page_title="Freight Budget Management", page_icon="🚚", layout="wide")
    freight_budget_dashboard.render()


if __name__ == "__main__":
    main()
