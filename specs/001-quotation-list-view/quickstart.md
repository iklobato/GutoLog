# Quickstart: Quotation List View

## Prerequisites
- Python 3.11+
- `uv` installed

## Setup
1. Create a virtual environment:
   - `uv venv`
2. Install dependencies:
   - `uv pip install -r requirements.txt`

## Run
- Launch the Streamlit app:
  - `uv run streamlit run app.py`

## Validate
- Open the app and select **Freight Budget Management** mode.
- Verify the quotation list shows the latest 100 quotations.
- Apply filters (status, customer, amount range, date range) and confirm results.
- Open a quotation to see full details and audit history.
- Confirm the quotation is locked for editing while open and released on save/close.
