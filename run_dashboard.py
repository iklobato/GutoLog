#!/usr/bin/env python3
"""
Script to run the Streamlit Working Hours Comparison Dashboard.

Make sure you have installed the requirements:
pip install -r requirements.txt

Then run this script or use:
streamlit run streamlit_app.py
"""

import os
import subprocess
import sys


def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['streamlit', 'pandas', 'plotly', 'numpy', 'openpyxl']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False

    return True


def check_data_file():
    """Check if the data file exists."""
    data_file = "files/Working_Hours_Comparison.xlsx"
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Please run the merge script first: python merge_data.py")
        return False

    return True


def main():
    """Main function to run the Streamlit app."""
    print("=== Streamlit Working Hours Dashboard ===")

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Check data file
    if not check_data_file():
        sys.exit(1)

    print("All checks passed! Starting Streamlit app...")
    print("The dashboard will open in your browser.")
    print("Press Ctrl+C to stop the application.")

    try:
        # Run Streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "localhost"]
        )
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")


if __name__ == "__main__":
    main()
