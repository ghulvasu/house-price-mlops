# src/loader.py
import pandas as pd
import os

def load_data():
    """
    This function loads the raw CSV data and prints a summary
    to help us understand the dataset structure.
    """
    # 1. Define the path (works on Windows/Mac/Linux)
    data_path = os.path.join('data', 'raw', 'Housing.csv')

    # 2. Check if the file exists before trying to read it
    if not os.path.exists(data_path):
        print(f"❌ Error: File not found at {data_path}")
        return None

    # 3. Load the CSV into a DataFrame
    print("\n... Loading Data ...")
    df = pd.read_csv(data_path)
    print("✅ Data loaded successfully!")

    # -------------------------------------------------------
    # 4. Display Basic Information (Rows, Columns, Types)
    # -------------------------------------------------------
    print("\n--- 1. DATA SHAPE ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- 2. COLUMN INFO (Types & Missing Values) ---")
    # info() prints directly, so we don't need print() around it
    df.info()

    # -------------------------------------------------------
    # 5. Display Statistical Description (Mean, Max, Min)
    # -------------------------------------------------------
    print("\n--- 3. STATISTICAL SUMMARY (Numerical Columns) ---")
    # We use .T to transpose (flip) the table so it's easier to read
    print(df.describe().T)

    print("\n--- 4. FIRST 5 ROWS ---")
    print(df.head())

    return df

if __name__ == "__main__":
    load_data()
# python .\src\loader.py 