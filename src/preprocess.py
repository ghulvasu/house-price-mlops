# src/preprocess.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_data():
    # ---------------------------------------------------------
    # 1. Define Paths (UNIVERSAL FIX)
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir) # Project Root

    print(f"📂 Project Root detected at: {base_dir}")

    input_path = os.path.join(base_dir, 'data', 'raw', 'Housing.csv')
    train_path = os.path.join(base_dir, 'data', 'processed', 'train.csv')
    test_path = os.path.join(base_dir, 'data', 'processed', 'test.csv')

    # 2. Load Data
    if not os.path.exists(input_path):
        print(f"❌ Error: Raw data not found at {input_path}")
        raise FileNotFoundError(f"{input_path} not found")

    df = pd.read_csv(input_path)

    # 3. Clean Data (No Scaling needed for Random Forest)
    df = df[df['price'] <= 9205000]

    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)

    status_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
    df['furnishingstatus'] = df['furnishingstatus'].map(status_map)

    # 4. Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 5. Save
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("✅ Preprocessing Complete")

if __name__ == "__main__":
    preprocess_data()