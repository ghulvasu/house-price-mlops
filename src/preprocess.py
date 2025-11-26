# src/preprocess.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    # 1. LOAD DATA
    # ----------------------------------------------------------------
    input_path = os.path.join('data', 'raw', 'Housing.csv')
    if not os.path.exists(input_path):
        print("❌ Error: Raw data not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Original Data: {len(df)} rows")

    # 2. REMOVE OUTLIERS
    # ----------------------------------------------------------------
    # We remove houses more expensive than 9.2 Million
    outlier_cutoff = 9205000
    df = df[df['price'] <= outlier_cutoff]
    
    print(f"Data after removing outliers: {len(df)} rows")

    # 3. ENCODE TEXT TO NUMBERS
    # ----------------------------------------------------------------
    # A. Yes/No Columns -> 1/0
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    
    def yes_no_map(x):
        return 1 if x == 'yes' else 0

    for col in binary_cols:
        df[col] = df[col].apply(yes_no_map)

    # B. Furnishing -> 0/1/2 (Ordinal Encoding)
    status_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
    df['furnishingstatus'] = df['furnishingstatus'].map(status_map)

    print("✅ Encoding Completed")

    # 4. SPLIT DATA (TRAIN vs TEST)
    # ----------------------------------------------------------------
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 5. SCALE FEATURES (0 to 1)
    # ----------------------------------------------------------------
    # Only scale the original numerical columns
    cols_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    
    scaler = MinMaxScaler()

    # Fit on Train, Transform on Test
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    print("✅ Scaling Completed")

    # 6. SAVE PROCESSED FILES
    # ----------------------------------------------------------------
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    
    train_path = os.path.join('data', 'processed', 'train.csv')
    test_path = os.path.join('data', 'processed', 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSUCCESS: Reverted to standard preprocessing.")
    print(f"Train set saved: {len(train_df)} rows")

if __name__ == "__main__":
    preprocess_data()