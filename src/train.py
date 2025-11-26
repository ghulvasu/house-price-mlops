# src/train.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    # ---------------------------------------------------------
    # 1. Define Paths (UNIVERSAL FIX)
    # ---------------------------------------------------------
    # Get the directory where THIS script (train.py) is located: src/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to get the Project Root: house-price-mlops/
    base_dir = os.path.dirname(current_dir)

    print(f"📂 Project Root detected at: {base_dir}")

    # Build paths dynamically
    train_path = os.path.join(base_dir, 'data', 'processed', 'train.csv')
    model_path = os.path.join(base_dir, 'models', 'model.pkl')
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # ---------------------------------------------------------
    # 2. Load Data
    # ---------------------------------------------------------
    if not os.path.exists(train_path):
        print(f"❌ Error: {train_path} not found.")
        raise FileNotFoundError(f"{train_path} does not exist")

    print(f"✅ Loading data from: {train_path}")
    train_df = pd.read_csv(train_path)
    
    y_train = np.log1p(train_df['price']) 
    X_train = train_df.drop(columns=['price'])

    # 3. Train Model
    print("Training RandomForest Model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 4. Check Metrics
    preds_log = model.predict(X_train)
    preds_actual = np.expm1(preds_log)
    y_actual = np.expm1(y_train)
    r2 = r2_score(y_actual, preds_actual)

    print(f"✅ Model Trained. Training R2 Score: {r2:.4f}")

    # 5. Save Model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_model()