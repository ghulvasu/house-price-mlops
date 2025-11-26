# src/evaluate.py
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model():
    # ---------------------------------------------------------
    # 1. Define Paths (UNIVERSAL FIX)
    # ---------------------------------------------------------
    # Get the directory where THIS script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to get the Project Root
    base_dir = os.path.dirname(current_dir)

    print(f"📂 Project Root detected at: {base_dir}")

    test_path = os.path.join(base_dir, 'data', 'processed', 'test.csv')
    model_path = os.path.join(base_dir, 'models', 'model.pkl')
    metrics_path = os.path.join(base_dir, 'metrics.json')

    # ---------------------------------------------------------
    # 2. Load Data and Model
    # ---------------------------------------------------------
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found")

    test_df = pd.read_csv(test_path)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 3. Predict
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    predictions_log = model.predict(X_test)
    predictions_actual = np.expm1(predictions_log)

    # 4. Calculate Metrics
    mse = mean_squared_error(y_test, predictions_actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions_actual)

    print(f"--- TEST RESULTS ---")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    # 5. Save Metrics
    metrics = {"rmse": rmse, "r2_score": r2}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Metrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate_model()