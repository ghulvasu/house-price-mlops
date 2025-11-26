# src/evaluate.py
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model():
    # 1. Define Paths
    test_path = os.path.join('data', 'processed', 'test.csv')
    model_path = os.path.join('models', 'model.pkl')
    metrics_path = 'metrics.json'

    # 2. Load Data and Model
    if not os.path.exists(test_path):
        print("❌ Error: Test data not found.")
        return

    test_df = pd.read_csv(test_path)
    
    # Load Model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 3. Prepare Data
    # Remember: 'price' in test.csv is still in real dollars (not log transformed)
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # 4. Make Predictions
    # The model predicts LOG price, so we must reverse it
    predictions_log = model.predict(X_test)
    predictions_actual = np.expm1(predictions_log)

    # 5. Calculate Metrics
    mse = mean_squared_error(y_test, predictions_actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions_actual)

    print("\n--- TEST SET EVALUATION ---")
    print(f"  RMSE: ${int(rmse):,}")
    print(f"  R2 Score: {r2:.4f}")

    # 6. Save Metrics to JSON (Crucial for DVC)
    metrics = {
        "rmse": rmse,
        "r2_score": r2
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Metrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate_model()