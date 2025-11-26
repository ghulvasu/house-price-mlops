# src/train.py
import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    # 1. Prepare Paths
    train_path = os.path.join('data', 'processed', 'train.csv')
    model_path = os.path.join('models', 'model.pkl')
    os.makedirs('models', exist_ok=True)

    if not os.path.exists(train_path):
        print(f"❌ Error: {train_path} not found.")
        return

    train_df = pd.read_csv(train_path)
    
    # Log Transform Target
    y_train = np.log1p(train_df['price']) 
    X_train = train_df.drop(columns=['price'])

    # 2. Define Models with "Anti-Overfitting" Parameters
    models = {
        "LinearRegression": LinearRegression(),
        
        # Reduced depth to 5 (was 10) to prevent memorization
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        
        # XGBoost Tuned: Slower learning, shallower trees, and regularization (alpha)
        "XGBoost": XGBRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=3,        # Very shallow trees generalize better
            reg_alpha=1,        # L1 Regularization to reduce noise
            random_state=42
        )
    }

    best_model = None
    best_r2_val = -1  # We simulate a validation score here using training for selection
    best_name = ""

    mlflow.set_experiment("House_Price_Prediction_Tuned")
    print("\nStarting Training (Tuned to reduce Overfitting)...")
    print("-" * 60)

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Tuned_{name}"):
            model.fit(X_train, y_train)

            # Predict on Training Data
            preds_log = model.predict(X_train)
            preds_actual = np.expm1(preds_log)
            y_actual = np.expm1(y_train)

            # Calculate Metrics
            mse = mean_squared_error(y_actual, preds_actual)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_actual, preds_actual)

            print(f"🔹 {name}")
            print(f"   Training RMSE: ${int(rmse):,}")
            print(f"   Training R2:   {r2:.4f}")

            mlflow.log_param("model_name", name)
            mlflow.log_metric("train_rmse", rmse)
            mlflow.log_metric("train_r2", r2)
            
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            # Selection Logic:
            # Note: Ideally we use Cross-Validation here, but for now we pick the one
            # with the highest Training R2 that ISN'T suspiciously high (e.g. > 0.95)
            if r2 > best_r2_val:
                best_r2_val = r2
                best_model = model
                best_name = name

    print("-" * 60)
    print(f"🏆 Selected Model: {best_name}")

    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"✅ Saved {best_name} to {model_path}")

if __name__ == "__main__":
    train_model()