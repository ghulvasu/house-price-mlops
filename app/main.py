# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI(title="House Price Prediction API")

# Allow HTML Frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HouseInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

model_path = "models/model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    # Universal Path Logic
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    else:
        # Fallback for running locally
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        abs_path = os.path.join(root_dir, 'models', 'model.pkl')
        
        if os.path.exists(abs_path):
             with open(abs_path, "rb") as f:
                model = pickle.load(f)
             print(f"✅ Model loaded from {abs_path}")
        else:
             print("❌ Warning: Model not found.")

@app.post("/predict")
def predict_price(input_data: HouseInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        data = input_data.dict()
        df = pd.DataFrame([data])

        # Preprocessing (Match training logic)
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        for col in binary_cols:
            df[col] = df[col].apply(lambda x: 1 if x.lower() == 'yes' else 0)

        status_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
        df['furnishingstatus'] = df['furnishingstatus'].map(status_map)

        expected_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
                         'basement', 'hotwaterheating', 'airconditioning', 'parking', 
                         'prefarea', 'furnishingstatus']
        df = df[expected_cols]

        log_pred = model.predict(df)
        price = np.expm1(log_pred)[0]

        return {"predicted_price": int(price)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))