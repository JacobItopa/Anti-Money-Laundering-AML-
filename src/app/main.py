import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="AML Fraud Detection API", description="API for scoring transactions for potential money laundering.")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Metrics - Path: {request.url.path} | Latency: {process_time:.4f} secs")
    response.headers["X-Process-Time"] = str(process_time)
    return response

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/advanced_xgboost.joblib')
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../../models/preprocessing_artifacts.joblib')

try:
    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)
    freq_maps = artifacts['freq_maps']
    expected_cols = artifacts['expected_cols']
except Exception as e:
    print(f"Error loading models: {e}")
    model, freq_maps, expected_cols = None, None, None

class TransactionInput(BaseModel):
    Timestamp: str
    From_Bank: str
    Account: str
    To_Bank: str
    Account_1: str
    Amount_Received: float
    Receiving_Currency: str
    Amount_Paid: float
    Payment_Currency: str
    Payment_Format: str

@app.post("/predict")
async def predict_fraud(transaction: TransactionInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded properly.")
    
    data = transaction.model_dump()
    col_mapping = {
        'From_Bank': 'From Bank',
        'To_Bank': 'To Bank',
        'Account_1': 'Account.1',
        'Amount_Received': 'Amount Received',
        'Receiving_Currency': 'Receiving Currency',
        'Amount_Paid': 'Amount Paid',
        'Payment_Currency': 'Payment Currency',
        'Payment_Format': 'Payment Format'
    }
    df = pd.DataFrame([data]).rename(columns=col_mapping)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    df['Amount Received'] = np.log1p(df['Amount Received'])
    df['Amount Paid'] = np.log1p(df['Amount Paid'])
    
    high_card_cols = ['Receiving Currency', 'Payment Currency', 'From Bank', 'To Bank']
    for col in high_card_cols:
        df[f'{col}_Freq'] = df[col].map(freq_maps[col]).fillna(0)
    
    format_col = f"Payment Format_{df['Payment Format'].iloc[0]}"
    df[format_col] = 1
    
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
            
    X = df[expected_cols]
    
    probability = model.predict_proba(X)[0, 1]
    prediction = int(model.predict(X)[0])
    
    return {
        "fraud_probability": float(probability),
        "is_laundering": bool(prediction),
        "flagged_for_review": bool(probability > 0.5)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
