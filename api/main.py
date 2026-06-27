from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os

app = FastAPI(title="Stock Price Forecaster API")

MODEL_PATH   = r"C:\Users\anand\stock-forecaster\model\lstm_stock_model.h5"
SEQUENCE_LEN = 60

model = None

@app.on_event("startup")
def load():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    else:
        print("WARNING: No model found. Train first using model/train.py")

def get_recent_data(ticker: str):
    df = yf.download(ticker, period="6mo", auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")
    df = df[["Close"]]
    df.columns = ["Close"]   # fix multi-level column issue
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)
    return df, scaled, scaler

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/history")
def history(ticker: str = "AAPL", days: int = 90):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")
    df = df[["Close"]]
    df.columns = ["Close"]   # fix multi-level column issue
    return {
        "ticker": ticker,
        "dates":  [str(d.date()) for d in df.index],
        "prices": [round(float(p), 2) for p in df["Close"].values.flatten()]
    }

@app.get("/predict")
def predict(ticker: str = "AAPL", days: int = 7):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="days must be between 1 and 30")

    df, scaled, scaler = get_recent_data(ticker)

    current_seq = scaled[-SEQUENCE_LEN:].reshape(1, SEQUENCE_LEN, 1)
    forecast    = []

    for _ in range(days):
        pred = model.predict(current_seq, verbose=0)[0][0]
        forecast.append(pred)
        current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

    prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    last_date      = df.index[-1]
    forecast_dates = [
        str((last_date + timedelta(days=i+1)).date())
        for i in range(days)
    ]

    return {
        "ticker":             ticker,
        "forecast_days":      days,
        "last_known_price":   round(float(df["Close"].iloc[-1]), 2),
        "forecast": [
            {"date": d, "price": round(float(p), 2)}
            for d, p in zip(forecast_dates, prices)
        ]
    }
