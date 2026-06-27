# Stock Price Forecaster

LSTM-powered stock price prediction with a 7-day forecast, REST API, and live interactive dashboard.

**Live demo:** https://huggingface.co/spaces/VaibhavAnand120/stock-price-forecaster

---

## Results

| Metric | Score |
|--------|-------|
| RMSE   | $4.66 |
| MAE    | $3.89 |
| Training data | 2018 – 2024 (AAPL) |
| Test split | 20% held-out data |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Model | TensorFlow, Keras, LSTM |
| Data | yfinance, Pandas, NumPy, scikit-learn |
| API | FastAPI, Uvicorn |
| Frontend | Gradio, Plotly |
| Deployment | HuggingFace Spaces |

---

## Project Structure

```
stock-price-forecaster/
├── model/
│   └── lstm_stock_model.keras   # trained LSTM model
├── api/
│   └── main.py                  # FastAPI backend
├── app.py                       # Gradio frontend (deployment)
├── requirements.txt
└── README.md
```

---

## How it Works

1. Downloads 5 years of daily closing prices via Yahoo Finance
2. Normalizes prices using MinMaxScaler
3. Creates 60-day sliding window sequences
4. Trains a 2-layer LSTM neural network
5. Forecasts next N days autoregressively
6. Serves predictions via FastAPI REST endpoint
7. Visualizes with interactive Plotly chart in Gradio

---

## How to Run Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the API**
```bash
python -m uvicorn api.main:app --reload
```
API runs at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

**3. Launch the Gradio UI**
```bash
python app.py
```
Opens at `http://localhost:7860`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API status and model load check |
| GET | `/history?ticker=AAPL&days=90` | Last N days of real prices |
| GET | `/predict?ticker=AAPL&days=7` | Next N days forecast |

---

## Sample API Response

```json
{
  "ticker": "AAPL",
  "forecast_days": 7,
  "last_known_price": 283.78,
  "forecast": [
    {"date": "2026-06-27", "price": 293.76},
    {"date": "2026-06-28", "price": 292.80},
    {"date": "2026-06-29", "price": 292.16}
  ]
}
```

---

## Dataset

Historical stock prices fetched via [yfinance](https://github.com/ranaroussi/yfinance) from Yahoo Finance.
No dataset file needed — data is downloaded automatically at runtime.
Supports any valid stock ticker: AAPL, TSLA, GOOGL, MSFT, AMZN, META, NVDA, NFLX.
