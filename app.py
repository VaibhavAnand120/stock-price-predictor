import gradio as gr
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

SEQUENCE_LEN  = 60
TICKERS       = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX"]

# load model once at startup
model = load_model("lstm_stock_model.keras")
print("Model loaded")

def forecast(ticker, days):
    try:
        # fetch data
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="6mo")
        if df.empty:
            return None, f"Could not fetch data for {ticker}"
        df = df[["Close"]]
        df.columns = ["Close"]

        # scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(df)

        # forecast autoregressively
        current_seq = scaled[-SEQUENCE_LEN:].reshape(1, SEQUENCE_LEN, 1)
        preds = []
        for _ in range(days):
            pred = model.predict(current_seq, verbose=0)[0][0]
            preds.append(pred)
            current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

        forecast_prices = scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()

        # dates
        last_date      = df.index[-1]
        forecast_dates = pd.date_range(last_date, periods=days + 1, freq="B")[1:]

        # connect last historical point to first forecast point
        last_90    = df[-90:]
        connect_x  = [last_90.index[-1]] + list(forecast_dates)
        connect_y  = [float(last_90["Close"].iloc[-1])] + list(forecast_prices)

        # plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=last_90.index,
            y=last_90["Close"].values.flatten(),
            mode="lines", name="Historical (last 90 days)",
            line=dict(color="#1D9E75", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=connect_x, y=connect_y,
            mode="lines+markers", name=f"{days}-Day Forecast",
            line=dict(color="#D85A30", width=2, dash="dash"),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f"{ticker} — {days}-Day Price Forecast",
            xaxis_title="Date", yaxis_title="Price (USD)",
            template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        summary = f"Last known price: ${float(df['Close'].iloc[-1]):.2f}\n\nForecast:\n"
        for i, (date, price) in enumerate(zip(forecast_dates, forecast_prices), 1):
            summary += f"  Day {i} ({str(date.date())}): ${price:.2f}\n"

        return fig, summary

    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(title="Stock Price Forecaster") as demo:
    gr.Markdown("""
    # 📈 Stock Price Forecaster
    LSTM-powered stock price prediction trained on 5 years of Yahoo Finance data.
    **RMSE: $4.66 | MAE: $3.89**
    """)

    with gr.Row():
        ticker_input = gr.Dropdown(choices=TICKERS, value="AAPL", label="Stock ticker")
        days_input   = gr.Slider(minimum=1, maximum=14, value=7, step=1, label="Forecast days")

    predict_btn = gr.Button("Forecast", variant="primary")
    chart       = gr.Plot(label="Price chart")
    summary     = gr.Textbox(label="Forecast summary", lines=12)

    predict_btn.click(
        fn=forecast,
        inputs=[ticker_input, days_input],
        outputs=[chart, summary]
    )

demo.launch()
