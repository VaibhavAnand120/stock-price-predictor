import gradio as gr
import requests
import plotly.graph_objects as go

API_URL = "http://localhost:8000"   # change to your deployed URL later

TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX"]

def forecast(ticker, days):
    try:
        # fetch history
        hist = requests.get(f"{API_URL}/history", params={"ticker": ticker, "days": 90}).json()
        # fetch forecast
        pred = requests.get(f"{API_URL}/predict", params={"ticker": ticker, "days": days}).json()

        if "detail" in pred:
            return None, f"Error: {pred['detail']}"

        # build plotly chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist["dates"], y=hist["prices"],
            mode="lines", name="Historical",
            line=dict(color="#1D9E75", width=2)
        ))

        forecast_dates  = [f["date"]  for f in pred["forecast"]]
        forecast_prices = [f["price"] for f in pred["forecast"]]

        # connect last historical point to first forecast point
        fig.add_trace(go.Scatter(
            x=[hist["dates"][-1]] + forecast_dates,
            y=[hist["prices"][-1]] + forecast_prices,
            mode="lines+markers", name=f"{days}-Day Forecast",
            line=dict(color="#D85A30", width=2, dash="dash"),
            marker=dict(size=7)
        ))

        fig.update_layout(
            title=f"{ticker} — {days}-Day Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        summary = (
            f"Last known price: ${pred['last_known_price']:.2f}\n"
            f"Forecast ({days} days):\n" +
            "\n".join([f"  {f['date']}: ${f['price']:.2f}" for f in pred["forecast"]])
        )
        return fig, summary

    except Exception as e:
        return None, f"Could not connect to API: {str(e)}\nMake sure FastAPI is running."

with gr.Blocks(title="Stock Price Forecaster") as demo:
    gr.Markdown("# 📈 Stock Price Forecaster\nLSTM-powered 7-day price prediction")

    with gr.Row():
        ticker_input = gr.Dropdown(choices=TICKERS, value="AAPL", label="Stock ticker")
        days_input   = gr.Slider(minimum=1, maximum=14, value=7, step=1, label="Forecast days")

    predict_btn = gr.Button("Forecast", variant="primary")
    chart       = gr.Plot(label="Price chart")
    summary     = gr.Textbox(label="Forecast summary", lines=10)

    predict_btn.click(
        fn=forecast,
        inputs=[ticker_input, days_input],
        outputs=[chart, summary]
    )

if __name__ == "__main__":
    demo.launch()
