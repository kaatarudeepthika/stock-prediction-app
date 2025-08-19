import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob

st.set_page_config(page_title="Stock Analysis App", layout="wide")
st.title("Stock Analysis & Forecasting App")

# Sidebar inputs
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Select Historical Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"])
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk", "1mo"])
forecast_days = st.sidebar.number_input("Days to Forecast", min_value=1, max_value=365, value=30)

# Download historical data
@st.cache_data
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    df.reset_index(inplace=True)
    return df

df = load_data(ticker, period, interval)

# Forecast function
def forecast_stock(df, days):
    if df.empty or len(df) < 10:
        return pd.Series()
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    forecast.index = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    return forecast

# Sentiment analysis function
def get_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        data = []
        for article in news[:10]:  # last 10 news
            title = article['title']
            sentiment = TextBlob(title).sentiment.polarity
            data.append({"Title": title, "Sentiment": sentiment})
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# Tabs
tab1, tab2, tab3 = st.tabs(["Historical Data", "Forecast", "Sentiment Analysis"])

with tab1:
    st.subheader(f"Historical Data for {ticker}")
    if not df.empty:
        st.dataframe(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data available.")

with tab2:
    st.subheader(f"{forecast_days}-Day Forecast for {ticker}")
    forecast = forecast_stock(df, forecast_days)
    if not forecast.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Close'))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='Forecast', line=dict(color='green')))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Forecast not available. Need more historical data.")

with tab3:
    st.subheader(f"Recent News Sentiment for {ticker}")
    sentiment_df = get_sentiment(ticker)
    if not sentiment_df.empty:
        st.dataframe(sentiment_df)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sentiment_df['Title'], y=sentiment_df['Sentiment']))
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No recent news available.")
