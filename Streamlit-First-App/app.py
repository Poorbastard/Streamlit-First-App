import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "NOK", "AAL", "TSLA", "FB", "ORCL")

selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction", 1, 50)
period = n_years * 365

@st.cache # This will cache the data so it won't have to run the code or download the code again

def load_data(ticker):
    data = yf.download(ticker, START, TODAY) # load data
    data.reset_index(inplace=True) # put the date in the very first column
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'stock_close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df = data[['Date', 'Close']]
df = df.rename(columns = {"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.write('forecast data plot')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

df = cross_validation(m, initial = '1800 days', period = '180 days', horizon = '1 days')
st.write(df.head())

df_p = performance_metrics(df)
st.write(df_p.tail())