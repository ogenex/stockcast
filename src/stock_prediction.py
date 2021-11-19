# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(
    page_title="Stock Price Forecast App",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

#logo_url = '../images/zetaris.200.png'
#st.image(logo_url, width=220)

st.title('Stock Price Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'TSLA', 'CBA.AX')
selected_stock = st.selectbox('Select stock for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('')

if st.checkbox('Show stock info'):
    st.subheader('Stock Info')
    stock = yf.Ticker(selected_stock)
    st.write(stock.info)

st.subheader('Historical Stock Price')
# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.update_layout(yaxis_title=selected_stock)
    fig.layout.update(xaxis_rangeslider_visible=True)
    fig.layout.update(legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)
	
plot_raw_data()

if st.checkbox('Show historical stock data'):
    st.subheader('Raw historical data')
    st.write(data.tail())


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# add horizontal spacing
st.write(' ')
st.write(' ')
st.write(' ')

st.subheader('Stock Price Forecast')

fig1 = plot_plotly(m, forecast)
# set y-axis label
fig1.update_layout(yaxis_title=selected_stock)
# set x-axis label to blank
fig1.update_layout(xaxis_title='')
st.plotly_chart(fig1, use_container_width=True)

# Show and plot forecast
if st.checkbox('Show stock forecast data'):
    st.subheader('Forecast data')
    st.write(forecast.tail())


# add horizontal spacing
st.write(' ')
st.write(' ')
st.write(' ')

# forecaset components
st.subheader('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)
