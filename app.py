import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# Streamlit app title and description
st.write("""
# Stock Recommendation Engine

"Unlock the secrets of the stock market with our expert insights."!

""")

# Sidebar dropdown to select stock ticker
selectbox = st.sidebar.selectbox(
    "*Select Ticker*",
    ["AAPL", "SPX","TATA","GOOG", "MSFT" ,"AMZN" ,"META" ,"TSLA","XOM", "BRK.A", "JNJ","BAC","JPM","WFC","C","GS","MS","BK","AXP","SCHW","V","MA"]
)

# Downloading historical stock data from Yahoo Finance
stock_data = yf.download(selectbox, start='2016-01-01', end='2021-10-01')
stock_data.head()

# Preparing data for model training
close_prices = stock_data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.8)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))

# Splitting data into training and testing sets
train_data = scaled_data[0: training_data_len, :]
x_train = []
y_train = []
for i in range(60, len(train_data)):  # Using a window of 60 days for training
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Preparing test data
test_data = scaled_data[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Building LSTM model
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

# Compiling and training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 1, epochs=3)

# Making predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculating RMSE
rmse = np.sqrt(np.mean(predictions - y_test)**2)

# Preparing data for visualization
data = stock_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions

# Displaying stock price chart
st.write('##',selectbox)
fig, ax = plt.subplots(figsize=(16,8))
ax.plot(train['Close'], label='Actual')
ax.plot(validation[['Close', 'Predictions']])
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Close Price USD ($)')
ax.set_title(selectbox)
st.pyplot(fig)

# Preparing to forecast future stock prices
prediction = []  # Empty list to store predictions
n_features=1
seq_size=60
current_batch = train_data[-seq_size:]  # Final data points in train set
current_batch = current_batch.reshape(1, seq_size, n_features)  # Reshape data
x = len(stock_data)-14

train = stock_data.iloc[:x]
test = stock_data.iloc[x:]

# User selects forecast duration
st.write(""   
         #FORECAST
          "")
genre = st.radio(
    "Predict for :",
    ('next day', 'next week', 'next month'))

if genre == 'next day':
    future=1
if genre == 'next week':
    future=2
if genre == 'next month':
    future=3

# Generating forecast predictions
for i in range(len(test) + future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

# Rescaling predictions to original scale
rescaled_prediction = scaler.inverse_transform(prediction)

# Creating a time series for predictions
time_series_array = test.index  # Get dates for test data
for k in range(0, future):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

# Capturing forecast data in a dataframe
df_forecast = pd.DataFrame(columns=["predicted"], index=time_series_array)
df_forecast.loc[:,"predicted"] = rescaled_prediction[:,0]

# Displaying forecast chart
st.line_chart(df_forecast)

# Function to generate stock recommendation
def generate_recommendation(df_forecast):
    # Get the current price and forecasted price from the dataframe
    current_price = stock_data['Close'][-1]
    forecasted_price = df_forecast['predicted'][-1]

    # Calculate the percentage change between current price and forecasted price
    percentage_change = (forecasted_price - current_price) / current_price * 100

    # Evaluate the stock based on percentage change
    recommendation_score = evaluate_stock(percentage_change)

    # Determine the recommended action based on the recommendation score
    if recommendation_score > 0.5:
        recommended_action = "Buy"
    elif recommendation_score < -0.25:
        recommended_action = "Sell"
    else:
        recommended_action = "Hold"

    recommendation = {
        "current_price": current_price,
        "forecasted_price": forecasted_price,
        "percentage_change": percentage_change,
        "recommendation_score": recommendation_score,
        "recommended_action": recommended_action
    }

    return recommendation

# Function to evaluate stock based on percentage change
def evaluate_stock(percentage_change):
    recommendation_score = 0

    # Assign higher score to stocks with positive percentage change and lower score to stocks with negative percentage change
    if percentage_change > 0:
        recommendation_score += 0.5 * percentage_change / 100
    elif percentage_change < 0:
        recommendation_score -= 0.5 * abs(percentage_change) / 100

    return recommendation_score

# Generating stock recommendation
recommendation = generate_recommendation(df_forecast)
st.write(recommendation)
