import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

print(tf.__version__)

st.set_page_config(page_title="Stock Price Forecast", layout="centered")

st.title("Stock Price Forecast using LSTM")

@st.cache_data
def load_data():
    df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    data = df['Close'].values.reshape(-1, 1)
    return df, data

df, data = load_data()

if st.checkbox("Show raw data"):
    st.write(df.tail())

model = load_model("lstm_stock_model.h5")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]  

X_test, y_test = [], []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
st.write(f"📉 RMSE on Test Set: {rmse:.2f}")

st.subheader("Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_prices, label="Actual Price", color="blue")
ax.plot(predicted_prices, label="Predicted Price", color="orange")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.success("Forecast Completed!")
