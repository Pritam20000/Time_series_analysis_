# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# # Set Streamlit Page Config
# st.set_page_config(page_title="Stock Price Forecast", layout="centered")

# # Title
# st.title("Stock Price Forecast using LSTM")

# # Load and preprocess data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
#     data = df["Close"].values.reshape(-1, 1)
#     return df, data

# df, data = load_data()

# # Show sample data
# if st.checkbox("Show raw data"):
#     st.write(df.tail())

# # Load the trained LSTM model
# model = load_model("lstm_stock_model.h5")

# # Scale data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)

# # Create input sequences
# X_test = []
# for i in range(60, len(scaled_data)):
#     X_test.append(scaled_data[i-60:i, 0])
# X_test = np.array(X_test)
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# # Predict
# predictions = model.predict(X_test)
# predicted_prices = scaler.inverse_transform(predictions)
# actual_prices = data[60:]

# # Plotting
# st.subheader("Actual vs Predicted Closing Prices")
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(actual_prices, label="Actual Price", color="blue")
# ax.plot(predicted_prices, label="Predicted Price", color="orange")
# ax.set_xlabel("Days")
# ax.set_ylabel("Price")
# ax.legend()
# st.pyplot(fig)

# st.success(" Forecast Completed!")

# # streamlit run streamlit_app.py



import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

# # Set Streamlit Page Config
# st.set_page_config(page_title="Stock Price Forecast", layout="centered")

# # Title
# st.title("Stock Price Forecast using LSTM")

# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Read CSV and parse dates
#     df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
    
#     # Ensure 'Close' is numeric, and drop rows with missing or invalid data
#     df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
#     df = df.dropna(subset=['Close'])  # Drop rows where 'Close' is NaN
    
#     # Convert 'Close' column to a numpy array and reshape
#     data = df['Close'].values.reshape(-1, 1)
#     return df, data

# df, data = load_data()

# # Show sample data
# if st.checkbox("Show raw data"):
#     st.write(df.tail())

# # Load the trained LSTM model
# model = load_model("lstm_stock_model.h5")

# # Scale data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)

# # Create input sequences for LSTM
# X_test = []
# for i in range(60, len(scaled_data)):
#     X_test.append(scaled_data[i-60:i, 0])
# X_test = np.array(X_test)
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# # Predict stock prices
# predictions = model.predict(X_test)
# predicted_prices = scaler.inverse_transform(predictions)
# actual_prices = data[60:]

# # Plotting
# st.subheader("Actual vs Predicted Closing Prices")
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(actual_prices, label="Actual Price", color="blue")
# ax.plot(predicted_prices, label="Predicted Price", color="orange")
# ax.set_xlabel("Days")
# ax.set_ylabel("Price")
# ax.legend()
# st.pyplot(fig)

# st.success("Forecast Completed!")
