import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)

# Train ARIMA model
train_data = data["Close"][:-30]
model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit()

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)

# Plot forecast
plt.plot(range(len(train_data), len(train_data) + 30), forecast, color='red', label="Forecast")
plt.plot(data["Close"], label="Actual Data", alpha=0.7)
plt.legend()
plt.show()
