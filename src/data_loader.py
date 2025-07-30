import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data(ticker, start="2015-01-01", end=None):
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    df = yf.download(ticker, start=start, end=end)[['Open', 'Close']]
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"❌ No data found for ticker '{ticker}' between {start} and {end}.")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    if len(X) == 0:
        raise ValueError(f"❌ Not enough data to create sequences for ticker '{ticker}'. Require at least 60 rows after preprocessing.")

    X, y = np.array(X), np.array(y)
    return X, y, scaler, df
