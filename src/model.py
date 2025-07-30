from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(X):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(2))  # 2 outputs: Open and Close
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
