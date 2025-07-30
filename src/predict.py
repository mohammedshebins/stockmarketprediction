import numpy as np

def predict_future(model, scaler, df, future_date):
    df = df.sort_index()
    df = df[['Open', 'Close']]

    last_60_days = df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    X_input = np.array([last_60_days_scaled])
    predicted = model.predict(X_input)

    # Inverse scale prediction to original prices (no concat!)
    predicted_prices = scaler.inverse_transform(predicted)

    predicted_open = predicted_prices[0][0]
    predicted_close = predicted_prices[0][1]

    return predicted_open, predicted_close
