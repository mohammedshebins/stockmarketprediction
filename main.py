import os
import sys
from datetime import datetime

# Ensure src/ is accessible for imports
sys.path.append(os.path.abspath("src"))

from src.data_loader import load_data
from src.model import build_model, train_model
from src.predict import predict_on_date, predict_and_plot, predict_future

# 🔹 Parameters
TICKER = "AAPL"
SEQ_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 32
FUTURE_DAYS = 100  # Number of days to forecast

# 🔹 Load and preprocess data
print("🔄 Loading data...")
X, y, scaler, df = load_data(TICKER, seq_length=SEQ_LENGTH, start="2015-01-01")

# 🔹 Build and train the LSTM model
print("🧠 Building and training the model...")
model = build_model(input_shape=(X.shape[1], X.shape[2]))
train_model(model, X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

# 🔹 Evaluate the model with RMSE and plot
print("\n📈 Predicting and evaluating on historical data...")
predict_and_plot(model, scaler, df, seq_length=SEQ_LENGTH)

# 🔹 Predict specific date (optional user input)
date_input = input("\n📅 Enter a date to predict (YYYY-MM-DD) or press Enter to skip: ").strip()
if date_input:
    try:
        datetime.strptime(date_input, "%Y-%m-%d")  # validate
        predict_on_date(model, df, scaler, date_input, seq_length=SEQ_LENGTH)
    except ValueError:
        print("❌ Invalid date format. Please use YYYY-MM-DD.")

# 🔹 Predict into the future (recursive multi-step)
print(f"\n🚀 Predicting next {FUTURE_DAYS} future days...")
predict_future(model, df, scaler, days=FUTURE_DAYS, seq_length=SEQ_LENGTH)
