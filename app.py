import streamlit as st
from datetime import datetime
from src.data_loader import load_data
from src.model import build_model
from src.predict import predict_future

st.title("📈 Stock Price Prediction using LSTM")

# Input: Stock ticker and date
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")
future_date = st.date_input("Select a future date", min_value=datetime.today())

if st.button("Predict"):
    st.info("Training model and predicting, please wait...")

    try:
        X, y, scaler, df = load_data(ticker)
        model = build_model(X)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        open_price, close_price = predict_future(model, scaler, df, future_date)

        st.success(f"📅 Prediction for {future_date.strftime('%Y-%m-%d')}:")
        st.write(f"🟢 Open Price: **${open_price:.2f}**")
        st.write(f"🔴 Close Price: **${close_price:.2f}**")

        # Evaluate model accuracy using R² Score
        from sklearn.metrics import r2_score

        y_pred = model.predict(X)
        y_true = y

        r2 = r2_score(y_true, y_pred)
        accuracy = r2 * 100

        st.markdown(f"✅ **Model Accuracy:** {accuracy:.2f}%")


    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
