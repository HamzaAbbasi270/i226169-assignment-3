import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial ML App", layout="wide")
st.title("📊 Financial Machine Learning App")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# --- Load Data ---
st.subheader(f"Downloading data for {ticker}")
data = yf.download(ticker, start=start_date, end=end_date)

# Flatten columns if MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [' '.join(col).strip() for col in data.columns.to_flat_index()]

# Show raw data to help debugging
st.write("Raw Data Sample:")
st.dataframe(data.head())

# --- Find Adjusted Close Column Robustly ---
adj_close_col = next((col for col in data.columns if 'adj close' in col.lower()), None)

if not adj_close_col:
    st.error("❌ 'Adj Close' column not found in the dataset. Please verify the ticker and date range.")
    st.stop()

# --- Feature Engineering ---
data['Return'] = data[adj_close_col].pct_change()
data['MA10'] = data[adj_close_col].rolling(window=10).mean()
data['MA50'] = data[adj_close_col].rolling(window=50).mean()
data.dropna(inplace=True)

# --- Sidebar Feature Selection ---
st.sidebar.subheader("Feature Selection")
possible_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'Return']
available_features = [f for f in possible_features if f in data.columns]
selected_features = st.sidebar.multiselect("Select Features", available_features, default=['MA10', 'MA50', 'Return'])

if not selected_features:
    st.warning("⚠️ Please select at least one feature.")
    st.stop()

# --- Model Preparation ---
X = data[selected_features]
y = data[adj_close_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
st.subheader("Model Training and Evaluation")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation Metrics ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# --- Plot Actual vs Predicted ---
st.subheader("📈 Actual vs Predicted")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.values, label='Actual', color='blue')
ax.plot(y_pred, label='Predicted', color='red')
ax.set_title("Actual vs Predicted Prices")
ax.legend()
st.pyplot(fig)

# --- Final Note ---
st.info("📌 You can extend this app with Kragle datasets or more advanced ML models as part of the assignment.")
