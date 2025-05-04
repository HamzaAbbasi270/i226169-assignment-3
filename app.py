import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial ML App", layout="wide")
st.title("📊 Financial Machine Learning App")

# --- Sidebar ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# --- Load Data ---
st.subheader(f"Downloading data for {ticker}")
data = yf.download(ticker, start=start_date, end=end_date)

# --- Flatten MultiIndex Columns First (to avoid KeyError) ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [' '.join(str(i) for i in col).strip() for col in data.columns.values]

# --- Check for 'Adj Close' column safely ---
columns_flat = [col.lower() for col in data.columns]
adj_close_col = None
for col in data.columns:
    if 'adj close' in col.lower():
        adj_close_col = col
        break

if not adj_close_col:
    st.error("❌ 'Adj Close' column not found in the downloaded data. Try a different stock or date range.")
    st.write("Columns found:", list(data.columns))
    st.dataframe(data.head())
    st.stop()

# --- Feature Engineering ---
data['Return'] = data[adj_close_col].pct_change()
data['MA10'] = data[adj_close_col].rolling(window=10).mean()
data['MA50'] = data[adj_close_col].rolling(window=50).mean()
data = data.dropna()

# --- Feature Selection ---
st.sidebar.subheader("Feature Selection")
all_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'Return']
available_features = [col for col in all_features if col in data.columns]
selected_features = st.sidebar.multiselect("Select Features", available_features, default=['MA10', 'MA50', 'Return'])

if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()

# --- Model Training ---
X = data[selected_features]
y = data[adj_close_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation ---
st.subheader("📈 Model Performance")
st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.4f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")

# --- Plot Actual vs Predicted ---
st.subheader("📉 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.values, label="Actual")
ax.plot(y_pred, label="Predicted")
ax.set_title("Actual vs Predicted")
ax.legend()
st.pyplot(fig)

# --- Completion Note ---
st.info("✅ You can expand this app with Kragle datasets or additional models for further analysis.")
