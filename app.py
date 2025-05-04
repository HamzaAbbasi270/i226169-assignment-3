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
st.subheader(f"Stock Data for {ticker}")
data = yf.download(ticker, start=start_date, end=end_date)

# Fix for multi-index columns (some yfinance versions do this)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]

# Ensure 'Adj Close' exists
adj_close_candidates = [col for col in data.columns if 'adj close' in col.lower()]
if not adj_close_candidates:
    st.error("'Adj Close' column not found. Please check the ticker or date range.")
    st.dataframe(data.head())
    st.stop()

adj_close_col = adj_close_candidates[0]  # Use the first match

st.dataframe(data.tail())

# --- Feature Engineering ---
data['Return'] = data[adj_close_col].pct_change()
data['MA10'] = data[adj_close_col].rolling(window=10).mean()
data['MA50'] = data[adj_close_col].rolling(window=50).mean()
data = data.dropna()

# --- User Select Features ---
st.sidebar.subheader("Feature Selection")
features = st.sidebar.multiselect("Select Features", ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'Return'], default=['MA10', 'MA50', 'Return'])
target = adj_close_col

# --- Train/Test Split ---
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
st.subheader("Model Training")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation ---
st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.4f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")

# --- Visualization ---
st.subheader("Actual vs Predicted")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.values, label='Actual')
ax.plot(y_pred, label='Predicted')
ax.set_title(f"Actual vs Predicted {adj_close_col}")
ax.legend()
st.pyplot(fig)

# --- Future Work Placeholder ---
st.info("📌 You can extend this app by integrating Kragle datasets or additional ML models as per the assignment.")
