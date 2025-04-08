import streamlit as st
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie
from sklearn.linear_model import LinearRegression

# --- Page Config ---
st.set_page_config(page_title="Dynamic Pricing System", page_icon="💸", layout="centered")

# --- Set Background Color to White and Text to Black ---
st.markdown(
    """
    <style>
        body {
            background-color: white;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Lottie Animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

# --- Title ---
st.markdown("<h1 style='color:green;'>Dynamic Pricing System using AI</h1>", unsafe_allow_html=True)

# --- Animation ---
if lottie_ai:
    st_lottie(lottie_ai, speed=1, height=200)

# --- Description ---
st.markdown("<h3 style='color:orange;'>Predict the best price based on real-time data! 🚀</h3>", unsafe_allow_html=True)

# --- Input Fields ---
st.markdown("<h4 style='color:black;'>Demand (1 - 5)</h4>", unsafe_allow_html=True)
demand = st.slider("Select Demand Level", 1, 5)

st.markdown("<h4 style='color:black;'>Competitor Price</h4>", unsafe_allow_html=True)
competitor_price = st.number_input("Competitor Price", min_value=0.0)

st.markdown("<h4 style='color:black;'>Stock Available</h4>", unsafe_allow_html=True)
stock_available = st.number_input("Stock Available", min_value=0)

st.markdown("<h4 style='color:black;'>Clicks on Product</h4>", unsafe_allow_html=True)
clicks = st.number_input("Clicks on Product", min_value=0, format="%d")

# --- Sample Data ---
st.markdown("<h4 style='color:#39ff14;'>Sample Data Used to Train the Model</h4>", unsafe_allow_html=True)

data = {
    'demand': [1, 2, 3, 4, 5],
    'competitor_price': [50, 60, 70, 80, 90],
    'stock_available': [100, 80, 60, 40, 20],
    'clicks': [10, 20, 30, 40, 50],
    'price': [55, 65, 75, 85, 95]
}
df = pd.DataFrame(data)
st.dataframe(df)

# --- Train Model ---
X = df[['demand', 'competitor_price', 'stock_available', 'clicks']]
y = df['price']
model = LinearRegression()
model.fit(X, y)

# --- Predict ---
input_data = np.array([[demand, competitor_price, stock_available, clicks]])
predicted_price = model.predict(input_data)[0]

# --- Output ---
st.success(f"🔥 Predicted Optimal Price: ₹{predicted_price:.2f}")
