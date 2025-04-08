
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from streamlit_lottie import st_lottie
import requests

# Lottie Animation Function
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animation
lottie_ai = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tno6cg2w.json")

# Sample training data
data = {
    'Demand': [1, 2, 3, 4, 5],
    'Competitor_Price': [90, 95, 100, 110, 120],
    'Stock': [5, 10, 15, 20, 25],
    'Clicks': [200, 400, 600, 800, 1000],
    'Price': [100, 110, 120, 130, 140]
}
df = pd.DataFrame(data)

# Train the model
X = df[['Demand', 'Competitor_Price', 'Stock', 'Clicks']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Custom CSS for background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #dbeafe, #f0f9ff);
    }
    </style>
    """, unsafe_allow_html=True)

# App title and animation
st.markdown("<h2 style='color: #4CAF50;'>Dynamic Pricing System using AI</h2>", unsafe_allow_html=True)
st_lottie(lottie_ai, speed=1, height=200)

st.markdown("Predict the best price based on real-time data! 🚀")

# Input fields
demand = st.slider("Demand (1 - 5)", 1, 5)
competitor_price = st.number_input("Competitor Price", value=100)
stock = st.number_input("Stock Available", value=10)
clicks = st.number_input("Clicks on Product", value=300)

# Prediction
if st.button("Predict Price"):
    with st.spinner("Calculating price..."):
        input_data = [[demand, competitor_price, stock, clicks]]
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted Price: ₹{round(predicted_price, 2)}")
        st.balloons()

# Show training data chart
st.subheader("Sample Data Used to Train the Model")
st.line_chart(df[['Price', 'Competitor_Price']])
