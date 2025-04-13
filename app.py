# © 2025 [Your Full Name]. All rights reserved.
# This project is licensed under the MIT License.
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Set page config with white background
st.set_page_config(page_title="Dynamic Pricing System", layout="centered")

# Inject white background using CSS
st.markdown("""
    <style>
        .main {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color: green;'>Dynamic Pricing System using AI</h1>", unsafe_allow_html=True)

# Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(lottie_ai, height=200)

# Orange subtitle
st.markdown("<h3 style='color: orange;'>Predict the best price based on real-time data! 🚀</h3>", unsafe_allow_html=True)

# Inputs
demand = st.slider("Demand (1 - 5)", 1, 5, key="demand_input")
competitor_price = st.number_input("Competitor Price", min_value=0.01, key="competitor_input")
stock_available = st.number_input("Stock Available", min_value=0, key="stock_input")
clicks = st.number_input("Clicks on Product", min_value=0, key="click_input")

# Sample data
data = {
    'demand': [1, 2, 3, 4, 5],
    'competitor_price': [50, 60, 70, 80, 90],
    'stock_available': [100, 80, 60, 40, 20],
    'clicks': [10, 20, 30, 40, 50],
    'price': [55, 65, 75, 85, 95]
}
df = pd.DataFrame(data)

# Train model
X = df[['demand', 'competitor_price', 'stock_available', 'clicks']]
y = df['price']
model = LinearRegression()
model.fit(X, y)

# Predict Button
if st.button("Predict Optimal Price"):
    user_input = pd.DataFrame([[demand, competitor_price, stock_available, clicks]], 
                              columns=['demand', 'competitor_price', 'stock_available', 'clicks'])
    prediction = model.predict(user_input)[0]
    st.success(f"🔥 Predicted Optimal Price: ₹{prediction:.2f}")

# Table heading in neon color
st.markdown("<h4 style='color: #39FF14;'>Sample Data Used to Train the Model</h4>", unsafe_allow_html=True)
st.dataframe(df)

# Graph
fig, ax = plt.subplots()
ax.plot(df['demand'], df['price'], marker='o', color='green')
ax.set_title('Demand vs Price')
ax.set_xlabel('Demand')
ax.set_ylabel('Price')
st.pyplot(fig)
