import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data (features and target)
data = {
    'Demand': [1, 2, 3, 4, 5],
    'Competitor_Price': [100, 110, 120, 130, 140],
    'Stock_Available': [50, 60, 70, 80, 90],
    'Clicks_on_Product': [10, 20, 30, 40, 50],
    'Price': [105, 115, 125, 135, 145]
}
df = pd.DataFrame(data)

# Train a simple model
X = df[['Demand', 'Competitor_Price', 'Stock_Available', 'Clicks_on_Product']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Dynamic Pricing System", page_icon="💰")
st.markdown("<h1 style='color: purple;'>Dynamic Pricing System Using AI 💰</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='color: orange;'>Predict the best price based on real-time data! 🚀</h4>", unsafe_allow_html=True)

st.markdown("<p style='color: black;'>Demand (1 - 5)</p>", unsafe_allow_html=True)
demand = st.slider("", 1, 5, 3)

st.markdown("<p style='color: black;'>Competitor Price</p>", unsafe_allow_html=True)
competitor_price = st.number_input("", min_value=0.0, format="%.2f")

st.markdown("<p style='color: black;'>Stock Available</p>", unsafe_allow_html=True)
stock_available = st.number_input("", min_value=0, format="%d")

st.markdown("<p style='color: black;'>Clicks on Product</p>", unsafe_allow_html=True)
clicks = st.number_input("", min_value=0, format="%d")

if st.button("Predict Price"):
    input_data = np.array([[demand, competitor_price, stock_available, clicks]])
    predicted_price = model.predict(input_data)[0]
    st.markdown(f"<h3 style='color: red;'>Predicted Price: ₹{predicted_price:.2f}</h3>", unsafe_allow_html=True)

# Show sample data
st.markdown("<h4 style='color: #39FF14;'>Sample Data Used to Train the Model</h4>", unsafe_allow_html=True)
st.dataframe(df)

# Add a chart
st.markdown("### Price vs Demand")
fig, ax = plt.subplots()
ax.plot(df['Demand'], df['Price'], marker='o')
ax.set_xlabel("Demand")
ax.set_ylabel("Price")
st.pyplot(fig)
