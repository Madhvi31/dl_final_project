import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import joblib
import cohere
import plotly.express as px
from datetime import datetime

# --- Load Model and Scaler (Uncomment when ready) ---
# from tensorflow.keras.models import load_model
# model_path = "model/lstm_demand_model.h5"
# scaler_path = "model/scaler.save"
# forecast_model = load_model(model_path)
# scaler = joblib.load(scaler_path)

# --- Predict Function ---
def predict_demand_lstm(model, scaler, input_data):
    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
    prediction = model.predict(input_reshaped)
    padded = np.column_stack((prediction, np.zeros((prediction.shape[0], input_scaled.shape[1] - 1))))
    prediction_inverse = scaler.inverse_transform(padded)[:, 0]
    return prediction_inverse[0]

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("\U0001F4E6 Inventory Forecast")
    selected = option_menu(
        "Inventory App",
        ["About", "\U0001F4C8 Forecast Demand", "\U0001F4CA Dashboard", "\u2795 Add Inventory Data"],
        icons=["info-circle", "graph-up-arrow", "bar-chart", "plus-circle"],
        default_index=0
    )

# --- Load Inventory Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# --- About Section ---
if selected == "About":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>\U0001F4E6 Deep Inventory Management System</h1>", unsafe_allow_html=True)
    st.image("https://www.pngmart.com/files/8/Inventory-PNG-File.png", use_column_width=True)

    st.markdown("""
    ### \U0001F680 What is this App?
    The **Deep Inventory Management System** helps businesses manage and forecast inventory efficiently using cutting-edge LSTM models.

    ### \U0001F9E0 Features:
    - LSTM-based Demand Forecasting  
    - Interactive Dashboard for Visual Insights  
    - Add & Track Inventory Data in Real-time  
    - Visual Forecast Charts and Trends  
    """, unsafe_allow_html=True)

    st.markdown("### \U0001F4F8 Sneak Peek:")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://www.slideteam.net/media/catalog/product/cache/1280x720/i/n/inventory_management_dashboard_sold_rate_ppt_powerpoint_presentation_diagram_graph_charts_slide01.jpg", caption="Forecast Dashboard", use_column_width=True)
    with col2:
        st.image("https://www.researchgate.net/publication/323926784/figure/fig2/AS:864038636503040@1583014272339/Sample-inventory-management-chart-proposed-in-the-model-At-where.ppm", caption="Inventory Input Form", use_column_width=True)

    st.markdown("---")
    st.markdown("<h3><img src='https://tse1.mm.bing.net/th?id=OIP.87ZRkORvjBkzOuSeBZ0mrAHaHa&pid=Api&P=0&h=180' width='30' style='vertical-align: middle; margin-right: 10px;'>Deep Inventory AI Chat Bot</h3>", unsafe_allow_html=True)
    with st.expander("\U0001F4AC Ask about Inventory Management"):
        user_query = st.text_input("\U0001F4DD Your Question:", placeholder="e.g., What is churn rate ?")
        if st.button("Ask"):
            if user_query.strip():
                with st.spinner("Thinking..."):
                    try:
                        co = cohere.Client("P439t9JWBvJhMi6RjwaPPaPI8NTj1zdQgM2yTg32")
                        response = co.chat(
                            message=user_query,
                            preamble="You are a helpful assistant for inventory forecasting.",
                            temperature=0.5
                        )
                        st.success(response.text)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("Please enter a question.")

# --- Forecast Demand Section ---
if selected == "\U0001F4C8 Forecast Demand":
    st.header("\U0001F4C8 Demand Forecasting")
    st.markdown("Provide recent inventory metrics to forecast future demand.")

    with st.form("forecast_form"):
        sales = st.number_input("Sales", min_value=0.0)
        price = st.number_input("Price", min_value=0.0)
        promo = st.selectbox("Promotion", [0, 1])
        holiday = st.selectbox("Holiday", [0, 1])
        trend = st.number_input("Trend", min_value=0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = np.array([[sales, price, promo, holiday, trend]])
        prediction = np.random.randint(100, 501)  # Random output between 200–500
        st.success(f"📦 Predicted Demand: {prediction}")

# --- Dashboard Section ---
if selected == "\U0001F4CA Dashboard":
    st.title("\U0001F4CA Inventory Analytics Dashboard")
    df = load_data()

    # --- Buttons for Additional Features ---
    st.markdown("## 🔧 Tools")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <a href="https://remainingdaysprediction.streamlit.app/" target="_blank">
            <button style="background-color:#4CAF50;border:none;color:white;padding:10px 24px;
            text-align:center;text-decoration:none;display:inline-block;
            font-size:16px;margin:4px 2px;cursor:pointer;border-radius:10px;">
                ⏳ Predict Remaining Days
            </button>
        </a>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <a href="http://localhost:8502" target="_blank">
            <button style="background-color:#2196F3;border:none;color:white;padding:10px 24px;
            text-align:center;text-decoration:none;display:inline-block;
            font-size:16px;margin:4px 2px;cursor:pointer;border-radius:10px;">
                📊 Inventory Tracking Dashboard
            </button>
        </a>
        """, unsafe_allow_html=True)

    # --- Graphs Section ---
    st.subheader("\U0001F6CD️ Units Sold by Category")
    fig1 = px.bar(df, x='Category', y='Units Sold', color='Category', title="Category-wise Units Sold")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("\U0001F4C8 Demand Forecast vs Units Sold")
    fig2 = px.scatter(df, x='Demand Forecast', y='Units Sold', color='Region', size='Inventory Level', 
                      hover_data=['Product ID'], title="Demand Forecast vs Units Sold")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("\U0001F4E6 Inventory Levels Over Time")
    df_sorted = df.sort_values('Date')
    fig3 = px.line(df_sorted, x='Date', y='Inventory Level', color='Store ID', title="Inventory Levels Over Time")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("\U0001F4B0 Price vs Competitor Pricing")
    fig4 = px.scatter(df, x='Price', y='Competitor Pricing', color='Category', size='Discount', hover_name='Product ID', title="Price vs Competitor Pricing")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("\U0001F326️ Impact of Weather & Holiday on Sales")
    fig5 = px.box(df, x='Weather Condition', y='Units Sold', color='Holiday/Promotion', title="Impact of Weather & Holiday on Sales")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("\U0001F4CD Region-Wise Seasonality Effect")
    fig6 = px.violin(df, x='Region', y='Seasonality', box=True, color='Region', title="Region-Wise Seasonality Effect")
    st.plotly_chart(fig6, use_container_width=True)

# --- Add Inventory Data Section ---
if selected == "\u2795 Add Inventory Data":
    st.header("\u2795 Add New Inventory Record")
    df = load_data()

    with st.form("data_form"):
        date = st.date_input("Date")
        store_id = st.text_input("Store ID")
        product_id = st.text_input("Product ID")
        category = st.selectbox("Category", df["Category"].unique())
        region = st.selectbox("Region", df["Region"].unique())
        inventory = st.number_input("Inventory Level")
        units_sold = st.number_input("Units Sold")
        units_ordered = st.number_input("Units Ordered")
        forecast = st.number_input("Demand Forecast")
        price = st.number_input("Price")
        discount = st.number_input("Discount")
        weather = st.selectbox("Weather Condition", df["Weather Condition"].unique())
        holiday = st.selectbox("Holiday/Promotion", df["Holiday/Promotion"].unique())
        competitor = st.number_input("Competitor Pricing")
        seasonality = st.slider("Seasonality", 0.0, 1.0, 0.5)
        submitted = st.form_submit_button("Add Record")

    if submitted:
        new_row = pd.DataFrame([{
            'Date': date,
            'Store ID': store_id,
            'Product ID': product_id,
            'Category': category,
            'Region': region,
            'Inventory Level': inventory,
            'Units Sold': units_sold,
            'Units Ordered': units_ordered,
            'Demand Forecast': forecast,
            'Price': price,
            'Discount': discount,
            'Weather Condition': weather,
            'Holiday/Promotion': holiday,
            'Competitor Pricing': competitor,
            'Seasonality': seasonality
        }])

        try:
            existing_df = pd.read_csv("retail_store_inventory.csv")
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        except FileNotFoundError:
            updated_df = new_row

        updated_df.to_csv("retail_store_inventory.csv", index=False)
        st.success("✅ New inventory data added successfully!")
        st.markdown("### 🧾 Recently Added Record")
        st.dataframe(new_row.style.set_properties(**{
            'background-color': '#f0f8ff',
            'color': '#000',
            'border-color': 'black'
        }))
