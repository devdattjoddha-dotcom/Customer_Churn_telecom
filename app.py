import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="centered")
st.title("📡 AI Customer Churn Predictor")
st.markdown("Use this tool to predict if a telecom customer is at high risk of canceling their service.")
st.divider()

# --- 2. LOAD FILES SAFELY ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('telco_churn_model.h5')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    feature_columns = pickle.load(open('features.pkl', 'rb'))
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_assets()
except Exception as e:
    st.error(f"⚠️ Error details: {e}")


# --- 3. CREATE WEB INPUTS FOR THE MOST IMPORTANT FEATURES ---
st.subheader("Customer Financials")
col1, col2, col3 = st.columns(3)
tenure = col1.slider("Tenure (Months)", 0, 75, 12)
monthly_charges = col2.number_input("Monthly Bill ($)", 20.0, 150.0, 70.0)
total_charges = col3.number_input("Total Charges ($)", 0.0, 10000.0, (tenure * monthly_charges))

st.subheader("Service Details")
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

st.divider()

# --- 4. THE PREDICTION BUTTON & LOGIC ---
if st.button("🔮 Predict Churn Risk", use_container_width=True):
    
    # We create a blank row of data filled with 0s matching our exact training columns
    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0] = 0.0 
    
    # A. Fill in the numerical data
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    
    # B. Translate the User's dropdown choices into the AI's One-Hot Encoded columns!
    if contract == "One year": input_data['Contract_One year'] = 1.0
    elif contract == "Two year": input_data['Contract_Two year'] = 1.0
    
    if internet_service == "Fiber optic": input_data['InternetService_Fiber optic'] = 1.0
    elif internet_service == "No": input_data['InternetService_No'] = 1.0
        
    if payment == "Electronic check": input_data['PaymentMethod_Electronic check'] = 1.0
    elif payment == "Mailed check": input_data['PaymentMethod_Mailed check'] = 1.0
    elif payment == "Credit card (automatic)": input_data['PaymentMethod_Credit card (automatic)'] = 1.0
        
    # C. Scale the ENTIRE dataset (since the scaler was fit on all columns in Colab)
    # Ensure columns match training order exactly
    input_data = input_data[feature_columns]
    scaled_data = scaler.transform(input_data)
    
    # D. Make the Prediction
    prediction_probability = model.predict(scaled_data)[0][0]
    # E. Display the beautiful result to the user!
    if prediction_probability > 0.5:
        st.error(f"🚨 **HIGH RISK!** This customer has a **{prediction_probability * 100:.2f}%** chance of canceling.")
        st.write("Recommendation: Send an automated email offering a 10% discount to switch to a 1-year contract.")
    else:
        st.success(f"✅ **SAFE.** This customer only has a **{prediction_probability * 100:.2f}%** chance of leaving.")
        st.write("Recommendation: No action needed.")
