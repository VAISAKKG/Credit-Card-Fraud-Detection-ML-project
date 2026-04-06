# ==========================================
# app.py
# Credit Card Fraud Detection (Professional UI)
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import os

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.markdown(
    "<h2 style='text-align:center;'>💳 Credit Card Fraud Detection</h2>",
    unsafe_allow_html=True
)

st.write("Upload transaction data to detect fraudulent activity.")

# =============================
# Load Model
# =============================
MODEL_PATH = "fraud_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found! Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# =============================
# File Upload
# =============================
uploaded_file = st.file_uploader(
    "📂 Upload CSV file (same format as dataset)",
    type=["csv"]
)

# =============================
# Prediction
# =============================
if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(data.head())

        # Check required columns
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        if st.button("🔍 Run Fraud Detection"):

            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1]

            data["Prediction"] = predictions
            data["Fraud Probability"] = probabilities

            st.subheader("✅ Results")
            st.dataframe(data)

            # Summary
            fraud_count = (predictions == 1).sum()
            legit_count = (predictions == 0).sum()

            st.markdown("### 📌 Summary")
            st.write(f"🚨 Fraud Transactions: **{fraud_count}**")
            st.write(f"✅ Legit Transactions: **{legit_count}**")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")