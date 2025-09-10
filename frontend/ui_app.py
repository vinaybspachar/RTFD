from datetime import datetime

import streamlit as st
import requests
import json
import sys
import os

# 🔹 Ensure Python can find the backend module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the DB utility from backend/
from backend.database import load_data_from_snowflake

# Page configuration
st.set_page_config(page_title="RTP Fraud Detection", page_icon="⚠️")

# App header
st.title("🚨 Real-Time Fraud Detection (APP & ATO + RTP Drain)")
st.markdown(
    "This tool detects potential **fraudulent transactions** "
    "using both Rule-Based and ML-Based models. SHAP explainability and LLM explanations are included."
)

# Function to fetch a single record from Snowflake
def get_customer_record(customer_id: str):
    query = f"""
        SELECT 
            Transaction_ID             AS transaction_id,
            timestamp                  AS ts,
            Transaction_Amount         AS transaction_amount,
            currency                   AS currency,
            Transaction_Type           AS transaction_type,
            Sender_Account             AS sender_account,
            origin_routing_number      AS origin_routing_number,
            Customer_ID                AS customer_id,
            origin_location            AS origin_location,
            device_id                  AS device_id,
            Receiver_Account           AS receiver_account,
            destination_routing_number AS destination_routing_number,
            destination_location       AS destination_location
        FROM C_U_RTPD.PUBLIC."DATABASE"
        WHERE Customer_ID = '{customer_id}'
        LIMIT 1
    """
    df = load_data_from_snowflake(query)
    if df.empty:
        return None
    return df.iloc[0].to_dict()

# Input Form
with st.form("fraud_form"):
    customer_id = st.text_input("Customer ID", value="CUST8084")

    transaction_type = "Member-to-External Transfer"
    st.text_input("Transaction Type", value=transaction_type, disabled=True)

    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=50000.0)

    device_type = "Web"
    st.text_input("Device Type", value=device_type, disabled=True)

    account_id = st.text_input("Account ID", value="ACC123456")
    device_id = st.text_input("Device ID", value="DEV98765")
    location = st.text_input("Location", value="New York, USA")

    # Two separate buttons
    detect_btn = st.form_submit_button("Detect Fraud")
    json_btn = st.form_submit_button("Show JSON Payload")

# ✅ Build record (common for both buttons)
if detect_btn or json_btn:
    record = get_customer_record(customer_id)

    if not record:
        st.error("❌ Customer ID not found in Snowflake")
    else:
        payload = {
            "Transaction_ID": record.get("transaction_id"),
            "timestamp": str(record.get("ts")),
            "Transaction_Amount": float(record.get("transaction_amount", 0)),
            "currency": record.get("currency"),
            "Transaction_Type": record.get("transaction_type"),
            "Customer_ID": record.get("customer_id"),
            "Device_Type": "Web",
            "originAccount": {
                "Sender_Account": record.get("sender_account"),
                "origin_routing_number": record.get("origin_routing_number"),
                "Customer_ID": record.get("customer_id"),
                "origin_location": record.get("origin_location"),
                "deviceId": record.get("device_id"),
            },
            "destinationAccount": {
                "Receiver_Account": record.get("receiver_account"),
                "destination_routing_number": record.get("destination_routing_number"),
                "destination_location": record.get("destination_location"),
            },
        }

        # Add ML/Rule-based expected fields
        ml_payload = {
            "transaction_amount": float(record.get("transaction_amount", 0)),
            "new_beneficiary_added": 1 if record.get("transaction_type") == "New Payee" else 0,
            "failed_login_attempts": 3 if record.get("device_id") == "DEV98765" else 0,
            "unusual_location": 1 if "Nigeria" in str(record.get("origin_location")) else 0
        }

        # 🎯 Show JSON payload
        if json_btn:
            st.subheader("📦 JSON Payload (from Snowflake)")
            st.code(json.dumps(payload, indent=4), language="json")

        # 🎯 Detect Fraud
        if detect_btn:
            try:
                final_payload = {**payload, **ml_payload}

                response = requests.post("http://localhost:8000/predict", json=final_payload)
                result = response.json()

                if response.status_code != 200:
                    st.error(f"❌ Error: {result.get('detail', 'Unknown error')}")
                else:
                    st.success("✅ Prediction Successful!")

                    col1, col2 = st.columns(2)
                    col1.metric("Rule-Based Result", result["rule_based_result"])
                    col2.metric("ML Prediction", result["ml_prediction"])

                    if "top_features" in result and result["top_features"]:
                        st.subheader("🔍 SHAP Explanation (Top 3 Features)")
                        st.json(result["top_features"])

                        # --- Optional: also log from UI side (extra safety) ---
                        with open("predictions.log", "a") as f:
                            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] "
                                    f"UI SHAP Features for Customer {payload['Customer_ID']}: {result['top_features']}\n")

                    if result["rule_based_result"] in ["APP Fraud", "ATO + RTP Drain"] or "APP Fraud" in result[
                        "ml_prediction"] or "ATO + RTP Drain" in result["ml_prediction"]:
                        st.warning("🚨 Email alert has been sent to staff.")
                    else:
                        st.info("✅ No fraud detected. Email alert not triggered.")


                    # ✅ Add this new block here
                    if "top_features" in response:
                        st.write("🔎 Top Features Contributing to Prediction:")
                        for feat, val in response["top_features"].items():
                            st.write(f"- {feat}: {val:.2f}")




            except Exception as e:
                st.error(f"⚠️ Internal Error: {str(e)}")
