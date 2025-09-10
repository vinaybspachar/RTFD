from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import shap
from database import load_data_from_snowflake

# Load model and encoders
model = joblib.load("xgb_model_multiclass.pkl")
encoders = joblib.load("encoders.pkl")
fraud_type_map = joblib.load("fraud_type_map.pkl")
fraud_type_map_inv = {v: k for k, v in fraud_type_map.items()}

# Load dataset
dataset = load_data_from_snowflake("SELECT * FROM FRAUD")
dataset.columns = [col.lower() for col in dataset.columns]

# Add datetime features
dataset["transaction_datetime"] = pd.to_datetime(dataset["transaction_datetime"], errors="coerce")
dataset["hour"] = dataset["transaction_datetime"].dt.hour
dataset["day"] = dataset["transaction_datetime"].dt.day
dataset["weekday"] = dataset["transaction_datetime"].dt.weekday

app = FastAPI()

class Transaction(BaseModel):
    Customer_ID: str
    Transaction_Type: str
    Transaction_Amount: float
    Device_Type: str

def log_prediction(customer_id, rule, ml):
    with open("predictions.log", "a") as f:
        f.write(f"[{datetime.now()}] Customer: {customer_id} | Rule: {rule} | ML: {ml}\n")

def send_email_alert(customer_id, fraud_type):
    msg = MIMEText(f"Fraud Alert Detected\n\nCustomer ID: {customer_id}\nFraud Type: {fraud_type}\nTimestamp: {datetime.now()}")
    msg['Subject'] = f"Fraud Alert - {fraud_type}"
    msg['From'] = "alert@example.com"
    msg['To'] = "staff@example.com"

    try:
        server = smtplib.SMTP("sandbox.smtp.mailtrap.io", 587)
        server.starttls()
        server.login("664b520ab3b6c3", "9b285cddfe1e97")
        server.sendmail(msg['From'], [msg['To']], msg.as_string())
        server.quit()
    except Exception as e:
        print("Email failed:", str(e))

@app.post("/predict")
def predict(txn: Transaction):
    try:
        data = txn.dict()
        cust_id = data["Customer_ID"].strip()
        cust_data = dataset[dataset["customer_id"] == cust_id]

        if cust_data.empty:
            raise HTTPException(status_code=404, detail="Customer ID not found")

        latest = cust_data.sort_values(by="transaction_datetime", ascending=False).iloc[0]

        input_data = {
            "transaction_type": data["Transaction_Type"],
            "transaction_amount": data["Transaction_Amount"],
            "location": latest["location"],
            "device_type": data["Device_Type"],
            "payment_method": latest["payment_method"],
            "failed_login_attempts": latest["failed_login_attempts"],
            "new_beneficiary_added": latest["new_beneficiary_added"],
            "unusual_location": latest["unusual_location"],
            "time_gap_between_transactions": latest["time_gap_between_transactions"],
            "transaction_frequency_per_day": latest["transaction_frequency_per_day"],
            "hour": latest["hour"],
            "day": latest["day"],
            "weekday": latest["weekday"],
            "is_rtp": 1 if data["Transaction_Type"].lower() == "rtp" else 0
        }

        # Rule-based logic
        rule_based_result = "None"
        if input_data["new_beneficiary_added"] == 1 and input_data["transaction_amount"] > 5000:
            rule_based_result = "APP Fraud"
        elif input_data["failed_login_attempts"] > 2 and input_data["unusual_location"] == 1:
            rule_based_result = "ATO + RTP Drain"

        # Encode categorical
        for field in ["transaction_type", "location", "device_type", "payment_method"]:
            encoder = encoders.get(field)
            if encoder and input_data[field] in encoder.classes_:
                input_data[field] = int(encoder.transform([input_data[field]])[0])
            else:
                raise HTTPException(status_code=400, detail=f"Unknown value for {field}: {input_data[field]}")

        input_df = pd.DataFrame([input_data])
        prediction_code = int(model.predict(input_df)[0])
        ml_prediction = fraud_type_map.get(prediction_code, "Unknown")

        # SHAP explanation
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        shap_contrib = shap_values.values[0][:, prediction_code]
        feature_contributions = pd.Series(shap_contrib, index=input_df.columns).abs().sort_values(ascending=False)
        top_features = feature_contributions.head(3).to_dict()

        # Logging + Alerts
        log_prediction(cust_id, rule_based_result, ml_prediction)
        if rule_based_result in ["APP Fraud", "ATO + RTP Drain"] or ml_prediction in ["APP Fraud", "ATO + RTP Drain"]:
            send_email_alert(cust_id, rule_based_result if rule_based_result != "None" else ml_prediction)

        return {
            "rule_based_result": rule_based_result,
            "ml_prediction": f"{ml_prediction} (ML-Based)",
            "top_features": top_features
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
