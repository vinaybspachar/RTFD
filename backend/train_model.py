import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from database import load_data_from_snowflake

# Load dataset from Snowflake
query = "SELECT * FROM DATABASE"
df = load_data_from_snowflake(query)
df.columns = [col.lower() for col in df.columns]

# Datetime features
df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
df["hour"] = df["transaction_datetime"].dt.hour
df["day"] = df["transaction_datetime"].dt.day
df["weekday"] = df["transaction_datetime"].dt.weekday

# Encode categoricals
categorical_cols = ["transaction_type", "location", "device_type", "payment_method"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Add is_rtp feature
rtp_index = None
if "rtp" in encoders["transaction_type"].classes_:
    rtp_index = list(encoders["transaction_type"].classes_).index("rtp")
df["is_rtp"] = (df["transaction_type"] == rtp_index).astype(int) if rtp_index is not None else 0

# Assign fraud labels based on business rules
def assign_label(row):
    if row["failed_login_attempts"] > 2 and row["unusual_location"] == 1:
        return 2  # ATO + RTP Drain
    elif row["new_beneficiary_added"] == 1 and row["transaction_amount"] > 5000:
        return 1  # APP Fraud
    else:
        return 0  # Normal

df["fraud_label"] = df.apply(assign_label, axis=1)
fraud_type_map = {0: "None", 1: "APP Fraud", 2: "ATO + RTP Drain"}

# Class distribution after labeling
print("Class Distribution After Reassignment:")
print(df["fraud_label"].value_counts(), "\n")

# Features and target
features = [
    "transaction_type", "transaction_amount", "location", "device_type", "payment_method",
    "failed_login_attempts", "new_beneficiary_added", "unusual_location",
    "time_gap_between_transactions", "transaction_frequency_per_day",
    "hour", "day", "weekday", "is_rtp"
]
X = df[features]
y = df["fraud_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Resampled Class Distribution:")
print(pd.Series(y_resampled).value_counts(), "\n")

# Train XGBoost model
model = XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42)
model.fit(X_resampled, y_resampled)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Model Accuracy:", model.score(X_test, y_test), "\n")

# Save model and encoders
joblib.dump(model, "xgb_model_multiclass.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(fraud_type_map, "fraud_type_map.pkl")
print("Model and encoders saved successfully.")
