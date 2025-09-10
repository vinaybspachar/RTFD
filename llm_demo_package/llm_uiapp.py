import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from litellm import completion

# 🔑 Load API key from .env
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# 🚨 Streamlit UI setup
st.set_page_config(page_title="LLM Fraud Rule Recommender", page_icon="⚠️", layout="centered")
st.title("🧠 LLM-Based Fraud Rule Recommender")
st.write("Upload a CSV or JSON file with recent transactions. The LLM will analyze patterns and suggest **new business rules** for APP Fraud and ATO Fraud detection.")

# 📂 File uploader
uploaded_file = st.file_uploader("Upload Transactions File", type=["csv", "json"])

def generate_response(user_input: str) -> str:
    """Send transactions to LLM and get new fraud rule recommendations"""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a fraud detection strategy assistant. "
                    "Analyze the following transactions and recommend **new business rules** "
                    "to detect APP Fraud and ATO Fraud. "
                    "⚠️ Do NOT repeat existing rules (like new beneficiary, failed logins, unusual location, high amount). "
                    "Instead, propose fresh, measurable, business-friendly rules "
                    "(e.g., device reuse patterns, velocity checks, cross-account activity). "
                    "Return ONLY valid JSON with the following keys:\n"
                    #"- new_app_rules (list of strings)\n"
                    #"- new_ato_rules (list of strings)\n"
                    "Generate new ATO and APP Rules based on the previous history data"
                    "Transactions:\n"
                )
            },
            {"role": "user", "content": user_input}
        ]

        response = completion(
            model="perplexity/sonar-reasoning",   # can also try "perplexity/sonar-pro"
            messages=messages,
            reasoning_effort="high",
            api_key=PERPLEXITY_API_KEY
        )

        return response["choices"][0]["message"]["content"]

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# 📊 Process file when uploaded
if uploaded_file is not None:
    if uploaded_file.type == "application/json":
        data = json.load(uploaded_file)
        st.subheader("📑 Uploaded JSON Preview")
        st.json(data)
        text_data = json.dumps(data, indent=2)

    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.subheader("📑 Uploaded CSV Preview")
        st.dataframe(df.head())
        text_data = df.to_csv(index=False)

    else:
        st.error("Unsupported file format. Please upload a CSV or JSON file.")
        text_data = None

    if text_data:
        if st.button("🚀 Recommend New Rules"):
            with st.spinner("LLM analyzing transactions and proposing new rules..."):
                result = generate_response(text_data)

            st.subheader("✅ New Fraud Rules Suggested")
            try:
                parsed = json.loads(result)  # Ensure valid JSON
                st.json(parsed)
            except Exception:
                st.write(result)  # Fallback if model fails to return JSON
