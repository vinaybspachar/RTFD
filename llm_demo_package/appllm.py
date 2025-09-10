"""
appllm.py

Streamlit app: Upload a JSON file of transactions (or a single transaction),
call an LLM to analyze and recommend an overall fraud classification,
report the most probable fraud type and a 2-3 sentence business-friendly explanation.

Requirements:
    pip install streamlit openai
Usage:
    export OPENAI_API_KEY="sk-..."
    (optional) export OPENAI_MODEL="gpt-4o-mini"   # or whichever model you prefer
    streamlit run appllm.py
"""

import os
import json
import streamlit as st
from typing import Any, Dict, List
import openai
from datetime import datetime

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change as needed
MAX_TRANSACTIONS_TO_SEND = 30  # limit to keep prompt size manageable

#Checking for opernapi key
if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY environment variable before running the app.")
else:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="LLM Fraud Recommender", page_icon="🛡️", layout="centered")

st.title("LLM-Based Fraud Detection — Overall Recommendation")
st.markdown(
    "Upload a JSON file (an array of transactions or a single transaction). "
    "The app will ask the LLM to analyze and give an overall fraud recommendation, the most probable fraud type, "
    "and a short (2–3 sentence) business-friendly explanation."
)

# Example expected JSON schema hint
st.info(
    "Expected file format: JSON array of transaction objects, each having keys like:\n\n"
    "`transaction_id, timestamp, origin_account, destination_account, amount, location, device_id, transaction_type, risk_flags`"
)

uploaded_file = st.file_uploader("Upload transactions JSON", type=["json"])

def build_prompt(transactions: List[Dict[str, Any]]) -> str:
    # Keep prompt concise, include only the most relevant fields
    picked = []
    for t in transactions[:MAX_TRANSACTIONS_TO_SEND]:
        # pick a handful of keys if present
        picked.append({
            "transaction_id": t.get("transaction_id") or t.get("Transaction_ID") or t.get("id"),
            "timestamp": t.get("timestamp") or t.get("time") or t.get("ts"),
            "origin_account": t.get("origin_account") or t.get("originAccount") or t.get("from_account"),
            "destination_account": t.get("destination_account") or t.get("destinationAccount") or t.get("to_account"),
            "amount": t.get("amount"),
            "location": t.get("location"),
            "device_id": t.get("device_id") or t.get("deviceId") or t.get("device"),
            "transaction_type": t.get("transaction_type") or t.get("type"),
            "risk_flags": t.get("risk_flags") or t.get("riskFlags") or t.get("flags"),
        })
    # Summarize prompt
    prompt = (
        "You are a concise fraud analyst assistant for a banking operations team. "
        "Given the following transactions (JSON array), give an overall recommendation whether fraud is likely or unlikely, "
        "and if likely, suggest the single most probable fraud type from this list:\n"
        "['Account Takeover', 'Card Not Present Fraud', 'Insider Fraud', 'Money Laundering', 'Social Engineering / Phishing', "
        "'New Payee Fraud', 'Synthetic Identity Fraud', 'Unauthorized Transfer', 'Mule Account', 'Other'].\n\n"
        "Return a strict JSON object with keys: recommendation (values: 'Likely', 'Unlikely', or 'Inconclusive'), "
        "most_probable_fraud (one of the fraud types above or 'None'), confidence (percentage 0-100), "
        "explanation (2-3 business-friendly sentences). Keep explanations short and actionable.\n\n"
        "Transactions:\n"
    )
    prompt += json.dumps(picked, default=str, indent=2)
    prompt += "\n\nDo NOT include internal chain-of-thought. Be concise and business-friendly."
    return prompt

def call_llm(prompt: str) -> Dict[str, Any]:
    # Use ChatCompletion or Chat API; this uses chat completion interface
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise fraud detection analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=450,
        n=1,
    )
    text = response["choices"][0]["message"]["content"].strip()
    # Try parse JSON from LLM response
    parsed = {}
    try:
        # Find first '{' to parse JSON robustly
        start = text.find("{")
        if start != -1:
            json_text = text[start:]
            parsed = json.loads(json_text)
        else:
            # fallback: attempt to parse entire response
            parsed = json.loads(text)
    except Exception as e:
        # if parsing fails, return raw text for debugging
        parsed = {"_raw_response": text, "_parse_error": str(e)}
    # attach metadata
    parsed["_llm_text"] = text
    parsed["_timestamp"] = datetime.utcnow().isoformat() + "Z"
    return parsed

if uploaded_file:
    try:
        data = json.load(uploaded_file)
        # normalize single object into list
        if isinstance(data, dict):
            transactions = [data]
        elif isinstance(data, list):
            transactions = data
        else:
            st.error("Uploaded JSON must be either an object or an array of objects.")
            st.stop()
    except Exception as exc:
        st.error(f"Failed to parse JSON: {exc}")
        st.stop()

    st.write(f"Loaded {len(transactions)} transaction(s). Showing first {min(len(transactions), MAX_TRANSACTIONS_TO_SEND)} for analysis.")

    # Show a compact preview
    preview = transactions[:MAX_TRANSACTIONS_TO_SEND]
    st.json(preview)

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not set. Set environment variable and restart.")
    else:
        with st.spinner("Asking LLM for fraud recommendation..."):
            prompt = build_prompt(transactions)
            llm_result = call_llm(prompt)

        # Display result
        st.subheader("LLM Recommendation (parsed)")
        if "_raw_response" in llm_result:
            st.warning("Could not parse JSON from LLM response; showing raw text.")
            st.text_area("Raw LLM response", llm_result.get("_llm_text", ""), height=300)
            st.json(llm_result)
        else:
            # show the structured fields in a business-friendly card
            rec = llm_result.get("recommendation", "Inconclusive")
            fraud = llm_result.get("most_probable_fraud", "None")
            confidence = llm_result.get("confidence", "")
            explanation = llm_result.get("explanation", "")

            st.metric(label="Recommendation", value=rec)
            st.write(f"**Most probable fraud:** {fraud}  \n**Confidence:** {confidence}")
            st.write("**Explanation (2-3 sentences):**")
            st.write(explanation)

            st.markdown("---")
            st.subheader("LLM full JSON output (for auditing)")
            st.json(llm_result)

        st.success("Done ✅")

# Footer / tips
st.markdown("---")
st.caption(
    "Tips: Keep uploads < few MB. If transactions are many, preprocess and send a sample or aggregated features. "
    "You can change OPENAI_MODEL env var to use a different model."
)
