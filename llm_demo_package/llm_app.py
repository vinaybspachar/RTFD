import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
import json
from datetime import datetime
from openai import OpenAI
import asyncio

# 🔑 Create OpenAI client (reads key from env variable OPENAI_API_KEY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="LLM Fraud Detection API", version="1.0")


def summarize_transactions(transactions: List[dict]) -> str:
    """Compact summary for faster LLM analysis"""
    return "\n".join(
        f"{txn.get('Transaction_ID')} | {txn.get('Transaction_Type')} | "
        f"{txn.get('Transaction_Amount')} {txn.get('currency')} | "
        f"{txn.get('Device_Type')} | "
        f"{txn.get('originAccount', {}).get('origin_location')} -> "
        f"{txn.get('destinationAccount', {}).get('destination_location')}"
        for txn in transactions[:10]
    )


def analyze_overall_with_llm(transactions: List[dict]) -> dict:
    txn_summary = summarize_transactions(transactions)

    prompt = f"""
    You are a fraud detection assistant.
    Review these transactions and provide ONE overall fraud assessment:
    - If fraud is likely, suggest the most probable fraud type.
    - Explain briefly (2-3 sentences) why.
    - Keep it business-friendly and concise.

    Transactions:
    {txn_summary}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # ⚡ fast + cheap
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
        )
        analysis = response.choices[0].message.content.strip()
    except Exception as e:
        analysis = f"⚠️ LLM unavailable: {str(e)}"

    return {
        "overall_fraud_recommendation": analysis,
        "analyzed_transactions": len(transactions[:10]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/upload_json")
async def upload_json(file: UploadFile = File(...)):
    try:
        content = await file.read()
        transactions = json.loads(content.decode("utf-8"))

        if not isinstance(transactions, list):
            raise HTTPException(status_code=400, detail="JSON must be a list of transactions")

        # Run the blocking LLM call in a background thread
        result = await asyncio.to_thread(analyze_overall_with_llm, transactions)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
