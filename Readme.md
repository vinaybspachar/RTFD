# 💳 Real-time Fraud Detection System (Credit Union)

A real-time fraud detection system built using **Python**, **Snowflake**, and **Streamlit UI**, designed to assist credit unions in identifying and flagging suspicious transactions.

---

## 🛠️ Tools & Technologies Used

- **IDE**: PyCharm / VS Code  
- **Backend**: FastAPI (exposed via Swagger UI)  
- **Frontend**: Streamlit  
- **Database**: Snowflake  
- **Python Libraries**: `pandas`, `scikit-learn`, `fastapi`, `streamlit`, etc.

---

## 📁 Project Structure

📦 project-root/
├── frontend/
│ ├── requirements.txt
│ ├── .env
│ └── ui_app.py
├── backend/
│ ├── requirements.txt
│ ├── database.py
│ ├── train_model.py
│ └── app.py


---

## 🚀 Getting Started

### 1. Clone or Download the Project

```bash
https://github.com/vinaybspachar/Fraud_RTFD/tree/main/PycharmProjects/PythonProject3

PycharmProjects/PythonProject3 (path)
cd fraud-detection-system
Or manually download and extract the ZIP file, then open it using VS Code or PyCharm.

2. Install Dependencies
Install backend and frontend requirements separately.

Backend
bash

cd backend
pip install -r requirements.txt
Frontend
bash
cd ../frontend
pip install -r requirements.txt

3. Configure Snowflake Database
Go to Snowflake and create a free account.

Inside the Snowflake UI:

Create a new database.

Note your account URL, username, password, and database name.

In the frontend/.env file, add your Snowflake credentials:

 
SNOWFLAKE_ACCOUNT=your_account_url
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_DATABASE=your_database_name

4. Verify Database Connection
Run the following script to test the Snowflake connection:

cd backend
python database.py

5. Train the Model
Run the model training script and check for accuracy output:

python train_model.py
6. Run the Backend (Swagger API)
Start the FastAPI backend:

bash
uvicorn app:app --reload
Access the API documentation at: http://localhost:8000/docs
7. Run the Frontend (Streamlit UI)
In a new terminal:

bash

cd frontend
streamlit run ui_app.py
Access the frontend at: http://localhost:8501

✅ Features
Train & evaluate fraud detection ML model.

View API documentation via Swagger UI.

Streamlit UI dashboard that connects to backend API.

Real-time database interaction with Snowflake.