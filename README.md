# AI-in-health-care-
 project management skills to improve patient outcomes and operational efficiency
AI-in-Healthcare-Project-Management/
│── data/
│   └── hospital_data.csv
│── notebooks/
│   └── model_training.ipynb
│── app.py
│── requirements.txt
│── README.md
│── screenshots/
import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000
data = {
    "Patient_ID": range(1, n+1),
    "Age": np.random.randint(20, 90, n),
    "Department": np.random.choice(["Cardiology", "Surgery", "Oncology", "Emergency", "ICU"], n),
    "Length_of_Stay": np.random.randint(1, 15, n),
    "Cost": np.random.randint(1000, 20000, n),
    "Satisfaction_Score": np.random.randint(1, 11, n),
    "Readmission": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    "Wait_Time": np.random.randint(10, 180, n)
}

df = pd.DataFrame(data)
df.to_csv("hospital_data.csv", index=False)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df[["Age", "Length_of_Stay", "Cost", "Wait_Time", "Satisfaction_Score"]]
y = df["Readmission"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
import streamlit as st
import plotly.express as px

st.title("🏥 AI in Healthcare - Project Management Dashboard")

# Upload data
df = pd.read_csv("data/hospital_data.csv")

# KPIs
avg_stay = round(df["Length_of_Stay"].mean(), 1)
avg_cost = round(df["Cost"].mean(), 2)
avg_satisfaction = round(df["Satisfaction_Score"].mean(), 1)

col1, col2, col3 = st.columns(3)
col1.metric("📅 Avg Length of Stay", avg_stay)
col2.metric("💰 Avg Cost", f"${avg_cost}")
col3.metric("😊 Avg Satisfaction", avg_satisfaction)

# Charts
st.subheader("Department-wise Satisfaction")
fig1 = px.box(df, x="Department", y="Satisfaction_Score", color="Department")
st.plotly_chart(fig1)

st.subheader("Readmission Rates by Department")
fig2 = px.histogram(df, x="Department", color="Readmission", barmode="group")
st.plotly_chart(fig2)
