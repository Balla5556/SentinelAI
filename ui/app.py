import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SentinelAI Command Center", layout="wide")

st.title("🛡️ SentinelAI — Fraud Analyst Command Center")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("System Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    risk_threshold = st.slider("Investigation Threshold", 0.0, 1.0, 0.5)
    st.info("Transactions above this score trigger the LangGraph Investigator.")

# Main Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 New Transaction")
    with st.form("txn_form"):
        t_id = st.text_input("Transaction ID", "TXN_9999")
        amount = st.number_input("Amount ($)", value=1200.0)
        category = st.selectbox("Category", ["electronics", "gas", "dining", "travel", "grocery"])
        location = st.text_input("Location", "London, UK")
        device = st.text_input("Device", "UNRECOGNIZED_LINUX")
        v1 = st.number_input("Velocity 1h", value=5)
        v24 = st.number_input("Velocity 24h", value=12)
        avg30 = st.number_input("Avg Amount 30d", value=150.0)
        dist = st.number_input("Dist from Home (km)", value=5000.0)
        foreign = st.selectbox("Foreign?", ["No", "Yes"])
        
        submit = st.form_submit_button("Analyze Transaction")

if submit:
    payload = {
        "transaction_id": t_id,
        "amount": amount,
        "merchant_category": category,
        "location": location,
        "device_fingerprint": device,
        "velocity_1h": v1,
        "velocity_24h": v24,
        "avg_amount_30d": avg30,
        "distance_from_home_km": dist,
        "foreign_transaction_flag": 1 if foreign == "Yes" else 0
    }
    
    with st.spinner("Running ML Prediction & Agentic Investigation..."):
        try:
            response = requests.post(f"{api_url}/investigate", json=payload)
            if response.status_code == 200:
                res = response.json()
                
                with col2:
                    st.subheader("🔍 Investigation Results")
                    
                    # Metric Cards
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk Score", f"{res['risk_score']:.2%}")
                    m2.metric("Recommendation", res['recommendation'])
                    m3.metric("Status", "Analyzed")
                    
                    # SHAP Drivers
                    st.markdown("### 📊 Model Drivers (SHAP)")
                    drivers_df = pd.DataFrame(res['shap_drivers'], columns=["Feature", "Impact"])
                    st.table(drivers_df)
                    
                    # Forensic Report
                    st.markdown("### 📝 Forensic Report")
                    st.info(res['forensic_report'])
                    
                    # Human-in-the-Loop Action
                    st.markdown("---")
                    st.markdown("### ⚖️ Analyst Decision")
                    c1, c2, c3 = st.columns(3)
                    if c1.button("✅ Approve", use_container_width=True):
                        st.success("Transaction Approved. Case Closed.")
                    if c2.button("❌ Reject", use_container_width=True):
                        st.error("Transaction Rejected. Account Frozen.")
                    if c3.button("🚩 Escalate", use_container_width=True):
                        st.warning("Case escalated to Senior Compliance Officer.")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
else:
    with col2:
        st.subheader("Waiting for transaction input...")
        st.write("Enter transaction details on the left to start the Predict $\rightarrow$ Explain $\rightarrow$ Investigate pipeline.")
