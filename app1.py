

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


from features import prepare_features

st.set_page_config(page_title="Churn Risk Predictor")

MODEL_PATH = Path(__file__).parent / "model" / "churn_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


pipe = load_model()

# ---- Risk tiers: REPLACE these with the thresholds and actions from your project ----
HIGH, MEDIUM = 0.60, 0.35
ACTIONS = {
    "High Risk": "Priority retention call this week; offer a contract or loyalty incentive.",
    "Medium Risk": "Include in the next retention campaign; check service quality.",
    "Low Risk": "No action needed; keep monitoring.",
}

YES_NO = ["Yes", "No"]
SERVICE = ["Yes", "No", "No internet service"]

st.title("Customer Churn Risk Predictor")
st.caption("Enter a customer's details to estimate their risk of leaving.")

with st.form("customer"):
    left, right = st.columns(2)

    with left:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12,
                           help="How long the customer has been with the company.")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.radio("Paperless billing", YES_NO, horizontal=True)
        monthly = st.number_input("Monthly charges ($)", min_value=18.0, max_value=120.0,
                                  value=70.0, step=0.5,
                                  help="Training data ranged from about 18 to 119.")
        st.subheader("Customer")
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        senior = st.radio("Senior citizen", YES_NO, index=1, horizontal=True)
        partner = st.radio("Has a partner", YES_NO, index=1, horizontal=True)
        dependents = st.radio("Has dependents", YES_NO, index=1, horizontal=True)

    with right:
        st.subheader("Services")
        phone = st.radio("Phone service", YES_NO, horizontal=True)
        multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online security", SERVICE, index=1)
        backup = st.selectbox("Online backup", SERVICE, index=1)
        protection = st.selectbox("Device protection", SERVICE, index=1)
        tech = st.selectbox("Tech support", SERVICE, index=1)
        tv = st.selectbox("Streaming TV", SERVICE, index=1)
        movies = st.selectbox("Streaming movies", SERVICE, index=1)

    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    addons = [security, backup, protection, tech, tv, movies]
    if internet == "No" and "Yes" in addons:
        st.warning("Internet service is 'No', but an internet add-on is 'Yes'. "
                   "Check these inputs.")
    if tenure == 0:
        st.info("Brand-new customer (0 months): the score is less reliable.")

    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    }])
    features = prepare_features(row)

    proba = float(pipe.predict_proba(features)[0, 1])   # column 1 = churn
    tier = "High Risk" if proba >= HIGH  else "Medium Risk" if proba >= MEDIUM else "Low Risk"

    st.divider()
    st.metric("Churn risk score", f"{proba:.0%}")
    st.progress(proba)
    st.subheader(f"Risk tier: {tier}")
    st.write(ACTIONS[tier])

    with st.expander("Debug: the row sent to the model"):
        st.dataframe(features)

    st.caption("Note: the model was trained with class weighting, so this is a risk "
               "score that ranks customers well, not an exact probability.")