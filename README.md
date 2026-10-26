# Churn-Risk-App


A Streamlit app that estimates a telecom customer's risk of churning.
Enter a customer's details and it returns a churn risk score, a risk tier
and a suggested retention action.

**Live app:** https://churn-risk-app-dao.streamlit.app/

## How it works
- Data: Telco Customer Churn dataset (7,043 customers, about 26% churned).
- Model: Logistic Regression with class weighting, inside a scikit-learn
  Pipeline (one-hot encoding + scaling + model).
- On a held-out 20% test set the model reached ROC-AUC of about 0.845.
- Risk tiers: High / Medium / Low using thresholds of 0.6 - 0.3.

## Run it locally
pip install -r requirements.txt
streamlit run app1.py

## Files
- app1.py: the Streamlit page
- features.py: shared feature preparation
- model/churn_model.joblib: the trained Pipeline
- requirements.txt: pinned library versions

## Limitations
- The model was trained with class weighting, so the output is a risk score
  that ranks customers well, not an exact probability.
- The Telco data is a single snapshot with no calendar dates, so it cannot
  show how the model behaves over time.
- Brand-new customers (tenure 0) are barely represented in the training
  data, so their scores are less reliable.
- Some inputs are demographic (gender, senior citizen, partner, dependents).
  A real deployment should review these for fairness.
- This is a learning project on public data, not a production tool.
