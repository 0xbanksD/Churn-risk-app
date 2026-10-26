import pandas as pd



MODEL_COLUMNS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'tenure_bucket', 'high_risk_contract']

def prepare_features(df):
    df = df.copy()
    for c in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6months", "6-12months", "1-2yrs", "2-4yrs", "4-6yrs"],
    ).astype(str)
    df["high_risk_contract"] = (df["Contract"] == "Month-to-month").astype(int)
    return df[MODEL_COLUMNS]