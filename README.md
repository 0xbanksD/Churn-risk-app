# SaaS Customer Churn Prediction

Predicting which telecom/subscription customers are likely to leave, explaining why, and putting the model into a working app, with a time-aware evaluation, a drift-monitoring check, and a survival analysis of *when* customers churn.

**Live demo:** https://churn-risk-app-dao.streamlit.app/

![App screenshot](<img width="1280" height="698" alt="images-app png" src="https://github.com/user-attachments/assets/ce36ea1c-4c2f-4bc2-831e-aa0001566a7a" />
) <img width="1280" height="699" alt="images2-app png" src="https://github.com/user-attachments/assets/204d79f1-a069-4bec-8b08-69c89eb33834" />


## The problem

Losing customers is expensive. This project builds a model that flags customers at risk of churning so a retention team can act early, and it tests how far the model can be trusted.

## Data

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data): 7,043 customers, 20 features, about 26% churned. The data file is **not** included in this repo; see "Run the notebook" below.

## Approach

1. **Cleaning and feature engineering:** fixed the blank `TotalCharges` values, merged the "No internet service" / "No phone service" levels into "No", and added `tenure_bucket` and `high_risk_contract` features.
2. **Modelling:** compared several models. The final model is a **Logistic Regression** (`class_weight="balanced"`, `max_iter=1000`) inside a scikit-learn `Pipeline` (one-hot encoding + scaling), so training and the app use identical preprocessing.
3. **Interpretation:** TODO: add 1-2 lines on your SHAP findings (top drivers of churn).
4. **App:** a Streamlit form that returns a churn risk score, a risk tier and a suggested action.

## Results (held-out test set, threshold 0.5)

| Evaluation | ROC-AUC | Recall | Precision |
|---|---|---|---|
| Random split | 0.845 | 0.703 | 0.557 |
| Time-based split (tenure proxy) | 0.756 | 0.774 | 0.662 |

The score is a **risk score, not a true probability**, because the model uses class weighting.

## Stretch goals

### 1. Deployed Streamlit app
`app1.py` loads the saved pipeline (`model/churn_model.joblib`) and `features.py` (shared with the notebook). Risk tiers: TODO-fill-in (for example High ≥ 0.xx, Medium ≥ 0.xx, Low below that).

### 2. Time-based split
The dataset has no dates, so tenure is used as a proxy: customers with longer tenure ("joined earlier") train the model and the newest 25% (tenure 0–9 months) test it. ROC-AUC fell from 0.845 to 0.756, and forward-chaining tuning did not improve it. Precision and PR-AUC rose only because churn is much more common in the test set (49.8% vs 18.2%). Removing the tenure features narrowed the ROC-AUC gap (0.818 vs 0.760), so part of the drop is an artefact of the proxy. I read this as "less effective on newer customers", not as proof of drift.

### 3. Survival analysis
Kaplan-Meier curves show month-to-month customers losing about 30% within 12 months (70.3% remain) and 41% within 24 months (58.6% remain), versus 99.1% and 97.8% remaining for one-year contracts (two-year contracts show 100% because customers are locked in). A Cox model (test concordance 0.845) found contract length, fiber optic service (hazard ratio about 2.3 vs DSL) and electronic or mailed check payment (about 1.8–1.9 vs automatic bank transfer) to be the strongest factors. The proportional-hazards check flagged contract type, monthly charges and mailed check, so hazard ratios are time-averaged and approximate.

### 4. Monitoring check
`check_batch()` scores a batch of new customers and compares it to a training baseline using a z-test on the share flagged (warning at 2, alert at 3) and the Population Stability Index of the scores (warning at 0.10, alert at 0.25). It returns OK, WARNING, ALERT, or TOO SMALL (under 100 rows). On a normal batch it returned OK (z = +0.8, PSI = 0.023). On a deliberately drifted batch (85% month-to-month contracts) it returned ALERT (z = +10.4, PSI = 0.352). It watches predictions, not accuracy, because churn labels arrive months later.

## Business recommendations

TODO: add 3 short bullets from your report (for example, target month-to-month customers in their first year, encourage annual contracts or autopay, review the fiber optic experience).

## Limitations

- Single snapshot, no signup dates: results show association, not cause.
- The time-based split uses a tenure proxy, which differs from the training group by construction.
- Risk scores are not calibrated probabilities.
- Contract type is each customer's current contract.

## Repository structure

```
├── app1.py                      # Streamlit app
├── features.py                  # shared preprocessing (notebook + app)
├── model/churn_model.joblib     # trained scikit-learn pipeline
├── Telco Customer churn.ipynb   # full analysis notebook
├── requirements.txt             # app dependencies (used by Streamlit Cloud)
├── requirements-notebook.txt    # notebook dependencies
├── images/                      # figures used in this README
└── reports/                     # written project report
```

## Run the app locally

```bash
git clone https://github.com/TODO-your-username/churn-app.git
cd churn-app
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app1.py
```

## Run the notebook

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the Kaggle link above and put it in a folder named `Project_DS/`.
2. `pip install -r requirements-notebook.txt`
3. Open the notebook and choose Run All.

## Author

Darlington Agho Osarodion, Abuja, Nigeria. TODO: add your LinkedIn link.
