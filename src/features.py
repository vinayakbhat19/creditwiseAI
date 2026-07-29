import numpy as np
import pandas as pd


def safe_ratio(numerator, denominator, epsilon=1e-6):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    safe_denominator = denominator.where(denominator.abs() > epsilon, np.nan)
    return (numerator / safe_denominator).replace([np.inf, -np.inf], np.nan)


def add_credit_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the feature engineering used during model training."""
    data = dataframe.copy()

    if "Applicant_Income" in data.columns:
        income = pd.to_numeric(data["Applicant_Income"], errors="coerce")
        data["Applicant_Income_log"] = np.log1p(income.clip(lower=0))

    income_columns = [
        c for c in ["Applicant_Income", "Coapplicant_Income", "Co_Applicant_Income"]
        if c in data.columns
    ]

    if income_columns:
        income_frame = data[income_columns].apply(pd.to_numeric, errors="coerce")
        data["Total_Income"] = income_frame.sum(axis=1, min_count=1)

        if "Loan_Amount" in data.columns:
            data["Loan_to_Income"] = safe_ratio(
                data["Loan_Amount"], data["Total_Income"]
            )

        if "Dependents" in data.columns:
            dependents = pd.to_numeric(data["Dependents"], errors="coerce")
            data["Income_per_Dependent"] = safe_ratio(
                data["Total_Income"], dependents.fillna(0) + 1
            )

    if {"Savings", "Loan_Amount"}.issubset(data.columns):
        data["Savings_to_Loan"] = safe_ratio(
            data["Savings"], data["Loan_Amount"]
        )

    if {"Collateral_Value", "Loan_Amount"}.issubset(data.columns):
        data["Collateral_Coverage"] = safe_ratio(
            data["Collateral_Value"], data["Loan_Amount"]
        )

    return data.replace([np.inf, -np.inf], np.nan)
