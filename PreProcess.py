import numpy as np
import pandas as pd
import streamlit as st
from utils.load_data import load_data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
     # -----------------------------
    # Feature Engineering
    # -----------------------------
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round().astype(int)
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).clip(lower=0, upper=60)

    df["DTI"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["LOAN_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    # -----------------------------
    # Missing Values
    # -----------------------------
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.6].index
    df.drop(columns=cols_to_drop, inplace=True)

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # -----------------------------
    # Standardize Categories
    # -----------------------------
    def merge_rare(series, threshold=0.01):
        freq = series.value_counts(normalize=True)
        rare = freq[freq < threshold].index
        return series.replace(rare, "Other")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = merge_rare(df[col])

    # -----------------------------
    # Outlier Handling
    # -----------------------------
    def winsorize(s):
        return np.clip(s, s.quantile(0.01), s.quantile(0.99))

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = winsorize(df[col])

    # -----------------------------
    # Income Brackets
    # -----------------------------
    df["INCOME_BRACKET"] = pd.qcut(
        df["AMT_INCOME_TOTAL"],
        q=[0, 0.25, 0.75, 1.0],
        labels=["Low", "Mid", "High"]
    )

    df["CODE_GENDER"] = df["CODE_GENDER"].replace({"XNA": "Other"})
    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].replace({"Academic degree": "Higher education"})
    df["NAME_FAMILY_STATUS"] = df["NAME_FAMILY_STATUS"].replace({"Unknown": "Other"})
    df["NAME_HOUSING_TYPE"] = df["NAME_HOUSING_TYPE"].replace({"Co-op apartment": "Other"})


    return df