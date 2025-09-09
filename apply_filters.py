# filters.py
import streamlit as st

def add_global_filters(df):
    st.sidebar.header("🌍 Global Filters")

    # --- Gender ---
    gender = st.sidebar.multiselect(
        "Gender",
        df["CODE_GENDER"].unique(),
        default=list(df["CODE_GENDER"].unique())
    )

    # --- Education ---
    education = st.sidebar.multiselect(
        "Education",
        df["NAME_EDUCATION_TYPE"].unique(),
        default=list(df["NAME_EDUCATION_TYPE"].unique())
    )

    # --- Family Status ---
    family_status = st.sidebar.multiselect(
        "Family Status",
        df["NAME_FAMILY_STATUS"].unique(),
        default=list(df["NAME_FAMILY_STATUS"].unique())
    )

    # --- Housing Type ---
    housing = st.sidebar.multiselect(
        "Housing Type",
        df["NAME_HOUSING_TYPE"].unique(),
        default=list(df["NAME_HOUSING_TYPE"].unique())
    )

    # --- Age Range ---
    min_age = int(df["DAYS_BIRTH"].apply(lambda x: -x/365).min())
    max_age = int(df["DAYS_BIRTH"].apply(lambda x: -x/365).max())
    age_range = st.sidebar.slider(
        "Age Range",
        min_age, max_age, (min_age, max_age)
    )

    # --- Income Bracket ---
    min_income = int(df["AMT_INCOME_TOTAL"].min())
    max_income = int(df["AMT_INCOME_TOTAL"].max())
    income_range = st.sidebar.slider(
        "Income Bracket",
        min_income, max_income, (min_income, max_income),
        step=10000
    )

    # Save in session_state
    st.session_state["filters"] = {
        "gender": gender,
        "education": education,
        "family_status": family_status,
        "housing": housing,
        "age_range": age_range,
        "income_range": income_range,
    }

    # --- Apply Filters ---
    df_filtered = df.copy()

    df_filtered = df_filtered[
        (df_filtered["CODE_GENDER"].isin(gender)) &
        (df_filtered["NAME_EDUCATION_TYPE"].isin(education)) &
        (df_filtered["NAME_FAMILY_STATUS"].isin(family_status)) &
        (df_filtered["NAME_HOUSING_TYPE"].isin(housing)) &
        (df_filtered["DAYS_BIRTH"].apply(lambda x: -x/365).between(age_range[0], age_range[1])) &
        (df_filtered["AMT_INCOME_TOTAL"].between(income_range[0], income_range[1]))
    ]

    return df_filtered
