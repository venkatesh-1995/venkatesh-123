import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.load_data import load_data
from PreProcess import preprocess_data
from apply_filters import add_global_filters

# ===================== Load dataset =====================
df = preprocess_data(df=load_data(r"D:\venkatesh python class\Project\application_train (1).csv"))

st.title("📌 Page 5 — Correlations, Drivers & Interactive Slice-and-Dice")

df_filtered = add_global_filters(df)


# ===================== KPI Calculations =====================
num_cols = df_filtered.select_dtypes(include=[np.number]).columns
corr = df_filtered[num_cols].corr()

corr_target = corr["TARGET"].drop("TARGET").sort_values()
top5_neg = corr_target.head(5)
top5_pos = corr_target.tail(5)

most_corr_income = corr["AMT_INCOME_TOTAL"].drop("AMT_INCOME_TOTAL").abs().idxmax()
most_corr_credit = corr["AMT_CREDIT"].drop("AMT_CREDIT").abs().idxmax()

corr_income_credit = corr.loc["AMT_INCOME_TOTAL", "AMT_CREDIT"]
corr_age_target = corr.loc["AGE_YEARS", "TARGET"]
corr_emp_target = corr.loc["EMPLOYMENT_YEARS", "TARGET"]
corr_family_target = corr.loc["CNT_FAM_MEMBERS", "TARGET"]

variance_explained_proxy = corr_target.abs().sort_values(ascending=False).head(5).sum()
num_high_corr = (corr_target.abs() > 0.5).sum()

# ===================== KPI Display =====================
st.subheader("🔑 Key Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Top 5 +Corr (TARGET)", ", ".join([f"{x} ({y:.2f})" for x,y in top5_pos.items()]))
    st.metric("Top 5 −Corr (TARGET)", ", ".join([f"{x} ({y:.2f})" for x,y in top5_neg.items()]))
    st.metric("Most correlated with Income", most_corr_income)
    st.metric("Most correlated with Credit", most_corr_credit)
    st.metric("Corr(Income, Credit)", f"{corr_income_credit:.2f}")

with col2:
    st.metric("Corr(Age, TARGET)", f"{corr_age_target:.2f}")
    st.metric("Corr(EmploymentY, TARGET)", f"{corr_emp_target:.2f}")
    st.metric("Corr(Family Size, TARGET)", f"{corr_family_target:.2f}")
    st.metric("Variance Explained (Top 5)", f"{variance_explained_proxy:.2f}")
    st.metric("# Features |corr| > 0.5", num_high_corr)

# ===================== Correlation & Drivers Visuals =====================
st.subheader("📊 Correlation & Risk Drivers")

# --- Row 1: Heatmap & Top 20 |Corr| with TARGET ---
col1, col2 = st.columns(2)
with col1:
    # Select only the columns you want
    selected_cols = ["TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]
    # Compute correlation only for those columns
    corr_subset = df[selected_cols].corr()
    # Plot heatmap
    st.write("### Heatmap (Correlation Table - Selected Columns)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_subset, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax, cbar=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)



with col2:
    st.write("### |Correlation| with TARGET (Top 20)")
    top_corr = corr_target.abs().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax, palette="viridis")
    ax.set_xlabel("Correlation (absolute)")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


# --- Row 2: Age vs Credit & Age vs Income ---
col1, col2 = st.columns(2)
with col1:
    st.write("### Scatter — Age vs Credit (split by TARGET)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_filtered, x="AGE_YEARS", y="AMT_CREDIT", hue="TARGET", alpha=0.5, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### Scatter — Age vs Income (split by TARGET)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_filtered, x="AGE_YEARS", y="AMT_INCOME_TOTAL", hue="TARGET", alpha=0.5, ax=ax)
    st.pyplot(fig)



# --- Row 3: Employment Years vs Default Rate & Credit by Education ---
col1, col2 = st.columns(2)
with col1:
    st.write("### Employment Years vs Default Rate (binned)")
    emp_bins = pd.cut(df_filtered["EMPLOYMENT_YEARS"], bins=20)
    emp_rate = df_filtered.groupby(emp_bins)["TARGET"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=emp_rate.index.astype(str), y=emp_rate.values, ax=ax, color="skyblue")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Employment Years (binned)")
    st.pyplot(fig)

with col2:
    st.write("### Credit by Education (summary)")
    edu_summary = df_filtered.groupby("NAME_EDUCATION_TYPE")["AMT_CREDIT"].describe()[["mean", "50%", "max"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    edu_summary[["mean", "50%", "max"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Credit Amount")
    st.pyplot(fig)


# --- Row 4: Income by Family Status & Pair Plot (Income, Credit, Annuity, Target) ---
col1, col2 = st.columns(2)
with col1:
    st.write("### Income by Family Status (summary)")
    fam_summary = df_filtered.groupby("NAME_FAMILY_STATUS")["AMT_INCOME_TOTAL"].describe()[["mean", "50%", "max"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    fam_summary[["mean", "50%", "max"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Income Amount")
    st.pyplot(fig)

with col2:
    st.write("### Pair Plot (Income, Credit, Annuity, Target) — Preview")
    pair_df = df_filtered[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","TARGET"]].dropna().sample(min(2000,len(df_filtered)))
    fig = sns.pairplot(pair_df, hue="TARGET", diag_kind="kde", plot_kws={'alpha':0.5})
    st.pyplot(fig)


col1,col2=st.columns(2)

with col1:
    st.write("### Default Rate by Gender (Filtered)")
    gender_rate = df_filtered.groupby("CODE_GENDER")["TARGET"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=gender_rate.index, y=gender_rate.values, ax=ax, palette="Set2")
    ax.set_ylabel("Default Rate (%)")
    st.pyplot(fig)

with col2:
    st.write("### Default Rate by Education (Filtered)")
    edu_rate = df_filtered.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=edu_rate.index, y=edu_rate.values, ax=ax, palette="Set2")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)
st.divider()

# ===================== Narrative Insights =====================
st.subheader("📝 Key Insights")
st.markdown("""
- ⚠️ **High LTI (>6) & DTI (>0.35)** segments have elevated default risk.  
- 👶 **Younger, low-income borrowers** with high credit exposure are the primary drivers of defaults.  
- 🎓 **Higher education and stable family status** appear protective.  
- 🏦 Policy considerations:  
    - Minimum income floors by contract type.  
    - Stricter review for applicants with **short employment history** and high LTI.  
    - Preferential pricing for **low-risk groups** (high education, stable family).  
""")
