import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load_data import load_data
from PreProcess import preprocess_data
from apply_filters import add_global_filters

# ===================== Load dataset =====================
df = preprocess_data(df=load_data(r"D:\venkatesh python class\Project\application_train (1).csv"))

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("Financial Health & Affordability")




# =================== side Filters=================
df_filtered=add_global_filters(df)


# -----------------------------
# KPIs
# -----------------------------
avg_income = df["AMT_INCOME_TOTAL"].mean()
med_income = df["AMT_INCOME_TOTAL"].median()
avg_credit = df["AMT_CREDIT"].mean()
avg_annuity = df["AMT_ANNUITY"].mean()
avg_goods = df["AMT_GOODS_PRICE"].mean()
avg_dti = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]).mean()
avg_lti = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).mean()
income_gap = df.loc[df["TARGET"] == 0, "AMT_INCOME_TOTAL"].mean() - df.loc[df["TARGET"] == 1, "AMT_INCOME_TOTAL"].mean()
credit_gap = df.loc[df["TARGET"] == 0, "AMT_CREDIT"].mean() - df.loc[df["TARGET"] == 1, "AMT_CREDIT"].mean()
pct_high_credit = (df["AMT_CREDIT"] > 1_000_000).mean() * 100

# KPI Display
st.subheader("🔑 Financial Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Annual Income", f"{avg_income:,.0f}")
col2.metric("Median Annual Income", f"{med_income:,.0f}")
col3.metric("Avg Credit Amount", f"{avg_credit:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Annuity", f"{avg_annuity:,.0f}")
col5.metric("Avg Goods Price", f"{avg_goods:,.0f}")
col6.metric("Avg DTI", f"{avg_dti:.2f}")

col7, col8, col9 = st.columns(3)
col7.metric("Avg Loan-to-Income (LTI)", f"{avg_lti:.2f}")
col8.metric("Income Gap (Non-def − Def)", f"{income_gap:,.0f}")
col9.metric("Credit Gap (Non-def − Def)", f"{credit_gap:,.0f}")

col10 = st.columns(1)[0]
col10.metric("% High Credit (>1M)", f"{pct_high_credit:.1f}%")


# -----------------------------
# Charts
# -----------------------------
st.subheader("📊 Financial Visuals")

# === Row 1: Income vs Credit Histograms ===
col1, col2 = st.columns(2)

with col1:
    st.write("### Income Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["AMT_INCOME_TOTAL"], bins=50, kde=False, ax=ax)
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.write("### Credit Distribution")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(df["AMT_CREDIT"], bins=50, kde=False, ax=ax)
    ax.set_xlabel("Credit Amount")
    ax.set_ylabel("Count")
    st.pyplot(fig)


col1, col2 = st.columns(2)

with col1:
    st.write("### Annuity Distribution")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(df["AMT_ANNUITY"], bins=50, kde=False, ax=ax)
    ax.set_xlabel("Annuity")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.write("### Income vs Credit")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(x="AMT_INCOME_TOTAL", y="AMT_CREDIT", 
                    data=df.sample(5000, random_state=42), alpha=0.5, ax=ax)
    ax.set_xlabel("Income")
    ax.set_ylabel("Credit")
    st.pyplot(fig)


# === Row 3: Scatter + Boxplot ===
col1, col2 = st.columns(2)

with col1:
    st.write("### Income vs Annuity")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(x="AMT_INCOME_TOTAL", y="AMT_ANNUITY", 
                    data=df.sample(5000, random_state=42), alpha=0.5, ax=ax)
    ax.set_xlabel("Income")
    ax.set_ylabel("Annuity")
    st.pyplot(fig)

with col2:
    st.write("### Credit by Target")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(x="TARGET", y="AMT_CREDIT", data=df, ax=ax)
    ax.set_xlabel("Target (0 = Non-default, 1 = Default)")
    ax.set_ylabel("Credit Amount")
    st.pyplot(fig)


# === Row 4: Boxplot + Heatmap ===
col1, col2 = st.columns(2)

with col1:
    st.write("### Income by Target")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(x="TARGET", y="AMT_INCOME_TOTAL", data=df, ax=ax)
    ax.set_xlabel("Target (0 = Non-default, 1 = Default)")
    ax.set_ylabel("Income")
    st.pyplot(fig)

with col2:
    st.write("### Correlation Heatmap (Financial Variables)")
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_fin = df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DTI", "LOAN_TO_INCOME", "TARGET"]].corr()
    sns.heatmap(corr_fin, cmap="coolwarm", annot=True, fmt=".2f", ax=ax, center=0)
    st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    st.write("### KDE / Density — Joint Income–Credit")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.kdeplot(
    x="AMT_INCOME_TOTAL", 
    y="AMT_CREDIT", 
    data=df.sample(5000, random_state=42), 
    fill=True, 
    cmap="mako", 
    thresh=0.05, 
    levels=50,
    alpha=0.8,
    ax=ax
)
    ax.set_xlabel("Income")
    ax.set_ylabel("Credit")
    st.pyplot(fig)


with col2:
    st.write("### Default Rate by Income Bracket")
    fig, ax = plt.subplots(figsize=(10, 4))
    income_default = df.groupby("INCOME_BRACKET")["TARGET"].mean() * 100
    sns.barplot(x=income_default.index, y=income_default.values, ax=ax)
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Income Bracket")
    st.pyplot(fig)

# -----------------------------
# Narrative Insights
# -----------------------------
st.subheader("📝 Insights")
st.markdown("""
- Default risk increases sharply when **Loan-to-Income (LTI) > 6** or **Debt-to-Income (DTI) > 0.35**, showing repayment stress.  
- **High-credit borrowers (>1M)** are fewer in number but show **lower default rates**, likely due to stronger repayment capacity.  
- A clear **income gap** exists — non-defaulters generally earn more than defaulters, highlighting affordability as a key driver.  
""")
