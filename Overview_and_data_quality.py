import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load_data import load_data  
from PreProcess import preprocess_data
from apply_filters import add_global_filters

# ===================== Load dataset =====================
df = preprocess_data(df=load_data(r"D:\venkatesh python class\Project\application_train (1).csv"))

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("🏦 Portfolio Risk Overview")


# ===================== Sidebar Filters =====================
df_filtered = add_global_filters(df)

# ===================== KPIs =====================
total_applicants = df_filtered["SK_ID_CURR"].nunique()
default_rate = df_filtered["TARGET"].mean() * 100
repaid_rate = (1 - df_filtered["TARGET"].mean()) * 100
total_features = df.shape[1]
avg_missing_pct = df_filtered.isnull().mean().mean() * 100
num_features = df.select_dtypes(include=[np.number]).shape[1]
cat_features = df.select_dtypes(include=["object"]).shape[1]
median_age = df_filtered["AGE_YEARS"].median()
median_income = df_filtered["AMT_INCOME_TOTAL"].median()
avg_credit = df_filtered["AMT_CREDIT"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Applicants", f"{total_applicants:,}")
col2.metric("Default Rate", f"{default_rate:.2f}%")
col3.metric("Repaid Rate", f"{repaid_rate:.2f}%")
col4.metric("Total Features", total_features)
col5.metric("Avg Missing per Feature", f"{avg_missing_pct:.2f}%")

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("# Numerical Features", num_features)
col7.metric("# Categorical Features", cat_features)
col8.metric("Median Age", f"{median_age} yrs")
col9.metric("Median Annual Income", f"${median_income:,.0f}")
col10.metric("Avg Credit Amount", f"${avg_credit:,.0f}")

st.divider()

# ===================== Charts =====================
st.subheader("📊 Key Distributions (Filtered Data)")
col1,col2=st.columns(2)

with col1:
    # 1. Target Distribution (Pie)
    fig, ax = plt.subplots()
    df_filtered["TARGET"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", startangle=90, 
    labels=["Repaid (0)", "Default (1)"], colors=["skyblue", "salmon"], ax=ax
)
    ax.set_ylabel("")
    st.pyplot(fig)

with col2:
# 2. Missing Values (Top 20)
   st.subheader("Top 20 Features by Missing %")
   missing = df_filtered.isnull().mean().sort_values(ascending=False).head(20) * 100
   fig, ax = plt.subplots(figsize=(8, 7))
   sns.barplot(x=missing.values, y=missing.index, ax=ax, palette="mako")
   ax.set_xlabel("Missing %")
   st.pyplot(fig)

col1,col2=st.columns(2)

with col1:

# 3. Age Histogram
   st.subheader("Age Distribution")
   fig, ax = plt.subplots()
   sns.histplot(df_filtered["AGE_YEARS"], bins=40, kde=True, ax=ax, color="teal")
   st.pyplot(fig)

with col2:
# 4. Income Histogram
   st.subheader("Income Distribution")
   fig, ax = plt.subplots()
   sns.histplot(df_filtered["AMT_INCOME_TOTAL"], bins=50, ax=ax, color="orange")
   st.pyplot(fig)


col1,col2=st.columns(2)

with col1:
# 5. Credit Histogram
   st.subheader("Credit Amount Distribution")
   fig, ax = plt.subplots()
   sns.histplot(df_filtered["AMT_CREDIT"], bins=50, ax=ax, color="purple")
   st.pyplot(fig)

with col2:
# 6. Gender Countplot
  st.subheader("Gender Distribution")
  fig, ax = plt.subplots()
  sns.countplot(x="CODE_GENDER", data=df_filtered, ax=ax,
              palette={"M": "#4e0cf3", "F": "#e61616","Other":"#40e616"})
  st.pyplot(fig)


col1,col2=st.columns(2)

with col1:
# 7. Family Status Countplot
   st.subheader("Family Status Distribution")
   fig, ax = plt.subplots()
   sns.countplot(y="NAME_FAMILY_STATUS", data=df_filtered,
              order=df_filtered["NAME_FAMILY_STATUS"].value_counts().index,
              palette="Set1", ax=ax)
   st.pyplot(fig)

with col2:
# 8. Education Countplot
  st.subheader("Education Distribution")
  fig, ax = plt.subplots()
  sns.countplot(y="NAME_EDUCATION_TYPE", data=df_filtered,
              order=df_filtered["NAME_EDUCATION_TYPE"].value_counts().index,
              palette="Set2", ax=ax)
  st.pyplot(fig)

col1,col2=st.columns(2)

with col1:

# 9. Family Status Countplot
  st.subheader("Family Status Distribution")
  fig, ax = plt.subplots()
  sns.countplot(y="NAME_FAMILY_STATUS", data=df_filtered, order=df_filtered["NAME_FAMILY_STATUS"].value_counts().index,palette="Set1" ,ax=ax)
  st.pyplot(fig)

with col2:
# 10. Education Countplot
  st.subheader("Education Distribution")
  fig, ax = plt.subplots()
  sns.countplot(y="NAME_EDUCATION_TYPE", data=df_filtered, order=df_filtered["NAME_EDUCATION_TYPE"].value_counts().index,palette="Set2",ax=ax)
  st.pyplot(fig)

st.divider()
# ===================== Narrative =====================
st.subheader("🔎 Insights & Red Flags")
st.markdown(f"""
- **Repaid vs Defaults:** Current filtered data shows ~{repaid_rate:.1f}% repaid vs ~{default_rate:.1f}% defaults.  
- **Income and Credit Distributions:** Both are **right-skewed** with long tails, and boxplots reveal strong outliers.  
- **Age Distribution:** Most loan demand comes from borrowers aged **30–50 years**, while risk is higher among the very young and elderly.  
""")

