import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load_data import load_data
from PreProcess import preprocess_data
from apply_filters import add_global_filters
# -----------------------------
# Load & Preprocess Data
# -----------------------------
df_raw = load_data(r"D:\venkatesh python class\Project\application_train (1).csv")
df = preprocess_data(df_raw)

st.title("📊 Page 2 — Target & Risk Segmentation")

df_filtered = add_global_filters(df)


# -----------------------------
# KPIs
# -----------------------------
total_defaults = int(df_filtered["TARGET"].sum())
default_rate = df_filtered["TARGET"].mean() * 100

# Group-wise default rates
def_rate_gender = df_filtered.groupby("CODE_GENDER")["TARGET"].mean() * 100
def_rate_edu = df_filtered.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100
def_rate_family = df_filtered.groupby("NAME_FAMILY_STATUS")["TARGET"].mean() * 100
def_rate_housing = df_filtered.groupby("NAME_HOUSING_TYPE")["TARGET"].mean() * 100

# Averages among defaulters
df_def = df[df["TARGET"] == 1]
avg_income_def = df_def["AMT_INCOME_TOTAL"].mean()
avg_credit_def = df_def["AMT_CREDIT"].mean()
avg_annuity_def = df_def["AMT_ANNUITY"].mean()
avg_emp_def = df_def["EMPLOYMENT_YEARS"].mean()

# Show KPIs
st.subheader("🔑 Key Risk Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Defaults", f"{total_defaults:,}")
col2.metric("Default Rate (%)", f"{default_rate:.2f}%")
col3.metric("Avg Income (Defaulters)", f"{avg_income_def:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Credit (Defaulters)", f"{avg_credit_def:,.0f}")
col5.metric("Avg Annuity (Defaulters)", f"{avg_annuity_def:,.0f}")
col6.metric("Avg Employment Years (Defaulters)", f"{avg_emp_def:.1f}")

col7, col8, col9 = st.columns(3)
col7.metric("Default Rate by Gender (%)", f"{def_rate_gender.mean():.2f}%")
col8.metric("Default Rate by Education (%)", f"{def_rate_edu.mean():.2f}%")
col9.metric("Default Rate by Family Status (%)", f"{def_rate_family.mean():.2f}%")

st.metric("Default Rate by Housing Type (%)", f"{def_rate_housing.mean():.2f}%")

# -----------------------------
# Charts using Matplotlib and Seaborn
# -----------------------------
st.subheader("📈 Risk Segment Visuals")
# -----------------------------
# Visualization Section
# -----------------------------
st.subheader("📊 Demographic & Financial Risk Profiles")

# === Row 1: Default Counts & Default % by Gender ===
col1, col2 = st.columns(2)

with col1:
    st.write("### Default vs Repaid Counts")
    fig, ax = plt.subplots()
    df_filtered["TARGET"].value_counts().rename({0: "Repaid", 1: "Default"}).plot(
        kind="bar", ax=ax, color=["#4CAF50", "#F44336"]
    )
    ax.set_ylabel("Count")
    ax.set_xlabel("Status")
    st.pyplot(fig)

with col2:
    st.write("### Default Rate by Gender (%)")
    fig, ax = plt.subplots()
    sns.barplot(
        x=def_rate_gender.index,
        y=def_rate_gender.values,
        ax=ax,
        palette=["#4C91AF", "#F44336", "#D10C7F"]
    )
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Gender")
    st.pyplot(fig)


# === Row 2: Education & Family Status ===
col1, col2 = st.columns(2)

with col1:
    st.write("### Default Rate by Education (%)")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=def_rate_edu.index, y=def_rate_edu.values, ax=ax, palette="Set2")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Education Type")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

with col2:
    st.write("### Default Rate by Family Status (%)")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=def_rate_family.index, y=def_rate_family.values, ax=ax, palette="Set2")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Family Status")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


# === Row 3: Housing & Income by Target ===
col1, col2 = st.columns(2)

with col1:
    st.write("### Default Rate by Housing Type (%)")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=def_rate_housing.index, y=def_rate_housing.values, ax=ax, palette="Set2")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Housing Type")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

with col2:
    st.write("### Income by Target (Boxplot)")
    fig, ax = plt.subplots()
    sns.boxplot(x="TARGET", y="AMT_INCOME_TOTAL", data=df, ax=ax, palette="Set3")
    ax.set_xticklabels(["Repaid", "Default"])
    ax.set_ylabel("Income")
    st.pyplot(fig)


# === Row 4: Credit & Age Distribution ===
col1, col2 = st.columns(2)

with col1:
    st.write("### Credit by Target (Boxplot)")
    fig, ax = plt.subplots()
    sns.boxplot(x="TARGET", y="AMT_CREDIT", data=df, ax=ax, palette="Set3")
    ax.set_xticklabels(["Repaid", "Default"])
    ax.set_ylabel("Credit Amount")
    st.pyplot(fig)

with col2:
    st.write("### Age vs Target (Violin Plot)")
    fig, ax = plt.subplots()
    sns.violinplot(
        x="TARGET", y="AGE_YEARS", data=df, ax=ax, inner="quartile", palette="Set2"
    )
    ax.set_xticklabels(["Repaid", "Default"])
    ax.set_ylabel("Age (Years)")
    st.pyplot(fig)


# === Row 5: Employment Years & Contract Type ===
col1, col2 = st.columns(2)

with col1:
    st.write("### Employment Years by Target")
    fig, ax = plt.subplots()
    sns.histplot(
        data=df,
        x="EMPLOYMENT_YEARS",
        hue="TARGET",
        bins=30,
        multiple="stack",
        ax=ax,
        palette="Set1"
    )
    ax.set_ylabel("Count")
    ax.set_xlabel("Employment Years")
    st.pyplot(fig)

with col2:
    st.write("### Contract Type vs Target")
    contract_dist = df.groupby(["NAME_CONTRACT_TYPE", "TARGET"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    contract_dist.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#F44336"])
    ax.set_ylabel("Count")
    ax.set_xlabel("Contract Type")
    st.pyplot(fig)

# -----------------------------
# Narrative Insights
# -----------------------------
st.subheader("📝 Insights")
st.markdown("""
- **Education**: Lower education levels (Secondary / Basic) tend to show **higher default rates** compared to higher education.  
- **Housing Type**: Applicants living in **Rented / with parents** show relatively higher risk vs those owning a house.  
- **Family Status**: Married applicants have lower default probability compared to single/divorced.  
- A possible hypothesis: **low income + high loan-to-income ratio** segments are driving defaults.  
""")
