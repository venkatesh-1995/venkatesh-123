import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load_data import load_data
from PreProcess import preprocess_data
from apply_filters import add_global_filters
# ===================== Load Data =====================
df = preprocess_data(df=load_data(r"D:\venkatesh python class\Project\application_train (1).csv"))

st.set_page_config(page_title="Applicant Profile Dashboard", layout="wide")
st.title("👨‍👩‍👧 Applicant Profile — Household & Human Factors")

# ===================== Sidebar Filters =====================
df_filtered=add_global_filters(df)



# ===================== KPIs =====================
male_pct = (df_filtered[df_filtered["CODE_GENDER"] == "M"].shape[0] / len(df_filtered)) * 100
female_pct = (df_filtered[df_filtered["CODE_GENDER"] == "F"].shape[0] / len(df_filtered)) * 100
avg_age_def = df_filtered.loc[df_filtered["TARGET"] == 1, "AGE_YEARS"].mean()
avg_age_nondef = df_filtered.loc[df_filtered["TARGET"] == 0, "AGE_YEARS"].mean()
with_children_pct = (df_filtered[df_filtered["CNT_CHILDREN"] > 0].shape[0] / len(df_filtered)) * 100
avg_family_size = df_filtered["CNT_FAM_MEMBERS"].mean()
married_pct = (df_filtered[df_filtered["NAME_FAMILY_STATUS"].str.contains("Married")].shape[0] / len(df_filtered)) * 100
pct_single = (df_filtered[df_filtered["NAME_FAMILY_STATUS"].str.contains("Single|Separated|Widow|Widower|Divorced")].shape[0] / len(df_filtered)) * 100
higher_edu = (df_filtered[df_filtered["NAME_EDUCATION_TYPE"].isin(["Higher education", "Academic degree", "Incomplete higher"])].shape[0] / len(df_filtered)) * 100
with_parents_pct = (df_filtered[df_filtered["NAME_HOUSING_TYPE"] == "With parents"].shape[0] / len(df_filtered)) * 100
working_pct = (df_filtered[df_filtered["OCCUPATION_TYPE"].notna()].shape[0] / len(df_filtered)) * 100
avg_emp_years = df_filtered["EMPLOYMENT_YEARS"].mean()

# Display KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("% Male", f"{male_pct:.1f}%")
col2.metric("% Female", f"{female_pct:.1f}%")
col3.metric("Avg Age (Defaulters)", f"{avg_age_def:.1f}")
col4.metric("Avg Age (Non-Defaulters)", f"{avg_age_nondef:.1f}")
col5.metric("% With Children", f"{with_children_pct:.1f}%")

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("Avg Family Size", f"{avg_family_size:.2f}")
col7.metric("% Married", f"{married_pct:.1f}%")
col8.metric("% Higher Education", f"{higher_edu:.1f}%")
col9.metric("% With Parents", f"{with_parents_pct:.1f}%")
col10.metric("% Working", f"{working_pct:.1f}%")

st.metric("Avg Employment Years", f"{avg_emp_years:.1f}")

st.divider()

# ===================== Graphs =====================
st.subheader("📊 Human Factors & Household Structure")
# -----------------------------
# Demographic Visuals
# -----------------------------
st.subheader("👥 Demographic Profiles")

# === Row 1: Age Distribution ===
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df_filtered["AGE_YEARS"], bins=40, kde=True, color="teal", ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(
        df_filtered,
        x="AGE_YEARS",
        hue="TARGET",
        bins=40,
        kde=True,
        ax=ax,
        palette={0: "skyblue", 1: "salmon"}
    )
    ax.set_title("Age Distribution by Target")
    st.pyplot(fig)


# === Row 2: Gender & Family Status ===
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x="CODE_GENDER", data=df_filtered, ax=ax, palette="Set2")
    ax.set_title("Gender Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(
        y="NAME_FAMILY_STATUS",
        data=df_filtered,
        order=df_filtered["NAME_FAMILY_STATUS"].value_counts().index,
        ax=ax,
        palette="Set1"
    )
    ax.set_title("Family Status Distribution")
    st.pyplot(fig)


# === Row 3: Education & Occupation ===
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(
        y="NAME_EDUCATION_TYPE",
        data=df_filtered,
        order=df_filtered["NAME_EDUCATION_TYPE"].value_counts().index,
        ax=ax,
        palette="Set2"
    )
    ax.set_title("Education Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    top_occ = df_filtered["OCCUPATION_TYPE"].value_counts().head(10)
    sns.barplot(x=top_occ.values, y=top_occ.index, ax=ax, palette="muted")
    ax.set_title("Top 10 Occupations")
    st.pyplot(fig)


# === Row 4: Housing & Children ===
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df_filtered["NAME_HOUSING_TYPE"].value_counts().plot.pie(
        autopct="%1.1f%%",
        startangle=45,
        ax=ax,
        colors=sns.color_palette("pastel")
    )
    ax.set_ylabel("")
    ax.set_title("Housing Type Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(x="CNT_CHILDREN", data=df_filtered, ax=ax, color="skyblue")
    ax.set_title("Children Count Distribution")
    st.pyplot(fig)


# === Row 5: Age vs Target & Correlation ===
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.boxplot(
        x="TARGET",
        y="AGE_YEARS",
        data=df_filtered,
        palette={'0': "skyblue", '1': "salmon"},
        ax=ax
    )
    ax.set_title("Age vs Target")
    ax.set_xticklabels(["Repaid", "Default"])
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    corr = df_filtered[["AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "TARGET"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation: Age, Children, Family Size, Target")
    st.pyplot(fig)

st.divider()

# ===================== Narrative =====================
st.subheader("🔎 Insights & Red Flags")
st.markdown(f"""
- **Life-stage patterns:** Younger applicants (<30) and older (>60) show higher default risk compared to middle-aged groups.  
- **Family structure:** {with_children_pct:.1f}% of applicants have children; larger families show slightly higher default risk.  
- **Education:** {higher_edu:.1f}% of applicants have higher education, typically showing lower risk.  
- **Occupation:** Default rates vary by job type — unstable/low-income jobs are riskier.  
- **Housing type:** {with_parents_pct:.1f}% Applicants living with parents or in rented housing may reflect higher financial vulnerability.  

""")

# ===================== Download =====================
st.download_button(
    "📥 Download Filtered Data (CSV)",
    df_filtered.to_csv(index=False).encode("utf-8"),
    "filtered_applicants.csv",
    "text/csv",
)
