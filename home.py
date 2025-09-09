import streamlit as st
from utils.load_data import load_data
from PreProcess import preprocess_data
from apply_filters import add_global_filters

st.set_page_config(page_title="Home Credit Default Risk Dashboard", page_icon="📊", layout="wide")

st.title("Home Credit Default Risk Dashboard")

st.markdown("""
Welcome to the **Home Credit Default Risk Dashboard** built with **Streamlit**.  
Use the sidebar to navigate between different analysis modules:
- Overview and Data Quality 
- Target & Risk Segmentation 
- Demographics & Household Profile  
- Financial Health & Affordability
- Correlations, Drivers & Interactive Slice-and-Dice
""")

st.divider()
st.subheader("Upload / Use Default Dataset")

# File uploader
uploaded_file = st.file_uploader("Upload your application_trained CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data(r"D:\venkatesh python class\Project\application_train (1).csv")

# Preprocess + apply filters
df = preprocess_data(df)
df_filtered = add_global_filters(df)

# Show preview
st.dataframe(df_filtered.head(10))
