import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def load_data(file_path="D:\venkatesh python class\Project\application_train (1).csv"):
    df = pd.read_csv(file_path)
    return df

