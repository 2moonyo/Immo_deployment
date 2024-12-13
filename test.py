from main import Preprocessing
import pandas as pd
import streamlit as st


data = pd.read_csv("/Users/irisvirus/Desktop/Becode/Python/Projects/Deployment/Immo_deployment/utils/properties_data_cleaned_05_12_14H30.csv")  # Replace with your dataset's file path

st.map(data["locality"])

