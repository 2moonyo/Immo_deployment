# Belgian Real Estate Price Predictor

A machine learning web application that predicts real estate prices in Belgium using CatBoost regression model. The application features an interactive map interface and provides detailed property analysis based on location and property characteristics.
**link**: https://2moonyo-immo-deployment-deploy-main-ao2emb.streamlit.app/

## Overview

This project consists of two main components:
1. A model training pipeline (`main_model_train.py`)
2. A Streamlit web application for deployment (`deploy_main.py`)

## Features

- Interactive property price prediction
- Location-based analysis with map visualization
- Detailed property characteristics input
- Real-time price estimates
- Region/Province/Locality-based searching
- Integration with Belgian income and geospatial data

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: CatBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Folium, Matplotlib, Seaborn
- **Geospatial Analysis**: GeoPandas
- **Model Explanation**: SHAP

## Installation

- pip install -r requirements.txt
 
 ## Usage

 - **Training the model**: python main_model_train.py

 - **Web Application**: streamlit run deploy_main.py

## Model Features

- The model takes into account various features including:

- **Location (Province, Region, Locality)**
- **Property Characteristics (Bedrooms, Living Area, etc.)**
- **Energy Certificate**
- **Building State**
- **Geographical Coordinates**
- **Local Income Data**
- **Property Type (House/Apartment)**
- **Additional Features (Garden, Pool, Terrace)**

## Model Training

- The training pipeline includes:

- **Data preprocessing and cleaning**
- **Feature engineering**
- **Model training using CatBoost**
- **Model evaluation with various metrics (RMSE, MAE, R²)**
- **SHAP value analysis for feature importance**

## Deployment Features

- The Streamlit application provides:

- **Interactive map interface**
- **Dynamic property price prediction**
- **Location-based filtering**
- **Detailed property input form**
- **Real-time updates**
- **Visual feedback**

## Data Sources

- The Streamlit application provides:

- **Belgian real estate data**
- **Regional income statistics**
- **Geospatial mapping data**
- **Postal code and municipality information**
