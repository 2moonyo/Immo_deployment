import geopandas as gpd
import folium
from streamlit_folium import folium_static
from catboost import CatBoostRegressor, Pool
import streamlit as st
import pandas as pd
import numpy as np
import os

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'catboost_model_6.0.cbm')

model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("/Users/irisvirus/Desktop/Becode/Python/Projects/Deployment/Immo_deployment/utils/exports/properties_data_cleaned_20241218_011836.csv")  
    data = data.dropna(subset=['latitude', 'longitude'])
    data = data.dropna(subset=['latitude', 'longitude'])
    return data

# Function to add CSS for custom styling
def add_custom_styles():
    st.markdown("""
        <style>
        /* Custom Font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        * {
            font-family: 'Roboto', sans-serif;
        }

        /* Header Styling */
        .stApp {
            background-color: #f7f8fa;
        }
        .main {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #002f6c;
            color: white;
        }
        .sidebar h1, .sidebar h2, .sidebar h3 {
            color: #ffd700; /* Gold color for headings */
        }
        .sidebar .widget-label {
            color: white;
        }

        /* Button Styling */
        button {
            background-color: #002f6c !important;
            color: white !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 10px 20px !important;
        }
        button:hover {
            background-color: #00509e !important;
        }

        /* Map Container Styling */
        .folium-map {
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

# Function to display Immoweb logo
def display_logo():
    st.sidebar.image(
        "https://www.immoweb.be/assets/images/logo-immoweb.svg", 
        use_column_width=True
    )


def plot_properties_on_map(filtered_data, zoom_level=10, center=None):
    # Create the base map centered on Belgium
    if center is None:
        center = [50.8503, 4.3517]  # Default to Belgium's center (Brussels)
    
    m = folium.Map(location=center, zoom_start=zoom_level)

    # Plot each property on the map
    for _, row in filtered_data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Price: €{row['price']:,.2f}\nLocality: {row['locality']}",
        ).add_to(m)

    return m

#Preprocessing
# Create comprehensive location mappings
def round_coordinates(lat, lon, precision=6):
        return (round(lat, precision), round(lon, precision))
    
  

def create_location_mappings(data):
    # Function to round coordinates to reduce precision and group similar locations
    # Mapping for all location-related information
    location_map = {}
    
    # Group by each location type
    location_groups = {
        'province': data.groupby('province'),
        'locality': data.groupby('locality'),
        'region': data.groupby('region'),
        'postal_code': data.groupby('postal_code'),
        #'gps_coordinates': data.groupby('gps_coordinates')
    }
    
    # Create comprehensive mappings
    for location_type, group in location_groups.items():
        location_map[f'{location_type}_to_localities'] = group['locality'].unique().to_dict()
        location_map[f'{location_type}_to_provinces'] = group['province'].unique().to_dict()
        location_map[f'{location_type}_to_regions'] = group['region'].unique().to_dict()
        location_map[f'{location_type}_to_postcodes'] = group['postal_code'].unique().to_dict()
        location_map[f'{location_type}_to_median_income'] = group['median_income'].mean().to_dict()
        if location_type != 'gps_coordinates':
            location_map[f'{location_type}_to_gps'] = group['gps_coordinates'].unique().to_dict()
            

    return location_map


def get_coordinates(data, location_type, location_name):
    """
    Retrieve the latitude and longitude of a given locality or region.

    """
    if location_type not in ['locality', 'region']:
        st.error("Invalid location type. Please select 'locality' or 'region'.")
        return None

    filtered_data = data[data[location_type] == location_name]
    
    if filtered_data.empty:
        st.warning(f"No matching {location_type} found for '{location_name}'.")
        return None

    # Get the first latitude and longitude
    latitude = filtered_data.iloc[0]['latitude']
    longitude = filtered_data.iloc[0]['longitude']
    return latitude, longitude

# Mapping of default localities and coordinates for regions
REGION_LOCALITY_MAPPING = {
    "Flanders": {"locality": "Antwerp", "latitude": 51.2194, "longitude": 4.4025},
    "Brussels": {"locality": "Brussels", "latitude": 50.8503, "longitude": 4.3517},
    "Wallonia": {"locality": "Charleroi", "latitude": 50.4114, "longitude": 4.4448},
}

# Capital cities for each province
PROVINCE_CAPITAL_MAPPING = {
    "Antwerp": {"locality": "Antwerp", "latitude": 51.2194, "longitude": 4.4025},
    "East Flanders": {"locality": "Ghent", "latitude": 51.0543, "longitude": 3.7174},
    "Flemish Brabant": {"locality": "Leuven", "latitude": 50.8798, "longitude": 4.7005},
    "Hainaut": {"locality": "Mons", "latitude": 50.4542, "longitude": 3.9523},
    "Liège": {"locality": "Liège", "latitude": 50.6337, "longitude": 5.5675},
    "Limburg": {"locality": "Hasselt", "latitude": 50.9307, "longitude": 5.3320},
    "Luxembourg": {"locality": "Arlon", "latitude": 49.6833, "longitude": 5.8167},
    "Namur": {"locality": "Namur", "latitude": 50.4674, "longitude": 4.8718},
    "Walloon Brabant": {"locality": "Wavre", "latitude": 50.7172, "longitude": 4.6114},
    "West Flanders": {"locality": "Bruges", "latitude": 51.2093, "longitude": 3.2247},
}

# Function to calculate mean/most frequent values for selected location type
def calculate_aggregates(data, group_by, selected_value):
    """Calculate mean or most frequent values for the given group."""
    group_data = data[data[group_by] == selected_value]
    if group_data.empty:
        return {
            "bedrooms": 3,  # Default value if no data available
            "living_area": 100,
            "surface_of_the_plot": 500,
            "terrace_surface": 20,
            "facades": 2,
            "garden_size": 100,
            "pool": False,
        }
    
    # Calculate means or fallback to defaults
    return {
        "bedrooms": round(group_data["bedrooms"].mean()) if "bedrooms" in group_data else 3,
        "living_area": round(group_data["living_area"].mean()) if "living_area" in group_data else 100,
        "surface_of_the_plot": round(group_data["surface_of_the_plot"].mean()) if "surface_of_the_plot" in group_data else 500,
        "terrace_surface": round(group_data["terrace_surface"].mean()) if "terrace_surface" in group_data else 20,
        "facades": round(group_data["facades"].mean()) if "facades" in group_data else 2,
        "garden_size": round(group_data["garden_size"].mean()) if "garden_size" in group_data else 100,
        "pool": group_data["pool"].mode()[0] if "pool" in group_data and not group_data["pool"].isnull().all() else False,
    }
def main():
    # Load data
    data = load_data()
  # Add a combined GPS coordinate column
    data['gps_coordinates'] = data.apply(
    lambda row: round_coordinates(row['latitude'], row['longitude']), 
    axis=1
    )
    print(data['gps_coordinates'][0],data['gps_coordinates'][1])
    # Create location mappings
    location_maps = create_location_mappings(data)
    
    # Streamlit app
    st.title("Interactive Property Price Prediction")
    
    # Sidebar for inputs
    st.sidebar.header("Location Selection")
    
    # Location selection dropdown
    location_type = st.sidebar.selectbox(
        "Select Location Type", 
        ["Locality"]
    )
    
    
    if location_type == "Locality":
        available_options = sorted(data['locality'].unique())
        selected_location = st.sidebar.selectbox(
            "Select Locality", 
            [""] + available_options
        )
        
        # If locality is selected, show related information
        if selected_location:
            coordinates = get_coordinates(data, location_type.lower(), selected_location)
            if coordinates:
                st.sidebar.write(f"Coordinates for {selected_location}:")
                st.sidebar.write(f"Latitude: {coordinates[0]}")
                st.sidebar.write(f"Longitude: {coordinates[1]}")
             # Filter data based on the selection
            filtered_data = data[data[location_type.lower()] == selected_location]
            # Plot the filtered properties on the map
            
            map_belgium = plot_properties_on_map(filtered_data)
            folium_static(map_belgium)
            
            # Province for this locality
            province = location_maps['locality_to_provinces'].get(selected_location, ["N/A"])[0]
            st.sidebar.write(f"Province: {province}")
            
            # Possible postal codes in this locality
            postcodes = location_maps['locality_to_postcodes'].get(selected_location, [])
            st.sidebar.write("Postal Codes:")
            st.sidebar.write(", ".join(map(str, sorted(postcodes))))
            
            # Median income for this locality
            median_income = location_maps['locality_to_median_income'].get(selected_location, "N/A")
            st.sidebar.write(f"Median Disposable Income per year: €{median_income:,.2f}")
    
    
    # Additional property details
    st.sidebar.header("Property Details")
    bedrooms = st.sidebar.number_input("Number of Bedrooms",min_value=1,max_value=10,value=3)
    livingArea = st.sidebar.number_input("Living Area (m²)",min_value=10,max_value=1000,value=150)
    energy_certifications = sorted(data['energy_certificate'].dropna().unique())
    energy_certificate = st.sidebar.selectbox("Energy Certification", [""] + energy_certifications)
    surface_of_the_plot = st.sidebar.number_input("Surface of the Plot (m²)", min_value=2, max_value=10000, value=200)
    building_state_options = (data["buildingState"].unique())
    building_state = st.sidebar.selectbox("Building State", [""] + building_state_options)
    region_selection = sorted(data["region"].unique())
    region = st.sidebar.selectbox("Region", [""]+region_selection)
    property_type_binary = sorted(data["property_type"].dropna().unique())
    property_type = st.sidebar.selectbox("Property Type", [""] + property_type_binary)
    terraceSurface = st.sidebar.number_input("Terrace Surface (m²)",min_value=0, max_value=500, value=0)
    facades = st.sidebar.number_input("Number of Facades", min_value=0, max_value=10, value=0)
    has_garden = st.sidebar.checkbox("Has Garden", value=False)
    garden_size = st.sidebar.number_input("Garden Size (m²)", min_value=0, max_value=10000, value=0, disabled=not has_garden)
    has_pool = st.sidebar.checkbox("Has Pool", value=False)

    # Prediction Button
    if st.sidebar.button("Predict Price"):
        try:
            # Initialize variables
            province = None
            locality = None
            median_income = None
            
            if location_type == "Province" and selected_location:
                province = selected_location
                locality = location_maps['province_to_localities'][0].get(province, [None])[0]
                median_income = location_maps['province_to_median_income'].get(province, np.nan)
                region = data["Province"] == data["region"]

            elif location_type == "Locality" and selected_location:
                locality = selected_location
                province = location_maps['locality_to_provinces'].get(locality, [None])[0]
                postal_code = location_maps['locality_to_postcodes'].get(locality, [np.nan])[0]
                median_income = location_maps['locality_to_median_income'].get(locality, np.nan)

            elif location_type == "Region" and selected_location:
                province = location_maps['region_to_provinces'].get(selected_location, [None])[0]
                locality = location_maps['region_to_localities'].get(selected_location, [None])[0]
                median_income = location_maps['region_to_median_income'].get(selected_location, np.nan)

            # Prediction function call
            predicted_price = predict_house_price(
                data,
                locality=locality,
                bedrooms=bedrooms,
                livingArea=livingArea,
                energy_certificate=energy_certificate,
                surface_of_the_plot=surface_of_the_plot,
                buildingState= building_state,
                region = region,
                facades=facades,
                has_garden=int(has_garden),
                terraceSurface = terraceSurface,
                property_type = property_type,
                garden_size=garden_size if has_garden else 0,
                has_pool=int(has_pool),
                province=province,
                median_income=median_income,
                coordinates=coordinates
                
                
            )

            # Display results
            st.subheader("Predicted House Price")
            st.write(f"€{predicted_price:,.2f}")

            st.write("Location Details:")
            st.write(f"Locality: {locality}")
            st.write(f"Province: {province}")
            st.write(f"Median Income: €{median_income:,.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# Prediction function (simplified version)
def predict_house_price(data,bedrooms,locality, livingArea,facades, energy_certificate, garden_size,surface_of_the_plot,coordinates,province=None, pool=0,median_income=np.nan, region=None, property_type="APARTMENT", buildingState="GOOD", terraceSurface=0, gps_coordinates=np.nan, **kwargs):
    # Debug print to understand input data
    print("Input Data:")
    print(f"Bedroom: {bedrooms}")
    print(f"Property_type: {property_type}")
    print(f"Locality: {locality}")
    print(f"Facades: {facades}")    
    print(f"Terrace Surface: {terraceSurface}")
    print(f"Building State: {buildingState}")
    print(f"Garden Surface: {garden_size}")
    print(f"Pool: {pool}")
    print(f"Living Area: {livingArea}")
    print(f"Surface of the plot: {surface_of_the_plot}")
    print(f"Energy Certificate: {energy_certificate}")
    print(f"Province: {province}")
    print(f"Median_income: {median_income}")
    print(f"Longitude: {coordinates[0]}")
    print(f"Latitude: {coordinates[1]}")
    print(f"Region: {region}")
    

    # Ensure categorical features have valid values
    def safe_category(value, default='Unknown'):
        return value if pd.notna(value) and value != '' else default

    # Prepare input data with safe categorical values
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'property_type': [safe_category(property_type)],
        'locality': [safe_category(locality)],
        'facades': [facades],
        'terraceSurface': [terraceSurface],
        'buildingState': [safe_category(buildingState)],
        'gardenSurface': [garden_size],
        'pool': [pool],
        'livingArea': [livingArea],
        'surfaceOfThePlot' : [surface_of_the_plot],
        'energy_certificate': [safe_category(energy_certificate)],
        'province': [safe_category(province)],
        'median_income': [median_income if pd.notna(median_income) else data['median_income'].median()],
        'latitude': [coordinates[0] if isinstance(coordinates[0], tuple) else np.nan],
        'longitude': [coordinates[1] if isinstance(coordinates[1], tuple) else np.nan],
        'region': [safe_category(region)],
        
        })

    # Categorical features used during training
    cat_features = ["locality", "energy_certificate", "region", "property_type", "province", "buildingState"]
    
    try:
        # Ensure all categorical features exist in the training data
        for feature in cat_features:
            if feature not in input_data.columns:
                raise ValueError(f"Missing categorical feature: {feature}")

        # Create a Pool object
        prediction_pool = Pool(
            data=input_data, 
            cat_features=cat_features
        )

        # Make prediction
        prediction = model.predict(prediction_pool)
        return prediction[0]

    except Exception as e:
        print(f"Prediction Error Details: {e}")
        raise
# Run the app
if __name__ == "__main__":
    main()