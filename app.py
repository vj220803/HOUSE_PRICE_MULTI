import streamlit as st
import joblib
import numpy as np


# Load models and preprocessing files
model = joblib.load(r"C:\\Users\\comp\\OneDrive\\Desktop\\HOUSI\\model_files\\lasso_model.pkl")
scaler = joblib.load(r"C:\\Users\\comp\\OneDrive\\Desktop\\HOUSI\\model_files\\scaler.pkl")
encoders = joblib.load(r"C:\\Users\\comp\\OneDrive\\Desktop\\HOUSI\\model_files\\encoders.pkl")
poly = joblib.load(r"C:\\Users\\comp\\OneDrive\\Desktop\\HOUSI\\model_files\\poly_transformer.pkl")
selected_features = joblib.load(r"C:\\Users\\comp\\OneDrive\\Desktop\\HOUSI\\model_files\\selected_features.pkl") 
# Page configuration
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }

        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        h1, h2, h3 {
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }

        .sidebar .sidebar-content {
            background-color: #e8f0fe;
            border-radius: 8px;
            padding: 20px;
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }

        .css-1aumxhk {
            display: none;
        }

        .price-card {
            background-color: #e0f7fa;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            font-size: 22px;
            color: #00796b;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color:#2c3e50;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("💡 *Powered by Polynomial Regression and ML Magic!*")

# Tabs
tab1, tab2 = st.tabs(["🏠 Predict Price", "ℹ️ About App"])

with tab1:
    with st.sidebar:
        st.header("Input Property Details")
        state = st.text_input("State", "Maharashtra")
        city = st.selectbox("City", ["Mumbai", "Pune", "Ahmedabad", "Delhi"])
        locality = st.selectbox("Locality", ["Locality_1", "Locality_2", "Locality_3"])
        property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Independent House"])
        bhk = st.slider("BHK", 1, 5, 2)
        size = st.number_input("Size in SqFt", min_value=100, max_value=10000, value=1000, step=100)
        price_per_sqft = st.slider("Price per SqFt", 1000, 10000, 5000)


        year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2010)
        furnished_status = st.selectbox("Furnished Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
        floor_no = st.slider("Floor Number", 0, 30, 2)
        total_floors = st.slider("Total Floors", 1, 50, 10)
        age_of_property = st.slider("Age of Property (years)", 0, 100, 10)
        nearby_schools = st.selectbox("Nearby Schools", ["Yes", "No"])
        nearby_hospitals = st.selectbox("Nearby Hospitals", ["Yes", "No"])
        public_transport = st.selectbox("Public Transport Access", ["Yes", "No"])
        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])
        amenities = st.selectbox("Amenities", ["Basic", "Premium", "Luxury"])
        facing = st.selectbox("Facing", ["East", "West", "North", "South"])
        owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
        availability_status = st.selectbox("Availability Status", ["Ready to Move", "Under Construction"])


    # Create feature dictionary
    input_data = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_per_SqFt": price_per_sqft,


        'Year_Built': year_built,
        'Furnished_Status': furnished_status,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age_of_property,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Public_Transport_Accessibility': public_transport,
        'Parking_Space': parking,
        'Security': security,
        'Amenities': amenities,
        'Facing': facing,
        'Owner_Type': owner_type,
        'Availability_Status': availability_status


    }

    # Fill missing features with default
    full_input = {feature: 0 for feature in selected_features}
    full_input.update(input_data)

    # Convert to array
    X_input = np.array([[full_input[feature] for feature in selected_features]])

    # Apply encoders
    for i, feature in enumerate(selected_features):
        if feature in encoders:
            encoder = encoders[feature]
            try:
                X_input[0, i] = encoder.transform([X_input[0, i]])[0]
            except:
                X_input[0, i] = 0
    
    
    # Scaling
    X_scaled = scaler.transform(X_input)

    # Polynomial transformation
    X_poly = poly.transform(X_scaled)

    # Prediction
    prediction = model.predict(X_poly)[0]

    # Display result
    st.markdown("### 🎯 Predicted House Price")
    st.markdown(f"<div class='price-card'>₹ {round(prediction, 2)} Lakhs</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("### 📘 About This App")
    st.markdown("""
    This House Price Predictor uses a trained **Lasso Regression** model with **Polynomial Features**  
    to estimate property prices based on multiple attributes.

    #### ⚙️ Features Used:
    - Location: State, City, Locality
    - Property Details: BHK, SqFt, Price per SqFt, Type
    - Other: Amenities, Age, Floors, Facing, etc.

    #### 🛠️ Built With:
    - Python, Streamlit
    - scikit-learn
    - Joblib (for model serialization)

    > 📌 Note: This is a demo app — predictions depend on model training data.
    """)

