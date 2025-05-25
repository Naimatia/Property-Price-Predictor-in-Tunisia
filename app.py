import streamlit as st
import pandas as pd
import joblib
import uuid

# Set page configuration
st.set_page_config(
    page_title="Property Price Predictor in Tunisia",
    page_icon="🏠",
    layout="centered"
)

# Add custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        opacity: 0.9;
    }
    .stApp > div {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title and description
st.title("🏠 Property Price Predictor in Tunisia")
st.markdown("Enter the property details below to predict its price in Tunisian Dinar (TND).")

# Load the trained model
try:
    model = joblib.load('property_price_model.pkl')
except FileNotFoundError:
    st.error("Model file 'property_price_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_property_prices_tunisia.csv')
    return df

df = load_data()

# Get unique cities and create city-to-region mapping
cities = sorted(df['city'].unique())
city_to_regions = {city: sorted(df[df['city'] == city]['region'].unique()) for city in cities}

# Create a form for better input organization
with st.form(key="property_form"):
    st.subheader("📝 Property Details")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Category dropdown (removed Terrains et Fermes as it’s not in the dataset)
        category = st.selectbox(
            "Category",
            options=["Appartements", "Villas", "Locaux industriels", "Offices", "Vacation Rentals", "Shared Housing"],
            help="Select the type of property."
        )

        # Type dropdown
        type_property = st.selectbox(
            "Type",
            options=["À Louer", "À Vendre"],
            help="Choose whether the property is for rent or sale."
        )

        # City dropdown
        city = st.selectbox(
            "City",
            options=cities,
            index=cities.index("Tunis") if "Tunis" in cities else 0,
            help="Select the city where the property is located."
        )

        # Region dropdown
        region = st.selectbox(
            "Region",
            options=city_to_regions[city],
            index=city_to_regions[city].index("Autres villes") if "Autres villes" in city_to_regions[city] else 0,
            help="Select the region within the chosen city."
        )

    with col2:
        # Room and bathroom sliders
        room_count = st.slider(
            "Number of Rooms",
            min_value=0.0, max_value=10.0, value=2.0, step=1.0,
            help="Number of rooms in the property (0 for non-residential)."
        )
        bathroom_count = st.slider(
            "Number of Bathrooms",
            min_value=0.0, max_value=5.0, value=1.0, step=1.0,
            help="Number of bathrooms in the property (0 for non-residential)."
        )

        # Size input (adjusted ranges based on dataset)
        size = st.number_input(
            "Size (m²)",
            min_value=20.0, max_value=1000.0, value=100.0, step=5.0,
            help="Enter the property size in square meters."
        )

    # Submit button for the form
    submitted = st.form_submit_button("🔍 Predict Price")

# Process prediction when form is submitted
if submitted:
    # Validate inputs
    if size <= 0:
        st.error("Size must be greater than 0.")
    elif room_count < 0 or bathroom_count < 0:
        st.error("Room count and bathroom count cannot be negative.")
    else:
        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            "category": [category],
            "room_count": [room_count],
            "bathroom_count": [bathroom_count],
            "size": [size],
            "type": [type_property],
            "city": [city],
            "region": [region]
        })

        # Make prediction
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"**Predicted Price**: {predicted_price:,.2f} TND")

            # Provide context based on type
            if type_property == "À Louer":
                st.markdown("This is an **estimated monthly rental price**.")
            else:
                st.markdown("This is an **estimated purchase price**.")

            # Add warnings for unusual predictions
            if type_property == "À Louer" and predicted_price > 10000:
                st.warning("The predicted rental price seems unusually high. Verify input details or model data.")
            elif type_property == "À Vendre" and predicted_price < 10000:
                st.warning("The predicted sale price seems unusually low. Verify input details or model data.")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Instructions and notes
with st.expander("ℹ️ Instructions & Notes"):
    st.markdown("""
    ### How to Use
    1. Select the **property category**, **type** (rent or sale), **city**, and **region**.
    2. Adjust the **number of rooms**, **bathrooms**, and **size** using the sliders or input field.
    3. Click **Predict Price** to get the estimated price in TND.
    4. Note: Predictions are based on a trained model using Tunisian property data. Ensure inputs are realistic for accurate results.
    5. For non-residential properties (e.g., Locaux industriels, Offices), room and bathroom counts can be set to 0 if not applicable.
    """)

# Footer
st.markdown(
    """
    <div style='text-align: center; padding-top: 20px;'>
        <p style='color: #666;'>Powered by Atia Naim 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)