import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import json
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Mansehra Flood Risk Dashboard", layout="wide")

FEATURES = ["dist_water", "elevation", "landcover", "rainfall", "slope"]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Risk Map", "Model Performance", "Feature Importance", "Predict Location"])

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_test_data():
    return pd.read_csv("test_data.csv")

@st.cache_data
def load_map_data():
    return pd.read_csv("map_data.csv")

@st.cache_data
def load_boundary():
    with open("mansehra_boundary.geojson") as f:
        return json.load(f)

model = load_model()
test_data = load_test_data()
map_data = load_map_data()
boundary = load_boundary()

if page == "Overview":
    st.title("Mansehra District Flood Risk Prediction Dashboard")
    st.markdown("""
    ### Machine Learning-Based Flood Risk Prediction for Mansehra District

    This dashboard shows flood risk assessment for Mansehra District, Khyber Pakhtunkhwa,
    built using satellite data and a Random Forest classification model.

    **Data Sources:**
    - Elevation & Slope: SRTM DEM
    - Rainfall: CHIRPS Daily Precipitation
    - Land Cover: ESA WorldCover
    - Water Bodies: JRC Global Surface Water
    - Historical Flood Data
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Test Samples", len(test_data))
    col2.metric("Model Accuracy", f"{accuracy_score(test_data['actual'], test_data['predicted'])*100:.1f}%")
    col3.metric("Flood-Risk Points (Actual)", int(test_data['actual'].sum()))

elif page == "Risk Map":
    st.header("Interactive Flood Risk Map - Mansehra District")
    st.markdown("Click anywhere on the map to see flood risk for the nearest known data point. Red = High Risk, Green = Low Risk.")

    center_lat, center_lon = 34.33, 73.24
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")

    folium.GeoJson(
        boundary,
        style_function=lambda x: {"fillColor": "transparent", "color": "blue", "weight": 3}
    ).add_to(m)

    for _, row in map_data.iterrows():
        color = "red" if row["flood_label"] == 1 else "green"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Elevation: {row['elevation']}m Rainfall: {row['rainfall']}mm Risk: {'HIGH' if row['flood_label']==1 else 'LOW'}"
        ).add_to(m)

    map_output = st_folium(m, width=1000, height=550)

    if map_output and map_output.get("last_clicked"):
        click_lat = map_output["last_clicked"]["lat"]
        click_lon = map_output["last_clicked"]["lng"]

        map_data["dist"] = np.sqrt((map_data["lat"] - click_lat)**2 + (map_data["lon"] - click_lon)**2)
        nearest = map_data.loc[map_data["dist"].idxmin()]

        input_df = pd.DataFrame([[nearest["dist_water"], nearest["elevation"], nearest["landcover"], nearest["rainfall"], nearest["slope"]]], columns=FEATURES)
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.subheader("Nearest Data Point to Your Click")
        col1, col2, col3 = st.columns(3)
        col1.metric("Elevation", f"{nearest['elevation']:.0f} m")
        col2.metric("Rainfall", f"{nearest['rainfall']:.2f} mm")
        col3.metric("Flood Risk Probability", f"{prob*100:.1f}%")

        if pred == 1:
            st.error("HIGH FLOOD RISK at this location")
        else:
            st.success("LOW FLOOD RISK at this location")

elif page == "Model Performance":
    st.header("Model Performance")
    acc = accuracy_score(test_data["actual"], test_data["predicted"])
    prec = precision_score(test_data["actual"], test_data["predicted"])
    rec = recall_score(test_data["actual"], test_data["predicted"])
    f1 = f1_score(test_data["actual"], test_data["predicted"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc*100:.1f}%")
    col2.metric("Precision", f"{prec*100:.1f}%")
    col3.metric("Recall", f"{rec*100:.1f}%")
    col4.metric("F1 Score", f"{f1*100:.1f}%")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(test_data["actual"], test_data["predicted"])
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Saved Confusion Matrices (All Models)")
    st.image("confusion_matrices.png")

elif page == "Feature Importance":
    st.header("Feature Importance (SHAP)")
    st.image("shap_summary.png", caption="SHAP Summary Plot")
    st.markdown("This shows which features (elevation, slope, rainfall, land cover, distance to water) most influence the model's flood risk predictions.")

elif page == "Predict Location":
    st.header("Predict Flood Risk for a Location")
    st.markdown("Enter values manually to get a flood risk prediction:")

    col1, col2 = st.columns(2)
    with col1:
        dist_water = st.number_input("Distance to Water (m)", min_value=0.0, value=500.0)
        elevation = st.number_input("Elevation (m)", min_value=0.0, value=1000.0)
        landcover = st.selectbox("Land Cover Class", [10, 20, 30, 40, 50, 60, 80], index=2)
    with col2:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=3.0)
        slope = st.number_input("Slope (degrees)", min_value=0.0, value=10.0)

    if st.button("Predict Flood Risk"):
        input_df = pd.DataFrame([[dist_water, elevation, landcover, rainfall, slope]], columns=FEATURES)
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        st.metric("Flood Risk Probability", f"{prob*100:.1f}%")
        if pred == 1:
            st.error("HIGH FLOOD RISK")
        else:
            st.success("LOW FLOOD RISK")
