import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from folium import plugins, folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
st.title("House Price Prediction with Clustering and Filtering")

uploaded_data_file = st.file_uploader("Upload JSON Dataset", type="json")
features = ['TotalArea', 'NrRooms', 'Balcony', 'Floor', 'NumberOfFloors', 'Lat', 'Lon', 'UpdateMonth',
            'GeoCluster']
if uploaded_data_file:
    df = pd.read_json(uploaded_data_file)

    # Data Cleaning
    df['UpdatedAt'] = pd.to_datetime(df['UpdatedAt'])

    # Feature Engineering
    df['UpdateMonth'] = df['UpdatedAt'].dt.month
    df['UpdateYear'] = df['UpdatedAt'].dt.year

    # Apply Filtering Conditions
    filtered_df = df[(df['Price'] >= 25000) & (df['Price'] <= 200000) & (df['NrRooms'] > 0)]

    # Standardize the features for clustering
    clustering_features_geo = ['Lat', 'Lon']
    scaler_geo = StandardScaler()
    scaled_data_geo = scaler_geo.fit_transform(filtered_df[clustering_features_geo])

    # Perform KMeans clustering based on 'Lat' and 'Lon'
    kmeans_geo = KMeans(n_clusters=25, random_state=42)
    filtered_df['GeoCluster'] = kmeans_geo.fit_predict(scaled_data_geo)

    # Visualization with Two Maps
    st.header("Map Visualizations")

    # Map 1: Original Data
    st.subheader("Original Data Map")
    m1 = folium.Map(location=[filtered_df['Lat'].mean(), filtered_df['Lon'].mean()], zoom_start=13)
    for idx, row in filtered_df.iterrows():
        folium.CircleMarker(location=[row['Lat'], row['Lon']],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_opacity=0.7).add_to(m1)
    st_data1 = st_folium(m1, width=700, height=500)
  # Map 1: Original Data
    st.subheader("Original Data Map 2")
    map3 = folium.Map(location=[filtered_df['Lat'].mean(), filtered_df['Lon'].mean()], tiles="OpenStreetMap",
                     zoom_start=7)

    heat_data = [[point.xy[0][0], point.xy[1][0]] for point in filtered_df.geometry]
    print(heat_data[0])
    plugins.HeatMap(heat_data, radius=12).add_to(map3)
    map3
    st_data3 = st_folium(map3, width=700, height=500)

    # Map 2: Clustered Data
    st.subheader("Clustered Data Map")
    m2 = folium.Map(location=[filtered_df['Lat'].mean(), filtered_df['Lon'].mean()], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m2)
    for idx, row in filtered_df.iterrows():
        folium.Marker(location=[row['Lat'], row['Lon']],
                      popup=f"Cluster: {row['GeoCluster']}").add_to(marker_cluster)
    st_data2 = st_folium(m2, width=700, height=500)


    # Model Training
    st.header("Train the Model")
    if st.button("Train Model"):
        target = 'Price'

        X_train, X_test, y_train, y_test = train_test_split(filtered_df[features], filtered_df[target], test_size=0.2,
                                                            random_state=42)
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)

        # Evaluation
        y_pred = rf_regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        st.write(f'MSE: {mse}, RMSE: {rmse}, R2: {r2}')

        # Save the model
        import joblib

        joblib.dump(rf_regressor, 'rf_model.pkl')
        st.success("Model trained and saved successfully!")

    # Prediction Section
    st.header("Predict the Price of an Apartment")

    total_area = st.number_input("Total Area (sq. meters)", value=50, min_value=10, max_value=500, step=1)
    nr_rooms = st.number_input("Number of rooms", value=2, min_value=1, max_value=10, step=1)
    balcony = st.number_input("Balcony", value=1, min_value=0, max_value=5, step=1)
    floor = st.number_input("Floor", value=1, min_value=1, max_value=30, step=1)
    number_of_floors = st.number_input("Number of floors in the building", value=5, min_value=1, max_value=50, step=1)
    lat = st.number_input("Latitude", value=filtered_df['Lat'].mean())
    lon = st.number_input("Longitude", value=filtered_df['Lon'].mean())

    # Create a dataframe with user inputs
    input_df = pd.DataFrame([[total_area, nr_rooms, balcony, floor, number_of_floors, lat, lon]],
                            columns=['TotalArea', 'NrRooms', 'Balcony', 'Floor', 'NumberOfFloors', 'Lat', 'Lon'])

    # Predict the GeoCluster for the input data
    scaled_input_geo = scaler_geo.transform(input_df[['Lat', 'Lon']])
    input_df['GeoCluster'] = kmeans_geo.predict(scaled_input_geo)
    input_df['UpdateMonth'] = 5  # Assuming current month
    input_df['UpdateYear'] = 2024  # Assuming current year

    input_df = input_df[features]

    if st.button("Predict"):
        # Load the trained model
        rf_regressor = joblib.load('rf_model.pkl')
        prediction = rf_regressor.predict(input_df)
        st.header(f"Predicted Price: {prediction[0]:,.2f} EUR")

    # Section for testing with custom input data
    st.header("Test the Model with Custom Data")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

        # Process the test data for prediction
        scaled_test_geo = scaler_geo.transform(test_df[['Lat', 'Lon']])
        test_df['GeoCluster'] = kmeans_geo.predict(scaled_test_geo)
        test_df['UpdateMonth'] = 5  # Assuming current month
        test_df['UpdateYear'] = 2024  # Assuming current year

        # Make predictions
        test_predictions = rf_regressor.predict(test_df[features])

        # Display the predictions
        st.write("Predictions:")
        st.write(test_predictions)
