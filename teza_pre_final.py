# -*- coding: utf-8 -*-
"""teza_pre-final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DWKLhypZjYHvijt_s51VYNLb1vCvntQe
"""

# House Price Prediction with Clustering and Filtering

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Data Loading
# Load the dataset. Replace the file path with your actual file path.
file_path = 'preprocessed.json'
df = pd.read_json(file_path)
df.head()

# Data Cleaning
# Convert 'UpdatedAt' to datetime format
df['UpdatedAt'] = pd.to_datetime(df['UpdatedAt'])

# Feature Engineering
# Extract month and year from 'UpdatedAt'
df['UpdateMonth'] = df['UpdatedAt'].dt.month
df['UpdateYear'] = df['UpdatedAt'].dt.year

# Apply Filtering Conditions
# Limit the price between 25000 and 200000 and remove properties with 0 rooms
filtered_df = df[(df['Price'] >= 25000) & (df['Price'] <= 200000) & (df['NrRooms'] > 0)]

# Standardize the features for clustering
clustering_features_geo = ['Lat', 'Lon']
scaler_geo = StandardScaler()
scaled_data_geo = scaler_geo.fit_transform(filtered_df[clustering_features_geo])

# Perform KMeans clustering based on 'Lat' and 'Lon'
kmeans_geo = KMeans(n_clusters=25, random_state=42)
filtered_df['GeoCluster'] = kmeans_geo.fit_predict(scaled_data_geo)

# Extended Exploratory Data Analysis (EDA)
# Summary statistics
print(filtered_df.describe())

# Correlation matrix
corr_matrix = filtered_df.corr(numeric_only=True)
print(corr_matrix)

# Heatmap for correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplots for categorical features
for feature in ['NrRooms', 'Balcony', 'Floor', 'NumberOfFloors', 'UpdateMonth', 'UpdateYear', 'GeoCluster']:
    sns.boxplot(x=feature, y='Price', data=filtered_df)
    plt.title(f'Price vs {feature}')
    plt.show()

# Visualizing the Clusters on a Scatter Plot
plt.scatter(filtered_df['Lat'], filtered_df['Lon'], c=filtered_df['GeoCluster'], cmap='rainbow')
plt.title('Clusters Based on Lat and Lon')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.colorbar().set_label('Cluster ID')
plt.show()

from folium import plugins, folium
from branca.utilities import split_six
import geopandas as geopandas

geometry = geopandas.points_from_xy(df.Lat, df.Lon)
geo_df = geopandas.GeoDataFrame(
    df[["Title", "Price", "Lon", "Lat"]], geometry=geometry
)

geo_df.head()
map = folium.Map(location=[filtered_df['Lat'].mean(), filtered_df['Lon'].mean()], tiles="OpenStreetMap", zoom_start=7)

heat_data = [[point.xy[0][0], point.xy[1][0]] for point in geo_df.geometry]
print(heat_data[0])
plugins.HeatMap(heat_data, radius=12).add_to(map)
map

import folium

# Create a colormap
cmap = plt.get_cmap('tab20', 20)

# Create a map object
m = folium.Map(location=[filtered_df['Lat'].mean(), filtered_df['Lon'].mean()], zoom_start=13)

# Add points to the map
for idx, row in filtered_df.iterrows():
    color = plt.cm.tab20(row['GeoCluster'] % 20)  # Cycle through 20 colors
    color = [int(x*255) for x in color[:3]]  # Convert to 0-255 RGB scale
    color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'  # Convert to hexadecimal

    folium.CircleMarker(location=[row['Lat'], row['Lon']],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=0.7).add_to(m)


# Show the map
m

features = ['TotalArea', 'NrRooms', 'Balcony', 'Floor', 'NumberOfFloors','Lat', 'Lon',  'UpdateMonth',  'GeoCluster','Price']

filtered_df[features].to_csv('sample.csv', index=False)

# Model Building
# Train a Random Forest model
target = 'Price'
X_train, X_test, y_train, y_test = train_test_split(filtered_df[features], filtered_df[target], test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Evaluation
# Evaluate the model
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, RMSE: {rmse}, R2: {r2}')

# # Hyperparameter Tuning
# # Grid search for hyperparameter tuning
# param_grid = {
#     'n_estimators': [50, 100],
#     'max_features': ['auto', 'sqrt'],
#     'max_depth': [5, None],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }
# grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# best_rf = RandomForestRegressor(**best_params, random_state=42)
# best_rf.fit(X_train, y_train)
# y_pred_best = best_rf.predict(X_test)
# r2_best = r2_score(y_test, y_pred_best)
# print(f'Best Params: {best_params}, Best R2: {r2_best}')