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

# Exploratory Data Analysis (EDA)
# Summary statistics
print(filtered_df.describe())

# Correlation matrix
print(filtered_df.corr(numeric_only=True))

# Visualizations (you can add more based on your needs)
sns.pairplot(filtered_df)
plt.show()

# Clustering Based on 'Lat' and 'Lon'
# Standardize the features for clustering
clustering_features_geo = ['Lat', 'Lon']
scaler_geo = StandardScaler()
scaled_data_geo = scaler_geo.fit_transform(filtered_df[clustering_features_geo])

# Perform KMeans clustering based on 'Lat' and 'Lon'
kmeans_geo = KMeans(n_clusters=3, random_state=42)
filtered_df['GeoCluster'] = kmeans_geo.fit_predict(scaled_data_geo)

# Model Building
# Train a Random Forest model
features = ['TotalArea', 'NrRooms', 'Balcony', 'Floor', 'NumberOfFloors', 'Lon', 'Lat', 'UpdateMonth', 'UpdateYear', 'GeoCluster']
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

# Hyperparameter Tuning
# Grid search for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(X_train, y_train)
y_pred_best = best_rf.predict(X_test)
r2_best = r2_score(y_test, y_pred_best)
print(f'Best Params: {best_params}, Best R2: {r2_best}')