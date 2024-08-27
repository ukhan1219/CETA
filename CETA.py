import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import networkx as nx
from datetime import datetime

# Load the graphml file
graph = nx.read_graphml("CETA\\filtered_export.graphml")

# Extract the relevant attributes into a dataframe
attributes = [
    'D_DT_START', 'BHC_IND', 'CHTR_TYPE_CD', 'FHC_IND',
    'INSUR_PRI_CD', 'MBR_FHLBS_IND', 'MBR_FRS_IND', 'SEC_RPTG_STATUS',
    'EST_TYPE_CD', 'BNK_TYPE_ANALYS_CD', 'ENTITY_TYPE', 'ACT_PRIM_CD', 'CITY', 'CNTRY_CD'
]

# Create a list of nodes and their attributes
data = []
for node, attr in graph.nodes(data=True):
    row = {key: attr.get(key, None) for key in attributes}
    row['ID_RSSD'] = attr.get('ID_RSSD', None)  # Use ID_RSSD for company identification
    row['D_DT_END'] = attr.get('D_DT_END', None)
    data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert date columns to datetime
df['D_DT_START'] = pd.to_datetime(df['D_DT_START'], errors='coerce').dt.tz_localize(None)
df['D_DT_END'] = pd.to_datetime(df['D_DT_END'], errors='coerce').dt.tz_localize(None)

# Function to calculate lifespan
def calculate_lifespan(start_date, end_date):
    if pd.isnull(end_date) or end_date.year >= 9999 or end_date > datetime.now():
        end_date = datetime.now()
    return (end_date - start_date).days / 365.25

# Function to check if the company still exists
def company_exists(end_date):
    if pd.isnull(end_date) or end_date.year >= 9999 or end_date > datetime.now():
        return "Yes"
    return "No"

# Calculate lifespan using the new function
df['lifespan'] = df.apply(lambda row: calculate_lifespan(row['D_DT_START'], row['D_DT_END']), axis=1)
df['exists'] = df.apply(lambda row: company_exists(row['D_DT_END']), axis=1)

# Drop rows where lifespan is NaN
df = df.dropna(subset=['lifespan'])

# Convert date columns to numerical values (number of days since a reference date)
reference_date = pd.Timestamp('1900-01-01')
df['D_DT_START'] = (df['D_DT_START'] - reference_date).dt.days
df = df.drop(columns=['D_DT_END'])

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'ID_RSSD':  # Do not encode ID_RSSD
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Split the data into features and target variable
X = df[attributes]
y = df['lifespan']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, df[['ID_RSSD', 'exists']], test_size=0.2, random_state=42
)

# Print out the IDs and existence status of companies in the training and testing sets
print("Training Set:")
print(id_train)
print("Testing Set:")
print(id_test)

# Train multiple Random Forest Regressors with different n_estimators
best_rmse = float('inf')
best_model = None
best_n_estimators = 0

for n_estimators in range(110, 301, 10):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_n_estimators = n_estimators

# Use the best model for final predictions
y_pred = best_model.predict(X_test)

# Calculate RMSE for the best model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Get feature importances
feature_importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': attributes,
    'importance': feature_importances * 100  # Convert to percentage
}).sort_values(by='importance', ascending=False)

# Print the features and their importance
print("Features and their importance in the model:")
print(feature_importance_df)

# Create a DataFrame to display the results
results = pd.DataFrame({
    'ID_RSSD': id_test['ID_RSSD'],
    'predicted_lifespan': y_pred,
    'exists': id_test['exists']
})

# Save the results to a CSV file
results.to_csv("C:\\Users\\xxbla\\OneDrive\\Documents\\VSCode\\CETA\\CETANN\\Usman\\predicted_lifespan_of_companies.csv", index=False)

# Display the results
print(results.head())

# Print RMSE
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Best number of estimators: {best_n_estimators}")
