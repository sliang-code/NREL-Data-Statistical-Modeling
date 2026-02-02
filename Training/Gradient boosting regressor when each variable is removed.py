import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  

# Load the data
df = pd.read_excel(
    'AZ_data_cleaned.xlsx',
    sheet_name='Electricity_ccooling_training',
    engine='openpyxl'
)

target_column = 'out.electricity.cooling.energy_consumption.kwh'
full_features = df.drop(columns=[target_column])
y = df[target_column]

categorical_cols = full_features.select_dtypes(include=['object']).columns.tolist()
numerical_cols = full_features.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Convert all categorical columns to string
for col in categorical_cols:
    full_features[col] = full_features[col].astype(str)

# Preprocessors
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# MSE results
mse_results = {}

# Loop through each column
for col_to_remove in tqdm(full_features.columns, desc="Computing MSEs"):
    X = full_features.drop(columns=[col_to_remove])
    
    # Recompute column 
    current_categorical = X.select_dtypes(include=['object']).columns.tolist()
    current_numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, current_numerical),
        ('cat', categorical_transformer, current_categorical)
    ])
    
    # Define model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_results[col_to_remove] = mse

# Print results 
sorted_results = sorted(mse_results.items(), key=lambda x: x[1])
for feature, mse in sorted_results:
    print(f"{feature}: MSE = {mse:,.2f}")