import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint

# Load data
df = pd.read_excel(
    'AZ_data_cleaned.xlsx',
    sheet_name='Total energy training_Temp adj',
    engine='openpyxl'
)

df= df[df['out.site_energy.net.energy_consumption.kwh'] <= 65000]

# Split into features and target
X = df.drop(columns=['out.site_energy.net.energy_consumption.kwh'])
y = df['out.site_energy.net.energy_consumption.kwh']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_cols:
    X[col] = X[col].astype(str)

# Preprocessors
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(max_iter=10000, random_state=42))
])

# Hyperparameter space
param_distributions = {
    'regressor__hidden_layer_sizes': [(64,), (32, 16), (64, 32), (128, 64, 32)],
    'regressor__activation': ['relu', 'tanh'],
    'regressor__solver': ['adam', 'lbfgs'],
    'regressor__alpha': uniform(1e-5, 1e-2),
    'regressor__learning_rate': ['constant', 'adaptive'],
}

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=30,  # increase for better tuning
    cv=5,
    verbose=2,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Fit
random_search.fit(X_train, y_train)

# Best model and evaluation
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Hyperparameters:", random_search.best_params_)
print("Test RMSE:", mse)
print("Test R²:", r2)