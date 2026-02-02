import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # for progress bar

df = pd.read_excel(
    'AZ_data_cleaned.xlsx',
    sheet_name='EC_Heating_included',
    engine='openpyxl'
) 

target = 'out.electricity.cooling.energy_consumption.kwh'
X_full = df.drop(columns=[target])
y = df[target]

categorical_cols = X_full.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_full.select_dtypes(include=['int64', 'float64']).columns.tolist()


for col in categorical_cols:
    X_full[col] = X_full[col].astype(str)


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


results = []

for col_to_remove in tqdm(X_full.columns, desc="Evaluating MSE without each feature"):
    X = X_full.drop(columns=[col_to_remove])
    
    
    cat_cols = [col for col in categorical_cols if col != col_to_remove]
    num_cols = [col for col in numerical_cols if col != col_to_remove]
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=5000,
            random_state=42
        ))
    ])
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    results.append((col_to_remove, mse))


results.sort(key=lambda x: x[1])
for col, mse in results:
    print(f"{col}: MSE = {mse:,.2f}")