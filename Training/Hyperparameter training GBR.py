import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint, zscore
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel(
    'AZ_data_cleaned.xlsx',
    sheet_name='EC_Heating_included',
    engine='openpyxl'
)

target_col = 'out.electricity.cooling.energy_consumption.kwh'

# quarter percentile remove method

Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

print(upper_bound)

df = df[df[target_col] <= upper_bound]


# z-score remove method

'''
z_scores = zscore(df[target_col])
threshold = 3  
df = df[np.abs(z_scores) <= threshold]
'''

# Split 
X = df.drop(columns=['out.electricity.cooling.energy_consumption.kwh'])
y = df['out.electricity.cooling.energy_consumption.kwh']


# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Convert all categoricals to string
for col in categorical_cols:
    X[col] = X[col].astype(str)

# Define transformers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Define pipeline with default GBR
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define hyperparameter search space
param_distributions = {
    'regressor__n_estimators': randint(100, 500),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__max_depth': randint(3, 10),
    'regressor__subsample': uniform(0.6, 0.4),
    'regressor__min_samples_split': randint(2, 10),
    'regressor__min_samples_leaf': randint(1, 10),
    'regressor__max_features': ['auto', 'sqrt', 'log2', None]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Best model and evaluation
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Hyperparameters:", random_search.best_params_)
print("Test MSE:", mse)
print("Test R²:", r2)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Cooling Energy Consumption',fontsize=16)
plt.ylabel('Predicted Cooling Energy Consumption', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()


preprocessor = best_model.named_steps['preprocessor']
regressor = best_model.named_steps['regressor']

# Get feature names after preprocessing
num_features = numerical_cols

cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)

all_features = np.concatenate([num_features, cat_features])

importances = regressor.feature_importances_

cat_feature_mapping = {
    cat: [f for f in cat_features if f.startswith(cat + "_")]
    for cat in categorical_cols
}

agg_importances = {}

for f in numerical_cols:
    agg_importances[f] = importances[list(all_features).index(f)]

for cat, dummy_vars in cat_feature_mapping.items():
    indices = [list(all_features).index(f) for f in dummy_vars]
    agg_importances[cat] = importances[indices].sum()

# Convert to DataFrame
agg_importance_df = pd.DataFrame({
    'Feature': list(agg_importances.keys()),
    'Importance': list(agg_importances.values())
}).sort_values(by='Importance', ascending=False)

agg_importance_df['Importance'] = agg_importance_df['Importance'] / agg_importance_df['Importance'].sum()


print("Top 20 Variables:")
print(agg_importance_df.head(20))


plt.figure(figsize=(12, 8))
plt.barh(agg_importance_df['Feature'].head(20)[::-1],
         agg_importance_df['Importance'].head(20)[::-1])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (Aggregated by Variable)')
plt.grid(True)
plt.tight_layout()
plt.show()
