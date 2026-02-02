import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from scipy.stats import randint as sp_randint, uniform as sp_uniform

# Load data
df = pd.read_excel(
    'AZ_data_cleaned.xlsx',
    sheet_name='Total energy training_Temp adj',
    engine='openpyxl'
)
target_col = 'out.site_energy.net.energy_consumption.kwh'

Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

print(upper_bound)

df = df[df[target_col] <= upper_bound]

# Split data
X = df.drop(columns=['out.site_energy.net.energy_consumption.kwh'])
y = df['out.site_energy.net.energy_consumption.kwh']

# Column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X[categorical_cols] = X[categorical_cols].astype(str)

# Transformers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

lgbm = LGBMRegressor(random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgbm)
])

# Hyperparameter
param_distributions = {
    'regressor__n_estimators': sp_randint(100, 500),
    'regressor__learning_rate': sp_uniform(0.01, 0.2),
    'regressor__max_depth': sp_randint(3, 10),
    'regressor__subsample': sp_uniform(0.6, 0.4),
    'regressor__colsample_bytree': sp_uniform(0.6, 0.4)
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Search
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions, 
    n_iter=30,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Best Params:", search.best_params_)
print("Tuned MSE:", mse)
print("Tuned R²:", r2)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Total Energy Consumption', fontsize = 16)
plt.ylabel('Predicted Total Energy Consumption', fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
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

# Add numerical features directly
for f in numerical_cols:
    agg_importances[f] = importances[list(all_features).index(f)]

# Sum categorical one-hot importances
for cat, dummy_vars in cat_feature_mapping.items():
    indices = [list(all_features).index(f) for f in dummy_vars]
    agg_importances[cat] = importances[indices].sum()

# Convert to DataFrame
agg_importance_df = pd.DataFrame({
    'Feature': list(agg_importances.keys()),
    'Importance': list(agg_importances.values())
}).sort_values(by='Importance', ascending=False)

agg_importance_df['Importance'] = agg_importance_df['Importance'] / agg_importance_df['Importance'].sum()

print(agg_importance_df.head(20))


plt.figure(figsize=(12, 8))
plt.barh(agg_importance_df['Feature'].head(20)[::-1],
         agg_importance_df['Importance'].head(20)[::-1])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (Aggregated by Variable)')
plt.grid(True)
plt.tight_layout()
plt.show()