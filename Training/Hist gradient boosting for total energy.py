import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel(
    'AZ_data_cleaned.xlsx',
    sheet_name='Total energy training_Temp adj',
    engine='openpyxl'
)

df= df[df['out.site_energy.net.energy_consumption.kwh'] <= 65000]

# Split into features and target
X = df.drop(columns=['out.site_energy.net.energy_consumption.kwh'])
y = df['out.site_energy.net.energy_consumption.kwh']

for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = HistGradientBoostingRegressor(
    max_iter=5000,         
    learning_rate=0.1,    
    max_depth=10,         
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", mse)
print("R²:", r2)