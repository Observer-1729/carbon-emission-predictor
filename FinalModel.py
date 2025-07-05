import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb

# Load data
data = pd.read_csv("test_dataset.csv", parse_dates=['timestamp'])
data['co2_lag1'] = data['co2_emission'].shift(1)
data['co2_lag2'] = data['co2_emission'].shift(2)
data['fuel_usage_lag1'] = data['fuel_usage'].shift(1)
data['co2_rolling_mean3'] = data['co2_emission'].rolling(window=3).mean()
data = data.dropna()

# Features and target
X = data[['fuel_usage', 'kiln_temperature', 'air_flow', 'raw_feed_rate', 'production_rate', 'co2_lag1', 'co2_lag2', 'fuel_usage_lag1', 'co2_rolling_mean3']]
y = data['co2_emission']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

sc_X = StandardScaler()
X_train = sc_X. fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}

model = xgb.XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred_best = best_model.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)
print(f"Improved Test RMSE: {rmse_best:.2f}")
print(f"Model R² Score (Accuracy): {r2_best:.2f}")

# Visualization
plt.figure(figsize=(15,5))
plt.plot(data['timestamp'].iloc[-len(y_test):], y_test.values, label='Actual CO₂ Emissions')
plt.plot(data['timestamp'].iloc[-len(y_test):], y_pred_best, label='Predicted CO₂ Emissions (XGBoost)')
plt.xlabel('Time')
plt.ylabel('CO₂ Emissions (tons/hour)')
plt.title('Actual vs Predicted CO₂ Emissions (XGBoost Model)')
plt.legend()
plt.show()
