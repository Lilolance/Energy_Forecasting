import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("entsoe_10_years_combined.csv", header=0, names=['interval', 'forecast', 'actual'], na_values='-')

print(df.head())
# Convert the start time to datetime
df['start_time'] = pd.to_datetime(df['interval'].str.split(' - ').str[0], format="%d.%m.%Y %H:%M")
df['end_time'] = pd.to_datetime(df['interval'].str.split(' - ').str[1], format="%d.%m.%Y %H:%M")

# plot the actual vs forecasted energy consumption for the first week
plt.figure(figsize=(12, 6))
plt.plot(df['start_time'], df['actual'], label='Actual', color='blue')
plt.plot(df['end_time'], df['forecast'], label='Forecast', color='orange')
plt.xlabel('Time')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption Actual vs Forecast')
plt.legend()
plt.show()

# engineer features: date, time index, weekday x hour, remove NA
# Remove rows where either forecast or actual is NaN
print("Number of missing rows actual:", df[['actual']].isna().sum())
df = df.dropna(subset=['actual'])

# Convert time to day, month, year, hour, minute
df['day'] = df['start_time'].dt.day #maybe redundant with dayofweek
df['dayofweek'] = df['start_time'].dt.dayofweek
df['month'] = df['start_time'].dt.month
df['year'] = df['start_time'].dt.year
df['hour'] = df['start_time'].dt.hour
df['quarter'] = df['start_time'].dt.minute // 15 

# time index numbering (with integers) each quarter hour from start to end
df['time_index'] = np.arange(len(df))

# add combine features: weekday and hour
df['weekday_hour'] = df['dayofweek'] * 24 + df['hour']

# custom MAPE function
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# compute MAPE on train set between actual and predicted by ENSOE-E
mape_ensoe = mape(df['actual'].values, df['forecast'].values)
print("MAPE by ENSOE-E:", mape_ensoe)

# initialize input and output
X = df[['day', 'dayofweek', 'month', 'year', 'hour', 'quarter', 'time_index', 'dayofweek']].values
y= df[['actual']].values.flatten() # flatten => 1D matrix 

#  Divide into test, cross-validation, and training sets in chronological order 
num_samples = X.shape[0]
train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size
X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:] 

import xgboost as xgb

# XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                              n_estimators=100,
                                learning_rate=0.1,
                                max_depth=6,
                                subsample=0.8,
                                eval_metric='rmse',
                                colsample_bytree=0.8,
                                random_state=42)
history = xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

# plot training and validation error
results = xgb_model.evals_result()
plt.figure(figsize=(12, 6))
plt.plot(results['validation_0']['rmse'], label='Train RMSE')
plt.plot(results['validation_1']['rmse'], label='Validation RMSE')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.title('XGBoost Training and Validation RMSE')
plt.legend()
plt.show()

# Optimize max_depth, n_estimators and learning rate using grid search with cross-validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=TimeSeriesSplit(n_splits=3), verbose=1)
grid_search.fit(X, y)

print("Best parameters found: ", grid_search.best_params_)

#get best model from grid search
best_xgb_model = grid_search.best_estimator_

# evaluate best model on train set
y_train_pred = best_xgb_model.predict(X_train)
print("MAPE on train set:", mape(y_train, y_train_pred))

# Evaluate best model on val set
y_val_pred = best_xgb_model.predict(X_val)
print("MAPE on val set:", mape(y_val, y_val_pred))

# Evaluate best model on test set
y_pred = best_xgb_model.predict(X_test)
print("MAPE on test set:", mape(y_test, y_pred))

# plot predicted vs actual for complete data set vs time
y_complete_pred = best_xgb_model.predict(X)
plt.figure(figsize=(12, 6))
plt.plot(df['start_time'], df['actual'], label='Actual')
plt.plot(df['start_time'], y_complete_pred, label='Predicted', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Energy Consumption')
plt.title('Predicted vs Actual Energy Consumption')
plt.legend()
plt.show()
