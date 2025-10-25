import numpy as np
import xgboost as xgb
import streamlit as st
import joblib
import streamlit as st
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

def train_xgboost_model(features, target, n_estimators=100, learning_rate=0.1, max_depth=6):
    model = xgb.XGBRegressor(objective='reg:squarederror', # sets loss function
                              n_estimators=n_estimators, 
                              learning_rate=learning_rate,
                              max_depth=max_depth,
                              subsample=0.8, # each tree uses 80% of data
                              eval_metric='rmse', 
                              colsample_bytree=0.8, # each tree uses 80% of features
                                random_state=42) # reproducibility
    # Train the model
    model.fit(features, target)    
    save_model(model)
    return model

def optimize_hyperparameters(model, train_features, train_target, val_features, val_target):
    # Optimize max_depth, n_estimators and learning rate using grid search with cross-validation
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # concatenate train and val for grid search
    grid_features = np.concatenate([train_features, val_features])
    grid_target = np.concatenate([train_target, val_target])


    # grid search with time series split
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=TimeSeriesSplit(n_splits=3), verbose=1)
    grid_search.fit(grid_features, grid_target)

    #get best model from grid search
    best_model = grid_search.best_estimator_
    save_model(model, 'xgboost_optimized_model.joblib')
    
    return best_model

def save_model(model, model_path='xgboost_model.joblib'):
    joblib.dump(model, model_path)

def load_model(model_path='xgboost_model.joblib'):
    return joblib.load(model_path)

###### What happens below ?

# def run_xgboost(start_date):
#     df = load_data(start_date)
#     features, target = preprocess_data(df)
#     model = train_xgboost_model(features, target)
#     save_model(model)
    
#     predictions = predict(model, features)
    
#     # Calculate MAPE
#     mape = np.mean(np.abs((target - predictions) / target)) * 100
    
#     return mape, predictions

# Example usage:
# start_date = '2025-01-01'
# mape, predictions = run_xgboost(start_date)
# print(f'MAPE: {mape}')
# print(predictions)