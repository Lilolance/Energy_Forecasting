# Models Documentation for Energy Consumption Forecasting Project

## Overview
This directory contains the implementation of the modeling approaches used in the quarter-hourly energy consumption forecasting project for Belgium. The primary focus is on the XGBoost model, which has shown promising results in predicting energy consumption based on historical data.

## Modeling Approaches

### XGBoost Model
- **File**: `xgboost_model.py`
- **Description**: This file implements the training process for the XGBoost model. It includes functions to preprocess the input data, train the model on historical energy consumption data, and save the trained model for future predictions.

### Prediction Functions
- **File**: `predict.py`
- **Description**: This file contains functions to load the trained XGBoost model and make predictions on new data. It takes the input features and returns the predicted energy consumption values along with performance metrics such as Mean Absolute Percentage Error (MAPE).

## Performance Metrics
The performance of the models is evaluated using the Mean Absolute Percentage Error (MAPE) metric. The results from the XGBoost model have been compared against a benchmark provided by ENTSO-E, demonstrating its effectiveness in forecasting energy consumption.

## Future Work
Further enhancements may include:
- Incorporating additional features such as weather data and holidays to improve model accuracy.
- Exploring advanced modeling techniques like LSTM or Transformer-based models to capture sequential dependencies in the data.

## Usage
To utilize the models, ensure that the necessary dependencies are installed as specified in the `requirements.txt` file. The models can be trained and evaluated through the Streamlit application, which provides an interactive interface for users to input historical data and visualize the results.