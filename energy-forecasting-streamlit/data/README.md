# Dataset Documentation for Energy Consumption Forecasting Project

## Overview
This dataset contains quarter-hourly energy consumption data for Belgium. It is used to train machine learning models to forecast future energy consumption based on historical patterns.

## Dataset Structure
The dataset is structured with the following key components:

- **Frequency**: The data is recorded at 15-minute intervals.
- **Predicted Consumption**: This column contains the forecasted energy consumption values.
- **Actual Consumption**: This column contains the actual recorded energy consumption values.

## Time Range
- The dataset spans from January 1, 2015, to January 1, 2025, providing a comprehensive historical context for training and testing the forecasting models.

## Derived Features
To enhance the predictive capabilities of the models, several derived features have been created from the raw data:

- **Periodic Features**:
  - Year
  - Month
  - Day
  - Weekday
  - Hour
  - Quarter
  - Interaction terms (e.g., hour x weekday)

- **Continuous Features**:
  - Time since the start of the dataset

## Usage
This dataset is utilized in various scripts and notebooks within the project to train models, evaluate performance, and visualize results. It is essential for understanding energy consumption trends and improving forecasting accuracy.

## Acknowledgments
The dataset is sourced from ENTSO-E and is crucial for research and development in energy forecasting and sustainability efforts in Belgium.