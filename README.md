📊 Quarter-Hourly Energy Consumption Forecasting Belgium
🧠 Project Overview
As someone interested in sustainability and thinking about how to reduce our impact as a society, energy production and consumption has been a hot topic both scientifically as politically. A recurring theme is the impact of wind and solar energy on the stability of the electricity grid (https://www.creg.be/sites/default/files/assets/Publications/Studies/F2866NL.pdf) and the compatibility of renewable energy sources (e.g. wind and solar) and nuclear energy. In order to, on the one hand, plan out a policy and, on the other hand, keep a stable day-to-day grid, it is necessary to have a correct prediction of our energy consumption (in this case Belgium).

The notebook "belgium_energy_consumption" explores predictive modeling for quarter-hourly time series data spanning from January 1, 2025 to October 20, 2025. The goal is to forecast future values using a variety of machine learning techniques, each leveraging different feature engineering strategies and model architectures.

The py "belgium_energy_consumption_XGBoost" explores the random forest model with XGboost applied to a larger data set than the notebook, spanning ???.

📁 Dataset
Frequency: 15-minute intervals (96 data points per day)

Time Range: January 1, 2025 – October 20, 2025

Features: Time-based features (month, day, weekday, hour, quarter) and derived combinations (time since start, hour x weekday)

🔍 Modeling Approaches
We experimented with three distinct models and evaluate using MAPE (Mean Absolute Percentage Error):

0. Forecasting benchmark by ENTSO-E

Data split into training set (70%), val set (15%) and test set (15%) with respectiv ENTSO-E MAPE 2.9%, 2.3% and 2.7%.

1. Linear Regression with Engineered Features
Applied sin/cos transformations to capture periodicity (e.g., month, hour of day, day of week)
Included higher-order polynomials to model nonlinear trends
Created combined features to capture interactions between time components such as time since start (to capture seasonal trends)
Optimized reg parameter
epochs=20, lambda=1, Adams optimizer

MAPError on (training, val, test): 6.8%, 6.7% and 7.1%.

Pros: Fast, interpretable, surprisingly effective with good feature engineering
Cons: Far off from benchmark

2. Random Forest with XGBoost
Used ensemble tree-based methods to capture complex nonlinear relationships
XGBoost provided boosted performance and better handling of feature importance
Hyperparameter optimization via GridSearch: learning rate=0.1, max depth=5, estimators=50 

MAPError on (training, val, test): 2.4%, 6.1% and 9.3%.
Pros: Very fast, robust to overfitting, handles missing data well, strong baseline
Cons: Test error far off from benchmark, high variance

3. Small Neural Network
Implemented a feedforward neural network with a few hidden layers
Trained on normalized features
epochs=100, nodes=(128,64,1), 20% dropout, adams optimizer

MAPError on (training, val, test): 2%, 7.5% and 13.5%
Pros: Flexible, capable of modeling subtle patterns, good generalization with tuning
Cons: Test error far off from benchmark, high variance

📈 Evaluation
Each model was evaluated using:

Train/Val/Test split
Mean Absolute Percentage Error (MAPE)
Visual comparison of predicted vs actual values

📌 Key Insights
Feature engineering significantly boosts linear model performance
Tree-based models are effective and efficient but require more data
Neural networks are effective, less efficient and require more data

🚀 Steps taken
Explored random forest and XGBoost (with hyperparamater optimalization) for a larger data set in "belgium_energy_consumption_XGBoost.py", spanning from January 1, 2015 until January 1, 2025.
    📈 Results
    MAPE  on (Train, Val, Test): 2.9%, 2.9% and 3.0%!

    📌 Insights
    Very fast and accurate, showing low bias and low variance. Although the MAPE is comparable with the benchmark MAPE, our predictions seem to still consistently underestimate the actual energy consumption (see "Actual_vs_PredictedwithXGBoost_10years")

🚀 Next Steps
Incorporate external features (e.g., weather, holidays)
Explore LSTM or Transformer-based models for sequential dependencies

## Files
- `belgium_energy_consumption.ipynb`: Main exploration notebook
- `belgium_energy_consumption_XGBoost.py`: Extended XGBoost implementation
- `merge_data.py`: Data preprocessing utilities