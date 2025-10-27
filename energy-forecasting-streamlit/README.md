


# 📊 Quarter-Hourly Energy Consumption Forecasting Belgium

## 🎯 Project Overview
This project focuses on forecasting quarter-hourly energy consumption in Belgium using machine learning techniques. The goal is to provide accurate predictions and it is motivated by the need for energy management and policy planning, particularly in the context of increasing reliance on renewable energy sources.

The result is an interactive machine learning application that predicts Belgium's energy consumption in 15-minute intervals. The project combines XGBoost modeling with a Streamlit interface to visualize and compare predictions against ENTSO-E's official day-ahead forecasts.

The XGBoost model has been selected after exploring different supervised ML models on a restricted data set (see notebooks).


## 📁 Project Structure
The project is organized into several directories and files:

- **streamlit_app.py**: Main entry point for the Streamlit application, displaying results
- **src/**: Contains the main source code for the model training and visualisation.
  - **data/**: Contains data loading and preprocessing utilities.
    - **loader.py**: Functions to load and preprocess the energy consumption dataset.
  - **xgboost_model.py**: Implements the XGBoost model training process.
  - **visualization.py**: Functions for visualizing results, including graphs and MAPE errors.
  - **utils.py**: Utility functions for data manipulation and formatting.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis.
  - **belgium_energy_consumption.ipynb**: Notebook documenting the analysis and results of different models for energy consumption forecasting.

- **scripts/**: Contains scripts for data preprocessing.
  - **merge_data.py**: Utilities for merging and cleaning the dataset.

- **data/**: Documentation about the dataset.
  - **README.md**: Information about the dataset structure and contents.

- **models/**: Contains the trained models.

- **requirements.txt**: Lists dependencies required for the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## 🔄 How It Works

### Data Pipeline
- **Input**: 10 years (2015-2025) of quarter-hourly consumption data
- **Processing**: 
  - Temporal feature extraction (hour, day, month, weekday)
  - Cyclic encoding for periodic patterns
  - Missing value handling
  - Data normalization
- **Output**: Cleaned dataset with engineered features

### Model Architecture
- **Algorithm**: XGBoost with hyperparameter optimization
- **Training Split**: 70% train, 15% validation, 15% test
- **Evaluation**: Mean Absolute Percentage Error (MAPE)
- **Benchmark**: ENTSO-E day-ahead forecasts

## 🚀 Getting Started
To run the Streamlit application, follow these steps:

### Web Interface
Built with Streamlit, offering:
- Interactive date range selection
- Real-time model training visualization
- Performance metrics comparison
- Interactive prediction plots

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Windows 10/11
- Git

### Installation
```powershell
# Clone repository
git clone <repository-url>
cd energy-forecasting-streamlit

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run src/streamlit_app.py
```

### Usage
Once the application is running, users can input a start date for historical data to train the XGBoost model. The application will display the MAPE errors and predicted results as graphs for the test set, allowing users to visualize the model's performance.

1. Open browser (default: http://localhost:8501)
2. Select training start date in sidebar
3. Click "Train Model" to:
   - View ENTSO-E benchmark metrics
   - Train & optimize XGBoost model
   - Compare prediction results

## 📊 Performance

### Current Results
| Metric | Training | Validation | Test |
|--------|-----------|------------|------|
| MAPE (ENTSO-E) | 2.9% | 2.3% | 2.7% |
| MAPE (XGBoost) | 2.9% | 2.9% | 3.0% |



## 📌 Key Insights
- The project demonstrates the effectiveness of machine learning in energy consumption forecasting.
- Feature engineering plays a crucial role in improving model accuracy.
- The Streamlit application provides an interactive interface for users to engage with the forecasting model.


## 🔮 Future Work
- Weather data integration
- Holiday & special events handling
- Multi-step forecasting capability
- Automated retraining pipeline
- Model ensemble approaches
