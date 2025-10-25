# 📊 Quarter-Hourly Energy Consumption Forecasting Belgium

## 🧠 Project Overview
This project focuses on forecasting quarter-hourly energy consumption in Belgium using machine learning techniques. The goal is to provide accurate predictions to aid in energy management and policy planning, particularly in the context of increasing reliance on renewable energy sources.

## 📁 Project Structure
The project is organized into several directories and files:

- **src/**: Contains the main source code for the Streamlit application and model training.
  - **streamlit_app.py**: Main entry point for the Streamlit application, displaying results and allowing user input for model training.
  - **data/**: Contains data loading and preprocessing utilities.
    - **loader.py**: Functions to load and preprocess the energy consumption dataset.
  - **features/**: Contains feature engineering functions.
    - **build_features.py**: Functions to create derived features from raw data.
  - **models/**: Contains model training and prediction scripts.
    - **xgboost_model.py**: Implements the XGBoost model training process.
    - **predict.py**: Functions to make predictions using the trained model.
  - **visualization.py**: Functions for visualizing results, including graphs and MAPE errors.
  - **utils.py**: Utility functions for data manipulation and formatting.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis.
  - **belgium_energy_consumption.ipynb**: Notebook documenting the analysis and results of energy consumption forecasting.

- **scripts/**: Contains scripts for data preprocessing.
  - **merge_data.py**: Utilities for merging and cleaning the dataset.

- **data/**: Documentation about the dataset.
  - **README.md**: Information about the dataset structure and contents.

- **models/**: Documentation about modeling approaches.
  - **README.md**: Descriptions of models and their performance.

- **tests/**: Contains unit tests for the application.
  - **test_streamlit_app.py**: Tests ensuring the Streamlit app functions correctly.

- **requirements.txt**: Lists dependencies required for the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## 🚀 Getting Started
To run the Streamlit application, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd energy-forecasting-streamlit
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/streamlit_app.py
   ```

## 📈 Usage
Once the application is running, users can input a start date for historical data to train the XGBoost model. The application will display the MAPE errors and predicted results as graphs for the test set, allowing users to visualize the model's performance.

## 📌 Key Insights
- The project demonstrates the effectiveness of machine learning in energy consumption forecasting.
- Feature engineering plays a crucial role in improving model accuracy.
- The Streamlit application provides an interactive interface for users to engage with the forecasting model.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.