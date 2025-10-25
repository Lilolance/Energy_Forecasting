import streamlit as st
import numpy as np
from src.data.loader import load_data, prepare_data_for_model, extract_features
from src.utils import get_train_val_test_split, mape
from src.models.xgboost_model import train_xgboost_model, optimize_hyperparameters
from src.visualization import plot_results
from datetime import date

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Energy Forecasting Belgium", layout="wide")
st.title("Quarter-Hourly Energy Consumption Forecasting Belgium")

# -------------------- SIDEBAR --------------------
st.sidebar.header("Model Controls")

# Date input
min_date = date(2015, 1, 1)
max_date = date(2025, 10, 20)
start_date = st.sidebar.date_input("Select a start date:", value=min_date, min_value=min_date, max_value=max_date)

# Confirm button
if st.sidebar.button("Confirm Start Date"):
    st.session_state.start_date = start_date
    st.success(f"Start date confirmed: {start_date.strftime('%Y-%m-%d')}")

# -------------------- CACHING --------------------
@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_prepared_data(data, start_date):
    return prepare_data_for_model(data, start_date)

@st.cache_data
def get_features_and_target(data):
    features = extract_features(data)
    target = data['actual'].values
    return features, target

@st.cache_resource
def get_trained_model(train_features, train_target):
    return train_xgboost_model(train_features, train_target)

@st.cache_resource
def get_optimized_model(_model, train_features, train_target, val_features, val_target):
    return optimize_hyperparameters(model, train_features, train_target, val_features, val_target)

# -------------------- MAIN LOGIC --------------------

if st.sidebar.button("Reset Date"):
    st.session_state.pop("start_date", None)

if "start_date" in st.session_state:

    # Fetch data and clip data
    data = get_data()
    data = get_prepared_data(data, start_date)
    if data.empty:
        st.error("No data loaded. Please check the data source.")
        st.stop()

    # Display benchmark
    st.subheader(f"Benchmark: ENTSO-E 1-Day Forecast spanning {min_date} to {max_date}")
    mape_ensoe = mape(data['actual'].values, data['forecast'].values)
    st.write(f"MAPE by ENTSO-E: {mape_ensoe:.2f}%")
    st.markdown("#### *ENTSO-E Forecast vs Actual Consumption*")
    plot_results(data['start_time'].values, data['actual'].values, data['forecast'].values)
    

    # -------------------- TRAINING --------------------

    if st.sidebar.button("Train Model"):

        st.subheader(f"Training model on historical data from {start_date} until {max_date}")

        try:
            # Split
            features, target = get_features_and_target(data)
            train_features, val_features, test_features = get_train_val_test_split(features) # 70-15-15 split
            train_target, val_target, test_target = get_train_val_test_split(target) # 70-15-15 split

            # Train
            model = get_trained_model(train_features, train_target) # n_est=100, max_depth=6, learning rate=0.1
            st.success("Model trained successfully!")

            # Optimize
            model = get_optimized_model(model, train_features, train_target, val_features, val_target)
            params_used = model.get_params()

            st.success(
                f"Model optimized successfully! "
                f"Optimal parameters: n_estimators={params_used['n_estimators']}, "
                f"learning_rate={params_used['learning_rate']}, "
                f"max_depth={params_used['max_depth']}"
            )
            
            # Predictions
            predictions_train = model.predict(train_features)
            predictions_val = model.predict(val_features)
            predictions_test = model.predict(test_features)
            predictions = np.concatenate([predictions_train, predictions_val, predictions_test])

        except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.stop()

        # MAPE scores
        mape_pred_train = mape(predictions_train, train_target)
        mape_pred_val = mape(predictions_val, val_target)
        mape_pred_test = mape(predictions_test, test_target)

        st.markdown("#### *Model Performance (MAPE)*")
        st.write(f"Training MAPE: {mape_pred_train:.2f}%")
        st.write(f"Validation MAPE: {mape_pred_val:.2f}%")
        st.write(f"Test MAPE: {mape_pred_test:.2f}%")

        st.markdown("#### *Predicted vs Actual Consumption*")
        plot_results(data['start_time'].values, data['actual'].values, predictions)
else:
    st.info("Please select and confirm a start date in the sidebar to begin. Consider taking a date close to 2025-10-20 for quick results.")



# -------------------- FOOTER --------------------

st.markdown("---")
st.write("Use the sidebar to select a start date and train the model. Results will appear below.")
