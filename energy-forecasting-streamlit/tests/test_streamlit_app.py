# import streamlit as st
# import pandas as pd
# import numpy as np
# from src.models.xgboost_model import train_xgboost_model
# from src.visualization import plot_results
# from src.data.loader import load_data

# def main():
#     st.title("Quarter-Hourly Energy Consumption Forecasting")
    
#     # Load the dataset
#     data = load_data()
    
#     # Display initial results
#     st.subheader("Initial Results")
#     st.write("Mean Absolute Percentage Error (MAPE) from previous models:")
#     st.write("Benchmark MAPE: 2.7%")
    
#     # User input for start date
#     start_date = st.date_input("Select a start date for historical data:", value=pd.to_datetime("2015-01-01"))
    
#     if st.button("Train Model"):
#         # Train the XGBoost model with the selected start date
#         historical_data = data[data['date'] >= start_date]
#         if historical_data.empty:
#             st.error("No data available for the selected date range.")
#         else:
#             model, train_mape = train_xgboost_model(historical_data)
#             st.success(f"Model trained successfully! Training MAPE: {train_mape:.2f}%")
            
#             # Make predictions
#             predictions = make_predictions(model, historical_data)
            
#             # Plot results
#             st.subheader("Predicted vs Actual Energy Consumption")
#             fig = plot_results(predictions, historical_data)
#             st.pyplot(fig)

# if __name__ == "__main__":
#     main()