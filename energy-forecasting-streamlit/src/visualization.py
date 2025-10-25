import streamlit as st
import matplotlib.pyplot as plt

def plot_results(time, actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(time, actual, label='Actual Consumption', color='blue')
    plt.plot(time, predicted, label='Predicted Consumption', color='orange')
    plt.title('Actual vs Predicted Energy Consumption')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption')
    plt.legend()
    st.pyplot(plt)
