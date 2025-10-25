import pandas as pd
import streamlit as st
import numpy as np
import os

from datetime import datetime

def load_data(file_path="src/data/entsoe_10_years_combined.csv"):
    """
    Load and preprocess raw data from CSV.
    Canonical preprocessing:
    - split interval column
    - parse dates
    - drop/mark missing
    - derive time features
    """
    # load CSV
    base_dir = os.path.dirname(__file__)  # Folder where loader.py lives
    file_path = os.path.join(base_dir, 'entsoe_10_years_combined.csv')
    df= pd.read_csv(file_path, header=0, names=['interval', 'forecast', 'actual'], na_values='-')

    # basic missing handling (adjust to your needs)
    df = df.dropna(subset=['actual'])

    # Split interval into start and end, convert to datetime
    df = df.copy()
    df['start_time'] = pd.to_datetime(df['interval'].str.split(' - ').str[0], format="%d.%m.%Y %H:%M")
    df['end_time'] = pd.to_datetime(df['interval'].str.split(' - ').str[1], format="%d.%m.%Y %H:%M")
    # derived time features (example)
    df['year'] = df['start_time'].dt.year
    df['month'] = df['start_time'].dt.month
    df['day'] = df['start_time'].dt.day
    df['weekday'] = df['start_time'].dt.weekday
    df['hour'] = df['start_time'].dt.hour
    df['minute'] = df['start_time'].dt.minute
    # quarter index (0-3 per hour)
    df['quarter'] = (df['minute'] // 15)
    # time index numbering (with integers) each quarter hour from start to end
    df['time_index'] = np.arange(len(df))
    # add combine features: weekday and hour
    df['weekday_hour'] = df['weekday'] * 24 + df['hour']

    return df.reset_index(drop=True)

def prepare_data_for_model(df, start_date):
    # df is pandas DataFrame and start_date is a datetime;
    """Filter rows on or after start_date (string or datetime)."""
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date, format="%Y-%m-%d")
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    

    return df[df['start_time'] >= start_datetime].reset_index(drop=True)

def extract_features(data, feature_cols=['year', 'month', 'day', 'weekday', 'hour', 'quarter', 'time_index', 'weekday_hour']):
    # extract only the specified feature columns and return as numpy array
    return data[feature_cols].values

def sin_cos_transform(data, column, period):
    # sin and cosine transformations for periodic features
    data[f'{column}_sin'] = np.sin(2 * np.pi * data[column] / period)
    data[f'{column}_cos'] = np.cos(2 * np.pi * data[column] / period)
    return data