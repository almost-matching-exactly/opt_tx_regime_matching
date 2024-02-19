#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 09:24:59 2023

@author: harshparikh
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import xgboost as xgb

outcome_function = xgb.XGBRegressor()
outcome_function.load_model("outcome_model.json")

# Define a function to calculate features based on the input DataFrame 'df_i'
def get_features_prop(df_i):
    """
    This function calculates features based on the input DataFrame 'df_i'.

    Args:
    - df_i: Input DataFrame containing relevant data.

    Returns:
    - DataFrame: A DataFrame containing computed features.
    """
    # Define a list of drugs of interest
    drugs_interest = ['Ddose_levetiracetam', 'Ddose_propofol']
    v = []

    # Iterate through time intervals in the DataFrame
    for t in range(0, df_i['time'].max() - 11, 6):
        # Calculate various features and store them in v_i
        tmax_1 = min(t + 1, 6) # last 1 hour
        tmax_6 = min(t + 1, 36) # last 6 hours
        tmax_12 = min(t + 1, 72) # last 12 hours
        v_i = [
            (df_i['IIC'].iloc[t - tmax_1 + 1:t + 1].mean() > 0.25),
            (df_i['IIC'].iloc[t - tmax_1 + 1:t + 1].mean() > 0.5),
            (df_i['IIC'].iloc[t - tmax_1 + 1:t + 1].mean() > 0.75),
            (df_i['IIC'].iloc[t - tmax_6 + 1:t + 1].mean() > 0.25),
            (df_i['IIC'].iloc[t - tmax_6 + 1:t + 1].mean() > 0.5),
            (df_i['IIC'].iloc[t - tmax_1 + 1:t + 1].mean() > 0.25) * (df_i['IIC'].iloc[t - tmax_6 + 1:t + 1].mean() > 0.25),
            (df_i['IIC'].iloc[t - tmax_6 + 1:t + 1].mean() > 0.25) * (df_i['IIC'].iloc[t - tmax_12 + 1:t + 1].mean() > 0.25),
        ]
        o_i = [
            df_i[drugs_interest[1]].iloc[t + 1:t + 7].sum()
        ]
        v_i = v_i + o_i
        v = v + [v_i]

    # Define column names for the resulting DataFrame
    columns = ['E in last 1h (>25%)',
               'E in last 1h (>50%)',
               'E in last 1h (>75%)',
               'E in last 6h (>25%)',
               'E in last 6h (>50%)',
               'E in last 1h (>25% ) AND E in last 6h (>25%)',
               'E in last 6h (>25%) AND E in last 12h (>25%)', 'Prop_Act']

    # Return the computed features as a DataFrame
    return pd.DataFrame(np.array(v), columns=columns)

def get_features_lev(df_i):
    """
    This function calculates features based on the input DataFrame 'df_i'.

    Args:
    - df_i: Input DataFrame containing relevant data.

    Returns:
    - DataFrame: A DataFrame containing computed features.
    """
    # Define a list of drugs of interest
    drugs_interest = ['Ddose_levetiracetam', 'Ddose_propofol']
    v = []

    # Iterate through time intervals in the DataFrame
    for t in range(0, df_i['time'].max() - 11, 6):
        # Calculate various features and store them in v_i
        tmax_1 = min(t + 1, 6) # last 1 hour
        tmax_6 = min(t + 1, 36) # last 6 hours
        tmax_12 = min(t + 1, 72) # last 12 hours
        v_i = [1,
            (df_i['IIC'].iloc[t - tmax_1 + 1:t + 1].mean() > 0.25) * (df_i[drugs_interest[0]].iloc[t - tmax_12 + 1:t + 1].mean() == 0),
            (df_i['IIC'].iloc[t - tmax_1 + 1:t + 1].mean() > 0.5) * (df_i[drugs_interest[0]].iloc[t - tmax_12 + 1:t + 1].mean() == 0),
            (df_i['IIC'].iloc[t - tmax_1 + 1:t + 1].mean() > 0.75) * (df_i[drugs_interest[0]].iloc[t - tmax_12 + 1:t + 1].mean() == 0),
            (df_i['IIC'].iloc[t - tmax_6 + 1:t + 1].mean() > 0.25) * (df_i[drugs_interest[0]].iloc[t - tmax_12 + 1:t + 1].mean() == 0),
            (df_i['IIC'].iloc[t - tmax_6 + 1:t + 1].mean() > 0.5) * (df_i[drugs_interest[0]].iloc[t - tmax_12 + 1:t + 1].mean() == 0),
            (df_i['IIC'].iloc[t - tmax_1 + 1:t + 1].mean() > 0.25) * (df_i['IIC'].iloc[t - tmax_6 + 1:t + 1].mean() > 0.25) * (df_i[drugs_interest[0]].iloc[t - tmax_12 + 1:t + 1].mean() == 0),
            (df_i['IIC'].iloc[t - tmax_6 + 1:t + 1].mean() > 0.25) * (df_i['IIC'].iloc[t - tmax_12 + 1:t + 1].mean() > 0.25) * (df_i[drugs_interest[0]].iloc[t - tmax_12 + 1:t + 1].mean() == 0),
        ]
        o_i = [
            df_i[drugs_interest[0]].iloc[t + 1:t + 7].sum()
        ]
        v_i = v_i + o_i
        v = v + [v_i]

    # Define column names for the resulting DataFrame
    columns = ['Baseline Dose',
               'E in last 1h (>25%)',
               'E in last 1h (>50%)',
               'E in last 1h (>75%)',
               'E in last 6h (>25%)',
               'E in last 6h (>50%)',
               'E in last 1h (>25% ) AND E in last 6h (>25%)',
               'E in last 6h (>25%) AND E in last 12h (>25%)',
               'Lev_Act']

    # Return the computed features as a DataFrame
    return pd.DataFrame(np.array(v), columns=columns)

def caliper_match(X, metric, caliper):
    """
    This function performs matching based on the caliper distance.

    Args:
    - X: Input data.
    - metric: Metric used for matching.
    - caliper: Maximum allowed distance for matching.

    Returns:
    - MG: Matching groups.
    - D: Distance matrix.
    """
    D = np.ones((X.shape[0], X.shape[1], X.shape[0])) * X.T
    D = (D - D.T)
    D = np.sum((D * (metric.reshape(-1, 1))) ** 2, axis=1)
    MG = (D <= caliper).astype(int)
    return MG, D

def outcome(x):
    """
    This function predicts an outcome based on the input 'x'.

    Args:
    - x: Input data.

    Returns:
    - float: Predicted outcome.
    """
    return outcome_function.predict(x)[0]

def normalize(x):
    x_ = x.copy(deep=True)
    for col in x_.columns:
        x_[col] = (x_[col].max() - x_[col]) / (x_[col].max() - x_[col].min()) 
    return x_