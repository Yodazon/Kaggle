### January 4 2024
### The purposes of this file is to clean up the 'script.ipynb' file in the same folder
### And to be able to modify all files in the zipfile for future forecasting
### Benefits: 
### Work with all files at once
### Less hassle of troubleshooting
### can be easier to plot for forecasting
###

### I am using this project as a learning oppurtunity to see how I can connect scripts together
### As well to learn about forecasting and using facebook's prophet python library



import pandas as pd
import numpy as np
import zipfile
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


###Purpose of this function is to import our csv files from the zip folder based on certain prefix
def import_csv_file(file_path, csv_file_name):
    dfs = {}

    #open zip file
    with zipfile.ZipFile(file_path, 'r') as zf:
        #Loop through csv files
        match_file =  [file for file in zf.namelist() if csv_file_name in file]

        for file in match_file:
            #read and store into the dictionary
            dfs[file] = pd.read_csv(zf.open(file))

    return dfs

###This functions normalizes the power
def normal(dict_dfs):
    dfs = dict_dfs

    for key, df in dfs.items():
        if 'Power' in df.columns:
            df['normalized Power'] = df['Power'] / df['Power'].mean()
        else:
            print(f"Warning: DataFrame {key} does not have a 'Power' column.")
    
    return dfs

###This function creates a new column called 'date' to be used later for mean and median values
def set_Date(dict_dfs):
    dfs_dates = dict_dfs

    for key, df in dfs_dates.items():
        df['Time'] = pd.to_datetime(df['Time'])
        df['date'] = df['Time'].dt.date
        df['date'] = pd.to_datetime(df['date']) 
    return dfs_dates

##This function takes the average of values of a given day
def mean(dict_dfs):
    dfs_avg = {}

    for key, df in dict_dfs.items():
        value = []
        for entry in df['date'].unique():
            x = df[df['date'] == entry].mean()
            value.append(x)
        dfs_avg[key] = pd.DataFrame(value)

    return dfs_avg 

##This function takes the median of values of a given day
def median(dict_dfs):
    dfs_median = {}

    for key, df in dict_dfs.items():
        value = []
        for entry in df['date'].unique():
            x = df[df['date'] == entry].median()
            value.append(x)
        dfs_median[key] = pd.DataFrame(value)

    return dfs_median

###This function sets up new dataframes that can be used for Prophet
def prophet_setup(dict_dfs, column_ds, column_y):
    dfs_prophet = {}

    for key, df in dict_dfs.items():
        temp_df = df[[column_ds, column_y]].copy()
        temp_df.rename(columns = {column_ds: 'ds', column_y: 'y'}, inplace=True)
        dfs_prophet[key] = temp_df
    return dfs_prophet

###This function is making + fitting a prophet dictionary
def prophet_library (dict_dfs):
    dict_dfs_fit = {}

    for key, df in dict_dfs.items():
        m = Prophet()
        dict_dfs_fit[key] = m.fit(df)
    
    return dict_dfs_fit

###Creating a dictionary of future
def prophet_future (dict_dfs):
    dict_dfs_future = {}
    for key,df in dict_dfs.items():
       dict_dfs_future[key] = df.make_future_dataframe(periods = 365)
    return dict_dfs_future 

###Making future predictions
def prophet_forecast (dict_dfs_lib, dict_dfs_future ):
    dict_dfs_forecast = {}

    for (key_lib, df), (key_future, future_df) in zip(dict_dfs_lib.items(), dict_dfs_future.items()):
        m = dict_dfs_lib[key_lib]
        forecast = m.predict(future_df)
        dict_dfs_forecast[key_future] = forecast
    
    return dict_dfs_forecast

def plot_prediciton(dict_dfs_fit, dict_dfs_forecast, type):
    fig ={}
    for (key_lib, df), (key_forecast, forecast_df) in zip (dict_dfs_fit.items(), dict_dfs_forecast.items()):
        plt.figure()  # Create a new figure
        df.plot(forecast_df)
        plt.title(f"{type} of {key_forecast}")
        fig[key_lib] = plt.gcf() # Get the current figure and store it in the dictionary

def plot_components(dict_dfs_fit, dict_dfs_forecast, type):
    fig ={}

    for (key_lib, df), (key_forecast, forecast_df) in zip (dict_dfs_fit.items(), dict_dfs_forecast.items()):
        plt.figure()  # Create a new figure
        df.plot_components(forecast_df)
        plt.title(f"{type} of {key_forecast} Components")
        fig[key_lib] = plt.gcf() # Get the current figure and store it in the dictionary




def plot_plotly_prediction(dict_dfs_fit, dict_dfs_forecast, type):
    fig = {}

    for (key_lib, df_fit), (key_forecast, df_forecast) in zip(dict_dfs_fit.items(), dict_dfs_forecast.items()):
        #plt.figure()  # Create a new figure
        plot_plotly(df_fit, df_forecast)
        plt.title(f"{type} of {key_forecast}")
        fig[key_lib] = plt.gcf()  # Get the current figure and store it in the dictionary

    return fig
        

def plot_plotly_components(dict_dfs_fit, dict_dfs_forecast, type):
    fig ={}

    for (key_lib, df), (key_forecast, forecast_df) in zip (dict_dfs_fit.items(), dict_dfs_forecast.items()):
        plt.figure()  # Create a new figure
        plot_components_plotly(df,forecast_df)
        plt.title(f"{type} of {key_forecast} Components")
        fig[key_lib] = plt.gcf() # Get the current figure and store it in the dictionary
