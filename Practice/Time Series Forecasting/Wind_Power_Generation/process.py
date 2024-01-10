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
    dfs_avg = {}

    for key, df in dict_dfs.items():
        value = []
        for entry in df['date'].unique():
            x = df[df['date'] == entry].median()
            value.append(x)
        dfs_avg[key] = pd.DataFrame(value)

    return dfs_avg 