import pandas as pd
import numpy as np
import warnings
import xgboost
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost 
import os
from google.cloud import storage


solar_data = pd.read_csv('gs://forecast_proj_resources/SolarMonthlyData_2657Custs.csv')


# Data preprocessing
# ======================================================================================
consumption_billrate = ['PK', 'LVP', 'SH', 'LVS','OP','OP1','OP2']
solar_data_consumption = solar_data[(solar_data['Netwk Bill Rate Type'].isin(consumption_billrate)) & (solar_data['Unit of me'] == 'KWH')].copy()
solar_data_consumption['Consumption Month'] = solar_data_consumption['Consumption Month'].apply(lambda x: "{:.4f}".format(x))
solar_data_consumption['Consumption Month'] = solar_data_consumption['Consumption Month'].astype('str')
solar_data_consumption['Consumption Month'] = pd.to_datetime(solar_data_consumption['Consumption Month'].apply(lambda x: dt.datetime.strptime(x, '%m.%Y')))
solar_data_consumption['house_type'] = 'solar'

generation_billrate = ['PGR', 'SGR','OGR','OGG','PGG','SGG']
solar_data_generation = solar_data[(solar_data['Netwk Bill Rate Type'].isin(generation_billrate)) & (solar_data['Unit of me'] == 'KWH')].copy()
solar_data_generation['Consumption Month'] = solar_data_generation['Consumption Month'].apply(lambda x: "{:.4f}".format(x))
solar_data_generation['Consumption Month'] = solar_data_generation['Consumption Month'].astype('str')
solar_data_generation['Consumption Month'] = pd.to_datetime(solar_data_generation['Consumption Month'].apply(lambda x: dt.datetime.strptime(x, '%m.%Y')))


solar_grouped_generation  = solar_data_generation.groupby(['Customer ID', 'Consumption Month']).agg({'Sum': 'sum'}).reset_index()
solar_grouped_consumption = solar_data_consumption.groupby(['Customer ID', 'Consumption Month']).agg({'Sum': 'sum'}).reset_index()

solar_net_consumption = solar_grouped_consumption.merge(solar_grouped_generation, on = [ 'Customer ID', 'Consumption Month'], how = 'left',
          suffixes=('_left', '_right'))

solar_net_consumption['Consumption'] = solar_net_consumption.fillna(0)['Sum_left'] - solar_net_consumption.fillna(0)['Sum_right']
solar_net_consumption.drop(['Sum_left','Sum_right'], axis=1, inplace=True)
df = solar_net_consumption.copy()

# Feature Engineering 
def create_lag_features(data, lag_steps):
    for lag in range(1, lag_steps + 1):
        data[f'lag_{lag}'] = data.groupby('Customer ID')['Consumption'].shift(lag)
    return data
df_long = create_lag_features(df, lag_steps=3)

def create_rolling_mean_features(data, window_size):
    data['rolling_mean'] = data.groupby('Customer ID')['Consumption'].transform(lambda x: x.shift(1).rolling(window=window_size).mean())
    return data
df_long = create_rolling_mean_features(df_long, window_size=2)


df_long.fillna(method='bfill', inplace=True)
df_long['Consumption Month'] = pd.to_datetime(df_long['Consumption Month'])



# splittting the data 
split_date = '2014-12-01'
train_data = df_long[df_long['Consumption Month'] < split_date]
test_data = df_long[df_long['Consumption Month'] == split_date]
# Separate features and target
X_train = train_data.drop(columns=['Customer ID', 'Consumption Month', 'Consumption'])
y_train = train_data['Consumption']
X_test = test_data.drop(columns=['Customer ID', 'Consumption Month', 'Consumption'])
y_test = test_data['Consumption']

from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    learning_rate=0.05,
    max_depth=5,
    subsample=1.0,
    colsample_bytree=0.8,
    n_estimators=300
)

bst = xgb_model.fit(X_train, y_train)

# SAVE MODEL
artifact_filename = 'model.bst'
# Save model artifact to local filesystem (doesn't persist)
local_path = artifact_filename
bst.save_model(local_path)
model_directory = 'gs://forecast_proj_resources'
storage_path = os.path.join(model_directory, artifact_filename)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(local_path)


