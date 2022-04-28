# Import necessary libraries
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

data_source = 'kaggle'

if data_source == 'alphavantage':
    # Receive my API Key
    api_key = 'Z5DPMU0ZYRJ1Y1NP'
    # Use AMD Ticker
    ticker = 'AAL'
    # Gather Data and save to file
    url_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&apikey={}'.format(ticker, api_key)
    file_to_save = 'stock_market_data-%s.csv'%ticker

    # Store date, values to Pandas DataFrame
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())

            # Extract market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])

            for k, v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                            float(v['4. close']),float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        print('Date saved to: %s'%file_to_save)
        df.to_csv(file_to_save)
    else:
        # Say file exists and read the CSV
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)
else:
    # Load data from Kaggle
    df = pd.read_csv(os.path.join('~/Downloads/archive/Stocks', 'aal.us.txt'), delimiter = ',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    print('Loaded data from Kaggle repository')


# Sort Data
df = df.sort_values('Date')
df.head()

# Create plot of Data
plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()

# Variables for high, low, mid-prices
high_prices = df.loc[:,'High'].to_numpy()
low_prices = df.loc[:,'Low'].to_numpy()
mid_prices = (high_prices+low_prices)/2.0

# Split into training data and test data
train_data = mid_prices[:900]
test_data = mid_prices[900:]

# Normalize data first by scaling
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

# Choose window size and fit data
smoothing_window_size = 500
for di in range(0,250,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])


# Reshape data to data_size again
train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

# Smoothen out the data using exponential moving average
EMA = 0.0
gamma = 0.1
for ti in range(250):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA
all_mid_data = np.concatenate([train_data,test_data],axis=0)