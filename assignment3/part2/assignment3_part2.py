#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np

url = "https://alpha-vantage.p.rapidapi.com/query"

querystring = {"function":"TIME_SERIES_WEEKLY","symbol":"MSFT","datatype":"csv"}

headers = {
    'x-rapidapi-key': "7c88ebd8b7mshc86f459ed07ac73p1b1930jsn0ced0c58307d",
    'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
    }

response = requests.request("GET", url, headers=headers, params=querystring)
print(response.text)

## Create a new csv file to save the output
MyFILE=open("Stock Price.csv","w")
MyFILE.write(response.text)
MyFILE.close()


############### PROCESS THE FILE ######################
import matplotlib.pyplot as plt
## Seaborn builds on top of matplotlib

import seaborn as sns
sns.set(style="darkgrid")

## Optional for setting sizes-----------
rc={'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16.0, 
    'axes.titlesize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16}

sns.set(rc=rc)
##----------------------------------------

MyFileName="/Users/didi/Desktop/gu/assignment3/Stock Price.csv"

############# clean the dataset #############
# (1) query the data types of the columns in the dataframe using the dataframe attribute .dtypes.
StockDF=pd.read_csv(MyFileName)
print(StockDF.head())
StockDF.dtypes
# The .dtypes attribute indicates that the data columns in your pandas dataframe are stored as several different data types as follows:
#    date as object: A string of characters that are in quotes.
#    max_temp as int64 64 bit integer. This is a numeric value that will never contain decimal points.
#    precip as float64 - 64 bit float: This data type accepts data that are a wide variety of numeric formats including decimals (floating point values) and integers. Numeric also accept larger numbers than int will.

# Plot Dates as Strings
# p1-open
stocknew = StockDF.head(30)
plt.figure(figsize = (15,8))
sns.lineplot(x="timestamp", y="open", data=stocknew).set(title="open price of MSFT")

# Now, look closely at the dates on the x-axis. When you plot a string field for the x-axis, Python gets stuck trying to plot the all of the date labels. Each value is read as a string, and it is difficult to try to fit all of those values on the x axis efficiently.
# You can avoid this problem by converting the dates from strings to a datetime object during the import of data into a pandas dataframe. Once the dates are converted to a datetime object, you can more easily customize the dates on your plot, resulting in a more visually appealing plot.

# Import Date Column into Pandas Dataframe As Datetime Object
StockDF=pd.read_csv(MyFileName,
                    parse_dates=['timestamp'], # To import the dates as a datetime object, you can use the parse_dates parameter of the pd.read_csv() function that allows you to indicate that a particular column should be converted to a datetime object:
                    index_col=['timestamp']) # If you have a single column that contain dates in your data, you can also set dates as the index for the dataframe using the index_col parameter:
print(StockDF.head())
StockDF.dtypes

# Summary Stats
StockDF.describe()

# 1. What was the change in price of the stock overtime?
# open
# Create figure and plot space
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.bar(StockDF.index.values,
       StockDF['open'],
       color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Open Price of MSFT",
       title="Open Price of MSFT from 1999 to 2021")

# Rotate tick marks on x-axis
plt.setp(ax.get_xticklabels(), rotation=45)

plt.show()

# close
# Create figure and plot space
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.scatter(StockDF.index.values,
        StockDF['close'],
        color='firebrick')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Close Price of MSFT",
       title="Close Price of MSFT from 1999 to 2021")

plt.show()

# open
StockDF.plot(y='open',legend=True,figsize=(10,4))
plt.ylabel('Opening price of MSFT')
plt.xlabel('date')
plt.title("Opening Price of MSFT from 1999 to 2021")

# high
StockDF.plot(y='high',legend=True,figsize=(10,4),color="orange")
plt.ylabel('High price of MSFT')
plt.xlabel('date')
plt.title("High price of MSFT from 1999 to 2021")

# low
StockDF.plot(y='low',legend=True,figsize=(10,4), color="purple")
plt.ylabel('Low price of MSFT')
plt.xlabel('date')
plt.title("Low price of MSFT from 1999 to 2021")

# volume
StockDF.plot(y='volume',legend=True,figsize=(10,4))
plt.ylabel('Volume')
plt.xlabel('date')
plt.title("Sales Volume for MSFT")


###### EDA ######################
# eda-1: moving average
ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = f"MA for {ma} days"
    StockDF[column_name] = StockDF['close'].rolling(ma).mean()


StockDF[['close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(figsize=(10,4)).set_title('MSFT')
plt.ylabel('stock price')

# eda-2: daily return
# We'll use pct_change to find the percent change for each day
StockDF['Daily Return'] = StockDF['close'].pct_change()

# Then we'll plot the daily return percentage
StockDF['Daily Return'].plot(figsize=(18,6), legend=True, linestyle='--', marker='o')
plt.ylabel('Daily Return')
plt.xlabel('date')
plt.title("Daily Return of MSFT")

# eda-3: daily return
# Note the use of dropna() here, otherwise the NaN values can't be read by seaborn
plt.figure(figsize=(12, 7))
sns.distplot(StockDF['Daily Return'].dropna(), bins=100, color='purple')
plt.ylabel('Daily Return')
plt.xlabel('number')
plt.title("Histogram of Daily Return of MSFT")
# Could have also done:
#AAPL['Daily Return'].hist()

#eda-4: correlation
sns.jointplot('volume', 'close', StockDF, kind='scatter', color='seagreen')
plt.ylabel('closed price of MSFT')
plt.figure(figsize=(12, 7))

#eda-5
sns.pairplot(StockDF.iloc[: , 0:5], kind='reg')

#eda-6
sns.heatmap(StockDF.iloc[: , 0:5].corr(), annot=True, cmap='summer')

#eda-7
# Let's start by defining a new DataFrame as a clenaed version of the oriignal tech_rets DataFrame
rets = StockDF.iloc[: , 8:9].dropna()
plt.figure(figsize=(10, 7))
area = np.pi * 20
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')
plt.title('Risk of investing in MSFT')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

################## models ##################
plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(StockDF['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
   
# Create a new dataframe with only the 'Close column 
data = StockDF.filter(['close'])
data = data.iloc[::-1]
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len

# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# PLOT-1
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Show the valid and predicted prices
valid

# PLOT-2
plt.figure(figsize=(16,6))
plt.plot(valid['close'], color = 'red', label = 'Real Price')
plt.plot(valid['Predictions'], color = 'blue', label = 'Predicted Price')
plt.title('MSFT Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MSFT Stock Price')
plt.legend()
plt.show()

################ 
plt.figure(figsize=(15, 5));
plt.plot(StockDF.open.values, color='red', label='open')
plt.plot(StockDF.close.values, color='green', label='low')
plt.plot(StockDF.low.values, color='blue', label='low')
plt.plot(StockDF.high.values, color='black', label='high')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('Stock Price of MSFT')
plt.xlabel('time [days]')
plt.ylabel('stock price')
plt.legend(loc='best')
plt.show()

from keras.models import Sequential
from keras.layers import Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

# The GRU architecture
regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=50, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
# Fitting to the training set
regressorGRU.fit(x_train,y_train,epochs=50,batch_size=150)

GRU_predicted_stock_price = regressorGRU.predict(x_test)
GRU_predicted_stock_price = scaler.inverse_transform(GRU_predicted_stock_price)

# Get the root mean squared error (RMSE)
rmse2 = np.sqrt(np.mean(((GRU_predicted_stock_price - y_test) ** 2)))
rmse2

# PLOT-1
train = data[:training_data_len]
valid2 = data[training_data_len:]
valid2['Predictions'] = GRU_predicted_stock_price

# Visualize the data
plt.figure(figsize=(16,6))
plt.title('GRU Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['close'])
plt.plot(valid2[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Show the valid and predicted prices
valid2

# Visualizing the results for GRU
# PLOT-2
plt.figure(figsize=(16,6))
plt.plot(valid2['close'], color = 'red', label = 'Real Price')
plt.plot(valid2['Predictions'], color = 'blue', label = 'Predicted Price')
plt.title('MSFT Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MSFT Stock Price')
plt.legend()
plt.show()

