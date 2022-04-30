import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import warnings
import matplotlib.dates as mpdates
from plotly.offline import iplot
import cufflinks as cf
import pandas as pd
from datetime import datetime
import pandas_datareader as data
import streamlit as st
from keras.models import load_model
import plotly.graph_objects as go
# Import necessary libraries

import pandas as pd
import yfinance as yf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp
from scipy.signal import argrelextrema


# start = '2010-01-01'
# end = '2019-12-31'

st.title('Stock Trend Prediction')

start1 = st.date_input(
     "Enter Start end",
)
end1 = st.date_input(
     "Enter Date end",
)
start2 = st.date_input(
     "Enter Resistance Support Date"
)
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start1, end1)

st.subheader('Data of stock')
st.write(df)

st.subheader("line chart")
st.subheader('closing price vs time chart ')
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('white')
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('candlestick chart')


fig = go.Figure(data = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
st.plotly_chart(fig)

st.subheader('closing price vs time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('white')
plt.plot(ma100,'red')
plt.plot(df.Close,'blue')
st.pyplot(fig)




st.subheader('closing price vs time chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
plt.plot(ma100, 'y')
plt.plot(ma200, 'r')

plt.plot(df.Close, 'blue')
ax.set_facecolor('white')
st.pyplot(fig)





data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


#loading the keras model

model = load_model('kerasmodel.h5')


#testing part
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)


y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test= y_test*scale_factor


#the project graph
st.subheader('Predictiosns vs Orginals')

fig2=plt.figure(figsize=(12,6))
ax = plt.axes()
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label="Predicted Price")

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
ax.set_facecolor('white')
st.pyplot(fig2)


import yfinance as yf
st.subheader('Resistance And Support')
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
# get stock prices using yfinance library
def get_stock_price(symbol):
    
    
    df = yf.download(symbol, start2, threads= False)
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
    return df
symbol = user_input
df = get_stock_price(symbol)

def is_support(df,i):
    cond1 = df['Low'][i] < df['Low'][i-1]   
    cond2 = df['Low'][i] < df['Low'][i+1]   
    cond3 = df['Low'][i+1] < df['Low'][i+2]   
    cond4 = df['Low'][i-1] < df['Low'][i-2]  
    return (cond1 and cond2 and cond3 and cond4) 
    
  
#determine bearish fractal
def is_resistance(df,i):
    
    cond1 = df['High'][i] > df['High'][i-1]   
    cond2 = df['High'][i] > df['High'][i+1]   
    cond3 = df['High'][i+1] > df['High'][i+2]   
    cond4 = df['High'][i-1] > df['High'][i-2]  
    return (cond1 and cond2 and cond3 and cond4)
    

# to make sure the new level area does not exist already
def is_far_from_level(value, levels, df):
    ave =  np.mean(df['High'] - df['Low'])    
    return np.sum([abs(value-level)<ave for _,level in levels])==0
# a list to store resistance and support levels
levels = []

for i in range(2, df.shape[0] - 2):
    if is_support(df, i):
        low = df['Low'][i]    
        if is_far_from_level(low, levels, df):
            levels.append((i, low))  
    elif is_resistance(df, i):
        high = df['High'][i]    
        if is_far_from_level(high, levels, df):
            
            levels.append((i, high))
            print(levels)

def plot_all(levels, df):
    
    global fig4,ax
    fig4,ax = plt.subplots(figsize=(16, 9)) 
    z = candlestick_ohlc(ax,df.values,width=0.6, colorup='green', 
    colordown='red', alpha=0.8)    
    date_format = mpl_dates.DateFormatter('%d %b %Y')
    ax.xaxis.set_major_formatter(date_format)
    for level in levels:
        
        
        plt.hlines(level[1], xmin = df['Date'][level[0]], xmax = 
      max(df['Date']), colors='blue', linestyle='--')
        

plot_all(levels,df)
ax.set_facecolor("white")
st.pyplot(fig4)


st.subheader('Channel Pattern')
# Get stock prices using yfinance library
def get_data(symbol, start):
  df = pd.DataFrame() 
  data =  yf.download(symbol, start)
  df = pd.concat([df, data], axis=0)
  df.rename(columns = {'Open':'open', 'High':'high','Low':'low','Close':'close','Adj Close':'adj close', 'Volume':'volume'}, inplace = True)
  df.columns = pd.MultiIndex.from_tuples([(symbol,'open'),(symbol,'high'),(symbol,'low'),(symbol,'close'),(symbol,'adj close'),(symbol,'volume')])
  df['row number'] = np.arange(1, len(df)+1)
  return df
get_data(user_input, start1)

class TrendlineSeeker:
  
    def __init__(self, data):
        self.data = data
        self.previousTrendlineBreakingPoint = 0
        self.currentTrendlineBreakingPoint = len(data) + 1
        self.refinedTrendlines = {}
    def getCurrentRange(self):
        df = self.data.copy().loc[(self.data['row number'] >= self.previousTrendlineBreakingPoint) & (self.data['row number'] < self.currentTrendlineBreakingPoint), :]
        return df
    def getNextRange(self):
        df = self.data.loc[self.data['row number']>= self.currentTrendlineBreakingPoint, :]
        return df
    def trendlineBreakingPoints(self, currentdf, slope, intercept):
        possibleTrendBreakingPoints = currentdf.loc[(currentdf.loc[:, (user_input, "close")] < 0.85*(slope*currentdf['row number'] + intercept)) | (currentdf.loc[:, (user_input, "close")] > 1.15*(slope*currentdf['row number'] + intercept)), 'row number']
        return possibleTrendBreakingPoints
    def upperBoundlinesForCurrentRange(self, currentdf):
        tempdf = currentdf.copy()
        slope = 0
        intercept = 0
        while len(tempdf) > 2:
           slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x=tempdf['row number'], y=tempdf.loc[:, (user_input, "close")])
           tempdf = tempdf.loc[(tempdf.loc[:, (user_input, "close")]> slope * tempdf['row number'] + intercept)]
        return slope, intercept
    def refineUpperBoundlinesForCurrentRange(self, possibleUpperBoundlineBreakingPoints):
        localPossibleUpperBoundlineBreakingPoints = possibleUpperBoundlineBreakingPoints
       
        i = 1
        while len(localPossibleUpperBoundlineBreakingPoints) > 0:
           self.currentTrendlineBreakingPoint = int(localPossibleUpperBoundlineBreakingPoints[0])
           if self.currentTrendlineBreakingPoint - self.previousTrendlineBreakingPoint < 24:
              self.currentTrendlineBreakingPoint = len(self.data) + 1 - i
              i += 1
           currentdf = self.getCurrentRange()
           slope, intercept = self.upperBoundlinesForCurrentRange(currentdf)
        
           localPossibleUpperBoundlineBreakingPoints = self.trendlineBreakingPoints(currentdf, slope, intercept)
           self.refinedTrendlines[str(self.previousTrendlineBreakingPoint)] = {'slope': slope, 'intercept': intercept, 'starting row': self.previousTrendlineBreakingPoint, 'ending row': self.currentTrendlineBreakingPoint - 1}
           self.previousTrendlineBreakingPoint = self.currentTrendlineBreakingPoint
           self.currentTrendlineBreakingPoint = len(self.data) + 1

    def main(self):
        i = 1
        while True:
          currentRange = self.getCurrentRange()
          if len(currentRange) <= 2: break
          upperline = self.upperBoundlinesForCurrentRange(currentRange)
          possibleUpperBoundlineBreakingPoints = self.trendlineBreakingPoints(currentRange, *upperline)
          if len(possibleUpperBoundlineBreakingPoints) == 0: 
            self.refinedTrendlines[str(self.previousTrendlineBreakingPoint)] = {'slope': upperline[0], 'intercept': upperline[1], 'starting row': self.previousTrendlineBreakingPoint, 'ending row': self.currentTrendlineBreakingPoint - 1}
            break
          else: self.refineUpperBoundlinesForCurrentRange(possibleUpperBoundlineBreakingPoints)
          i += 1
        plt1.plot(self.data.index,self.data.loc[:, (user_input, "close")], label='price action')
        for key, value in self.refinedTrendlines.items():
            plt1.plot(self.data.index.values[value['starting row']:value['ending row']], value['slope']*self.data['row number'][value['starting row']:value['ending row']] + value['intercept'])
        plt1.legend(loc='best')
        fig5 = plt1.show()
        st.pyplot(fig5)

if __name__ == '__main__':
   data = get_data(user_input, start1)
   ts = TrendlineSeeker(data)
   ts.main()

st.set_option('deprecation.showPyplotGlobalUse', False)