# Import necessary libraries
import math
import numpy as np
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.dates as mpl_dates
from mplfinance.original_flavor import candlestick_ohlc
import streamlit as st
from keras.models import load_model
import plotly.graph_objects as go
import yfinance as yf
import scipy as sp
from sklearn.preprocessing import MinMaxScaler


# Website Title
st.title('StockBot - Chart Pattern Analyzer')

st.header("Details of Stock")

# User Inputs
START_DATE = st.date_input(
     "Enter Start Date",
)
END_DATE = st.date_input(
     "Enter End Date",
)
RES_SUP_DATE = st.date_input(
     "Enter Resistance Support Date"
)
TICKER = st.text_input('Enter Stock Ticker', 'AAPL')

st.header("Details of Chart Pattern Analysis")
st.subheader("Enter Channel Validation Window")
CHANNEL_VALIDATION_FRAME = st.number_input("To check for a feasible channel pattern in given number of days", 0, 10000, 48)
st.subheader("Enter Channel Intercepting Threshold in percentage")
CHANNEL_INTERCEPTING_THRESHOLD = st.number_input("The percentage of channel pattern feasible", 0, 100, 10)
st.subheader("Enter Slope difference between trendlines of channel in degree")
SLOPE_DIFF_THRESHOLD = st.number_input("A pair of support and resistance is considered parallel if their slope difference is less than the threshold.", 0, 90, 8)
st.subheader("Enter Deviation Threshold in percentage")
DEVIATION_THRESHOLD = st.number_input("A channel pattern is broken if a ticker is p percentage above the value on the resistance line or below the support line.", 0, 100, 15)

df = data.DataReader(TICKER, 'yahoo', START_DATE, END_DATE)
st.header(f'Data of {TICKER} stock')
st.write(df)

st.header("Analysis of different types of stock charts")

# Interactive Candlestick Chart
st.header('Candlestick Chart')
fig = go.Figure(data = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
st.plotly_chart(fig)

# Closing Price vs Time Chart
st.header('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('white')
plt.plot(df.Close,'b')
st.pyplot(fig)

st.header('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('white')
plt.plot(ma100,'red')
plt.plot(df.Close,'blue')
st.pyplot(fig)

st.header('Closing Price vs time chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
plt.plot(ma100, 'y')
plt.plot(ma200, 'r')
plt.plot(df.Close, 'blue')
ax.set_facecolor('white')
st.pyplot(fig)

# Stock Price Prediction
data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler=MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# loading the keras model
model = load_model('kerasmodel.h5')

# testing part
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

# Predicted Price vs Original Price Graph
st.header('Predicted Price vs Original Price')

fig2=plt.figure(figsize=(12,6))
ax = plt.axes()
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label="Predicted Price")

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
ax.set_facecolor('white')
st.pyplot(fig2)

# Resistance and Support Levels
st.header('Resistance and Support Levels')

# get stock prices using yfinance library
def get_stock_price(symbol):   
    df = yf.download(symbol, RES_SUP_DATE, threads= False)
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
    return df
symbol = TICKER
df = get_stock_price(symbol)

def is_support(df,i):
    cond1 = df['Low'][i] < df['Low'][i-1]   
    cond2 = df['Low'][i] < df['Low'][i+1]   
    cond3 = df['Low'][i+1] < df['Low'][i+2]   
    cond4 = df['Low'][i-1] < df['Low'][i-2]  
    return (cond1 and cond2 and cond3 and cond4)     
  
# determine bearish fractal
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

# Channel Pattern Analysis
st.header('Channel Pattern Analysis')

# Get stock prices using yfinance library
def get_data(symbol, start, end):
  df = pd.DataFrame() 
  data =  yf.download(symbol, start =start, end=end)
  df = pd.concat([df, data], axis=0)
  df.rename(columns = {'Open':'open', 'High':'high','Low':'low','Close':'close','Adj Close':'adj close', 'Volume':'volume'}, inplace = True)
  df.columns = pd.MultiIndex.from_tuples([(symbol,'open'),(symbol,'high'),(symbol,'low'),(symbol,'close'),(symbol,'adj close'),(symbol,'volume')])
  df['row number'] = np.arange(1, len(df)+1)
  return df

# Class to identify trendlines in stock chart
class trendlineDetector:  
    def __init__(self, symbols, data, flag='all'):
        self.symbols = symbols
        self.start = START_DATE
        self.end = END_DATE
        self.upperValidationWindow = CHANNEL_VALIDATION_FRAME
        self.validationWindowLower = CHANNEL_VALIDATION_FRAME
        self.upperDeviationThreshold = DEVIATION_THRESHOLD/100
        self.lowerDeviationThreshold = DEVIATION_THRESHOLD/100
  
        if flag == 'all':
            self.flag = ['upper', 'lower']
        else:
            self.flag = [flag]
        self.data = data
        self.previousTrendlineInterceptingPoint = 0
        self.currentTrendlineInterceptingPoint = len(self.data) + 1
        self.refinedTrendlines = {}
    def getCurrentSymbol(self):
        for symbol in self.symbols:
            yield symbol
    def getCurrentRange(self):
        df = self.data.copy().loc[(self.data['row number'] >= self.previousTrendlineInterceptingPoint) & (self.data['row number'] < self.currentTrendlineInterceptingPoint), :]
        return df
    def trendlineInterceptingPoints(self, symbol, flag, currentdf, slope, intercept):
        if flag == 'upper':
           distortionThreshold = self.upperDeviationThreshold
        if flag == 'lower':
           distortionThreshold = self.lowerDeviationThreshold
        possibleTrendInterceptingPoints = currentdf.loc[(currentdf.loc[:, (symbol, "close")] < (1-distortionThreshold)*(slope*currentdf['row number'] + intercept)) | (currentdf.loc[:, (symbol, "close")] > (1+distortionThreshold)*(slope*currentdf['row number'] + intercept)), 'row number']
        return possibleTrendInterceptingPoints
    def currentRangeTrendlines(self, symbol, flag, currentdf):
        tempdf = currentdf.copy()
        slope = 0
        intercept = 0
        while len(tempdf) > 2:
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x=tempdf['row number'], y=tempdf.loc[:, (TICKER, "close")])
            if flag == 'upper':
              tempdf = tempdf.loc[(tempdf.loc[:, (symbol, "close")] > slope * tempdf['row number'] + intercept)]
            if flag == 'lower':
              tempdf = tempdf.loc[(tempdf.loc[:, (symbol, "close")]< slope * tempdf['row number'] + intercept)]
        return slope, intercept
    def refineCurrentRangeTrendlines(self, symbol, flag, possibleTrendlineInterceptingPoints):
        if flag == 'upper':
           validationWindow = self.upperValidationWindow
        if flag == 'lower':
           validationWindow = self.validationWindowLower
        localPossibleTrendlineInterceptingPoints = possibleTrendlineInterceptingPoints
        i = 1
        while len(localPossibleTrendlineInterceptingPoints) > 0:
            self.currentTrendlineInterceptingPoint = int(localPossibleTrendlineInterceptingPoints[0])
            if self.currentTrendlineInterceptingPoint - self.previousTrendlineInterceptingPoint < validationWindow:
                self.currentTrendlineInterceptingPoint = len(self.data) + 1 - i
                i += 1
            currentdf = self.getCurrentRange()
            slope, intercept = self.currentRangeTrendlines(symbol, flag, currentdf)
            localPossibleTrendlineInterceptingPoints = self.trendlineInterceptingPoints(symbol, flag, currentdf, slope, intercept)
        self.refinedTrendlines[symbol][flag][str(self.previousTrendlineInterceptingPoint)] = {'slope': slope, 'intercept': intercept,'starting row': self.previousTrendlineInterceptingPoint, 'ending row': self.currentTrendlineInterceptingPoint - 1}
        self.previousTrendlineInterceptingPoint = self.currentTrendlineInterceptingPoint
        self.currentTrendlineInterceptingPoint = len(self.data) + 1
    def graphTrend(self):
        fig, axs = plt.subplots(len(self.symbols))
        pltCount = 0
        for symbol, values in self.refinedTrendlines.items():
            for flag, trendlines in values.items():
                for period, trendline in trendlines.items():
                    tempAxis = axs[pltCount] if len(self.symbols) > 1 else axs
                    tempAxis.plot(self.data.index,self.data.loc[:, (symbol, "close")], label=f'{symbol} price action')
                    tempAxis.plot(self.data.index.values[trendline['starting row']:trendline['ending row']],trendline['slope']*self.data['row number'][trendline['starting row']:trendline['ending row']] + trendline['intercept'], label=f'{self.data.index.values[trendline["starting row"]]}-{self.data.index.values[trendline["ending row"]-1]}')
            pltCount += 1
        chart = plt.show()
        st.pyplot(chart)

    def runTrendlineDetector(self, graph=True):
        for symbol in self.symbols:           
            self.refinedTrendlines[symbol] = {}
            for flag in self.flag:
                self.refinedTrendlines[symbol][flag] = {}
                while True:
                    currentRange = self.getCurrentRange()
                    if len(currentRange) <= 2: break
                    trendline = self.currentRangeTrendlines(symbol, flag, currentRange)
                    possibleTrendlineInterceptingPoints = self.trendlineInterceptingPoints(symbol, flag, currentRange, *trendline)
                    if len(possibleTrendlineInterceptingPoints) == 0:
                       self.refinedTrendlines[symbol][flag][str(self.previousTrendlineInterceptingPoint)] = {'slope': trendline[0], 'intercept': trendline[1], 'starting row': self.previousTrendlineInterceptingPoint, 'ending row': self.currentTrendlineInterceptingPoint - 1}
                       break
                    else:
                        self.refineCurrentRangeTrendlines(symbol, flag, possibleTrendlineInterceptingPoints)
                self.previousTrendlineInterceptingPoint = 0
                self.currentTrendlineInterceptingPoint = len(self.data) + 1
        if graph: self.graphTrend()
    
class channelDetector(trendlineDetector):
    def __init__(self, symbols, data):
        super().__init__(symbols, data, flag='all')
        self.channelValidationFrame = CHANNEL_VALIDATION_FRAME
        self.slopeDiffThreshold = SLOPE_DIFF_THRESHOLD*math.pi/180
        self.channelInterceptingThreshold = CHANNEL_INTERCEPTING_THRESHOLD/100
        self.channels = {}
    def runChannelDetector(self, graph=False):
        self.runTrendlineDetector(graph=False)
        trendlines = self.refinedTrendlines
        for symbol in self.symbols:
            self.channels[symbol] = {}
            upperTrends = trendlines[symbol]['upper']
            lowerTrends = trendlines[symbol]['lower']
            for ku, vu in upperTrends.items():
                uSlope, uIntercept, uStart, uEnd = vu['slope'], vu['intercept'], vu['starting row'], vu['ending row']
                for kl, vl in lowerTrends.items():
                    lSlope, lIntercept, lStart, lEnd = vl['slope'], vl['intercept'], vl['starting row'], vl['ending row']
                    if (np.abs(uSlope-lSlope) < self.slopeDiffThreshold) and (uIntercept-lIntercept > 0) and range(max(uStart, lStart), min(uEnd, lEnd)+1):
                       start = np.min((uStart, lStart))
                       end = np.max((uEnd, lEnd))
                       upperViolation = self.channelInterceptingPoints(symbol, uSlope, uIntercept, start, end, self.channelInterceptingThreshold, flag='upper')
                       lowerViolation = self.channelInterceptingPoints(symbol, lSlope, lIntercept, start, end, self.channelInterceptingThreshold, flag='lower')
                       if upperViolation.any() or lowerViolation.any():
                           continue
                       else:
                           channelId = f'{start}'
                           self.channels[symbol][channelId] = {}
                           self.channels[symbol][channelId]['start'] = start
                           self.channels[symbol][channelId]['end'] = end
                           self.channels[symbol][channelId]['slopeUpper'] = uSlope
                           self.channels[symbol][channelId]['slopeLower'] = lSlope
                           self.channels[symbol][channelId]['interceptUpper'] = uIntercept
                           self.channels[symbol][channelId]['interceptLower'] = lIntercept
                    else:
                        continue
        if graph == True:
            self.graphChannel()
    def channelInterceptingPoints(self, symbol, slope, intercept, start, end, tolerance, flag):
        if flag == 'upper':
            violationCondition = self.data.loc[:, (symbol,'close')].iloc[start:end] > (1 + tolerance)*(slope*self.data['row number'].iloc[start:end]+1.0*intercept)
        if flag == 'lower':
            violationCondition = self.data.loc[:, (symbol,'close')].iloc[start:end] < (1 - tolerance)*(slope*self.data['row number'].iloc[start:end]+1.0*intercept)
        return self.data.iloc[start:end].loc[violationCondition, 'row number']
    def graphChannel(self):
        fig, axs = plt.subplots(len(self.symbols))
        pltCount = 0
        for symbol, values in self.channels.items():
            tempAxis = axs[pltCount] if len(self.symbols) > 1 else axs
            tempAxis.plot(self.data.index,self.data.loc[:, (symbol, "close")], label=f'{symbol} price action') 
            for channelId, channelProperties in values.items():
                  tempAxis.plot(
self.data.index.values[channelProperties['start']:channelProperties['end']], channelProperties['slopeUpper']*self.data['row number'][channelProperties['start']:channelProperties['end']] + channelProperties['interceptUpper'])
                  tempAxis.plot(
self.data.index.values[channelProperties['start']:channelProperties['end']],channelProperties['slopeLower']*self.data['row number'][channelProperties['start']:channelProperties['end']] + channelProperties['interceptLower'])
            pltCount += 1
        chart = plt.show()
        st.pyplot(chart)  

if __name__ == '__main__':
    data = get_data(TICKER, START_DATE, END_DATE)
    st.header('Stock Chart Trendlines')
    ts = trendlineDetector([TICKER], data, flag='all')
    ts.runTrendlineDetector(graph = True)
    st.header('Detected Channel Patterns')
    cs = channelDetector([TICKER], data)
    cs.runChannelDetector(graph=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
