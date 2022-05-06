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
st.title('StockBot - Stock Chart Analyzer')

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

# Display Stock OHLC Data
df = data.DataReader(TICKER, 'yahoo', START_DATE, END_DATE)
st.subheader(f'Data of {TICKER} stock')
st.write(df)

st.subheader("Analysis of different types of stock charts")

# Interactive Candlestick Chart
st.subheader('Candlestick Chart')
fig = go.Figure(data = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
st.plotly_chart(fig)

# Closing Price vs Time Chart
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('white')
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('white')
plt.plot(ma100,'red')
plt.plot(df.Close,'blue')
st.pyplot(fig)

st.subheader('Closing Price vs time chart with 100MA and 200MA')
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
st.subheader('Predicted Price vs Original Price')

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
st.subheader('Resistance and Support Levels')

# get stock prices using yfinance library
def get_stock_price(symbol):   
    df = yf.download(symbol, START_DATE, threads= False)
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
st.subheader('Channel Pattern Analysis')

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
class trendlineSeeker:  
    def __init__(self, symbols, data, flag='all'):
        self.symbols = symbols
        self.start = START_DATE
        self.end = END_DATE
        self.validationWindowUpper = 48
        self.validationWindowLower = 48
        self.distortionThresholdUpper = 0.15
        self.distortionThresholdLower = 0.15
  
        if flag == 'all':
            self.flag = ['upper', 'lower']
        else:
            self.flag = [flag]
        self.data = data
        self.previousTrendlineBreakingPoint = 0
        self.currentTrendlineBreakingPoint = len(self.data) + 1
        self.refinedTrendlines = {}
    def getCurrentSymbol(self):
        for symbol in self.symbols:
            yield symbol
    def getCurrentRange(self):
        df = self.data.copy().loc[(self.data['row number'] >= self.previousTrendlineBreakingPoint) & (self.data['row number'] < self.currentTrendlineBreakingPoint), :]
        return df
    def trendlineBreakingPoints(self, symbol, flag, currentdf, slope, intercept):
        if flag == 'upper':
           distortionThreshold = self.distortionThresholdUpper
        if flag == 'lower':
           distortionThreshold = self.distortionThresholdLower
        possibleTrendBreakingPoints = currentdf.loc[(currentdf.loc[:, (symbol, "close")] < (1-distortionThreshold)*(slope*currentdf['row number'] + intercept)) | (currentdf.loc[:, (symbol, "close")] > (1+distortionThreshold)*(slope*currentdf['row number'] + intercept)), 'row number']
        return possibleTrendBreakingPoints
    def trendlineForCurrentRange(self, symbol, flag, currentdf):
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
    def refineTrendlineForCurrentRange(self, symbol, flag, possibleTrendlineBreakingPoints):
        if flag == 'upper':
           validationWindow = self.validationWindowUpper
        if flag == 'lower':
           validationWindow = self.validationWindowLower
        localPossibleTrendlineBreakingPoints = possibleTrendlineBreakingPoints
        i = 1
        while len(localPossibleTrendlineBreakingPoints) > 0:
            self.currentTrendlineBreakingPoint = int(localPossibleTrendlineBreakingPoints[0])
            if self.currentTrendlineBreakingPoint - self.previousTrendlineBreakingPoint < validationWindow:
                self.currentTrendlineBreakingPoint = len(self.data) + 1 - i
                i += 1
            currentdf = self.getCurrentRange()
            slope, intercept = self.trendlineForCurrentRange(symbol, flag, currentdf)
            localPossibleTrendlineBreakingPoints = self.trendlineBreakingPoints(symbol, flag, currentdf, slope, intercept)
        self.refinedTrendlines[symbol][flag][str(self.previousTrendlineBreakingPoint)] = {'slope': slope, 'intercept': intercept,'starting row': self.previousTrendlineBreakingPoint, 'ending row': self.currentTrendlineBreakingPoint - 1}
        self.previousTrendlineBreakingPoint = self.currentTrendlineBreakingPoint
        self.currentTrendlineBreakingPoint = len(self.data) + 1
    def vizTrend(self):
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

    def runTrendlineSeeker(self, viz=True):
        # self.get_data()
        for symbol in self.symbols:
            # print("symbol -", symbol)            
            self.refinedTrendlines[symbol] = {}
            # print("refinedTrendlines[symbol] -", self.refinedTrendlines[symbol])
            for flag in self.flag:
                self.refinedTrendlines[symbol][flag] = {}
                # print("refinedTrendlines[symbol][flag] -", self.refinedTrendlines[symbol][flag])
                while True:
                    currentRange = self.getCurrentRange()
                    # print("currentRange -", currentRange)
                    if len(currentRange) <= 2: break
                    trendline = self.trendlineForCurrentRange(symbol, flag, currentRange)
                    # print("trendline -", trendline)
                    possibleTrendlineBreakingPoints = self.trendlineBreakingPoints(symbol, flag, currentRange, *trendline)
                    # print("possibleTrendlineBreakingPoints -", possibleTrendlineBreakingPoints)
                    if len(possibleTrendlineBreakingPoints) == 0:
                       self.refinedTrendlines[symbol][flag][str(self.previousTrendlineBreakingPoint)] = {'slope': trendline[0], 'intercept': trendline[1], 'starting row': self.previousTrendlineBreakingPoint, 'ending row': self.currentTrendlineBreakingPoint - 1}
                       break
                    else:
                        self.refineTrendlineForCurrentRange(symbol, flag, possibleTrendlineBreakingPoints)
                self.previousTrendlineBreakingPoint = 0
                self.currentTrendlineBreakingPoint = len(self.data) + 1
        if viz: self.vizTrend()
    
class channelSeeker(trendlineSeeker):
    def __init__(self, symbols, data):
        super().__init__(symbols, data, flag='all')
        self.channelValidationWindow = 48
        self.slopeDiffThreshold = 8*math.pi/180
        self.channelBreakingThreshold = 0.1
        self.channels = {}
    def runChannelSeeker(self, viz=False):
        self.runTrendlineSeeker(viz=False)
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
                       upperViolation = self.channelBreakingPoints(symbol, uSlope, uIntercept, start, end, self.channelBreakingThreshold, flag='upper')
                       lowerViolation = self.channelBreakingPoints(symbol, lSlope, lIntercept, start, end, self.channelBreakingThreshold, flag='lower')
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
        if viz == True:
            # print(self.channels)
            self.vizChannel()
    def channelBreakingPoints(self, symbol, slope, intercept, start, end, tolerance, flag):
        if flag == 'upper':
            violationCondition = self.data.loc[:, (symbol,'close')].iloc[start:end] > (1 + tolerance)*(slope*self.data['row number'].iloc[start:end]+1.0*intercept)
        if flag == 'lower':
            violationCondition = self.data.loc[:, (symbol,'close')].iloc[start:end] < (1 - tolerance)*(slope*self.data['row number'].iloc[start:end]+1.0*intercept)
        # print(self.data.loc[:, (symbol, 'close')].iloc[start:end]-(1 - tolerance)*(slope*self.data['row number'].iloc[start:end]+1.0*intercept))
        return self.data.iloc[start:end].loc[violationCondition, 'row number']
    def vizChannel(self):
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
    st.subheader('Stock Chart Trendlines')
    ts = trendlineSeeker([TICKER], data, flag='all')
    ts.runTrendlineSeeker(viz = True)
    st.subheader('Detected Channel Patterns')
    cs = channelSeeker([TICKER], data)
    cs.runChannelSeeker(viz=True)
st.set_option('deprecation.showPyplotGlobalUse', False)