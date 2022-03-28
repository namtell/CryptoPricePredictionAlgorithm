import cryptocompare
coin = cryptocompare.get_coin_list(format=False)

#list the currencies you wish to analyze here
coin_top10 = ['Bitcoin', 'Ethereum ', 'Bitcoin Cash / BCC', 'Ripple', 'Litecoin', 'NEM', 'DigitalCash', 'ZCash','Monero', 'IOTA', 'Ethereum Classic', 'Nxt', 'Stellar']
coin_top10_sym = []
for key, value in coin.items():
    if value['CoinName'] in coin_top10:
        print(key, value['CoinName'])
        coin_top10_sym.append(key)
        
import pandas as pd
import requests
from pandas import json_normalize

#determine how many rows of data here
lim = '2000'

df = pd.DataFrame()
for i in coin_top10_sym:
    URL = 'https://min-api.cryptocompare.com/data/histohour?fsym='+i+'&tsym=USD&limit='+lim
    data = requests.get(URL)
    json_data = data.json()
    table = json_normalize(json_data, 'Data').set_index('time')
    table.index = pd.to_datetime(table.index ,unit='s')
    df = pd.concat([df, table.high], axis=1)
df.columns = coin_top10_sym

# Performing Dickey-Fuller stationary test

from statsmodels.tsa.stattools import adfuller

for i in df.columns: 
    x = df[i].values
    result = adfuller(x)
    print('\033[1m' + i + '\033[0m')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    
# Perform differencing to stationalize the series

# Creat difference function, with default value of lag 1
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# Differencing the dataset 
df_diff = pd.DataFrame()
for i in df.columns:
    df_diff[i] = difference(df[i])
    
for i in df.columns: 
    x = df_diff[i].values
    result = adfuller(x)
    print('\033[1m' + i + '\033[0m')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    
from itertools import permutations
com = list(permutations(coin_top10_sym, 2))
pair_list = []
count = 0
for i in com:
    globals()[str(i[0]) + '_' + str(i[1])] = df_diff[[com[count][0], com[count][1]]]
    count = count + 1
    pair_list.append(str(i[0]) + '_' + str(i[1]))
    
# Performing Granger Causality Test

import statsmodels.tsa.stattools as sm
lag = 24
cor = {}
for i in pair_list:
    cor[i] = sm.grangercausalitytests(eval(i), lag)
    
for key, values in cor.items():
    print('\n')
    print('\033[1m' + key + '\033[0m')
    for i in range(1, lag+1):
        print('lag', i, '=', values[i][0]['lrtest'][1])
        
# Printing top correlated coins results

# Manually going through the test results to identify pairs with lowest lag values
top = ['ETC_XMR', 'ETC_XEM', 'XEM_XMR', 'LTC_XMR', 'LTC_XEM', 'MIOTA_XMR', 'ZEC_XMR', 'BTC_XMR', 'BTC_XEM', 'XLM_XMR']
for key, values in cor.items():
    if key in top:
        print('\n')
        print('\033[1m' + key + '\033[0m')
        for i in range(1, lag+1):
            print('lag', i, '=', values[i][0]['lrtest'][1])
            
# Scaling and visualizing BTC_XMR

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
df_plot = pd.DataFrame(index=df.index)
df_plot['BTC'] = sc_x.fit_transform(df['BTC'].values.reshape(-1,1))
df_plot['XMR'] = sc_x.fit_transform(df['XMR'].values.reshape(-1,1))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.plot(df.index, df_plot['BTC'], color='blue')
plt.plot(df.index, df_plot['XMR'], color='green')
plt.ylim((-4, 4))
plt.legend(loc='lower left')
plt.xlabel('Time')
plt.ylabel('Prices (scaled)')
plt.title('BTC/XMR Prices over Time')
plt.show()

import cryptocompare
import requests
import json
import pandas as pd
from pandas import json_normalize
import datetime
from pytrends.request import TrendReq
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import demjson
import eventregistry


# Login to Google
pytrend = TrendReq()
# Get today date
today_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Getting price data of BTC and XMR using Cryptocompare API

# Indicate how many rows of data here
lim = '2000' 
pair = ['BTC', 'XMR']
coins = pd.DataFrame()
# Making API call, normalize JSON file, and put it in Dataframe
for i in pair:
    URL = 'https://min-api.cryptocompare.com/data/histohour?fsym='+i+'&tsym=USD&limit='+lim 
    data = requests.get(URL)
    json_data = data.json()
    table = json_normalize(json_data, 'Data').set_index('time') # Set index
    table.index = pd.to_datetime(table.index ,unit='s') # Make datetime object
    coins = pd.concat([coins, table.high], axis=1)
coins.columns = pair

from_date = coins.index[0].strftime('%Y-%m-%d') # Get date where the query started

# Importing market cap data already stored in csv file

mc = pd.read_csv('mc_coins.csv').set_index('date') # Set index
mc.index = pd.to_datetime(mc.index) # Make datetime object
mc = mc.resample('1h').pad() # Upsample from daily to hourly

# Query stock data from Yahoo! Financial using pandas_datareader

# Specify start and end time here
start = datetime.datetime(2022, 1, 4)
end = datetime.datetime(2022, 3, 24)

amd = web.DataReader('AMD', 'yahoo', start, end)
nvda = web.DataReader('NVDA', 'yahoo', start, end)

# Cleaning and resampling data
amd = amd.resample('1h').pad().drop(['Open', 'Low', 'Close', 'Adj Close', 'Volume'], axis='columns')
nvda = nvda.resample('1h').pad().drop(['Open', 'Low', 'Close', 'Adj Close', 'Volume'], axis='columns')
amd.columns = ['amd']
nvda.columns = ['nvda']

# Getting data from Google Trends using Pytrends API
 
# keyword = 'cryptocurrency', cathegory = 16 (news), timeframe- limit range to 8 months to get daily data
pytrend.build_payload(kw_list=['cryptocurrency'], cat=16, timeframe=from_date+' '+today_date) # Build payload
ggtrends_1 = pytrend.interest_over_time()
ggtrends_1 = ggtrends_1.resample('1h').pad().drop(['isPartial'], axis='columns') # Upsample daily to hourly
ggtrends_1.columns = ['gg_crypto']

# keyword = 'bitcoin price', cathegory = 0 (all), timeframe- limit range to 8 months to get daily data
pytrend.build_payload(kw_list=['bitcoin price'], cat=0, timeframe=from_date+' '+today_date)  
ggtrends_2 = pytrend.interest_over_time()
ggtrends_2 = ggtrends_2.resample('1h').pad().drop(['isPartial'], axis='columns')
ggtrends_2.columns = ['gg_bitcoin_p']

# keyword = 'monero price', cathegory = 0 (all), timeframe- limit range to 8 months to get daily data
pytrend.build_payload(kw_list=['monero price'], cat=0, timeframe=from_date+' '+today_date)  
ggtrends_3 = pytrend.interest_over_time()
ggtrends_3 = ggtrends_3.resample('1h').pad().drop(['isPartial'], axis='columns')
ggtrends_3.columns = ['gg_monero_p']

# keyword = 'bitcoin wallet', cathegory = 0 (all), timeframe- limit range to 8 months to get daily data
pytrend.build_payload(kw_list=['bitcoin wallet'], cat=0, timeframe=from_date+' '+today_date)  
ggtrends_4 = pytrend.interest_over_time()
ggtrends_4 = ggtrends_4.resample('1h').pad().drop(['isPartial'], axis='columns')
ggtrends_4.columns = ['gg_bitcoin_w']

# keyword = 'monero wallet', cathegory = 0 (all), timeframe- limit range to 8 months to get daily data
pytrend.build_payload(kw_list=['monero wallet'], cat=0, timeframe=from_date+' '+today_date)  
ggtrends_5 = pytrend.interest_over_time()
ggtrends_5 = ggtrends_5.resample('1h').pad().drop(['isPartial'], axis='columns')
ggtrends_5.columns = ['gg_monero_w']

# keyword = 'gpu mining', cathegory = 0 (all), timeframe- limit range to 8 months to get daily data
pytrend.build_payload(kw_list=['gpu mining'], cat=0, timeframe=from_date+' '+today_date)  
ggtrends_6 = pytrend.interest_over_time()
ggtrends_6 = ggtrends_6.resample('1h').pad().drop(['isPartial'], axis='columns')
ggtrends_6.columns = ['gg_gpu']

# Joining data frames
df = pd.concat([coins, amd, nvda, mc, ggtrends_1, ggtrends_2, ggtrends_3, ggtrends_4, ggtrends_5, ggtrends_6], axis=1).dropna(how='any')
df.to_csv('cap1_df.csv')

# Feature Scaling
df_scaled = df.copy(deep=True)

sc_x = StandardScaler()
df_scaled['BTC'] = sc_x.fit_transform(df_scaled['BTC'].values.reshape(-1,1))
df_scaled['XMR'] = sc_x.fit_transform(df_scaled['XMR'].values.reshape(-1,1))
df_scaled['mc_bitcoin'] = sc_x.fit_transform(df_scaled['mc_bitcoin'].values.reshape(-1,1))
df_scaled['mc_monero'] = sc_x.fit_transform(df_scaled['mc_monero'].values.reshape(-1,1))
df_scaled['gg_crypto'] = sc_x.fit_transform(df_scaled['gg_crypto'].values.reshape(-1,1))
df_scaled['gg_bitcoin_p'] = sc_x.fit_transform(df_scaled['gg_bitcoin_p'].values.reshape(-1,1))
df_scaled['gg_monero_p'] = sc_x.fit_transform(df_scaled['gg_monero_p'].values.reshape(-1,1))
df_scaled['gg_bitcoin_w'] = sc_x.fit_transform(df_scaled['gg_bitcoin_w'].values.reshape(-1,1))
df_scaled['gg_monero_w'] = sc_x.fit_transform(df_scaled['gg_monero_w'].values.reshape(-1,1))
df_scaled['gg_gpu'] = sc_x.fit_transform(df_scaled['gg_gpu'].values.reshape(-1,1))
df_scaled['amd'] = sc_x.fit_transform(df_scaled['amd'].values.reshape(-1,1))
df_scaled['nvda'] = sc_x.fit_transform(df_scaled['nvda'].values.reshape(-1,1))

plt.plot(df_scaled.index, df_scaled['BTC'], color='blue')
plt.plot(df_scaled.index, df_scaled['XMR'], color='green')
plt.show()

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# Import dataframe 

df = pd.read_csv('cap1_df.csv', index_col=0)

# Standardize the data

sc_x = StandardScaler()
df_scaled = pd.DataFrame(sc_x.fit_transform(df), index=df.index, columns=df.columns)

# Perform differencing to stationalize the series

# Creat difference function, with default value of lag 24
def difference(dataset, interval=24):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# Differencing the dataset
inter_d = 24
df_diff = pd.DataFrame(index=df.index)
for i in df.columns:
    data = difference(df_scaled[i], inter_d)
    data = pd.Series(np.append(np.repeat(np.nan, inter_d), data), index=df.index, name=i)
    df_diff = pd.concat([df_diff, data], ignore_index=False, axis=1)
    
# Perform Dickey-fuller test to test the differenced series for stationality

diff = df_diff.dropna()

for i in diff.columns: 
    x = diff[i].values
    result = adfuller(x)
    print('\033[1m' + i + '\033[0m')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    
# Splitting data to test/train sets

X = diff[['XMR', 'amd', 'nvda', 'gg_crypto','gg_bitcoin_p', 'gg_monero_p', 'gg_bitcoin_w', 'gg_monero_w','gg_gpu']]
#Roughly 80% of data in train and 20% in test
Ytrain = pd.DataFrame(diff[:1500]['BTC'])
Ytest = pd.DataFrame(diff[1500:]['BTC'])

# Performing ElasticNet Regression 

en = ElasticNet(alpha=0.1, normalize=False)
en.fit(X[:1500], Ytrain)
coef = list(en.coef_)
count = 0
for i in X.columns:
    print(i, ':', coef[count])
    count = count + 1
    
# Based on the ElasticNet results, manually identifying selected features (features with non-zero coefficients)

sig = ['XMR', 'gg_bitcoin_p', 'gg_monero_p', 'gg_gpu']

# Constructing dataframe

X_sig = pd.DataFrame(index=X.index, columns=[sig])
for i in sig:
    X_sig[i] = X[i]
    count = count + 1

# Save dataframe
X_sig[:1500].to_csv('X_train.csv')
X_sig[1500:].to_csv('X_test.csv')
Ytrain.to_csv('Y_train.csv')
Ytest.to_csv('Y_test.csv')

import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as sm
import statsmodels.tsa.arima_model as ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf
from pandas import tseries
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Import/clean dataframes

X_train = pd.read_csv('X_train.csv', index_col=0)
X_test = pd.read_csv('X_test.csv', index_col=0)
Y_train = pd.read_csv('Y_train.csv', index_col=0)
Y_test = pd.read_csv('Y_test.csv', index_col=0)
X = pd.concat([X_train, X_test])
Y = pd.concat([Y_train, Y_test])

X_train.index = pd.to_datetime(X_train.index)
X_test.index = pd.to_datetime(X_test.index)
Y_train.index = pd.to_datetime(Y_train.index)
Y_test.index = pd.to_datetime(Y_test.index)

# Plotting ACF and PACF

# ACF
plot_acf(Y)
plt.title('ACF', loc='center')
plt.xlim((0, 50))
plt.ylabel('ACF')
plt.xlabel('Lag(s)')
plt.show()

# PACF
plot_pacf(Y, lags=10)
plt.title('PACF', loc='center')
plt.ylabel('PACF')
plt.xlabel('Lag(s)')
plt.show()

# ARIMAX

# Building AR 2 with exogeneous variables
arima_d = ARIMA.ARIMA(endog=Y_train['BTC'], exog=X_train, order=[1,0,0])
arima_results_d = arima_d.fit()
print(arima_results_d.summary())

# Plotting the fitted values
plt.plot(np.arange(len(Y_train)), Y_train, color='purple', label='Actual')
plt.plot(np.arange(len(arima_results_d.fittedvalues)), arima_results_d.fittedvalues, color='green', label='Fitted Value')
plt.xlim((0, 400))
plt.title('ARIMAX Fitted Values')
plt.xlabel('Time')
plt.ylabel('Differenced Price')
plt.legend()
plt.show()
print(X_test)

# Out-of-sample prediction
exog_d = X_test
arima_results_ofs_d = arima_results_d.predict(exog=exog_d, start=exog_d.index[0], end=exog_d.index[-1])
plt.plot(np.arange(len(Y_test)), Y_test, color='purple', label='Actual')
plt.plot(np.arange(len(arima_results_ofs_d)), arima_results_ofs_d, color='green', label='Prediction')
#plt.xlim((0, 50))
plt.title('ARIMAX Out-of-Sample Prediction')
plt.xlabel('Time')
plt.ylabel('Differenced Price')
plt.legend()
plt.show()

# Metric

# Calculating mean squared error for steps 1-100
expected = Y_test[:99]
predictions = arima_results_ofs_d[:99]
mse = mean_squared_error(expected, predictions)
print('Mean Squared Error')
print('Steps 1-100: %f' % mse)

# Calculating mean squred error for steps 101-373
expected = Y_test[100:]
predictions = arima_results_ofs_d[100:]
mse = mean_squared_error(expected, predictions)
print('Steps 101-373: %f' % mse)
