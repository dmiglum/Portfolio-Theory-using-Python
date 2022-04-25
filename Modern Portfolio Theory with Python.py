# Modern Portfolio Theory with Python

import pandas_datareader.data as web
import datetime

import calendar

import numpy as np, pandas as pd

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# list of tickers required from yahoo finance
tickers = ['^SP500TR','EEM','IEF' , 'AGG','^BCOM']

start = datetime.datetime(2011, 9, 29)

end = datetime.datetime(2021, 10, 1)
data = web.DataReader(tickers, 'yahoo', start, end)

# use adjusted close as daily price
data = data['Adj Close']
data.head(3)

# get dividends for AGG
div = web.DataReader('AGG', 'yahoo-dividends', start, end)
div.head(3)

# Extract Long Volatility
# percentage change is already in excel, so we can skip one step
data_lv = pd.read_excel('long-vol.xlsx')  
data_lv.columns = data_lv.iloc[2] #setting column names
data_lv = data_lv[3:].set_index('ReturnDate')['Index'] #setting 'ReturnDate' as index and keeping ['Index'] column

# set date as index
data_lv.index = pd.to_datetime(data_lv.index)

# upsample month returns to daily return by averaging
data_lv = data_lv.resample('24h').ffill()
days = [calendar.monthrange(idx.year, idx.month)[1] for idx, x in data_lv.iteritems()]
data_lv = data_lv/days

# Extracting Gold Data
data_gold = pd.read_excel('gold.xlsx', sheet_name = 'Daily_Indexed')
data_gold = data_gold[['Name', 'US dollar']].set_index('Name')
data_gold.dropna(inplace = True)

# Combining data
data = pd.concat([data, div['value']], axis = 1)
data.AGG = data.AGG + data.value.fillna(0)
data.drop(columns = 'value', inplace= True)

# combine yahoo, bcom, gold
df = pd.concat([data, data_gold, data_lv], axis = 1).dropna()
df.columns = ['SNP', 'EEM', 'IEF', 'AGG', 'BCM', 'GLD', 'LOV']

#find % difference and log difference
df = df.pct_change().apply(lambda x: np.log(1+x))
df.head(3)

# compile price data only
df_px = pd.concat([data,data_gold,data_lv],axis=1).dropna()
df_px.columns = ['SNP', 'EEM', 'IEF', 'AGG', 'BCM', 'GLD', 'LOV']

# Calculating Variance
df.var() * 365.25

# Calculating Standard deviation (volatility)
sd = df.std()*np.sqrt(365.25)
sd.plot(kind='bar')
plt.title('Standard Deviation (Volatility)')
''' 
Volatility observations 
Fixed income (i.e. IEF, AGG ) have the lowest. Although interesting that investment 
grade bonds have lower vol than EEM however the dividends are distributed
Commodities (BCM, LOV) have twice as much. Equity has 3 times as much as fixed income
'''
# create the covariance matrix. 
# this will be used later in portfolio construction
cov_matrix = df.cov()
cov_matrix
sns.heatmap(df.corr(), annot = True)
'''
Correlation observations
Equities (SNP, EEM) have a high correlation
Equities (SNP, EEM) have a slight negative correlation to the risk-free return (IEF) (? due to substitution effect?)
Fixed Income (IEF, AGG) have a correlation
Fixed Income (IEF, AGG) have no to negative correclation with commodities
Commodities and gold have a small correlation
long vol is least correlated to all portfolios
'''

# Expected Return
# get average daily return times 365.25 in the year that is extracted from dataset
e_r = df.mean()*365.25

# plot exp return
e_r.plot(kind='bar')
plt.title('Expected Annual Return');
'''
Return Observations
SNP highest, Commodities lowest.
Risk-free rate(IEF) is still higher than investment grade bonds
long-vol still outperforms commodities
although EEM has the highest vol it doesnt have the highest returns which should not be expected because higher volatility or risk is associated with more returns
'''
# Positive and negative months
for col in df.columns:
    new_col = col + '_sign'
    df[new_col] = np.sign(df[col])

signs = []
for col in df.columns:
    if '_sign' in col:
        signs.append(df[col].value_counts(normalize = True))

df_signs = pd.concat(signs, axis = 1)

# MPT Portfolio Construction
'''
For Modern Portfolio Theory there are a few assumptions about investor's objective:
1) maximise returns
2) minimise risk / volatility / standard deviation
    
In order to make an optimal portfolio is the maximum return for a given level of risk. The following steps is used to determine the optimal portfolio given our constrained assets above.
1) initiate no. of assets and portfolios to simulate.
2) for each random portfolio (created with random weights)
3) save the return - using weighted average to determine
4) save the volatility - using covariance matrix multiplied by associated weights.
'''

p_ret = []
p_weights = []
p_vol = []
# set number of assets and portfolios to simulate
num_assets = len(df_px.columns)
num_portfolios = 1000  #in an article, 10,000 portfolios are used - but that takes too much time to compute

# for each portfolio find a return and volatility
for portfolio in range(num_portfolios):
    # create random weights
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    
    returns = np.dot(weights, e_r)
    p_ret.append(returns)
    
    var = cov_matrix.mul(weights, axis = 0).mul(weights, axis = 1).sum().sum()
    ann_sd = np.sqrt(var * 365.25)
    p_vol.append(ann_sd)

data = {'returns': p_ret, 'volatility': p_vol}

for counter, symbol in enumerate(df_px.columns.tolist()): #adding a weight of each asset in each portfolio
    data[symbol + '_weight'] = [w[counter] for w in p_weights]

portfolios = pd.DataFrame(data)

# Finding portfolio with lowest volatility and plotting results
min_var_port = portfolios.loc[portfolios['volatility'].idxmin()] #idxmin gives index of the minimum value

portfolios.plot.scatter(x='volatility',y='returns',grid=True, marker='o', s=10, alpha=0.3,figsize=[10,10])
plt.scatter(x=min_var_port[1],y=min_var_port[0], color='r', marker='*', s=500)
plt.title('Efficient Frontier and least risky portfolio');

# Optimal Risky Portfolio    
'''
Sharpe Ratio = (E(R_i) - rf) / sigma_i
Sharpe Ratio is the ratio of return to volatility. The larger the ratio, the more return per unit vol.
rf is the risk free rate of 0.08% since the time horizon for each portfolio is 1 month before the portfolio is re-adjusted
'''
rf = np.log(1+.0008)

portfolios['sharpe'] = (portfolios['returns'] - rf) / portfolios['volatility']
sharpe_max = portfolios.loc[portfolios['sharpe'].idxmax()]

# plot results
portfolios.plot.scatter(x='volatility',y='returns',grid=True, marker='o', s=10, alpha=0.3,figsize=[10,10])
plt.scatter(x=min_var_port[1],y=min_var_port[0], color='r', marker='*', s=500);
plt.scatter(x=sharpe_max[1],y=sharpe_max[0], color='y', marker='*', s=500);
plt.title('Efficient Frontier and least risky / max sharpe portfolio');

# Capital Allocation Line, Utility function, and Optimised Portfolio
'''
Capital allocation line:  E(R_p) = rf + ( (E(R_i) - rf) / sigma_i ) * sigma_p
This line implies all the possible allocations of portfolio. i.e. different combinations of 
the sharpe optimised portfolio and a risk-free asset (e.g. T-bill) (from risk free to most risk).

There is a linear relationship in returns, because the portfolio only has 2 components and as the 
risky portfolio weight decreases the returns decrease monotonously. Linear relationship in sd due 
to the risk-free having 0 vol so the decrease in volatility from the optimal portfolio to risk free is monotonous.

Utility Function:  U = E(R) - 0.5A*(sigma^2)
This function is from an economic model. Higher return leads to higher utility. A is the coefficient of 
risk aversion. If A is small then we are less risk averse. We assume 25 < A < 35.
Expected return is the level of utility and is discounted by level of risk aversion. Here we will use 35 as a conservative level

Final optimised portfolio
To come up with the final optimised portfolio below, we find the intersection of the Capital allocation line and Utility function
Assumptions: 1) investors always want to reduce risk, 2) investors want the greatest return for risk
'''
cal_x = []
cal_y = []
utility = []
a = 35
max_returns = portfolios.returns.max()

for er in np.linspace(rf, max_returns, 1000):
    sd = (er - rf) / ((sharpe_max[0] - rf)/sharpe_max[1])
    u = er - .5*a*(sd**2)
    cal_x.append(sd)
    cal_y.append(er)
    utility.append(u)

data2 = {'utility': utility, 'cal_y':cal_y, 'cal_x': cal_x} #creating a dictionary
cal = pd.DataFrame(data2)  #converting dictionary to a DataFrame
cal.head()

investors_port = cal.iloc[cal['utility'].idxmax()]

portfolios.plot.scatter(x='volatility',y='returns',grid=True, marker='o', s=10, alpha=0.3,figsize=[10,10])
plt.scatter(x=min_var_port[1],y=min_var_port[0], color='r', marker='*', s=500)
plt.scatter(x=sharpe_max[1],y=sharpe_max[0], color='y', marker='*', s=500)
plt.plot(cal_x, cal_y, color='purple')
plt.plot(investors_port[2], investors_port[1], '*', color='lightgreen')
plt.title('Efficient Frontier and least risky / max sharpe portfolio / max utility portfolio');

# Utility Adjusted Portfolio
'''
The point on the CAL is a weighted combination of complete sharpe optimised or rf (cash) 
portfolios. The final portfolio mix is found by the ratio of return vs the sharpe optimised
Then then compare the returns of the Capital Allocation Line for the same risk.
'''
# find weight of sharpe optimised portfolio
pct_risk = investors_port[2]/sharpe_max[1]

# find final returns, vol, weights of 
# sharpe optimised portfolio components
risk = sharpe_max[2:]*pct_risk

# find final returns, vol, weights of rf
risk_free = pd.Series([(1-pct_risk)], index=['Cash'])

port_fin = pd.concat([investors_port,risk,risk_free], axis=0).rename({'cal_y':'returns', 'cal_x':'volatility'})
port_fin

# Validating greatest utility, return, volatility
'''
There is alternative to using the Capital Allocation Line. Here, the utility for each
portfolio can be calculated, solving for the maximum utility.
It can be seen the CAL allocates with higher return with even lower volatility. Also the 
point on the Efficient Frontier does not maximise return to volatility.
'''
portfolios['utility'] = portfolios['returns'] - .5*a*(portfolios['volatility']**2)
portfolios.loc[portfolios['utility'].idxmax()]
cal.utility.describe()
