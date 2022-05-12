#!/usr/bin/env python
# coding: utf-8

# # Markowitz Portfolio Optimization
# 
# From Wikipedia, the free encyclopedia
# 
# In finance, **the Markowitz model** ─ put forward by Harry Markowitz in 1952 ─ is a portfolio optimization model; it assists in the selection of the most efficient portfolio by analyzing various possible portfolios of the given securities. Here, by choosing securities that do not 'move' exactly together, the HM model shows investors how to reduce their risk. The HM model is also called mean-variance model due to the fact that it is based on expected returns (mean) and the standard deviation (variance) of the various portfolios. It is foundational to Modern portfolio theory.
# 
# 
# **References:**
# 
# Markowitz portfolio optimization: https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/
# 
# JPX dataset EDA: https://www.kaggle.com/code/anurag2405/jpx-tokyo-stock-exchange-prediction

# ## Markowitz is alive!!! ###
# 
# Harry Markowitz was the 1990 Nobel Memorial Prize winner in Economic Sciences.
# 
# I wanted to provide an introduction to Markowitz portfolio optimization which still remains relevant today.
# 
# The explanation of the methodology is simple: the optimization will select the weights of the portfolio which gives the minimum variance OR the maximum Sharpe Ratio. This idea can be used for ranking the stocks in order to reduce the standard deviation of the return of the day and get a better sharpe value.

# # Error message
# 
# For some reason I had to install scipy version 1.4.1
# 
# There is an error message about matrix dimension that I could not figure out how to solve. Internet connection needs to be on.
# 
# **ValueError: `f0` passed has more than 1 dimension.**

# In[ ]:


# For some reason I had to install scipy version 1.4.1
# There is an error message about matrix dimension that I could not figure out how to solve
# pip install --upgrade scipy==1.4.1


# In[ ]:


pip install --upgrade scipy==1.4.1


# In[ ]:


# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# import random
#
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import scipy.optimize as sco

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


stocks = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
df_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
df2_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")


# In[ ]:


stocks


# In[ ]:


stocks.info()


# In[ ]:


stocks.describe()


# In[ ]:


stocks.SecuritiesCode.nunique()


# In[ ]:


stocks_code = stocks.SecuritiesCode.unique()


# In[ ]:


stocks_code


# In[ ]:


# stocks["Section/Products"].value_counts()


# In[ ]:


# stocks = stocks[stocks['NewMarketSegment'].notna()]


# In[ ]:


# fig = px.pie(stocks,names="Section/Products", title='Stock Indices')
# fig.show()


# In[ ]:


# fig = px.pie(stocks,names="NewMarketSegment", title='Market Segment')
# fig.show()


# In[ ]:


# fig = px.pie(stocks,names="33SectorName", title='Sector')
# fig.show()


# ## Markowitz optimization and the Efficient Frontier ##
# 
# Let's bring Markowitz to this competition. 
# 
# It is said that it's better to select N stocks with weights equal to 1/N than trying to select stocks based on portfolio optimization. Hence, many improvements have been suggested to the model, but I'll stick to the basic model.

# In[ ]:


# Mean and covariance matrix calculated over the last 56 days. It could be changed...
# df2_prices['Close_lag1'] = df2_prices.groupby(['SecuritiesCode'])['Close'].shift(1)
df2_prices['Return'] = df2_prices.groupby(['SecuritiesCode'])['Close'].pct_change()
#------------------------------------------------------------------------------------
# Select only N assets. N = 2000 takes time (~ 20 min)
#------------------------------------------------------------------------------------
N_ASSETS = 200
sec_code = np.random.choice(stocks_code, size=N_ASSETS, replace=False)
df2_prices = df2_prices.loc[df2_prices['SecuritiesCode'].isin(sec_code)]
#------------------------------------------------------------------------------------
mean_ret = df2_prices.groupby(['SecuritiesCode'])['Return'].mean()
# Pivot table 
ret_pivot = df2_prices.pivot(index='Date', columns='SecuritiesCode', values='Return')


# In[ ]:


df2_prices


# In[ ]:


mean_ret


# In[ ]:


ret_pivot


# In[ ]:


# Covariance matrix
cov_ret = ret_pivot.cov()


# In[ ]:


cov_ret


# In[ ]:


def rand_weight(n):
    # Produces n random weights that sum to 1
    k = np.random.rand(n)
    # k = np.random.randint(-1, 1, n)
    return k / sum(k)


# In[ ]:



# Mean returns and coariance matrix of returns
ret_MEAN = np.asmatrix(mean_ret)
ret_COV = np.asmatrix(cov_ret)
# Generate one vector of random weights
w = np.asmatrix(rand_weight(mean_ret.shape[0]))
# Calculate return and std dev
mu = w * ret_MEAN.T
sigma = np.sqrt(w * ret_COV * w.T)


# In[ ]:


print(w)
print(ret_COV.shape)
print(ret_MEAN.shape)


# In[ ]:


def random_portfolio(ret_MEAN, ret_COV, n_assets):
    ''' 
    Returns the mean and standard deviation of returns for a portfolio
    '''
    w = np.asmatrix(rand_weight(n_assets))
    # Return
    mu = w * ret_MEAN.T 
    # STDev.
    sigma = np.sqrt(w * ret_COV * w.T)
    
    return mu, sigma


# In[ ]:


# Number of portfolios to simulate
n_portfolios = 1000
# Number of stocks (n = 2000)
n_assets = mean_ret.shape[0] 

means, stds = np.column_stack([
    random_portfolio(ret_MEAN, ret_COV, n_assets) 
    for _ in range(n_portfolios)
])


# In[ ]:


fig = plt.figure()
plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of random-weights generated portfolios')


# In[ ]:



'''
The above constraint is saying that sum of x should be equal to 1. You can think of the ‘fun’ part construction as ‘1’ on the right side of equal sign has been moved to the left side of the equal sign.
np.sum(x) == 1 has become np.sum(x)-1
And what does this mean? It simply means that the sum of all the weights should be equal to 1. You cannot allocate more than 100% of your budget in total.
“bounds” is giving another limit to assign random weights, by saying any weight should be inclusively between 0 and 1. You cannot give minus budget allocation to a stock or more than 100% allocation to a stock.
'''

def min_variance(mean_returns, cov_matrix):
    num_assets = mean_returns.shape[1] # len(mean_returns)
    args = (mean_returns, cov_matrix)
    # constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # bound = (0.0,1.0)
    # bounds = tuple(bound for asset in range(num_assets))
    
    # result = sco.minimize(x * cov_matrix * x.T, num_assets*[1./num_assets,], args=args)
    
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args)
                        # , method='SLSQP', bounds=bounds, constraints=constraints)

    return result

# min_vol = min_variance(p, C)


# In[ ]:


# Number of stocks (n = 2000 total)
n_assets = mean_ret.shape[0]

# Return random floats in the half-open interval [0.0, 1.0)
weights = np.random.random(size = n_assets) 
# Normalize to unity
# The /= operator divides the array by the sum of the array and rebinds "weights" to the new object
weights /= np.sum(weights) 

# returns = weights * mean_returns.T # np.sum(mean_returns*weights )
# std =  np.sqrt(weights * cov_matrix * weights.T) # np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Function for computing portfolio return
def portfolio_returns(weights):
    return weights * ret_MEAN.T # (np.sum(ret_MEAN * weights))

# Function for computing standard deviation of portfolio returns
def portfolio_sd(weights):
    return np.sqrt(weights @ ret_COV @ weights.T) # np.sqrt(np.transpose(weights) @ (ret_COV) @ weights)

# User defined Sharpe ratio function
# Negative sign to compute the negative value of Sharpe ratio ( - minimization -> maximization)
def sharpe_fun(weights):
    return - (portfolio_returns(weights) / portfolio_sd(weights))


# In[ ]:


x = portfolio_returns(weights)
y = portfolio_sd(weights)
z = sharpe_fun(weights)


# In[ ]:


# Portfolio Return
print(x)
# STDev of portfolio
print(y)
# Sharpe ratio
print(z)


# In[ ]:


# We use an anonymous lambda function
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# This creates n tuples of (0, 1), all of which exist within a container tuple
# We essentially create a sequence of (min, max) pairs
bounds = tuple(
  (0, 1) for w in weights
)

# Repeat the list with the value (1 / n) n times, and convert list to array
equal_weights = np.array(
    [1 / n_assets] * n_assets, ndmin = 2)

# Minimization results
max_sharpe_results = sco.minimize(
  # Objective function
  fun = sharpe_fun, 
  # Initial guess, which is the equal weight array
  x0 = equal_weights, 
  method = 'SLSQP',
  bounds = bounds, 
  constraints = constraints
)



# In[ ]:


# weights
print(max_sharpe_results['x'])
# max weight
print(np.max(max_sharpe_results['x']))
# min weight
print(np.min(max_sharpe_results['x']))
# Sum of weights should be equal to one
print(np.sum(max_sharpe_results['x']))


# In[ ]:


'''
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = weights * mean_returns.T # np.sum(mean_returns*weights )
    std =  np.sqrt(weights * cov_matrix * weights.T) # np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std, returns

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

# STD of returns
x = portfolio_volatility(w, ret_MEAN, ret_COV)
print(x)


def min_variance(mean_returns, cov_matrix):
    num_assets = mean_returns.shape[1] # len(mean_returns)
    args = (mean_returns, cov_matrix)
    # constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # bound = (0.0,1.0)
    # bounds = tuple(bound for asset in range(num_assets))
    
    # result = sco.minimize(x * cov_matrix * x.T, num_assets*[1./num_assets,], args=args)
    
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args)
                        # , method='SLSQP', bounds=bounds, constraints=constraints)

    return result

# min_vol = min_variance(p, C)
'''


# # Conclusion #
# 
# **Work in progress. **
# 
# Portfolio optimization could be used to reduce the variance and increase the Sharpe ratio.
