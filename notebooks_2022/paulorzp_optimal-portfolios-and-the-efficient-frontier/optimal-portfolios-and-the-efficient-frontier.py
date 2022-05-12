#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing necessary packages
import pandas as pd
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
from scipy.optimize import minimize, Bounds, LinearConstraint, linprog


# #### Load Train Set

# In[ ]:


# Loading Stock Returns
path = "../input/jpx-tokyo-stock-exchange-prediction/"
usecols = ["Date","SecuritiesCode","Target"]
df = pd.read_csv(f"{path}train_files/stock_prices.csv", usecols=usecols)
df = df[df.Date>"2021-08-01"]
df = df.pivot(index='Date', columns='SecuritiesCode', values='Target')
df = df.fillna(0)


# In[ ]:


def absHighPass(df, absThresh):
    c = df.columns.values
    a = np.abs(df.values)
    np.fill_diagonal(a, 0)
    i = np.where(a >= absThresh)[0]
    i = sorted(i)
    return df.loc[c[i],c[i]]

def absHigh(df, num):
    c = df.columns.values
    a = np.abs(df.values)
    np.fill_diagonal(a, 0)
    i = (-a).argpartition(num, axis=None)[:num]
    i, _ = np.unravel_index(i, a.shape)
    i = sorted(i)
    return df.loc[c[i],c[i]]

def selLow(df, num):
    c = df.columns.values
    a = df.values
    np.fill_diagonal(a, 0)
    i = (a).argpartition(num, axis=None)[:num]
    i, _ = np.unravel_index(i, a.shape)
    i = sorted(i)
    return df.loc[c[i],c[i]]

corr = df.corr()


# In[ ]:


mat = absHigh(corr,8)
mask = np.triu(np.ones_like(mat))
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_title("High Correlations", fontsize = 24)
sns.heatmap(mat, annot=True, mask=mask, cmap="viridis")
plt.show();


# In[ ]:


mat = selLow(corr,10)
mask = np.triu(np.ones_like(mat))
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_title("Low Correlations", fontsize = 24)
sns.heatmap(mat, annot=True, mask=mask, cmap="viridis")
plt.show();


# **Calculating the Efficient Frontier**
# 
# There are 2 inputs we must compute before finding the Efficient Frontier for our stocks: annualized rate of return and covariance matrix.
# 
# Annualized rate of return is calculated by multiplying the daily percentage change for all of the stocks with the number of business days each year (252).

# In[ ]:


# Calculate annualized average return for each asset
# Annualized average return = Daily average return * 252 business days.
ra = np.mean(df,axis=0)*252

# Create a covariance matrix
covar = df.cov()*252

# Calculate annualized volatility for each asset
vols = np.sqrt(252)*df.std()

# Create weights array
weights = np.concatenate([np.linspace(start=2, stop=1, num=200),
                          np.zeros(1600),
                          np.linspace(start=-1, stop=-2, num=200)])

# Calculate Sharpe Ratio for each asset
sr = (ra/vols).reset_index().rename(columns={0: 'SR'})
sr['Rank0'] = sr["SR"].rank(method="first",ascending=False).astype('int')-1
sr = sr.sort_values('Rank0')
sr['weights'] = weights

#Select Top and Botton Sharpe Ratios
Top200SR = sr.SecuritiesCode.values[:200]
Bot200SR = sr.SecuritiesCode.values[-200:]


# #### Top and Botton Sharpe Ratios Assets

# In[ ]:


Top200SR, Bot200SR


# In[ ]:


# Covariance matrix annualized of Top Sharpe Ratios
cov_port = df[Top200SR].cov()*252
# Returns annualized
ret_port = np.mean(df[Top200SR],axis=0)*252


# Next, we should define some functions that we will use later in our calculation.
# 
# * Rate of return is the annualized rate of return for the whole portfolio.
# * Volatility is the risk level, defined as the standard diviation of return.
# * Sharpe ratio is risk efficiency; it assesses the return of an investment compared to its risk.

# In[ ]:


#Define frequently used functions.
# r is each stock's return, 
# w is the portion of each stock in our portfolio, 
# c is the covariance matrix

from numba import jit

# Rate of return
@jit(forceobj=True)
def ret(r,w):
    return r.dot(w)

# Risk level or volatility
@jit(forceobj=True)
def vol(w,c):
    return np.sqrt(np.dot(w,np.dot(w,c)))

@jit(forceobj=True)
def sample_opt(c, r, w):
    # Round Expected volatility
    _vi = int(vol(w, c)*1e5)/1e5
    # Round Expected return
    _ri = int(ret(r, w)*1e4)/1e4
    return (_vi,_ri)


# #### Optimizing Risk and Sharpe Ratio

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n#Create x0, the first guess at the values of each asset's weight.\nw0 = np.linspace(start=1, stop=0, num=cov_port.shape[1])\nx0 = w0/np.sum(w0)\n# All weights between 0 and 1\nbounds = Bounds(0, 1)\n# The second boundary is the sum of weights.\nlinear_constraint = LinearConstraint(np.ones((cov_port.shape[1],), dtype=int),1,1)\noptions = {'xtol': 1e-07, 'gtol': 1e-07, 'barrier_tol': 1e-07, 'maxiter': 1000}\n\n# Find a portfolio with the minimum risk.\ndef min_risk(_cov):  \n    #Define a function to calculate volatility\n    fvol = lambda w: np.sqrt(np.dot(w,np.dot(w,_cov)))\n    res = minimize(fvol,x0,method='trust-constr', \n                   constraints=linear_constraint, \n                   bounds=bounds)\n    return res.x\n\n# Find a portfolio with the highest Sharpe Ratio.\ndef max_sr(_ret,_cov):\n    #Define 1/Sharpe_ratio\n    isharpe = lambda w: np.sqrt(np.dot(w,np.dot(w,_cov)))/_ret.dot(w)\n    res = minimize(isharpe,x0,method='trust-constr',\n                          constraints = linear_constraint,\n                          bounds = bounds,\n                          options = options)\n    return res.x\n \n\n#These are the weights of the assets in the portfolio with the lowest level of risk possible.\nw_minr = min_risk(cov_port)\nopt_risk_ret = ret(ret_port,w_minr)\nopt_risk_vol = vol(w_minr,cov_port)\nprint(f'Min. Risk = {opt_risk_vol*100:.3f}% => Return: {(opt_risk_ret*100):.3f}%  Sharpe Ratio = {opt_risk_ret/opt_risk_vol:.2f}')\n\n#These are the weights of the assets in the portfolio with the highest Sharpe ratio.\nw_sr_top = max_sr(ret_port,cov_port)\nopt_sr_ret = ret(ret_port,w_sr_top)\nopt_sr_vol = vol(w_sr_top,cov_port)\nprint(f'Max. Sharpe Ratio = {opt_sr_ret/opt_sr_vol:.2f} => Return: {(opt_sr_ret*100):.2f}%  Risk: {opt_sr_vol*100:.3f}%')")


# #### Efficient Frontier Optimizer

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Efficient Frontier Optimizer\n\nfrontier_y = np.linspace(opt_risk_ret*.35, opt_sr_ret*1.2, 50)\nfrontier_x = []\nsr_opt_set = set()\n\nx0 = w_sr_top\nbounds = Bounds(0, 1)\n\n@jit(forceobj=True)\ndef callbackF(w):\n    global sr_opt_set, ret_port, cov_port\n    sr_opt_set.add( sample_opt(cov_port, ret_port, w) )\n\n@jit(forceobj=True)\ndef check_sum(w):\n    #return 0 if sum of the weights is 1\n    return np.sum(w)-1\n\nfor possible_return in frontier_y:\n    cons = ({'type':'eq', 'fun': check_sum},\n            {'type':'eq', 'fun': lambda w: ret(ret_port, w) - possible_return})\n\n    #Define a function to calculate volatility\n    fun = lambda w: np.sqrt(np.dot(w,np.dot(w,cov_port)))\n    result = minimize(fun,x0,method='SLSQP', bounds=bounds, constraints=cons, callback=callbackF)\n    frontier_x.append(result['fun'])\n\nfrontier_x = np.array(frontier_x)\ndt_plot = pd.DataFrame(sr_opt_set, columns=['vol', 'ret'])\nvol_opt = dt_plot['vol'].values\nret_opt = dt_plot['ret'].values\nsharpe_opt = ret_opt/vol_opt")


# In[ ]:


# Plot Efficient Frontier

triang = tri.Triangulation(vol_opt, ret_opt)
triang.set_mask(np.hypot(vol_opt[triang.triangles].mean(axis=1),
                         ret_opt[triang.triangles].mean(axis=1))<.01)

fig1 = plt.figure(figsize=(16,6))
ax1 = fig1.add_subplot(111)
tcf = ax1.tricontourf(triang, sharpe_opt)
fig1.colorbar(tcf, label='Sharpe Ratio')
ax1.tricontour(triang, sharpe_opt, colors=None)
plt.xlim([frontier_x.min()-0.01,frontier_x.max()-0.012])
plt.title('Efficient Frontier', fontsize=24)
plt.xlabel('Risk/Volatility')
plt.ylabel('Return')
plt.plot(opt_sr_vol, opt_sr_ret,'r*', markersize=20, label='Highest Sharpe Ratio') # red star
plt.plot(opt_risk_vol,  opt_risk_ret, 'ro', markersize=12, label='Minimum Risk') # red dot
plt.plot(frontier_x, frontier_y, 'r--', linewidth=3, label='Efficient Frontier') # red dashed line
plt.legend(loc="upper left", frameon=False)
plt.show();


# #### Optimize Sharpe Ratio for Top and Botton Assets

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Optimize Sharpe Ratio for Top and Botton Assets\n\n# Covariance matrix annualized of Sharpe Ratios\ncov_port = df[np.concatenate([Top200SR,Bot200SR])].cov()*252\n# Returns annualized\nret_port = np.mean(df[np.concatenate([Top200SR,Bot200SR])],axis=0)*252\n\n#Create x0, the first guess at the values of each asset's weight.\nx0 = np.linspace(start=1, stop=-1, num=cov_port.shape[1])\n\n# All weights between -1 and 1\nbounds = Bounds(-1, 1)\n\n# The second boundary is the sum of weights.\nlinear_constraint = LinearConstraint(np.ones((cov_port.shape[1],), dtype=int),0,0)\n\n#These are the weights of the asset in the portfolio.\nw_sr_port = max_sr(ret_port,cov_port)\nw_sr_all = np.concatenate([w_sr_port[:200],np.ones(1600)*np.abs(w_sr_port).min(), w_sr_port[-200:]])\nopt_ret = ret(ret_port,w_sr_port)\nopt_vol = vol(w_sr_port,cov_port)\nprint(f'Sharpe Ratio = {opt_ret/opt_vol:.2f} Risk = {opt_vol*100:.7f}% => Return: {(opt_ret*100):.3f}%')")


# #### Save Results

# In[ ]:


sr['w_opt'] = w_sr_all
sr = sr.sort_values('SecuritiesCode')
sr['Target'] = ra.values*sr['w_opt']
sr['Rank'] = sr['w_opt'].rank(method='first',ascending=False).astype('int')-1
rank = sr.set_index('SecuritiesCode')['Rank'].to_dict()
trgt = sr.set_index('SecuritiesCode')['Target'].to_dict()
sr.to_csv("sharpe_ratio_opt.csv")


# In[ ]:


# Utilities 

def calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
    weights_mean = weights.mean()
    df = df.sort_values(by='Rank')
    purchase = (df['Target'][:portfolio_size]  * weights).sum() / weights_mean
    short    = (df['Target'][-portfolio_size:] * weights[::-1]).sum() / weights_mean
    return purchase - short

def calc_spread_return_sharpe(df, portfolio_size=200, toprank_weight_ratio=2):
    grp = df.groupby('Date')
    min_size = grp["Target"].count().min()
    if min_size<2*portfolio_size:
        portfolio_size=min_size//2
        if portfolio_size<1:
            return 0, None
    buf = grp.apply(calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio, buf

def add_rank(df, col_name="pred"):
    df["Rank"] = df.groupby("Date")[col_name].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df


# #### Preview LB Score

# In[ ]:


# Preview LB Score
sub = pd.read_csv(f"{path}supplemental_files/stock_prices.csv", usecols=usecols)
sub["Rank"] = sub["SecuritiesCode"].map(rank)
print("Score =" ,calc_spread_return_sharpe(sub)[0])


# #### Submission

# In[ ]:


import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (prices, _, _, _, _, submission) in iter_test:
    submission["Rank"] = submission["SecuritiesCode"].map(rank)
    env.predict(submission)

