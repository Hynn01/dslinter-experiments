#!/usr/bin/env python
# coding: utf-8

# # Be careful! Target is not what it claims to be!

# ### In Data description it is claimed:
# > #### "*You can calculate the Target column from the Close column; it's the return from buying a stock the next day and selling the day after that.*"     
# 

# #### However, the direct calculations of return from the Close column tells us that **more than 2% of Target in train set rejects this claim**.

# ## Let's check it

# ----------------------------

# ### 1. Calculations

# In[ ]:


import numpy as np 
import pandas as pd 


# #### Firstly, let's look at the stock_prices data from train_files folder

# In[ ]:


df = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
df.head()


# We need Date, SecuritiesCode, and Close columns only to check the claim 

# #### I've prepared a simple function to calculate the Target column by it's definition from the Data description

# In[ ]:


def pivot_pct_calculation(data: pd.DataFrame, periods: int = 1, shift: int = 0, dropna: bool = True) -> pd.DataFrame:
    td = pd.pivot_table(df, index='Date', columns='SecuritiesCode', values='Close', dropna=dropna)
    td = td.pct_change(periods)
    return td.shift(shift)


# It must be considered, that for normal calculation of the daily return we should use the default value shift=0.      
# However, since we have to calculate the return of 'next day after tomorrow', we use shift=-2.

# #### Let us apply it for our data:

# In[ ]:


pct_df = pivot_pct_calculation(df, shift=-2, dropna=False)
pct_df.head()


# #### From this table we can spot the shifted return for each security for each day

# ### 2. For now we have our calculations and need to compare our calculated values with givens ones

# #### To do that we have to unpivot our table and bring it back to the initial look. Then,don't forget to drop NA Target values and reset index to move Date to the columns

# In[ ]:


calculated_target = pd.melt(pct_df, ignore_index=False, value_name='Target').reset_index().sort_values(['Date','SecuritiesCode']).dropna(subset=['Target']).reset_index(drop=True)
print(f'Calculated_target shape: {calculated_target.shape}')
calculated_target.head()


# #### Also we have to drop from the table that was provided for us in the train dataset the values we couldn't calculate. It is clear that we couldn't calculate the return for days we don't have Close price yet.

# In[ ]:


given_target = df.loc[df.Date <= calculated_target.Date.max(), ['Date', 'SecuritiesCode', 'Target']].sort_values(['Date','SecuritiesCode']).dropna(subset=['Target']).reset_index(drop=True)
print(f'Given_target shape: {given_target.shape}')
given_target.head()


# #### On this stage it sounds like the shapes of our dataframes aren't equal:

# In[ ]:


calculated_target.shape[0] == given_target.shape[0]


# It occurs because not all securities started tradind the first day of our observation. However, having the Close prices for them we calculated 'preliminary/shifted' return rate for the day before the first traid day.     

# #### To manage this issue we have to calculate the first trading date for each security and drop calculated returns for the previous days.

# In[ ]:


min_gd = given_target.groupby('SecuritiesCode')['Date'].min()
calculated_target['MinDate'] = calculated_target['SecuritiesCode'].map(min_gd)


# Let's see the Min Date for one of these securities:

# In[ ]:


calculated_target[calculated_target.SecuritiesCode == 9519].head(3)


# It's clear that we have calculated return for 2017-02-22 while the first trade date was 2017-02-23. We have to drop all these cases.

# In[ ]:


calculated_target = calculated_target[calculated_target.Date >= calculated_target.MinDate].drop(columns=['MinDate']).reset_index(drop=True)
print(f'Calculated_target shape: {calculated_target.shape}')
calculated_target.head()


# #### Let's compare the shapes again

# In[ ]:


calculated_target.shape[0] == given_target.shape[0]


# Let's also compare index, Dates and SecuritiesCode for these two tables. Are they identical?

# In[ ]:


print((calculated_target.index != given_target.index).sum() == 0)
print((calculated_target.Date != given_target.Date).sum()==0)
print((calculated_target.SecuritiesCode != given_target.SecuritiesCode).sum()==0)


# ### 3. Finally, we have two tables with given Target values and calculated Target values. Thus, we are ready to compare the results

# #### To do that let us make the table with differences, comparing the values with 10**(-10) precision:

# In[ ]:


diff_df = calculated_target.rename(columns= {'Target': 'Calculated'})
diff_df = diff_df.merge(given_target, on=['Date', 'SecuritiesCode']).rename(columns= {'Target': 'Given'})
diff_df['Diff'] = diff_df[['Calculated', 'Given']].apply(lambda x: abs(x[0] - x[1])>10**(-10), axis=1)
diff_df.head()


# #### Looks pretty nice, doesn't it? It sounds like our calculations provide us the correct values. For the first 5 rows the Diff columns consists from 'False' only! But ...

# -------------------------------------------------------

# ## Ooooooooppppppssssssss!

# In[ ]:


print(f'There are {diff_df[diff_df.Diff].shape[0]} cases where our calculations do not fit the provided values \nwhich is {diff_df[diff_df.Diff].shape[0]/diff_df.shape[0]*100:.1f}% of all values')


# ### Re: There are 54921 cases where our calculations do not fit the provided values,which is 2.4% of all values

# #### Maybe our calculations were incorrect? Let's check them really manually

# In[ ]:


diff_df[diff_df.Diff].head()


# We can see that the first case is the security with code 1407 for the date of '2017-01-04'. The provided value of return is -0.003437 but our calculated value is -0.003390

# We need 3 date to reculculate this value

# In[ ]:


df.loc[df.SecuritiesCode == 1407].head(3)


# #### Recall the definition:
# > #### *You can calculate the Target column from the Close column; it's the return from buying a stock the next day and selling the day after that*.

# * Buying the stock the next day we have Close price 885
# * Selling the stock the day after that we have Close price 882

# In[ ]:


manually_calculated = (882 - 885)/885
manually_calculated


# ## The Final Gong. Dououounnnnngggg!

# In[ ]:


print(f'The manually calculated Target {manually_calculated:6f}')
print(f"The Target, calculeted by our function {diff_df.loc[(df.SecuritiesCode == 1407) & (df.Date == '2017-01-04'), 'Calculated'].values[0]:6f}")
print(f"The Target, provided in Target column of train dataset {diff_df.loc[(df.SecuritiesCode == 1407) & (df.Date == '2017-01-04'), 'Given'].values[0]:6f}")


# ### We can see that our calculations are correct but they do not match the provided values of Target.

# # Conclusion

# ### The Target column values in train dataset do not match the provided definition in more than 2% cases.
# ### Let us investigate the reasons

# --------------------------------

# ### PS:

# ### In the next notebook we continue the topic of Target calculations from the Close feature, applying 'not cumprod' approach for Close price adjustment
# See ***Adjusted Target. Still doesn't match the Given one***     
# https://www.kaggle.com/code/vasiliisitdikov/adjusted-target-still-doesn-t-match-the-given-one
