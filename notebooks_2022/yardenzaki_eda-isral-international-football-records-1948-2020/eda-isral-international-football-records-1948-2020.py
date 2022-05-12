#!/usr/bin/env python
# coding: utf-8

# # EDA ISRAL International Football Records 1948-2020
# > #### DataSet  by THOMAS KONSTANTIN 
# ###### https://www.kaggle.com/datasets/thomaskonstantin/israel-international-football-records-19482020?select=1948-2020_IFD.csv

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas_profiling
import plotly.express as px

# for random forest model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


#  # *** Filters ***

# In[ ]:


tournament = "UEFA European Championship" #"UEFA European Championship" / FIFA World Cup
since = 1950


# # read data:
# 

# In[ ]:


df = pd.read_csv("../input/israel-international-football-records-19482020/1948-2020_IFD.csv")
# Remove Nan:
df = df.dropna(how='any', axis=0)


# # Add columns
# ###### year , Israel Scored, Oponent Scored

# In[ ]:


#df["year"] = df["Date"].apply(lambda x: (str(x)[6:11]))
df["year"] = pd.DatetimeIndex(df['Date']).year
df["month"] = pd.DatetimeIndex(df['Date']).month
idx = df["Match"].str.split(" v ")
df["idx"] = idx.apply(lambda x: x[0].startswith("Israel"))
df["first_scored"] = df["Score"].apply(lambda x: int(str(x)[0:1]))
df["second_scored"] = df["Score"].apply(lambda x: int(str(x)[2:3]))
df['Israel Scored'] = np.where(df['idx'] == True, df['first_scored'] , df['second_scored'])
df['Oponent Scored'] = np.where(df['idx'] == False, df['first_scored'] , df['second_scored'])
df = df.drop(columns = ["idx","first_scored","second_scored"])


# # Data Validation

# ## info:

# In[ ]:


df.dtypes


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## Missing Values

# In[ ]:


df.columns[df.isnull().any()]


# In[ ]:


df.isnull().sum()


# # Profile

# In[ ]:


profiling = pandas_profiling.ProfileReport(df)
profiling.to_file("profiling.html")
profiling


# # Filter Data to Include only "meaningful" competitions

# In[ ]:


meaning_comp_lst = ["FIFA World Cup","UEFA European Championship"]
df = df [df.Competition.isin(meaning_comp_lst)]
df.head(150)


# * ## How Many goals Israel scored in each competition ?

# In[ ]:


mask = df["Competition"].isin([tournament])
mask2 = df["year"].astype(int) > since
tour_df = df[(mask) & (mask2)]
tour_df = tour_df.reset_index()
#tour_df.head(150)


# * ## Wins / losses

# In[ ]:


# Stats:
Result_grouped = tour_df.groupby(["Result"])["Result"].count()


# In[ ]:


W = Result_grouped.loc["W"]
L = Result_grouped.loc["L"]
D = Result_grouped.loc["D"]


# In[ ]:


f"since {since}, Isral had {W} wins,{L} losses and {D} draws in {tournament}"


# * ## How many goals did Israel score?

# In[ ]:


tour_goals = tour_df.groupby(["Competition"])["Israel Scored"].sum()
f"For {tournament}, Israel scored {tour_goals.iloc[0]} goals since {since}"


# * ## How many goals did the oponents score?

# In[ ]:


tour_goals = tour_df.groupby(["Competition"])["Oponent Scored"].sum()
f"For {tournament}, The oponents scored {tour_goals.iloc[0]} goals since {since}"


# * Best Matches:

# In[ ]:


f"Israel {tournament} Best Matches since {since} :"


# In[ ]:


best_df=tour_df.nlargest(5, 'Israel Scored')
best_df = best_df.loc[:,["Match","Score","year"]]
best_df.style.hide_index()


# * Worst Matches:

# In[ ]:


f"Israel {tournament} Worst Matches since {since} :"


# In[ ]:


worst_df=tour_df.nlargest(5, 'Oponent Scored')
worst_df = worst_df.loc[:,["Match","Score","year"]]
worst_df.style.hide_index()


# * Goals vs. year

# In[ ]:


#%%writefile SeriesToPD.py
Result_grouped = tour_df.groupby(["year"])["Israel Scored"].sum()
### 1-liner Series to DF and then plotting it:
Result_grouped = Result_grouped.to_frame().reset_index(level=0)
#Result_grouped
#Result_grouped.columns

# plot with Plotly
fig = px.bar(Result_grouped, x="year", y="Israel Scored", color="Israel Scored", title="Goals scored by Israel vs. year")
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    )
)

fig.show()


# * Seasonal Shape

# In[ ]:



W_dict={"W":1,"D":0,"L":0}
D_dict={"W":0,"D":1,"L":0}
L_dict={"W":0,"D":0,"L":1}
tour_df["W"]=tour_df["Result"].map(W_dict)
tour_df["L"]=tour_df["Result"].map(L_dict)
tour_df["D"]=tour_df["Result"].map(D_dict)
W_Result_grouped = tour_df.groupby(["month"])["W"].sum()
D_Result_grouped = tour_df.groupby(["month"])["D"].sum()
L_Result_grouped = tour_df.groupby(["month"])["L"].sum()

seriess = [W_Result_grouped,D_Result_grouped,L_Result_grouped]
colors = ["SkyBlue","IndianRed","green"]
lables = ["Wins","Draws","Losses"] 
get_ipython().run_line_magic('matplotlib', 'inline')
for i,s in enumerate (seriess):
    xx = np.asarray(tuple(s.index))
    yy = np.asarray(tuple(s))
    width = 0.3
    offset = 0.3-i*0.3
    plt.bar(xx-offset, yy, label=lables[i], color = colors[i],width=width)
months=range(13)    
plt.xticks(months)    
plt.legend()
plt.xlabel('Month')
plt.ylabel('Match Result')

plt.title(f'Seasonal shape of the israeli national team: {tournament} , since {since}')
plt.show()


# # Random Forest Model
# ## features: 'month' & 'Result'
# 

# In[ ]:


# Choose target and features
y = df["Score"]
_features = ['month', 'Result']
X = df[_features]


# In[ ]:


# Train / Test Split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
one_hot_encoded_train_X = pd.get_dummies(train_X).reset_index().drop(columns='index')
#print(one_hot_encoded_train_X.columns)
# print(one_hot_encoded_train_X)
one_hot_encoded_train_Y = pd.get_dummies(train_y).reset_index().drop(columns='index')
# print(one_hot_encoded_train_Y.head(3))
#print(one_hot_encoded_train_Y.columns)

one_hot_encoded_val_X = pd.get_dummies(val_X).reset_index().drop(columns='index')
one_hot_encoded_val_Y = pd.get_dummies(val_y).reset_index().drop(columns='index')
# print(one_hot_encoded_val_X)
# decode 
#s2 = one_hot_encoded_train_Y.idxmax(axis=1)
#print(s2)
#print((s2 == train_y).all())


# In[ ]:


def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()


# In[ ]:


# Define Model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(one_hot_encoded_train_X, one_hot_encoded_train_Y)
#print(one_hot_encoded_val_X)
_predictions = forest_model.predict(one_hot_encoded_val_X)


# In[ ]:


#for p in _predictions:
    #print(p)
    #print(np.argmax(p, axis=-1))
    #print(one_hot_encoded_train_Y.columns[np.argmax(p, axis=-1)])
    #print("--------------------------------------")
print(get_mae(one_hot_encoded_val_Y, _predictions))


# In[ ]:


to_pred = pd.DataFrame(columns=['month','Result_D','Result_L','Result_W'])
to_pred.loc[to_pred.shape[0]] =  [10, 0, 0,1]
#one_hot_encoded_to_pred = pd.get_dummies(to_pred).reset_index().drop(columns='index')
#one_hot_encoded_to_pred.head()
prediction = forest_model.predict(to_pred)
print(one_hot_encoded_train_Y.columns[np.argmax(prediction, axis=-1)])

