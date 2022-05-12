#!/usr/bin/env python
# coding: utf-8

# # EDA USING SEABORN
# 
# This notebook attempts to carry out an exhaustive EDA with beginner friendly tutorial using SEABORN . This notebook is still in progress and further feature engineering , model building will be added to this eventually !
# So let's hop on and explore the Spaceship Titanic Dataset!! 

# # DATA FIELD DESCRIPTION
# train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# 
# * PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# * HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# * CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# * Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# * Destination - The planet the passenger will be debarking to.
# * Age - The age of the passenger.
# * VIP - Whether the passenger has paid for special VIP service during the voyage.
# * RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# * Name - The first and last names of the passenger.
# * Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# In[ ]:


import pandas as pd
import numpy as np

import os
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


# In[ ]:


DATAPATH = '../input/spaceship-titanic'


train_df = pd.read_csv(os.path.join(DATAPATH,'train.csv'))


# In[ ]:


train_df.head()


# # Quick EDA
# 
# A quick eda can be carried out using pandas, we can observe the following - 
# 
# 
# > * **Transported** - Almost half or majority of the passengers are transported to another dimensions i.e. value True <br>
# * **HomePlanet** -  The passengers come from 3 unique homeplanets. Majority of them come from *Earth* 😇
# * **CryoSleep** - Almost 60% of passengers prefer to travel without cryosleep . Note passengers who are in cryosleep are confined to cabins.
# * **Destination** - 3 unique destinations and 70% are travelling to *TRAPPIST-1e*	 planet :)
# * **Age** - Mean age of passengers 29 and median age 27 . Slightly right skewed. Highest age being 79 and min age is 0 . 25% of pasenngers age below 19 . 
# * **VIP** - 95% of passengers are non-VIP .
# * **Services** -  Out of all services average amount spend is highest for Food Court. Since food being a necessity .
# 
# 
# 

# In[ ]:


train_df.describe(include='all')


# # DETAILED EDA 
# 
# * We can see only 6 columns out of 14 are numeric majority of them categorical.

# # Missing Values Count

# In[ ]:


# % of Missing Values in each column
print('% of Missing Values for each Columns')
print('======================================')
train_df.apply(lambda x :(x.isnull().sum()/train_df.shape[0])*100)


# In[ ]:


# if we drop all na amount of data loss

print('Amount of data loss on dropping on all Na ~25%\n',1-(train_df.dropna().shape[0]/train_df.shape[0]))


# # Adding Columns / Splitting Columns 
# 
# Based on Data field description we can split the column `PassengerId` and `Cabin`  to derive further feature columns for analysis.
# 
# * `PassengerId` split into :
#   * `Group_Num` 
#   * `Group_Id`
# 
# * `Cabin` split into:
#   * `deck`
#   * `num`
#   * `side`
# 

# In[ ]:


#Splitting Columns
train_df[['GroupNum','Group_Id']] =train_df['PassengerId'].str.split('_', expand=True)

train_df[['deck','num','side']] = train_df['Cabin'].str.split('/', expand=True)


# **Combining Columns**
# 
# Adding columns of Luxury Services to create a new column in dataframe as `TotalBill` 
# 
# **NOTE** : NA columns are considered as 0 while summing 

# In[ ]:


train_df['TotalBill'] =  train_df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)


# In[ ]:


train_df.info()


# # Univariate Analysis
# 
# 
# * AGE - We see a high count of age near 0 bin followed by a dip and then another rise.
# * TOTALBILL - Majority 50% of total bill near 0,median slightly above 0 . Lot of outliers present
# * DECK - deck F,G having majority of passengers, deck T and A having minimum.
# * SIDE - Almost equal number of passengers for both sides.
# 

# #### Numeric
# 

# In[ ]:


train_df.describe()


# **HISTOGRAMS - AGE** 
# 
# 

# In[ ]:


sns.distplot(train_df['Age'], kde=False, bins=20, hist=True) #bydefault hist,kde is true
plt.show()


# **RUG PLOT**
# 
# A rug plot is a plot of data for a single quantitative variable, displayed as marks along an axis. It is used to visualise the distribution of the data. As such it is analogous to a histogram with zero-width bins, or a one-dimensional scatter plot.

# In[ ]:


sns.distplot(train_df['RoomService'], kde=False, bins=20 , rug=True, )
plt.show()


# **BOXPLOT**

# In[ ]:


sns.boxplot(y=train_df.TotalBill , palette='Paired_r' )


# Most of the amenities will have 75% of data in 0 so not plotting their histograms.

# #### Categorical
# 
# https://www.geeksforgeeks.org/how-to-create-different-subplot-sizes-in-matplotlib/
# 

# In[ ]:


from matplotlib import figure
fig,ax = plt.subplots(2,2, figsize=(15,10))
sns.countplot(train_df['HomePlanet'] , palette='Paired_r', ax=ax[0][0])
sns.countplot(train_df['CryoSleep'] , palette='Paired_r', ax=ax[0][1])
sns.countplot(train_df['Destination'] , palette='Paired_r', ax=ax[1][0])
sns.countplot(train_df['VIP'] , palette='Paired_r', ax=ax[1][1])
plt.show()


# In[ ]:


from matplotlib import figure
fig,ax = plt.subplots(1,3, figsize=(20,5))
sns.countplot(train_df['deck'] , palette='Paired_r', ax=ax[0])
sns.countplot(train_df['side'] , palette='Paired_r', ax=ax[1])
sns.countplot(train_df['Transported'] , palette='Paired_r', ax=ax[2])
plt.show()


# # Bivariate Analysis
# 
# We now go through analysing relationship between 2 variables. 
# 
# * AGE vs Transported  - age in 0-10 have higher number of transported passengers as compared to non-transported.
# * TotalBill vs AGE - Passengers with age less than 13 years have 0 total bill. 
# * HomePlanet vs Transported - Passengers from "Europa" planet has highest prob of getting transported ~0.7
# * CryoSleep vs Transported - Passengers in Cryosleep i.e. True have 0.8 prob of getting transported.
# * Destination vs Transported - Cancrei 55 e has 0.6 prob highest followed by others at 0.5 .
# * Deck vs Transported  - Deck B, C passengers have highest prob ~0.7 of getting transported.
# * VIP vs Transported  - Non- VIP passengers have higher prob of getting transported >0.5.
# * Side vs Transported - 's' side passengers have higher prob of getting transported >0.5
# 
# * TotalBill vs VIP - On an average VIP passengers have higher bill amount
# 
# * TotalBill vs Deck - ODeck B and C having high bill amounts as compared to other decks
# * Cryosleep vs Deck - The prob of being put on cryosleep is high for deck B and G and if we recall prob of crysleep being transported is high and same can be observed when we see deck wise prob of being transported we see Deck B,C ,g having high prob so a connection between deck, cryosleep and transported there.

# **HISTPLOT WITH HUE**

# In[ ]:


sns.histplot(x=train_df.Age , hue=train_df.Transported, palette='Paired_r',kde=True)
plt.show()


# In[ ]:


train_df['dummy'] = 0 #dummy variable is required when we want to plot hue without any x axis variable 
                      #that is only two variable in plot instead of 3 (x,y,hue)
sns.violinplot(y='Age' ,x='dummy',hue ='Transported' , data = train_df,split=True).set_title('Violin Plot for Age vs Transported ')
plt.show()


# **JOINTPLOT**

# In[ ]:


sns.jointplot(x=train_df.Age, y=train_df.TotalBill, palette='Paired_r', hue=train_df.Transported)


# In[ ]:


train_df[train_df.Age<13.0].TotalBill.sum() #Person with Age less than 13 have 0 total bills


# **COUNTPLOT AND MEAN PROBABILITIES OF BEING TRANPORTED**
# 
# We now plot countplots and mean probabilities of being transported for each category. This helps us better understand which category class have higher probability of getting transported on an average basis.

# In[ ]:


#PLottting mean prob for single feature 'HomePlanet'
notNull_df = train_df[(train_df.HomePlanet.isnull() == False) & (train_df.Transported.isnull() == False)][['HomePlanet','Transported']].reset_index()

temp = notNull_df.groupby('HomePlanet').mean()
sns.barplot(x=temp.index,y=temp.Transported).set_title('Mean probability to be Transported vs HomePlanet')
plt.show()


# In[ ]:


#Custom Function to plot Count of Categories wrt Target Column and Mean probability for each class in each category.
def plot_cat_prob(columnsList , data, ax_row, targetCol):
  sns.set_context("paper", font_scale=1.2)   
  #initialise plot ax
  f,ax = plt.subplots(ax_row, 2, figsize=(18, 38))

  #iterate 
  for r in range(ax_row):
    col_name = columnsList[r]
    sns.countplot(x=data[col_name], hue=train_df[targetCol], palette='RdPu',ax=ax[r][0])
    ax[r][0].set_title(f'Count of {col_name} wrt {targetCol}', fontsize=18)

    notNull_df = data[(data[col_name].isnull() == False) & (data[targetCol].isnull() == False)][[col_name,targetCol]].reset_index()
    temp = notNull_df.groupby(col_name).mean()

    sns.barplot(x=temp.index,y=temp[targetCol], ax=ax[r][1], palette='mako')
    ax[r][1].set_title(f'Mean probability to be {targetCol} vs {col_name}', fontsize=20)


# In[ ]:


#Calling custom function to plot 
plot_cat_prob(['HomePlanet','CryoSleep','Destination','VIP','deck','side'],train_df,6,targetCol='Transported')


# **VIOLINPLOT -  Total Bill vs VIP** 
# 

# In[ ]:


train_df['dummy'] = 0 #dummy variable is required when we want to plot hue without any x axis variable 
                      #that is only two variable in plot instead of 3 (x,y,hue)
sns.violinplot(y='TotalBill' ,x='dummy',hue ='VIP' , data = train_df,split=True,inner='quartile').set_title('Violin Plot for TotalBill vs VIP ')
plt.show()


# **STRIPPLOT - DECK VS Total Bill**
# 

# In[ ]:


sns.violinplot(y='TotalBill' ,x='deck', data = train_df,split=True).set_title('Violin Plot for Age vs Transported ')
sns.stripplot(y='TotalBill', x='deck',data=train_df)


# **FACTORPLOT - CRYOSLEEP vs DECK**
# 

# In[ ]:


sns.factorplot(y='CryoSleep', x='deck',data=train_df,kind='bar')


# # Multivariate Analysis
# 
# **CORRELATION HEATMAP**

# In[ ]:


train_df.drop(columns=['dummy','PassengerId','Name','Cabin'],inplace=True)


# In[ ]:


train_df.info()


# In[ ]:


#Custom Function to generate label encodings for object and bool columns
def make_label_encoder(data):
  
  cat_col = data.select_dtypes(['object','bool']).columns
  transformed_df = pd.DataFrame()

  for col in cat_col:
    le = LabelEncoder()
    x = le.fit_transform(data[col])
    transformed_df[col] = x

  numeric_col = data.drop(columns=cat_col)
  return pd.concat([transformed_df,numeric_col],axis=1)

   


# In[ ]:


#Transform dataset
train_df_transformed = make_label_encoder(train_df)


# In[ ]:


corr = train_df_transformed.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(25, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True)
plt.show()


# # > **THANK YOU!!**
