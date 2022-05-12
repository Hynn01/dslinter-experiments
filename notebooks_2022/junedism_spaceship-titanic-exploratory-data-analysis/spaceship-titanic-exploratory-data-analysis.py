#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# In[ ]:


# Color Setting

colors = ['#8EBAD9', '#FFF56D']
tcolor_dict = {
    True : "#8EBAD9",
    False : "#FFF56D"
}
mycmp = LinearSegmentedColormap.from_list("MyCmp", ['#8EBAD9', '#FFFFFF', '#FFF56D'], N=100)


# ## Loading the Data
# 
# #### Load the Spaceship Titanic Data

# In[ ]:


data_train = pd.read_csv("../input/spaceship-titanic/train.csv")

# Let's take a look at the data
data_train.head()


# ## Understanding the Data - Categorical Data
# #### Let's do some simple Exploratory Data Analysis on the categorical data in the dataset

# In[ ]:


# Let's see the shape of the data
print("The data has a dimension of", data_train.shape[0], "Rows, and", data_train.shape[1], "columns")


# >### Data Field Descriptions
# >
# >- `PassengerId` - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# >- `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
# >- `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# >- `Cabin` - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# >- `Destination` - The planet the passenger will be debarking to.
# >- `Age` - The age of the passenger.
# >- `VIP` - Whether the passenger has paid for special VIP service during the voyage.
# >- `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# >- `Name` - The first and last names of the passenger.
# >- `Transported` - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# In[ ]:


# Basic information about the data
data_train.info()


# In[ ]:


data_train.describe()


# In[ ]:


# Check if the data has missing value
data_train.isnull().sum().sort_values()


# ### Let's create a dataframe to see better representation of the magnitude of the missing data

# In[ ]:


# Calculate the percentage of missing data from each column
nan_ratio = []
for col in data_train.columns:
    nan_item = []
    nan_item.append(col)
    nan_item.append(data_train[col].isnull().sum())
    nan_item.append(str(round(100 * data_train[col].isnull().sum() / data_train.shape[0], 2)) + '%')
    nan_ratio.append(nan_item)
    
df_nan = pd.DataFrame(nan_ratio, columns=["Column", "NaN count", "NaN ratio"]).set_index("Column")
df_nan = df_nan.sort_values("NaN ratio", ascending=False)
df_nan.astype(object).T


# > ### Findings
# >* All columns except `PassengerId` and `Transported` has some degree of missing values problem
# >* The percentage of missing values are generally low, with magnitude ranging from 2 - 2.5%

# ### Class Imbalance Check!
# Let's create a pie chart to see if the data we have has a class imbalance problem

# In[ ]:


# Plot a pie chart to check if we have class imbalance problem

fig, ax = plt.subplots(figsize=(6,4))
ax.pie(x = data_train["Transported"].value_counts(), autopct="%1.1f%%",
        pctdistance=.75, startangle=24, textprops={"fontsize":12},
        colors=colors, wedgeprops={'edgecolor':'#383838'});
# ax.text(1.5,.1,"There are approximately same amount of positive and negative class,\n so we don't have class imbalance problem.",
#         size=15)
ax.set_title("Target Class Distribution", fontdict={'fontsize':14})
ax.legend(['Transported', 'Not-Transported'], bbox_to_anchor=(.8, .77))

centre_circle = plt.Circle((0,0),0.55,fc='white', ec='#383838')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.show()


# > ### Findings
# >* There are approximately same amount of Positive (Transported) class and Negative (Not-Transported) class, so we don't have class imbalance problem
# 
# <br>
# We need to dive deeper into the data to unearth more insight from the data and see which categories of the passengers who got transported and who don't.
# 
# First let's try to analyze the features in the data

# ### Age Distribution
# Let's see the age distribution among the passengers

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18, 4))
fig.subplots_adjust(wspace=.1)
bins = np.int8(data_train["Age"].max())

ax[0].hist(data_train["Age"], bins=bins, edgecolor="k", alpha=.6, zorder=2)
ax[0].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax[0].set_title("Age Distribution")

ax[1].hist(data_train[data_train["Transported"]]["Age"], bins=bins, alpha=.6, histtype='stepfilled', edgecolor='k',
        color=colors[0], zorder=2, label="Transported")
ax[1].hist(data_train[~data_train["Transported"]]["Age"], bins=bins, alpha=.6, histtype='stepfilled', edgecolor='k',
        color=colors[1], zorder=2, label="Not-Transported")
ax[1].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax[1].set_title("Age Distribution by Target Class")
ax[1].legend()

plt.show();


# > ### Findings
# >* Most passengers are in the age range of 20 to 29, therefore this age range has the most transported passengers
# >* From the histogram plot on the right we can see each class Age Distribution are pretty much on top of each other, so we can say that the amount of Transported passengers are equally distributed between each age range. Therefore it seems that there is little to no correlation between Age and Transported since we can't see the difference between Transported and Not-Transported class
# >* There are considerable amount of transported passengers that in the age range of 0 to 1

# #### Let's double check our findings

# In[ ]:


# Check which age range has the most passengers and most transported passengers
age_list = []
for i in range(0, 80, 10):
    item = []
    item.append("%d-%d" % (i, i+9))
    item.append(data_train[(data_train["Age"] >= i) & (data_train["Age"] < i+10)]["Age"].count())
    item.append(data_train[(data_train["Transported"]) & (data_train["Age"] >= i) & (data_train["Age"] < i+10)]["Age"].count())
    item.append(data_train[(~data_train["Transported"]) & (data_train["Age"] >= i) & (data_train["Age"] < i+10)]["Age"].count())
    age_list.append(item)
    
age_list = pd.DataFrame(age_list, columns=["AgeRange", "Count", "TransCount", "notTransCount"]).set_index("AgeRange").transpose()
age_list


# In[ ]:


# Check the correlation of Age and Transported feature
data_cluster = data_train[["Age", "Transported"]].corr()
print("Age and Transpoted Correlation coefficient : {:.4f}".format(data_cluster.loc['Age','Transported']))


# The `Age` and `Transported` has a correlation value very near to zero, which indicates that there is very little correlation between the two feature. We might as well dropping the `Age` column since it won't give us any information about the Target variable

# ### Understanding the Relationship between HomePlanet, Destination and Transported

# In[ ]:


# Create dataframe grouped by HomePlanet and Transported
y_home = data_train.groupby(['HomePlanet', 'Transported'])['PassengerId'].count().reset_index()
y_home.rename(columns={'PassengerId' : 'Count'}, inplace=True)

# Create dataframe grouped by Destination and Transported
y_dest = data_train.groupby(['Destination', 'Transported'])['PassengerId'].count().reset_index()
y_dest.rename(columns={'PassengerId' : 'Count'}, inplace=True)


# In[ ]:


### Ignore the messy code, i'll fix it later ###

x = np.arange(3)
width=.24

gs_kw = dict(width_ratios=[1,1], height_ratios=[.4, 1.6])
fig, ax = plt.subplots(2, 2, figsize=(14,7), gridspec_kw=gs_kw)
fig.subplots_adjust(wspace=.25, hspace=.3)

# ---------------------------------------------------------------------------------------
bar1 = ax[0,0].barh(x, y_home.groupby(["HomePlanet"])["Count"].sum(), zorder=2, hatch='//////', ec="#383838", color=tcolor_dict[True])
ax[0,0].bar_label(bar1, padding=4)
ax[0,0].set_yticks(x, list(y_home["HomePlanet"].unique()))
ax[0,0].set_xlim(right=5200)
ax[0,0].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax[0,0].set_title("Number of Passenger by HomePlanet")

# ---------------------------------------------------------------------------------------
bar1 = ax[0,1].barh(x, y_dest.groupby(["Destination"])["Count"].sum(), zorder=2, hatch='//////', ec="#383838", color=tcolor_dict[True])
ax[0,1].bar_label(bar1, padding=4)
ax[0,1].set_yticks(x, list(y_dest["Destination"].unique()))
ax[0,1].set_xlim(right=6700)
ax[0,1].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax[0,1].set_title("Number of Passenger by Destination")

# ---------------------------------------------------------------------------------------
# HomePlanet vs Transported barchart
bar1 = ax[1, 0].bar(x-.12, y_home[y_home["Transported"]]["Count"], width, zorder=2, color=tcolor_dict[True], ec="#383838")
bar2 = ax[1, 0].bar(x+.12, y_home[~y_home["Transported"]]["Count"], width, zorder=2, color=tcolor_dict[False], ec="#383838")
ax[1, 0].bar_label(bar1)
ax[1, 0].bar_label(bar2)
ax[1, 0].set_xticks(x, list(y_home["HomePlanet"].unique()))
ax[1, 0].set_xlabel("HomePlanet", size=12)
ax[1, 0].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax[1, 0].set_title("Transported vs HomePlanet")
ax[1, 0].legend(["Transported", "Not-Transported"])

# ---------------------------------------------------------------------------------------
# Destination vs Transported barchart
bar1 = ax[1, 1].bar(x-.12, y_dest[y_dest["Transported"]]["Count"], width, zorder=2, color=tcolor_dict[True], ec="#383838")
bar2 = ax[1, 1].bar(x+.12, y_dest[~y_dest["Transported"]]["Count"], width, zorder=2, color=tcolor_dict[False], ec="#383838")
ax[1, 1].bar_label(bar1)
ax[1, 1].bar_label(bar2)
ax[1, 1].set_xticks(x, list(y_dest["Destination"].unique()))
ax[1, 1].set_xlabel("Destination", size=12)
ax[1, 1].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax[1, 1].set_title("Transported vs Destination")


plt.show();


# > ### Findings
# >* Most of the passengers came from planet Earth
# >* Relative to other Planet, passengers from Earth are the most Transported
# >* Most of the passengers are leaving to TRAPPIST-1e
# >* Relative to other Destination, passengers leaving to TRAPPIST-1e are the most Transported

# In[ ]:


# Create dataframe grouped by HomePlanet, Destination and Transported

y_homedest = data_train.groupby(['HomePlanet', 'Destination', 'Transported'])['PassengerId'].count().reset_index()
y_homedest['HomeDest'] = y_homedest['HomePlanet'] + ' to ' + y_homedest['Destination']
y_homedest.rename(columns={'PassengerId' : 'Count'}, inplace=True)
y_homedest = y_homedest.sort_values(["Count", "HomePlanet"], ascending=True)


# In[ ]:


y = np.arange(stop=18, step=2)
height = .86


fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.barh(y-.43, y_homedest[~y_homedest["Transported"]]["Count"], color=tcolor_dict[False], ec="#383838", height=height,
               zorder=2, label="Not-Transported")
bar2 = ax.barh(y+.43, y_homedest[y_homedest["Transported"]]["Count"], color=tcolor_dict[True], ec="#383838", height=height,
               zorder=2, label="Transported")
ax.bar_label(bar1, padding=5)
ax.bar_label(bar2, padding=5)
ax.set_yticks(y, y_homedest["HomeDest"].unique())
ax.grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax.set_title("Transported Passengers by HomePlanet and Destination")
ax.set_xlim(right=2000)
ax.legend(bbox_to_anchor=(1, .15))

plt.show();


# ### Understanding the Relationship between CryoSleep, VIP and Transported

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].pie(x = data_train["CryoSleep"].value_counts(), autopct="%1.1f%%",
        pctdistance=.5, startangle=24, textprops={"fontsize":12},
        colors=colors, wedgeprops={'edgecolor':'#383838'});
ax[0].set_title("CryoSleep Distribution", fontdict={'fontsize':14})
ax[0].legend(['Not-CryoSlepp', 'CryoSleep'], bbox_to_anchor=(.8, .77))


ax[1].pie(x = data_train["VIP"].value_counts(), autopct="%1.1f%%",
        pctdistance=.5, startangle=24, textprops={"fontsize":12},
        colors=colors, wedgeprops={'edgecolor':'#383838'}, explode=[0,.5]);
ax[1].set_title("VIP Distribution", fontdict={'fontsize':14})
ax[1].legend(['Not-VIP', 'VIP'], bbox_to_anchor=(.8, .77))

plt.show()


# > ### Findings
# >* About two-thirds of the passengers are enjoying their trip without going into cryosleep state
# >* A very little amount of passengers are having the benefit of the VIP service

# #### Let's see how those two features correlate to the Transported feature

# In[ ]:


y_cs = data_train.groupby(["CryoSleep", "Transported"])["PassengerId"].count().reset_index()
y_cs.rename(columns={"PassengerId" : "Count"}, inplace=True)

y_vip = data_train.groupby(["VIP", "Transported"])["PassengerId"].count().reset_index()
y_vip.rename(columns={"PassengerId" : "Count"}, inplace=True)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
x = np.arange(2)


# plot
bar1 = ax[0].bar(x-.12, y_cs[y_cs["Transported"]]["Count"], width=width, label="Transported",
                 color=tcolor_dict[True], ec="#383838", zorder=2)
bar2 = ax[0].bar(x+.12, y_cs[~y_cs["Transported"]]["Count"], width=width, label="Not-Transported",
                 color=tcolor_dict[False], ec="#383838", zorder=2)
ax[0].bar_label(bar1)
ax[0].bar_label(bar2)
ax[0].set_xticks(x, ["Not-CryoSleep", "CryoSleep"])
ax[0].set_ylim(top=4000)
ax[0].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax[0].set_title("Transported vs CryoSleep")
ax[0].legend()

bar1 = ax[1].bar(x-.12, y_vip[y_vip["Transported"]]["Count"], width=width, label="Transported",
                 color=tcolor_dict[True], ec="#383838", zorder=2)
bar2 = ax[1].bar(x+.12, y_vip[~y_vip["Transported"]]["Count"], width=width, label="Not-Transported",
                 color=tcolor_dict[False], ec="#383838", zorder=2)
ax[1].bar_label(bar1)
ax[1].bar_label(bar2)
ax[1].set_xticks(x, ["Not-VIP", "VIP"])
ax[1].set_ylim(top=4600)
ax[1].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax[1].set_title("Transported vs VIP")
ax[1].legend()

plt.show();


# > ### Findings
# >* It seems passengers in the cryosleep state are more likely to be transpoted, and vice versa
# >* Whether passengers have VIP benefit or not, they have about the same chance to be transported

# ### Understanding the Relationship between Cabin and Transported
# Cabin feature has a low granulity with deck, num, and side data as constituent aggregated into one feature. We're going to increase the granulity of `Cabin` feature by dividing the three aggregated data into individual feature. The reason behind why we dismantling the `Cabin` feature is because as the granularity increases, the more information is available for analysis.

# In[ ]:


# Create new dataframe with deck, num, and side as columns

df_cabin = data_train["Cabin"].str.extract("(.*?)/(.*?)/(.)")
df_cabin.columns = ["Deck", "Num", "Side"]
df_cabin.info()


# In[ ]:


# We want to see if there is any correlation between the three new features and the Transported feature,
# so we're adding the transported feature into the newly created dataframe

df_cabin["Transported"] = data_train["Transported"]
df_cabin.dropna(inplace=True)
df_cabin.head(4)


# In[ ]:


# Let's do some observation on the newly created features

print("List of Decks :", df_cabin["Deck"].unique())
print("List of Nums :", df_cabin["Num"].unique(), "| # of unique val :", df_cabin["Num"].nunique())
print("Sides :", df_cabin["Side"].unique(), "| P = Port, S = Starboard")


# #### Let's analyze how the three newly created features correlates to the Transported feature

# In[ ]:


y_deck = df_cabin.groupby(["Deck", "Transported"])["Num"].count().reset_index()
y_deck.rename(columns = {"Num" : "Count"}, inplace=True)

y_side = df_cabin.groupby(["Side", "Transported"])["Num"].count().reset_index()
y_side.rename(columns = {"Num" : "Count"}, inplace=True)


# In[ ]:


y = np.arange(stop=16, step=2)
y = y[::-1]
fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.barh(y+.43, y_deck[y_deck["Transported"]]["Count"], height=height, color=tcolor_dict[True],
              ec="#383838", zorder=2, label="Transported")
bar2 = ax.barh(y-.43, y_deck[~y_deck["Transported"]]["Count"], height=height, color=tcolor_dict[False],
              ec="#383838", zorder=2, label="Not-Transported")
ax.bar_label(bar1, padding=5)
ax.bar_label(bar2, padding=5)
ax.set_yticks(y, y_deck["Deck"].unique())
ax.set_xlim(right=1700)
ax.set_ylabel("Deck", fontsize=13)
ax.grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax.set_title("Transported vs Deck")
ax.legend()

plt.show();


# > ### Findings
# >* Passengers are more likely to be transported if they placed in Deck **B** and **C** with **73.4%** and **68%** chance of being transported respectivelty
# >* Deck T has the least passengers with only 5 passengers total

# In[ ]:


y_side


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
x = np.arange(2)

bar1 = ax.bar(x-.12, y_side[y_side["Transported"]]["Count"], width=width, zorder=2, color=tcolor_dict[True],
              label="Transported", ec="#383838")
bar2 = ax.bar(x+.12, y_side[~y_side["Transported"]]["Count"], width=width, zorder=2, color=tcolor_dict[False],
              label="Not-Transported", ec="#383838")
ax.bar_label(bar1)
ax.bar_label(bar2)
ax.set_xticks(x, ["Port", "Starboard"])
ax.set_xlabel("Side", fontsize=13)
ax.set_ylim(top=2600)
ax.grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax.set_title("Transported vs Side")
ax.legend(bbox_to_anchor=(1,1))

plt.show();


# > ### Findings
# >* Nothing attractive to caught my attention here, only fact that there's slightly more passengers that got Transported in Starboard Side

# In[ ]:


df_cabin["Num"] = df_cabin["Num"].astype(int)


# In[ ]:


df_cabin["Num"].describe()


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 5))

ax.hist(df_cabin[df_cabin["Transported"]]["Num"], bins=150, alpha=.6, histtype='stepfilled', edgecolor='k',
        color=tcolor_dict[True], zorder=2)
ax.hist(df_cabin[~df_cabin["Transported"]]["Num"], bins=150, alpha=.6, histtype='stepfilled', edgecolor='k',
        color=tcolor_dict[False], zorder=2)
ax.grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
ax.set_title("Transported vs Num")

plt.show();


# > ### Findings
# >* The Sum feature seems quite uninformative at distinguishing the Transported class since the two histogram are mostly overlap. So we can expect a little to no correlation between Transported and Num features
# >*

# In[ ]:


data_cluster = df_cabin[["Num", "Transported"]].corr()
print("Num and Transported Correlation Coefficient {:.4f}".format(data_cluster.loc["Num", "Transported"]))


# The `Num` and `Transported` has a correlation value very near to zero, which indicates that there is very little correlation between the two feature. We might as well dropping the `Num` column since it won't give us any information about the Target variable

# ## Understanding the Data - Continuous Data
# #### Let's do some simple Exploratory Data Analysis on the continuous data in the dataset

# In[ ]:


data_train.head(4)


# There are 5 continuous features in the dataset namely RoomService, FoodCourt, ShoppingMall, Spa, VRDeck

# In[ ]:


data_train[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].describe()


# #### Defining some function to help with our vizualisation

# In[ ]:


## Change big numbers into human readable format
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.1f%s' % (num, ['', 'k', 'M', 'G', 'T', 'P'][magnitude])


## x axis tick label
def expenses_xtick_lbl(bins, col):
    spend_range=[]
    mx = 0
    mn = 0
    for i in range(10):
        mx = mx + data_train[col].max()/10
        #spend_range.append(("%s - %s"% (human_format(mn), human_format(mx))))
        spend_range.append(">"+human_format(mn))
        mn = mx + 1
        
    return spend_range

## define the position where x axis tick label placed
def xticks_position(col):
    tick = np.arange(data_train[col].max(), step=data_train[col].max()/10)
    tick = tick + (data_train[col].max()/10)/2
    
    return tick


# In[ ]:


fig, ax = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
fig.supylabel("Passengers Count", x=.085, size=14)
fig.supxlabel("Money Spent", y=.0, size=14)
fig.suptitle("Continuous Features Distribution", size=14)
fig.subplots_adjust(wspace=.08, hspace=.45)
bins = 10
xt = np.arange(10)


x = 0
for i in range(2):
    for j in range(3):
        if i == 1 and j == 2:
            ax[i, j].remove()
        else:
            c,e, bars = ax[i, j].hist(data_train[cols[x]], bins=bins, zorder=2, color=tcolor_dict[True], ec='k')
            ax[i, j].bar_label(bars, padding=5)
            ax[i, j].grid(linestyle='--', linewidth=0.5, color='gray', zorder=0)
            ax[i, j].set_ylim(top=9500)
            ax[i, j].set_title(cols[x])
            xtick_pos = xticks_position(cols[x])
            xtick_lbl = expenses_xtick_lbl(bins, cols[x])
            ax[i, j].set_xticks(xtick_pos, xtick_lbl, rotation=45, ha='center')
            x = x + 1


# > ### Findings
# >* **To Be Continued**

# In[ ]:





# In[ ]:


# TBD

# threshold = 0.0
# data_cluster = data_train.corr()
# mask = data_cluster.where(abs(data_cluster) >= threshold).isna()
# sns.heatmap(data_cluster, annot=True, mask=mask, cmap=mycmp);

