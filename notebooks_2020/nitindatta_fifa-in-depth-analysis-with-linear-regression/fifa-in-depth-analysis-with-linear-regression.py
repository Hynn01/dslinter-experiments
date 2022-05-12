#!/usr/bin/env python
# coding: utf-8

# <font color="red" size=5><center>FIFA - EDA and Linear Regression Starter</center></font>

# # Introduction
# 
# <a class="anchor" id="toc"></a>
# 
# FIFA 19 is a football simulation video game developed by EA Vancouver as part of Electronic Arts' FIFA series. It is the 26th installment in the FIFA series, and was released on 28 September 2018 for PlayStation 3, PlayStation 4, Xbox 360, Xbox One, Nintendo Switch, and Microsoft Windows
# 
# 
# *Source: [Wikipedia](https://en.wikipedia.org/wiki/FIFA_19)*
# 
# This game did not recieve much popular reviews. However, we will have a deep analysis in this notebook.
# 
# <font color="red" size=3>Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>
# 
# ![Image](https://images.daznservices.com/di/library/GOAL/d2/e5/fifa-19-ultimate-team_19gk2cvl4sd11kpd72z9ufwi1.jpg?t=1305899489&quality=60&w=1600)

# # Table of Contents
# 1. [Glimpse of the data at hand](#1)
# 2. [Data Cleaning](#2)
# 3. [Data Analysis](#3)
# 4. [Exploratory Data Analysis](#4)
# 5. [Modelling](#5)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# <a id ="1" > </a>
# # 1. Glimpse of the data at hand

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from collections import Counter
import missingno as msno

import warnings
warnings.filterwarnings('ignore')
import plotly
sns.set_style('darkgrid')


# In[ ]:


df=pd.read_csv('../input/data.csv')


# In[ ]:


df.head().T


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# <a id="2"></a> 
# # 2. Data Cleaning

# In[ ]:


df.drop(['Unnamed: 0','Photo','Flag','Club Logo'],axis=1,inplace=True)


# In[ ]:


msno.bar(df.sample( 18207 ),(28,10),color='red')


# In[ ]:


df.isnull().sum()


# `48` is repeating many times lets check if all of them are same `ID` or not

# In[ ]:


missing_height = df[df['Height'].isnull()].index.tolist()
missing_weight = df[df['Weight'].isnull()].index.tolist()
if missing_height == missing_weight:
    print('They are same')
else:
    print('They are different')


# As they are same I am assuming it will be same for other all columns too. 
# 
# We will see it soon. 

# In[ ]:


df.drop(df.index[missing_height],inplace =True)


# In[ ]:


df.isnull().sum()


# As we can see, we guessed right and now we have fewer null values.

# In[ ]:


df.drop(['Loaned From','Release Clause','Joined'],axis=1,inplace=True)


# <a id="3"></a>
# # 3. Data Analysis

# In[ ]:


#Number of countries available and top 5 countries with highest number of players
print('Total number of countries : {0}'.format(df['Nationality'].nunique()))
print(df['Nationality'].value_counts().head(5))
print('--'*40)
print("\nEuropean Countries have most players")


# In[ ]:


#Total number of clubs present and top 5 clubs with highest number of players
print('Total number of clubs : {0}'.format(df['Club'].nunique()))
print(df['Club'].value_counts().head(5))


# In[ ]:


#Player with maximum Potential and Overall Performance
print('Maximum Potential : '+str(df.loc[df['Potential'].idxmax()][1]))
print('Maximum Overall Perforamnce : '+str(df.loc[df['Overall'].idxmax()][1]))


# In[ ]:


pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
print('BEST IN DIFFERENT ASPECTS :')
print('_________________________\n\n')
i=0
while i < len(pr_cols):
    print('Best {0} : {1}'.format(pr_cols[i],df.loc[df[pr_cols[i]].idxmax()][1]))
    i += 1


# In[ ]:


#Cleaning some of values so that we can interpret them 
def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]

        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value

df['Value'] = df['Value'].apply(value_to_int)
df['Wage'] = df['Wage'].apply(value_to_int)


# In[ ]:


df.head().T


# In[ ]:


#Top earners
print('Most valued player : '+str(df.loc[df['Value'].idxmax()][1]))
print('Highest earner : '+str(df.loc[df['Wage'].idxmax()][1]))
print("--"*40)
print("\nTop Earners")


# <a id="4"></a> <br>
# # 4. Exploratory Data Analysis

# In[ ]:


sns.jointplot(x=df['Age'],y=df['Potential'],
              joint_kws={'alpha':0.1,'s':5,'color':'red'},
              marginal_kws={'color':'red'})


# `Potential` tends to fall as you grow old

# Source of below visualisations: [fifa19-analysis](https://www.kaggle.com/dczerniawko/fifa19-analysis)

# In[ ]:


player_features = (
    'Acceleration', 'Aggression', 'Agility', 
    'Balance', 'BallControl', 'Composure', 
    'Crossing', 'Dribbling', 'FKAccuracy', 
    'Finishing', 'GKDiving', 'GKHandling', 
    'GKKicking', 'GKPositioning', 'GKReflexes', 
    'HeadingAccuracy', 'Interceptions', 'Jumping', 
    'LongPassing', 'LongShots', 'Marking', 'Penalties'
)

from math import pi
idx = 1
plt.figure(figsize=(15,45))
for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    # number of variable
    categories=top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(10, 3, idx, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1


# In[ ]:


sns.lmplot(data = df, x = 'Age', y = 'SprintSpeed',lowess=True,scatter_kws={'alpha':0.01, 's':5,'color':'green'}, 
           line_kws={'color':'red'})


# As the `age` increases the `sprint speed` decreases

# In[ ]:


sns.lmplot(x = 'BallControl', y = 'Dribbling', data = df,col = 'Preferred Foot',scatter_kws = {'alpha':0.1,'color':'orange'},
           line_kws={'color':'red'})


# `Left Footed Players` vs `Right Footed Players`

# In[ ]:



sns.jointplot(x=df['Dribbling'], y=df['Crossing'], kind="hex", color="#4CB391");


# `Crossing` vs `Dribbling`

# In[ ]:


value = df.Value
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

sns.relplot(x="Age", y="Potential", hue=value/100000, 
            sizes=(40, 400), alpha=.5,
            height=6, data=df);


# Relation between `potential` and `age` with respected `value` of players

# In[ ]:


corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="YlGnBu")


# **Lowest correlation** among the goalkeeping side with other columns and high among themselves 
# 
# **High correlation** between `Dribbling`, `Volleys`, `Passing` etc...

# In[ ]:


plt.rcParams['figure.figsize'] = (20, 7)
plt.style.use('seaborn-dark-palette')

sns.boxenplot(df['Overall'], df['Age'], hue = df['Preferred Foot'], palette = 'rocket')
plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)
plt.show()


# We will have comparisions for `Age`, `Overall`, `Potential`, `Accelaration`, `SprintSpeed`, `Agility` , `Stamina`, `Strength`

# In[ ]:


cols = ['Age','Overall','Potential','Acceleration','SprintSpeed',"Agility","Stamina",'Strength','Preferred Foot']
df_small = df[cols]


# In[ ]:


df_small.head()


# In[ ]:


sns.pairplot(df_small, hue ='Preferred Foot',palette=["black", "red"],plot_kws=dict(s=50, alpha =0.8),markers=['^','v'])


# ### From the plot we can infer many things. 
# * Firstly, most of the people are right footed xD
# * Some unusual behavior in `Potential` vs `Overall`
# * `Accelaration` and `SprintSpeed` follow a proper linear relationship
# * `Agility` vs `SprintSpeed`, `Agility` vs `Accelaration` and `Agility` vs `Stamina` have somewhat linear relationship

# <a id="5"></a> <br>
# # 5.Modelling

# In[ ]:


df=pd.read_csv('../input/data.csv')


# In[ ]:


#DROP UNNECESSARY VALUES
drop_cols = df.columns[28:54]
df = df.drop(drop_cols, axis = 1)
df = df.drop(['Unnamed: 0','ID','Photo','Flag','Club Logo','Jersey Number','Joined','Special','Loaned From','Body Type', 'Release Clause',
               'Weight','Height','Contract Valid Until','Wage','Value','Name','Club'], axis = 1)
df = df.dropna()
df.head()


# In[ ]:


#Turn Real Face into a binary indicator variable
def face_to_num(df):
    if (df['Real Face'] == 'Yes'):
        return 1
    else:
        return 0
    
#Turn Preferred Foot into a binary indicator variable
def right_footed(df):
    if (df['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0

#Create a simplified position varaible to account for all player positions
def simple_position(df):
    if (df['Position'] == 'GK'):
        return 'GK'
    elif ((df['Position'] == 'RB') | (df['Position'] == 'LB') | (df['Position'] == 'CB') | (df['Position'] == 'LCB') | (df['Position'] == 'RCB') | (df['Position'] == 'RWB') | (df['Position'] == 'LWB') ):
        return 'DF'
    elif ((df['Position'] == 'LDM') | (df['Position'] == 'CDM') | (df['Position'] == 'RDM')):
        return 'DM'
    elif ((df['Position'] == 'LM') | (df['Position'] == 'LCM') | (df['Position'] == 'CM') | (df['Position'] == 'RCM') | (df['Position'] == 'RM')):
        return 'MF'
    elif ((df['Position'] == 'LAM') | (df['Position'] == 'CAM') | (df['Position'] == 'RAM') | (df['Position'] == 'LW') | (df['Position'] == 'RW')):
        return 'AM'
    elif ((df['Position'] == 'RS') | (df['Position'] == 'ST') | (df['Position'] == 'LS') | (df['Position'] == 'CF') | (df['Position'] == 'LF') | (df['Position'] == 'RF')):
        return 'ST'
    else:
        return df.Position

#Get a count of Nationalities in the Dataset, make of list of those with over 250 Players (our Major Nations)
nat_counts = df.Nationality.value_counts()
nat_list = nat_counts[nat_counts > 250].index.tolist()

#Replace Nationality with a binary indicator variable for 'Major Nation'
def major_nation(df):
    if (df.Nationality in nat_list):
        return 1
    else:
        return 0

#Create a copy of the original dataframe to avoid indexing errors
df1 = df.copy()

#Apply changes to dataset to create new column
df1['Real_Face'] = df1.apply(face_to_num, axis=1)
df1['Right_Foot'] = df1.apply(right_footed, axis=1)
df1['Simple_Position'] = df1.apply(simple_position,axis = 1)
df1['Major_Nation'] = df1.apply(major_nation,axis = 1)

#Split the Work Rate Column in two
tempwork = df1["Work Rate"].str.split("/ ", n = 1, expand = True) 
#Create new column for first work rate
df1["WorkRate1"]= tempwork[0]   
#Create new column for second work rate
df1["WorkRate2"]= tempwork[1]
#Drop original columns used
df1 = df1.drop(['Work Rate','Preferred Foot','Real Face', 'Position','Nationality'], axis = 1)
df1.head()


# In[ ]:


#Split ID as a Target value
target = df1.Overall
df2 = df1.drop(['Overall'], axis = 1)

#Splitting into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2, target, test_size=0.2)

#One Hot Encoding
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
print(X_test.shape,X_train.shape)
print(y_test.shape,y_train.shape)


# In[ ]:


#Applying Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#Finding the r2 score and root mean squared error
from sklearn.metrics import r2_score, mean_squared_error
print('r2 score: '+str(r2_score(y_test, predictions)))
print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, predictions))))


# Permutation importance is the first tool for understanding a machine-learning model, and involves shuffling individual variables in the validation data (after a model has been fit), and seeing the effect on accuracy.

# In[ ]:


perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())
#Top 3 important features are Potential, Age & Reactions 


# In[ ]:


#Visualising the results
plt.figure(figsize=(18,10))
sns.regplot(predictions,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':'0.7'},line_kws={'color':'black','alpha':0.5})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Linear Prediction of Player Rating")
plt.show()


# <font color="chocolate" size=+2.5><b>My Other Kernels</b></font>
# 
# Click on the button to view kernel...
# 
# 
# <a href="https://www.kaggle.com/nitindatta/eda-with-r3-id" class="btn btn-success" style="color:white;">EDA wirh R3*ID</a><br><br>
# 
# <a href="https://www.kaggle.com/nitindatta/ensemble-learning-part-1" class="btn btn-success" style="color:white;">Ensemble Learning Part 1</a><br><br>
# 
# <a href="https://www.kaggle.com/nitindatta/ensemble-learning-part-2" class="btn btn-success" style="color:white;">Ensemble Learning Part 2</a><br><br>
# 
# <a href="https://www.kaggle.com/nitindatta/students-performance-in-exams-eda-in-depth" class="btn btn-success" style="color:white;">Students performance in Exams- EDA in depth 📊📈</a><br><br>
# 
# <a href="https://www.kaggle.com/nitindatta/pulmonary-embolism-dicom-preprocessing-eda" class="btn btn-success" style="color:white;">🩺Pulmonary Embolism Dicom preprocessing & EDA🩺</a><br><br>
# 
# <a href="https://www.kaggle.com/nitindatta/first-kaggle-submission" class="btn btn-success" style="color:white;">Titanic: Machine Learning from Disaster</a><br><br>
# 
# <a href="https://www.kaggle.com/nitindatta/graduate-admission-chances" class="btn btn-success" style="color:white;">📖 Graduate Admission Chances 📕 📔</a><br><br>
# 
# <a href="https://www.kaggle.com/nitindatta/flower-classification-augmentations-eda" class="btn btn-success" style="color:white;">Flower_Classification+Augmentations+EDA</a><br><br>
# 
# <a href="https://www.kaggle.com/nitindatta/storytelling-with-gwd-pre-print-data" class="btn btn-success" style="color:white;">Storytelling with GWD pre_print data</a><br><br>
# 
# 
# ### If these kernels impress you,give them an <font size="+2" color="red"><b>Upvote</b></font>.<br>
# 
# <a href="#toc" class="btn btn-primary" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to TOP</a>
