#!/usr/bin/env python
# coding: utf-8

# # Aim of this notebook is to explore this dataset in much simple and clean way.

# # **Import**

# In[ ]:


import numpy as np 
import pandas as pd 
from statsmodels.graphics.gofplots import qqplot
import plotly.express as px
import seaborn as sns
sns.set(style = "darkgrid")
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


# # **Read Dataset**

# In[ ]:


sample = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
print("Sample")
display(sample.head(2))
print()
print("Train")
display(train.head(2))
print()
print("Test")
display(test.head(2))


# # **Overall view of dataset**
# * train 900000 rows, test 700000 rows
# * no nan values in our datasets
# * 31 features
# * f_00 - f_06 and f_19 - f_26 and f_28 := float columns
# * f_07 - f_18 and f_29 - f_30 := int columns
# * f_27 := object column
# * target columns := binary (0/1) and target is almost balanced #0: 462161 and #1: 437839

# In[ ]:


print("Sample, train, test")
print(sample.shape, train.shape, test.shape)
print()
print("No of null values")
print(sample.isnull().sum().sum(), train.isnull().sum().sum(), test.isnull().sum().sum())
print()
features = test.drop("id", axis=1).columns.tolist()
print(features)
print()
print(train.info())


# In[ ]:


print(train.target.value_counts())
sns.countplot(x=train['target'])
plt.show()


# # **Here we look at no of unique value in each columns**
# 
# 
# * 700000+900000 = 1600000 [train+test]
# * all float columns has different values in each row <br>
# * all int columns has total at most 17 different unique values ( so these are some sort of categorical variables))<br>
# * f_27 which is an object column(as has string entry) has total 1181880 unique values 160000-1181880=418120 repetitions
# * In int columns there are many unique values whose frequency is less than 1%(see below). We can combine them to create new feature.

# In[ ]:


full_data = pd.concat([train[features],test[features]], axis=0)
print(full_data.shape)
print()
list(zip(full_data.columns, full_data.dtypes, full_data.nunique()))


# In[ ]:


cat_features = [i for i in features if full_data[i].nunique() <= 17]
num_features = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28']
print("features with no of unique values less than equal to 17")
print(cat_features)
print()
print("% of unique values")
for feat in cat_features:
    print(feat,":")
    a = full_data[feat].value_counts()*100/full_data.shape[0]
    print(a)
    print("="*40)
    print()


# In[ ]:


print("f_27 :")
print(full_data.f_27.value_counts())


# # **f_27**
# **The f_27 column contains string of length 10 characters, Let's try to explore these encoding.**
# 
# We first created a new dataframe from f_27 by splitting these strings into 10 columns of each characters.<br>
# We notice following things of this encoding:
# * f0, f2, f5 : contains only two characters A,B  (can be used to create new features)
# * f1, f3, f4, f6, f8, f9: all contains characters from A to O 
# * f7: contains charactes from A to T
# * f1, f3, f4, f6, f8, f9 : all has same distribution of characters 
# * except f7 which has almost same frequency of each character

# In[ ]:


data_f_27 = pd.DataFrame([list(i) for i in sorted(full_data.f_27.value_counts().index.values)])
data_f_27.columns = ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9"]
data_f_27.head(3)


# In[ ]:


for i in range(10):
    print(data_f_27.groupby([f"f{i}"]).count().iloc[:,0])
    print("="*40)
    print()


# 
# * f0 and f5 have very similar distribution while f2 has just opposite distribution

# In[ ]:


plt.figure(figsize=(12,6))
for i in [0,2,5]:
    d= data_f_27[f"f{i}"].value_counts()
    plt.plot(d,label=f"f{i}")
plt.legend()
plt.show()
plt.figure(figsize=(12,6))
for i in [1,3,4,6,7,8,9]:
    d= data_f_27[f"f{i}"].value_counts()
    plt.plot(d, label=f"f{i}")
plt.legend()
plt.show()


# <code>px.treemap()</code> is used to visualize proportions for multiple columns at at time.

# In[ ]:


fig  = px.treemap(data_f_27.sample(20), path= data_f_27.columns.tolist() ) 
fig.show()


# * Parallel Sets represents contribution of columns on each other. <br>
# * It is used to represent inter-connection among columns.<br>
# * Note: It works only for Object and int data type columns.<br>
# * We can set color value based on a column which can be int/float type.
# > We have created two plots:-
# 1. In first plot we have taken full_data i.e. train+test 
# 1. In second plot we have taken only train data with target as color

# In[ ]:


px.parallel_categories(data_f_27.sample(200)) # train+test


# In[ ]:


train_f_27= pd.DataFrame([list(i) for i in train.f_27.value_counts().index.values])
train_f_27.columns = ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9"]
train_f_27["target"] = train.target
train_f_27.head(3)


# In[ ]:


px.parallel_categories(train_f_27.head(800),color="target") # train


# # features = cat_features + num_features + f_27
# * cat_features := 14
# * num_features := 16
# * f_27

# In[ ]:


display(full_data[cat_features].head(2))

display(full_data[num_features].head(2))

display(full_data[["f_27"]].head(2))


# ## Categorial features(features which has no of unique values less than equal to 17)
# * both train and test set have same distribution 
# * both train and test set don't follow normal distribution
# 
# Q-Q plot also known as (Quantile-Quantile plot) is used to check whether our data follows normal distribution or not.
# If our plot lies on the red line(y=x) then it is normally distributed. It it don't lie on the y=x line then our feature is not normally distributed.

# In[ ]:


print("histplot"," "*3,"Kde plot"," "*3, "Boxplot"," "*3,"QQplot train"," "*3,"QQplot test")
fig, axes = plt.subplots(14,5, figsize=(25,60))
axes = axes.flatten()
for i in range(0,len(axes),5):
    col = cat_features[i//5]
    ax = axes[i]
    train[col].hist(ax= ax,bins=20, color="r",alpha=.5, label="train")
    test[col].hist(ax= ax,bins=20, color="b", alpha=.5, label="test")
    
    sns.kdeplot(train[col], color="red", label="train", ax=axes[i+1])
    sns.kdeplot(test[col],  color="green", label="test", ax=axes[i+1])
    axes[i+1].legend()
    
    sns.boxplot(data=train[col], color="red",ax=axes[i+2])
    sns.boxplot(data= test[col],  color="green", ax=axes[i+2])
    axes[i+2].legend() 
    
    t1= (train[col].values - train[col].values.mean())/ train[col].values.std()
    t2= (test[col].values - test[col].values.mean())/ test[col].values.std()
    qqplot(t1,line="s",ax=axes[i+3])
    qqplot(t2,line="s",ax=axes[i+4])
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'f{cat_features[i//5]}', loc = 'right', fontsize = 12)
    ax.legend()
    fig.suptitle("distribution of train-test cat_features")
    fig.tight_layout()  
plt.show()


# ## numerical features
# * both train and test set have same distribution 
# * both train and test set follow normal distribution <b>[with slight deviation from normal behaviour for f_25 and f_26]</b>

# In[ ]:


print("histplot"," "*3,"Kde plot"," "*3, "Boxplot"," "*3,"QQplot train"," "*3,"QQplot test")
fig, axes = plt.subplots(16,5, figsize=(25,70))
axes = axes.flatten()
for i in range(0,len(axes),5):
    col = num_features[i//5]
    ax = axes[i]
    train[col].hist(ax= ax,bins=20, color="r",alpha=.5, label="train")
    test[col].hist(ax= ax,bins=20, color="b", alpha=.5, label="test")

    sns.kdeplot(train[col], color="red", label="train", ax=axes[i+1])
    sns.kdeplot(test[col],  color="green", label="test", ax=axes[i+1])
    axes[i+1].legend()
    
    sns.boxplot(data=train[col], color="red",ax=axes[i+2])
    sns.boxplot(data= test[col],  color="green", ax=axes[i+2])
    axes[i+2].legend()    
    
    t1= (train[col].values - train[col].values.mean())/ train[col].values.std()
    t2= (test[col].values - test[col].values.mean())/ test[col].values.std()
    qqplot(t1,line="s",ax=axes[i+3])
    qqplot(t2,line="s",ax=axes[i+4])
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'{num_features[i//5]}', loc = 'right', fontsize = 12)
    ax.legend()
    fig.suptitle("distribution of train-test num_features")
    fig.tight_layout()   
plt.show()


# ## Now for numerical columns we will see if there is any outlier, if present then we will remove it.

# In[ ]:


def check_outlier(data,col_name):
    """
    input:= data, column name
    output:= Lower wishker and Upper wishker 
    """
    Q3 = data[col_name].quantile(0.75)
    Q1 = data[col_name].quantile(0.25)
    IQR = Q3-Q1 
    print("75%:", Q3)
    print("25%",Q1)
    print("IQR:",IQR)
    
    LW = Q1 - 1.5*IQR 
    UW = Q3 + 1.5*IQR 
    print("Lower and Upper Wishker: ",LW, UW)
    print("Min and Max value: ", np.min(data[col_name]),np.max(data[col_name]))
    print("Full data:", data.shape)
    print("No of outliers: ",data[(data[col_name]<LW) | (data[col_name]>UW)].shape)
    
    sns.boxplot(x=data[col_name])
    sns.stripplot(x=data[col_name], color="0.5")
    plt.show()
    return LW, UW
    


# In[ ]:


for c in num_features:
    print("Column: ",c)
    LW, UW= check_outlier(train,c)
    print("After removing outliers")
    train=train[(train[c]>= LW) & (train[c]<= UW)]
    sns.boxplot(x=train[c])
    sns.stripplot(x=train[c], color="0.5")
    plt.show()
    print("="*40)


# ## Plot after removing outliers from train set.
# * We can see that, now our train set is not following normal distribution in tail region, because we have removed outliers(from tails). But now our dataset is much more stable.

# In[ ]:


print("histplot"," "*3,"Kde plot"," "*3, "Boxplot"," "*3,"QQplot train"," "*3,"QQplot test")
fig, axes = plt.subplots(16,5, figsize=(25,70))
axes = axes.flatten()
for i in range(0,len(axes),5):
    col = num_features[i//5]
    ax = axes[i]
    train[col].hist(ax= ax,bins=20, color="r",alpha=.5, label="train")
    test[col].hist(ax= ax,bins=20, color="b", alpha=.5, label="test")

    sns.kdeplot(train[col], color="red", label="train", ax=axes[i+1])
    sns.kdeplot(test[col],  color="green", label="test", ax=axes[i+1])
    axes[i+1].legend()
    
    sns.boxplot(data=train[col], color="red",ax=axes[i+2])
    sns.boxplot(data= test[col],  color="green", ax=axes[i+2])
    axes[i+2].legend()    
    
    t1= (train[col].values - train[col].values.mean())/ train[col].values.std()
    t2= (test[col].values - test[col].values.mean())/ test[col].values.std()
    qqplot(t1,line="s",ax=axes[i+3])
    qqplot(t2,line="s",ax=axes[i+4])
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'{num_features[i//5]}', loc = 'right', fontsize = 12)
    ax.legend()
    fig.suptitle("distribution of train-test num_features")
    fig.tight_layout()   
plt.show()


# # Feature correlation
# * correlation between numerical features <b>[ there is some correlation between (f_28, f_2) (f_28, f_3) (f_28, f_5) and (f_25, f_23) ]</b>
# * correlation between categorical features <b>[ there is no correlation among categorical features ]</b>

# In[ ]:


feat = num_features 
fig, ax = plt.subplots(1,2,figsize=(32,11))         # Sample figsize in inches
ax[0].title.set_text("train")
ax[1].title.set_text("test")
sns.heatmap(train[feat].corr().abs(), cmap="viridis", linewidths=.5, ax=ax[0], annot=True, fmt=".2f")
sns.heatmap(test[feat].corr().abs(), cmap="viridis",linewidths=.5, ax=ax[1], annot=True, fmt=".2f")
plt.show()

## threshold of .2
fig, ax = plt.subplots(1,2,figsize=(32,11))         # Sample figsize in inches
ax[0].title.set_text("train")
ax[1].title.set_text("test")
sns.heatmap(train[feat].corr().abs()>.2, cmap="coolwarm", linewidths=.5, ax=ax[0],annot=True, fmt=".2f")
sns.heatmap(test[feat].corr().abs()>.2, cmap="coolwarm",linewidths=.5, ax=ax[1],annot=True, fmt=".2f")
plt.show()


# In[ ]:


feat = cat_features 
fig, ax = plt.subplots(1,2,figsize=(32,11))         
ax[0].title.set_text("train")
ax[1].title.set_text("test")
sns.heatmap(train[feat].corr().abs(), cmap="viridis", linewidths=.5, ax=ax[0], annot=True, fmt=".2f")
sns.heatmap(test[feat].corr().abs(), cmap="viridis",linewidths=.5, ax=ax[1], annot=True, fmt=".2f")
plt.show()
## threshold of .2
fig, ax = plt.subplots(1,2,figsize=(32,11))         
ax[0].title.set_text("train")
ax[1].title.set_text("test")
sns.heatmap(train[feat].corr().abs()>.2, cmap="coolwarm", linewidths=.5, ax=ax[0],annot=True, fmt=".2f")
sns.heatmap(test[feat].corr().abs()>.2, cmap="coolwarm",linewidths=.5, ax=ax[1],annot=True, fmt=".2f")
plt.show()


# In[ ]:


print(train.shape, test.shape, sample.shape) # final dataset after removing outliers


# # If you like my work please do upvote!
# **<span style="color:#444160;"> Thanks!🙂</span>**<br>
# .<br>
# .<br>
# .
# 
# <img src="https://media.giphy.com/media/SfYTJuxdAbsVW/giphy.gif" width=70%>

# In[ ]:




