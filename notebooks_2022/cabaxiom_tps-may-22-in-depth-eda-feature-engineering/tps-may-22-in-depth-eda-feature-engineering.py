#!/usr/bin/env python
# coding: utf-8

# # TPS MAY 22 - Exploratory Data Analysis

# > For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. The data has various feature interactions that may be important in determining the machine state.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif

sns.set_style('darkgrid')


# In[ ]:


train_df = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test_df = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")


# In[ ]:


display(train_df.head())
display(test_df.head())


# In[ ]:


print("Train Rows:", train_df.shape[0], "Train Cols:", train_df.shape[1])
print("Test Rows:", test_df.shape[0], "Test Cols:", test_df.shape[1])


# In[ ]:


train_df.info()


# In[ ]:


print("Missing values - train:", train_df.isna().sum().sum())
print("Missing values - test:", test_df.isna().sum().sum())


# In[ ]:


print("Unique train IDs:", train_df["id"].nunique())
print("Train ID range:", train_df["id"].min(), "to", train_df["id"].max())
print("Unique test IDs:", test_df["id"].nunique())
print("Test ID range:", test_df["id"].min(), "to", test_df["id"].max())


# ## Target

# In[ ]:


def val_count_df(df, column_name, sort=True):
    value_count = df[column_name].value_counts(sort=sort).reset_index().rename(columns={column_name:"Value Count","index":column_name}).set_index(column_name)
    value_count["Percentage"] = df[column_name].value_counts(sort=sort,normalize=True)*100
    value_count = value_count.reset_index()
    return value_count


# In[ ]:


target_count = val_count_df(train_df, "target")
display(target_count)
target_count.set_index("target").plot.pie(y="Value Count", figsize=(10,7), legend=False, ylabel="");


# In[ ]:


from statsmodels.stats.proportion import proportions_ztest

#perform one proportion z-test
zval, pval = proportions_ztest(count=target_count.loc[0,"Value Count"], nobs=target_count.loc[:,"Value Count"].sum(), value=0.5)
if pval > 0.05:
    print("p-value = {0:0.03f}".format(pval), "- p > 0.05, we reject the null hypothesis that the classes are not balanced (50% split).")
else:
    print("p-value = {0:0.20f}".format(pval), "- p < 0.05, we accept the null hypothesis that the classes are not balanced (50% split).")


# **Observations:**
# - For our purposes the targets are balanced, although statistically the proportions are not a perfect 50:50 split.

# ## Features

# In[ ]:


feature_cols = [col for col in train_df.columns if "f_" in col]
dtype_cols = [train_df[i].dtype for i in feature_cols]
dtypes = pd.DataFrame({"features":feature_cols, "dtype":dtype_cols})
float_cols = dtypes.loc[dtypes["dtype"] == "float64", "features"].values.tolist()
int_cols = dtypes.loc[dtypes["dtype"] == "int64", "features"].values.tolist()


# ## Feature Correlations + Interactions

# In[ ]:


plt.subplots(figsize=(25,20))
sns.heatmap(train_df[feature_cols + ["target"]].corr(),annot=True, cmap="RdYlGn", fmt = '0.2f', vmin=-1, vmax=1, cbar=False);


# ### Float Correlations

# In[ ]:


plt.subplots(figsize=(10,10))
sns.heatmap(train_df[float_cols + ["target"]].corr(),annot=True, cmap="RdYlGn", fmt = '0.2f', vmin=-1, vmax=1, cbar=False);


# We plot the different features values against each other, with the target 0 or 1 shown by the colours blue and orange respecitvely.

# In[ ]:


sns.pairplot(train_df[float_cols+["target"]].sample(5000), hue="target", plot_kws=dict(linewidth=1, s=10,  alpha=0.9, edgecolors="face"));


# It's a little hard to see as theses so many features, so lets look at a subset of features:

# In[ ]:


selected_float_cols = ["f_19","f_20","f_21","f_22","f_25","f_26"]
sns.pairplot(train_df[selected_float_cols+["target"]].sample(20000), hue="target", plot_kws=dict(linewidth=1, s=10,  alpha=0.9, edgecolors="face"));


# In[ ]:


selected_float_cols = ["f_00","f_01","f_02","f_04","f_05"]
sns.pairplot(train_df[selected_float_cols+["target"]].sample(20000), hue="target", plot_kws=dict(linewidth=1, s=10,  alpha=0.9, edgecolors="face"));


# **Observations:**
# - We can see definite distinct patches of blue and orange, particularly for some features.

# ### Integer Correlations

# In[ ]:


plt.subplots(figsize=(10,10))
sns.heatmap(train_df[int_cols + ["target"]].corr(),annot=True, cmap="RdYlGn", fmt = '0.2f', vmin=-1, vmax=1, cbar=False);


# In[ ]:


g = sns.PairGrid(train_df[int_cols+["target"]].sample(10000),hue="target")
g = g.map_upper(sns.histplot, discrete=(True,True))
g = g.map_lower(sns.kdeplot,shade=True)
g = g.map_diag(plt.hist)
g.tight_layout()
plt.show()


# In[ ]:


#The same plot but without using the target for hue
g = sns.PairGrid(train_df[int_cols+["target"]].sample(10000))
g = g.map_upper(sns.histplot, discrete=(True,True))
g = g.map_lower(sns.kdeplot,shade=True)
g = g.map_diag(sns.histplot, binwidth=1)
g.tight_layout()
plt.show()


# Looking at a subset of features

# In[ ]:


selected_int_cols = ["f_08","f_09","f_10","f_11","f_12"]
g = sns.PairGrid(train_df[selected_int_cols].sample(10000))
g = g.map_upper(sns.histplot, discrete=(True,True))
g = g.map_lower(sns.kdeplot,shade=True)
g = g.map_diag(sns.histplot, binwidth=1)
g.tight_layout()
plt.show()


# ## Distributions

# ### Float Features Distribtuions:

# In[ ]:


plt.subplots(figsize=(25,35))
for i, column in enumerate(float_cols):
    plt.subplot(6,3,i+1)
    #plt.hist(train_df.loc[train_df["target"] == 0, column], bins = 100, color="blue")
    #plt.hist(train_df.loc[train_df["target"] == 1, column], bins = 100, color="orange")
    sns.histplot(data=train_df, x=column, hue="target")
    plt.title(column)


# **Observations:**
# - Normal distributions, symetrical around mean of 0.
# - The distributions have different scales (different standard deviations) 

# ### Int Features Distribution

# In[ ]:


plt.subplots(figsize=(25,30))
for i, column in enumerate(int_cols):
    val_count = train_df[column].value_counts()
    ax = plt.subplot(5,3,i+1)
    #sns.barplot(x=val_count.index,y=val_count.values)
    ax.bar(val_count.index, val_count.values)
    ax.set_xticks(val_count.index)
    plt.title(column)


# In[ ]:


plt.subplots(figsize=(25,30))
for i, column in enumerate(int_cols):
    val_count = train_df[[column,"target"]].value_counts().rename("value_counts").reset_index()
    plt.subplot(5,3,i+1)
    ax = sns.barplot(data = val_count, x=column, y="value_counts", hue="target")
    ax.set_xlabel(None)
    plt.title(column)


# **Observations:**
# 
# - All integer values >= 0
# - Most features have 14 possible values, but normally only ~8 values are frequent.
# - f_29 only has 2 possible values 0 or 1.
# - f_30 only has 3 possible values 0,1,2.

# ## f_27

# f_27 is special as its the only string column. Each string contains 10 uppercase letters.

# In[ ]:


train_df["f_27"].head(10)


# In[ ]:


# Checking ALL values have 10 characters
train_df["f_27"].apply(lambda x:len(x)).unique()


# In[ ]:


import string
alphabet_upper = list(string.ascii_uppercase)

char_counts = []
for character in alphabet_upper:
    char_counts.append(train_df["f_27"].str.count(character).sum())


# In[ ]:


char_counts_df = pd.DataFrame({"Character": alphabet_upper, "Character Count": char_counts})
char_counts_df = char_counts_df.loc[char_counts_df["Character Count"] > 0]
print(np.sum(char_counts))
#No other hidden characters

plt.subplots(figsize=(20,7))
sns.barplot(data = char_counts_df, x="Character", y="Character Count", color="blue");
plt.title("Total number of characters in f_27 - train");


# **Observations:**
# 
# There are 20 possible characters (A - T).

# In[ ]:


char_counts_df = char_counts_df.set_index("Character", drop=False)


# In[ ]:


for i in range(10):
    char_counts_df["character"+str(i+1)] = train_df["f_27"].str[i].value_counts()
char_counts_df = char_counts_df.fillna(0)


# In[ ]:


f,ax = plt.subplots(figsize=(20,30))
character_cols = [i for i in char_counts_df.columns if "character" in i]
for i, column in enumerate(character_cols):
    ax = plt.subplot(5,2,i+1)
    ax = sns.barplot(data = char_counts_df, x="Character", y=column, color="blue");
    plt.title("Character value counts in position: " +str(i+1));
    ax.set_ylabel("Character Count")


# **Observations:**
# - Characters 1,3 and 6 in the string can only take values A and B.
# - Character 8 in the string can be any value (A-T) with mostly equal probability.

# # Feature Engineering - Starter

# We can use the string from f_27 to create new features.
# 
# The most obvious features to add is to create a seperate feature for all 10 character positions in f_27. So our new features would be:
# - f_27_char1
# - f_27_char2
# - ...
# - f_27_char10

# In[ ]:


#f_27_char1 would be:
train_df["f_27"].str[0].head()


# We could then use these features as categorical featues. However we need to answer the question of whether the closeness of characters is important (e.g. is A more similar to B than A is to C). The answer is probably yes as:
# 1. The alphabet does have an order
# 2. We can see from the frequency count of characters that some order is present (e.g. for character 10: freq C > freq D > freq E > freq F > freq G...)
# 
# So lets create encode the characters ordinally (A=0, B=1, etc.):

# In[ ]:


# for example char1 would be
train_df["f_27"].str[0].apply(lambda x: ord(x) - ord("A")).head()


# Lets create these 10 features.
# 
# In addition we add the number of unique characters as a feature for example "AABAABAABA" has 2 unique features.

# In[ ]:


#For example:
display(train_df["f_27"].head(4))
#Would become:
train_df["f_27"].apply(lambda x: len(set(x))).head(4)


# In[ ]:


def feature_engineer(df):
    new_df = df.copy()
    
    for i in range(10):
        new_df["f_27_"+str(i)] = new_df["f_27"].str[i].apply(lambda x: ord(x) - ord("A"))
    
    new_df["unique_characters"] = new_df["f_27"].apply(lambda x: len(set(x)))
    
    new_df = new_df.drop(columns=["id", "f_27"])
    return new_df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df_2 = feature_engineer(train_df)\ntest_df_2 = feature_engineer(test_df)')


# In[ ]:


f27_cols = [i for i in train_df_2.columns if "f_27" in i]
plt.subplots(figsize=(10,10))
sns.heatmap(train_df_2[f27_cols].corr(),annot=True, cmap="RdYlGn", fmt = '0.2f', vmin=-1, vmax=1, cbar=False);


# In[ ]:


g = sns.PairGrid(train_df_2[f27_cols+["target"]].sample(1000))

g = g.map_lower(sns.histplot, discrete=(True,True))
g = g.map_diag(sns.histplot, binwidth=1)

g = g.map_upper(sns.kdeplot, shade=True)
g.tight_layout()


# ## PCA

# In[ ]:


int_cols = [col for col in train_df_2.columns if (train_df_2[col].dtype == "int" and col != "target")]
float_cols = [col for col in train_df_2.columns if train_df_2[col].dtype == "float"]


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
standardised_train = ss.fit_transform(train_df_2[int_cols + float_cols])

pca = PCA(random_state = 10, whiten = True)
pca.fit(standardised_train)

X_PCA = pca.transform(standardised_train)

PCA_df = pd.DataFrame({"PCA_1" : X_PCA[:,0], "PCA_2" : X_PCA[:,1],  "target":train_df_2["target"]})
    
plt.figure(figsize=(14, 14))
sns.scatterplot(data = PCA_df, x = "PCA_1", y = "PCA_2", hue = "target", s=3);


# In[ ]:


f,ax = plt.subplots(figsize=(10,7))
sns.lineplot(x=range(1,len(pca.explained_variance_ratio_)+1), y=np.cumsum(pca.explained_variance_ratio_));
ax = sns.lineplot(x=[0,41],y=[0,1], color="red", dashes=True, linestyle = "--" )
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance");


# In[ ]:


ss_test = ss.transform(test_df_2[int_cols + float_cols])
X_test_PCA = pca.transform(ss_test)
PCA_test_df = pd.DataFrame({"PCA_1" : X_test_PCA[:,0], "PCA_2" : X_test_PCA[:,1]})

PCA_df["data"] = "Train"
PCA_test_df["data"] = "Test"
PCA_combined = pd.concat([PCA_df,PCA_test_df])

plt.figure(figsize=(14, 14))
sns.scatterplot(data = PCA_combined, x = "PCA_1", y = "PCA_2", hue = "data", s=3);


# **Observations:**
# - All components useful
# - Train test follow the same distribution

# ## Assessing Feature Performance

# ### Mutual Information

# Firsty lets consider Mutual Information:
# 
# > The mutual information (MI) between two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other. If you knew the value of a feature, how much more confident would you be about the target? - https://www.kaggle.com/code/ryanholbrook/mutual-information
# 
# MI is similar to the correlation between feature and target only MI can detect non-linear relationships.

# In[ ]:


y = train_df_2["target"]
X = train_df_2.drop(columns="target")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n#Adapted from https://www.kaggle.com/code/ryanholbrook/mutual-information\ndef make_mi_scores(X, y, discrete_features):\n    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features) #Our target variable is discrete 0 or 1 so we use mutual_info_classif\n    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)\n    mi_scores = mi_scores.sort_values(ascending=False)\n    return mi_scores\n\nmi_scores = make_mi_scores(X, y, discrete_features = X.dtypes == int) #in this case all discrete cols are ints')


# In[ ]:


f,ax = plt.subplots(figsize=(20,15))
sns.barplot(y=mi_scores.index, x=mi_scores.values, color="blue");


# **Warning:**
# 
# Mutual Information can't detect interactions between features. It is a univariate metric. The focus of this TPS is feature interactions, so these MI scores (and the plots below) may not be a good predictor of actual feature importance. However they still may be useful for experimenting with created features.

# Now lets estimate the importance of the features visually. We can see how the graphs are linked to the MI score:

# ### Importance of integer features

# For each feature - including those we just created we plot the mean target value for each integer - including those we just created. We remove any integer values with low value counts (<200) for clarity.

# In[ ]:


f,ax = plt.subplots(figsize=(20,80))
for i, column in enumerate(int_cols):
    temp_df = train_df_2.groupby([column])["target"].mean()
    temp_df_2 = train_df_2[column].value_counts()
    temp_df = temp_df[temp_df_2 > 200]
    plt.subplot(15,2,i+1)
    ax = sns.barplot(x=temp_df.index, y=temp_df.values, color="blue")
    ax.set_ylim([0,1])
    plt.ylabel("mean target value")
    plt.xlabel(None)
    plt.title("Feature: " + column)


# **Observations:**
# - The newly created features appear to very useful.

# ### Importance of float features

# To assess the performance of float features we:
# 1. Sort values by the feature value
# 2. Calculate the rolling mean of the target values with a large window size
# 3. Plot the feature value against the rolling mean of the target

# In[ ]:


f,ax = plt.subplots(figsize = (25,40))
for n, col in enumerate(float_cols):
    temp_df = pd.DataFrame({col: train_df_2[col].values, "target":train_df_2["target"]})
    temp_df = temp_df.sort_values(col).reset_index(drop=True)
    temp_df["rolling_mean"] = temp_df["target"].rolling(10000, center=True).mean()
    
    ax = plt.subplot(8,2,n+1)
    sns.scatterplot(data=temp_df,x=col,y="rolling_mean",s=3)
    ax.set_ylim([0.15,0.85])
    ax.set_ylabel(None)


# **Observations:**
# - Mean target levels varies non-linearally with feature values
# - Features 19-28 look the most useful.

# ## **Work in progress**

# In[ ]:





# In[ ]:





# In[ ]:




