#!/usr/bin/env python
# coding: utf-8

# # EnerjiSA Üretim Hackathon - Wind Force
# ## Doç. Dr. Oetker
# 
# - **Gökay Aydoğan**  
# - **Onur Hakkı Eyüboğlu**  
# - **Oğuzhan Kır**
# 
# ![Rüzgar Türbini Balıkesir](https://www.enerjisauretim.com.tr/_assets/images/gallery/balikesir_res_5.jpg)

# Note1: We did very, very detailed EDA and feature engineering in 2 weeks. I wanted to share some of the visualizations we made on this notebook. Since the second stage of the competition will be in a similar format, we can share a small part of what we do. After the second stage, we will share our detailed EDA and feature engineering processes.
# 
# Note2: I did not plot all the columns in the data to avoid too many graphics.
# 
# Note3: Thanks to everyone who contributed to the competition.

# # Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
import missingno as msno
from statsmodels.tsa.stattools import adfuller
from matplotlib import cycler
import warnings

warnings.filterwarnings("ignore")


# # Read data

# In[ ]:


featuresDF = pd.read_csv("../input/enerjisa-uretim-hackathon/features.csv")
powerDF = pd.read_csv("../input/enerjisa-uretim-hackathon/power.csv")
submissionCSV = pd.read_csv("../input/enerjisa-uretim-hackathon/sample_submission.csv")


# # Preprocess - EDA

# In[ ]:


#This function converts the dates in the Timestamp column to datetime and turns them into an index.
def set_time_index(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")
    
    #change dtype to float32
    df = df.astype(np.float32)
    return df

featuresDF = set_time_index(featuresDF)
powerDF = set_time_index(powerDF)
allDF = pd.concat([featuresDF, powerDF], axis=1)


# In[ ]:


#This function divides the data according to the dates given in the submission file.
def train_submission_split(df, power=True):
    
    split_date = "2021-08-15"
    trainDF = df.loc[df.index < split_date]
    submissionxDF = df.loc[df.index >= split_date]
    if power==False:
        submissionxDF = submissionxDF.drop("Power(kW)", axis=1)

    return trainDF, submissionxDF

trainDF, submissionxDF = train_submission_split(allDF, power=False)


# In[ ]:


#This function prints a summary of the data.
def print_summary(df):
    print("Df.shape: ", df.shape, "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")
    print("Summary:\n", df.describe(), "\n")
    print("İnfo:\n", df.info())
    print("Columns:\n", df.columns, "\n")

print_summary(allDF)


# We wrote a function where we can examine the general information in the data at once. The first thing that caught our eye here was that there were values like 99999 in the columns. We also thought that too much missing data would cause problems.

# In[ ]:


"""
This function visualizes how many missing values are in the columns in the data, where the missing values are located in the data, 
and how strongly the presence or absence of one variable affects the presence of the other with a heatmap plot.
"""
def plot_missing_values(df):
    msno.bar(df)
    plt.show()
    msno.matrix(df, freq='M')
    plt.show()
    msno.heatmap(df)
    
plot_missing_values(allDF[allDF.columns[-20:]])


# When we drew the columns in the data with the train and test parts side by side, we clearly understood that the high values that caught our eye greatly distorted the distribution. We thought these were manually added noises rather than a sensor error.

# In[ ]:


trainDF,submissionxDF = train_submission_split(allDF,power=True)

fig, axes = plt.subplots(nrows=19, ncols=4, dpi=60, figsize=(30,50))
for i, ax in enumerate(axes.flatten()):
    if trainDF.columns[i] != "Power(kW)":
        data = trainDF[trainDF.columns[i]]
        data2 = submissionxDF[submissionxDF.columns[i]]
        ax.plot(data, color='red', linewidth=0.2)
        ax.plot(data2, color='blue', linewidth=0.2)
        ax.set_title(trainDF.columns[i])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

plt.tight_layout();


# We converted very high values such as 99999 to missing values. Later, when we plotted the distributions again, we were able to see the real outliers. However, we didn't want to play around too much in case these were sensor errors. For this reason, we made manual corrections by investigating the wind turbines and estimating how much the values could take.

# In[ ]:


#This function converts the outlier values we observe in the data to NaN values.
def outlier_to_NaN(df):
    df = df.replace(99999,np.NaN)
    for i in allDF.columns:
        if "Temp" in i:
            df[i] = df[i].apply(lambda x: x if (x>0) else np.nan)
    df["Torque"] = df["Torque"].apply(lambda x: 105 if (x>105) else x)
    for col in df:
        if "Tower Acceleration Normal" in col:
            df[col][df[col]>450] = np.NaN
        if "Moment D Direction" in col:
            df[col][df[col]<-1000] = np.NaN
        if "Moment D Filtered" in col:
            df[col][df[col]<-1000] = np.NaN
        if "Pitch Offset-2 Asymmetric Load Controller" in col:
            df[col][df[col]>0.15] = np.NaN
        if "Pitch Offset Tower Feedback" in col:
            df[col][df[col]<-0.010] = np.NaN
        if "External Power Limit" in col:
            df[col][df[col]<3050] = np.NaN
        if "Tower Accelaration Normal Raw" in col:
            df[col][df[col]<-2000] = np.NaN
        if "Blade-2 Actual Value_Angle-B" in col:
            df[col][df[col]<-250] = np.NaN
            df[col][df[col]>250] = np.NaN
        if "Blade-1 Actual Value_Angle-B" in col:
            df[col][df[col]<-250] = np.NaN
            df[col][df[col]>250] = np.NaN
        if "Blade-3 Actual Value_Angle-B" in col:
            df[col][df[col]<-50] = np.NaN
            df[col][df[col]>50] = np.NaN
        if "Pitch Offset-1 Asymmetric Load Controller" in col:
            df[col][df[col]<-0.15] = np.NaN
        if "Tower Accelaration Lateral Raw" in col:
            df[col][df[col]>1000] = np.NaN
        if "Pitch Offset-3 Asymmetric Load Controller" in col:
            df[col][df[col]<-0.05] = np.NaN
    return df
allDF = outlier_to_NaN(allDF)
trainDF,submissionxDF = train_submission_split(allDF,power=True)


# In[ ]:


fig, axes = plt.subplots(nrows=19, ncols=4, dpi=60, figsize=(30,50))
for i, ax in enumerate(axes.flatten()):
    if trainDF.columns[i] != "Power(kW)":
        data = trainDF[trainDF.columns[i]]
        data2 = submissionxDF[submissionxDF.columns[i]]
        ax.plot(data, color='red', linewidth=0.2)
        ax.plot(data2, color='blue', linewidth=0.2)
        ax.set_title(trainDF.columns[i])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

plt.tight_layout();


# After the distribution of the data looked as we wanted, we started to examine the values in the power column.

# In[ ]:


trainDF["Power(kW)"].plot(figsize=(30,5),color="b",alpha=0.8,title="Power(kW)",fontsize=15,lw=0.3);


# We examined the distributions of the features in the data, and according to our review, the distributions in most features had a skewed normal distribution. Since tree-based gradient boosting algorithms such as XGBoost work better on normally distributed data, we then transform the distribution of these features to approximate the normal distribution.

# In[ ]:


#This function plots the distribution and probability plot of the given column.
def plot_dist(df, target):

    fig, axes = plt.subplots(ncols=2, figsize=(24, 6), dpi=100)

    sns.kdeplot(df[target], label=target, fill=True, ax=axes[0])
    axes[0].axvline(df[target].mean(), 
                    label='Mean', color='r', 
                    linewidth=2, linestyle='-')
    
    axes[0].axvline(df[target].median(), 
                    label='Median', color='b', 
                    linewidth=2, linestyle='-')
    
    axes[0].axvline(df[target].min(), 
                    label='Min', 
                    linewidth=2, linestyle='--')
    
    axes[0].axvline(df[target].max(), 
                    label='Max', linewidth=2, 
                    linestyle='--')
    
    axes[0].legend(prop={'size': 15})
    probplot(df[target], plot=axes[1])
    
    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=12.5)
        axes[i].tick_params(axis='y', labelsize=12.5)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
    axes[0].set_title(f'{target} Distribution', 
                      fontsize=15, pad=12)
    
    axes[1].set_title(f'{target} Probability Plot', 
                      fontsize=15, pad=12)
    return

for feature in allDF.columns[-5:]:
    plot_dist(allDF, feature)


# We investigated whether the power generation is stationary with the dickey fuller test. Since the time series is stationary, we did not need seasonal analysis.

# In[ ]:


#This function checks whether the time series is stationary with the Dickey Fuller test.
def check_adfuller(ts):
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
    if result[4]['1%'] > result[0]:
        print('Time series is stationary')
    else:
        print('Time series is Non-stationary')

#This function softens the time series with rolling mean,std properties and allows it to be displayed together.
def check_mean_std(ts, col):
    rolmean = ts.rolling(window=24).mean()
    rolstd = ts.rolling(window=24).std()
    plt.figure(figsize=(22,10))
    orig = plt.plot(ts, color='red', label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.title(f'Rolling Mean & Standard Deviation {col}')
    plt.legend()
    
selected_columns = ["Power(kW)"]

for i in selected_columns:
    check_adfuller(allDF[i].dropna(axis = 0))
    check_mean_std(allDF[i], i)


# In[ ]:


#This function plots correlation heatmap of data
def heatmap_and_corr(df, selected_columns):
    cmap = sns.diverging_palette(250, 15, s=75, 
                                 l=40, n=9, 
                                 center="light", 
                                 as_cmap=True)

    matrix = df[selected_columns].corr(method="pearson")
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sns.heatmap(matrix, mask=mask, cmap=cmap, 
                square=True, annot=True, fmt=".2f", ax=ax)


selected_columns = allDF.columns[:10]
heatmap_and_corr(allDF, selected_columns)


# In[ ]:


# This function visualizes how many unique data are in features in train and test data.
def show_unique_values(df,percent, ascending):
    frame = dict()
    for i in df.columns:
        frame[i] = df[i].nunique()
    result = pd.DataFrame(data = frame.values(), index = frame.keys(),columns = ["Val"])
    result = result.sort_values("Val", ascending= ascending)
    if percent == True:
        result = result/result.max()
    return result


def plot_together(n_th, percent = True, ascending=True):
    ax = show_unique_values(trainDF, percent, ascending).head(n_th).plot.barh(alpha = 0.7)
    show_unique_values(submissionxDF, percent, ascending).head(n_th).plot.barh(ax = ax, color="red", alpha = 0.7)
    plt.legend(["Train", "Test"])
    return

plot_together(20, percent = False, ascending=True)

