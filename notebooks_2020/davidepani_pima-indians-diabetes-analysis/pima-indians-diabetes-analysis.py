#!/usr/bin/env python
# coding: utf-8

# # Pima Indians Diabetes Analysis
# ## Davide Pani
# ### M.Sc. Computer Engineering (Data Science)

# ***

# ## Index

# - [1. Introduction](#1.-Introduction)
#     - [1.1 Features description](#1.1-Features-description)
#     - [1.2 Software used for the analysis](#1.2-Software-used-for-the-analysis)
# - [2. Exploratory Data Analysis](#2.-Exploratory-data-analysis)
#     - [2.1 Dataset loading and overview](#2.1-Dataset-loading-and-overview)
#     - [2.2 Histograms](#2.2-Histograms)
#     - [2.3 Statistics](#2.3-Statistics)
#     - [2.4 Correlation Analysis](#2.4-Correlation-Analysis)
#     - [2.5 Boxplots and Crosstabs](#2.5-Boxplots-and-Crosstabs)
# - [3. Data Preprocessing](#3.-Data-Preprocessing)
#     - [3.1 Split into training set and test set](#3.1-Split-into-training-set-and-test-set)
#     - [3.2 Imputation of missing values](#3.2-Imputation-of-missing-values)
#     - [3.3 Standardization](#3.3-Standardization)
#     - [3.4 Principal Component Analysis](#3.4-Principal-Component-Analysis)
#     - [3.5 SMOTE](#3.5-SMOTE)
# - [4. Classification](#4.-Classification)
#     - [4.1 k-Nearest Neighbors](#4.1-k-Nearest-Neighbors)
#         - [4.1.1 Model Selection](#4.1.1-Model-Selection)
#         - [4.1.2 Model Evaluation](#4.1.2-Model-Evaluation)
#     - [4.2 Linear Discriminant Analysis](#4.2-Linear-Discriminant-Analysis)
#         - [4.2.1 Assumption Checking](#4.2.1-Assumption-Checking)
#     - [4.3 Support Vector Machine](#4.3-Support-Vector-Machine)
#         - [4.3.1 Model Selection](#4.3.1-Model-Selection)
#         - [4.3.2 Model Evaluation](#4.3.2-Model-Evaluation)
#     - [4.4 Decision Tree](#4.4-Decision-Tree)
#         - [4.4.1 Model Selection](#4.4.1-Model-Selection)
#         - [4.4.2 Model Evaluation](#4.4.2-Model-Evaluation)
#     - [4.5 Random Forest](#4.5-Random-Forest)
#         - [4.5.1 Model Selection](#4.5.1-Model-Selection)
#         - [4.5.2 Model Evaluation](#4.5.2-Model-Evaluation)
# - [5. Comparison](#5.-Comparison)
# - [6. Conclusions](#6.-Conclusions)

# ***

# ## 1. Introduction

# The goal of this notebook is to analyse the **Pima Indians Diabetes** dataset:
# 
# > This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# The objective of this dataset is to predict whether a patient has or not diabetes, so this is a **supervised binary classification** problem.
# 
# I also report here some additional information about the topic of our analysis, Diabetes, that I found at the following link: https://www.idf.org/aboutdiabetes/what-is-diabetes.html.
# 
# #### What is Diabetes?
# 
# > Diabetes is a chronic disease that occurs when the pancreas is no longer able to make insulin, or when the body cannot make good use of the insulin it produces.
# Insulin is a hormone made by the pancreas, that acts like a key to let glucose from the food we eat pass from the blood stream into the cells in the body to produce energy. All carbohydrate foods are broken down into glucose in the blood. Insulin helps glucose get into the cells. 
# Not being able to produce insulin or use it effectively leads to raised glucose levels in the blood (known as hyperglycaemia). Over the long-term high glucose levels are associated with damage to the body and failure of various organs and tissues.

# ### 1.1 Features description

# In our dataset, each patient is described by the following 9 features:
# 
# - **Pregnancies**: number of times pregnant.
# - **Glucose**: plasma glucose concentration a 2 hours in an oral glucose tolerance test.
# - **BloodPressure**: diastolic blood pressure (mm Hg).
# - **SkinThickness** triceps skin fold thickness (mm).
# - **Insulin**: 2-Hour serum insulin (mu U/ml).
# - **BMI**: body mass index (weight in kg/(height in m)^2).
# - **DiabetesPedigreeFunction**: diabetes pedigree function.
# - **Age**: age (years).
# - **Diabetes**: class variable (1:diabetes, 0:no diabetes). 

# ### 1.2 Software used for the analysis

# The analysis has been conducted using Python 3.7.1 and Jupyter Notebook 5.7.4, which is a browser-based interactive programming environment.
# 
# In particular I used the following Python packages:
# - **Pandas**: it provides fast, flexible, and expressive data structures designed to work with relational data.
# - **NumPy**: it provides a large library of high-level mathematical functions for operating on arrays and matrices.
# - **Scikit-learn**: it is a Machine Learning library that features many classification, regression and clustering algorithms.
# - **Imbalanced-learn**: it provides methods to deal with imbalanced dataset in machine learning and pattern recognition.
# - **Matplotlib**: it is a plotting library which provides a MATLAB-like interface. 
# - **Seaborn**: it is a visualization library based on matplotlib. It provides a high-level interface for drawing statistical graphics.

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import imblearn
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# For a better rendering of the plots that will be shown during the analysis I initialized some parameters of the Pyplot module:

# In[ ]:


matplotlib.rcParams[ 'figure.dpi' ] = 350.0
matplotlib.rcParams[ 'axes.linewidth' ] = 1.0
matplotlib.rcParams[ 'axes.grid' ] = True
matplotlib.rcParams[ 'legend.borderpad' ] = 0.5
matplotlib.rcParams[ 'legend.framealpha' ] = 1.0
matplotlib.rcParams[ 'legend.frameon' ] = True
matplotlib.rcParams[ 'legend.fancybox' ] = False
matplotlib.rcParams[ 'legend.borderaxespad' ] = 0.5
matplotlib.rcParams[ 'grid.linewidth' ] = 1.0
matplotlib.rcParams[ 'grid.alpha' ] = 0.5
matplotlib.rcParams[ 'grid.linestyle' ] = '--'
matplotlib.rcParams[ 'lines.linewidth' ] = 2.0

import warnings
warnings.filterwarnings( 'ignore' )


# In order to make the results of this notebook reproducible every time it is run, I also set a random state seed to be used in the functions that produce random outcomes:

# In[ ]:


RANDOM_STATE_SEED = 0


# ***

# ## 2. Exploratory Data Analysis

# Exploratory Data Analysis is an important step prior to the training of machine learning models. It consists of the exploration of the dataset by means of statistical techniques and graphs in order to visualize the distribution of the data, figure out eventual relationships among the features and detect the presence of missing values and outliers.

# ### 2.1 Dataset loading and overview

# First, I loaded the dataset from the csv file into a pandas DataFrame:

# In[ ]:


df = pd.read_csv( '../input/pima-indians-diabetes-database/diabetes.csv' )
cols = [ "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Diabetes" ]
df.columns = cols


# The DataFrame object provides several methods that allows to explore the main characteristics of our dataset.
# 
# Let's take a look at some of them:

# In[ ]:


df.info()


# The dataset is composed by **768 rows** and **9 columns**. <br>
# We can also  see that there are no null values in each of the columns.

# Let's now take a look at the first rows of the dataset:

# In[ ]:


df.head()


# We can see that all the features are **numerical**, except for the target variable Diabetes that is **categorical**.

# ### 2.2 Histograms

# Using histograms we can quickly visualize the distributions of the values of the features:

# In[ ]:


df.hist( bins=40, figsize=( 7.0, 5.0 ) )
plt.tight_layout( True )
plt.show()


# The plot shows that some features, like Insulin and SkinThickness, present a peak of values at 0, which is not an admissible value for those features. So, probably, those values have been inserted where the corresponding value was missing.
# 
# Let's check how many 0 values are present in the dataset:

# In[ ]:


df.drop( "Diabetes", axis=1 ).isin( [ 0 ] ).sum()


# We can see that, apart from DiabetesPedigreeFunction and Age, all the other columns have 0 values. Among them, the only one that could actually have admissible 0 values is Pregnancies. So, I will consider all the other values as missing and I will have to deal with them in the preprocessing phase.
# 
# For now, I will simply annotate the columns that present missing values and I will replace them with NaN values, in order to prevent wrong assumptions and biased plots in the following of the analysis:

# In[ ]:


columns_with_missing_values = [ "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI" ]

for col in columns_with_missing_values:
    df[ col ] = df[ col ].replace( to_replace=0, value=np.NaN )


# Let's check how the histograms have changed after the replacement:

# In[ ]:


df.hist( bins=40, figsize=( 7.0, 5.0 ) )
plt.tight_layout( True )
plt.show()


# The plot shows also that the class variable Diabetes is slightly unbalanced toward 0. We can deduce that in our dataset there are more samples of people that have not been diagnosed with diabetes than the ones with a positive diagnosis.
# 
# Let's count exactly how many samples belong to the two classes:

# In[ ]:


num_diabetes = df[ "Diabetes" ].sum()
num_no_diabetes = df.shape[ 0 ] - num_diabetes
perc_diabetes = num_diabetes / df.shape[ 0 ] * 100
perc_no_diabetes = num_no_diabetes / df.shape[ 0 ] * 100

print( "There are %d (%.2f%%) people who have diabetes and the remaining %d (%.2f%%) who have not been diagnosed with the desease." % ( num_diabetes, perc_diabetes, num_no_diabetes, perc_no_diabetes ) )

def plot_diabetes_value_counts( normalize ):
    plt.grid( False )
    df.Diabetes.value_counts( normalize=normalize ).plot( kind="bar", grid=False, color=[ sns.color_palette()[ 0 ], sns.colors.xkcd_rgb.get( 'dusty orange' ) ] )
    plt.xticks( [ 0, 1 ], [ 'No', 'Yes' ], rotation=0 )
    plt.xlabel( "Diabetes" )
    
    if ( normalize == False ):
        plt.ylabel( "Count" )
    else:
        plt.ylabel( "Percentage" )    
        
    return
    
plt.subplot( 1, 2, 1 )
plot_diabetes_value_counts( False )
plt.subplot( 1, 2, 2 )
plot_diabetes_value_counts( True )
plt.tight_layout( True )
plt.show()


# ### 2.3 Statistics

# The DataFrame object provides also the method `describe` that allows to generate descriptive statistics about the features of the dataset.
# 
# It shows the following measures separately for each feature:
# - **count**: the number of non-null observations.
# - **mean**: the mean of the values.
# - **std**: the standard deviation of the values.
# - **min**: the minimum of the values.
# - **max**: the maximum of the values.
# - **25%**: the lower percentile.
# - **50%**: the median.
# - **75%**: the upper percentile.

# In[ ]:


df.describe().round( 2 )


# From the table we can see that some features, like SkinThickness and Insulin, have a maximum value that is far away from mean of that feature. So, probably, there are some outliers among the samples. We will explore more in depth the presence of outliers using Boxplots later in section [2.5 Boxplots and Crosstabs](#2.5-Boxplots-and-Crosstabs).

# ### 2.4 Correlation Analysis

# Correlation Analysis is a statistical method for investigating the relationship between two numerical variables. We can do that by computing the Pearson's correlation coefficient, which measures the strength of the linear relationship between two variables. 
# 
# It is defined as following:
# 
# ${ \displaystyle \rho_{ X, Y } = \frac{ cov( X, Y ) }{ \sigma_X \sigma_Y } }$
# 
# The coefficient can only take values between -1 and 1:
# - the closer the value to 1, the higher the positive linear relationship.
# - the closer the value to 0, the lower the linear relationship.
# - the closer the value to -1, the higher the negative linear relationship.
# 
# In order to visually investigate the correlation among all the features of our dataset, I will display the heatmap of the correlation matrix, which is a square matrix that contains the Pearson's coefficients computed for all the pairs of variables:

# In[ ]:


plt.figure( figsize=( 5.5, 5.0 ) )
plt.grid( False )
plt.xticks( range( df.shape[ 1 ] ), df.columns[ 0: ], rotation=0 )
plt.yticks( range( df.shape[ 1 ] ), df.columns[ 0: ], rotation=0 )
sns.heatmap( df.corr(), cbar=True, annot=True, square=False, fmt='.2f', cmap=plt.cm.Blues, robust=False, vmin=0 )
plt.show()


# From the correlation matrix we can see that there no highly correlated features, but there are still some features that seem to have some sort of relationship as suggested by the correlation index value around 0.5. 
# 
# Those features are:
# 
# - **Age** - __Pregnancies__: this is reasonable as the number of pregnancies can only increase as the age increases.
# - **Glucose** - __Diabetes__: we can deduce that an higher glucose concentration is related to an higher probability of being diagnosed with diabetes. This is also confirmed by the additional information about diabetes reported in the introduction.
# - **Glucose** - __Insulin__: we can deduce that when there is an higher level of glucose in the blood, the body produces more Insulin.
# - **BMI** - __SkinThickness__: people with an higher Body Mass Index seem to have a thicker skin.
# 
# Let's explore more in depth the relationships of the latest two pairs of features using a scatterplot matrix:

# In[ ]:


sns.pairplot( df.dropna(), vars=[ 'Glucose', 'Insulin', 'BMI', 'SkinThickness' ], size=1.5, diag_kind='kde', hue='Diabetes' )
plt.tight_layout( False )
plt.show()


# As expected, the plot confirms what we have discovered looking at the correlation matrix: there seems to be a positive linear relationship between Insulin and Glucose and also between BMI and SkinThickness, where the correlation is even more clear.
# 
# By looking at the diagonal we can also see the distribution of the features separated by class. We can see that people with diabetes tend to have higher values in each of those features.

# ### 2.5 Boxplots and Crosstabs

# Boxplots are a type of chart that provide a visual summary of the distribution and skewness of the data through displaying the data quartiles.
# 
# They divide the data into sections, each containing approximately 25% of the data:
# - Lower whisker: represents the lower 25% of the scores, excluding outliers.
# - Lower quartile: 25% of the scores fall below the lower quartile.
# - Median: it is shown by the line that divides the box into two parts. Half the scores are greater than or equal to this value and half are less.
# - Upper quartile: 75% of the scores fall below the upper quartile.
# - Upper whisker: represents the upper 25% of the scores, excluding outliers.
# 
# Boxplots allows also to visually detect outliers as they are located outside the whiskers.
# 
# Let's take a look at the boxplots for each of the features:

# In[ ]:


plt.figure( figsize=( 7.0, 5.0 ) )

for i in range( 8 ):
    plt.subplot( 2, 4, i + 1 )
    plt.grid( False )
    sns.boxplot( x='Diabetes', y=df.columns[ i ], data=df )
    plt.xticks( [ 0, 1 ], [ 'No', 'Yes' ], rotation=0 )

plt.tight_layout( True )
plt.show()


# As expected, many features contain values that are far away from the others.
# 
# Looking at the boxplots we can also see which are the features that differ the most between the two classes and so, the ones that can be more relevant for detecting the presence or not of the desease.
# 
# As we have already discovered in the previous sections, Glucose and Insulin seem to play a key role in Diabetes.
# 
# Let's now investigate more closely the distributions of these features separately for the two classes using Crosstabs:

# In[ ]:


pd.crosstab( pd.cut( df.Glucose, bins=25 ), df.Diabetes ).plot( kind='bar', figsize=( 6.5, 3.2 ) )
plt.ylabel( "Frequency" )
plt.show()


# The plot shows clearly that high values of Glucose are strictly related to diabetes. In particular, values higher than 160 have been almost esclusively measured in people diagnosed with the desease.
# 
# As the previous analysis highlighted, probably Glucose is a relevant feature for discriminanting among the two classes.
# 
# Let's now look more in depth at Insulin:

# In[ ]:


pd.crosstab( pd.cut( df.Insulin, bins=25 ), df.Diabetes ).plot( kind='bar', figsize=( 6.5, 3.4 ), yticks=[ 0, 10, 20, 30, 40, 50, 60, 70 ] )
plt.ylabel( "Frequency" )
plt.show()


# Although people with diabetes seems to have higher values of Insulin than the healthy ones, the trend is not as clear as with Glucose. In fact, we can see that the frequency of people belonging to the two classes becomes almost equal as the measured insulin increases.

# ***

# ## 3. Data Preprocessing

# ### 3.1 Split into training set and test set

# I will now split the dataset into a training set, which will contain 80% of the samples of the original dataset, and a test set, which will contain the remaining 20%.
# 
# This step is necessary in order to preserve a set of data samples to be used exclusively for testing purposes at the end of the training procedure, preventing any leakage of information before that phase. For this reason, also all the preprocessing steps that I will perform in this section will use esclusively the information contained in the training set. Otherwise, the final evaluation of the models would be overly optimistic and it would not reflect their actual performances.
# 
# I will use a stratified split so that, in both sets, the ratio between the samples belonging to the different classes is the same as in the original dataset.

# In[ ]:


from sklearn.model_selection import train_test_split

df_X = df.drop( [ "Diabetes" ], axis=1 )
df_y = df.Diabetes

X_train, X_test, y_train, y_test = train_test_split( df_X, df_y, test_size=0.20, random_state=RANDOM_STATE_SEED, shuffle=True, stratify=df_y )

train_size = np.shape( X_train )[ 0 ]
train_num_diabetes = np.sum( y_train )
train_num_no_diabetes = train_size - train_num_diabetes
train_perc_diabetes = train_num_diabetes / train_size * 100
train_perc_no_diabetes = train_num_no_diabetes / train_size * 100

test_size = np.shape( X_test )[ 0 ]
test_num_diabetes = np.sum( y_test )
test_num_no_diabetes = test_size - test_num_diabetes
test_perc_diabetes = test_num_diabetes / test_size * 100
test_perc_no_diabetes = test_num_no_diabetes / test_size * 100

print( "The training set is composed by %d samples: %d (%.2f%%) with diabetes and %d (%.2f%%) without diabetes." % ( train_size, train_num_diabetes, train_perc_diabetes, train_num_no_diabetes, train_perc_no_diabetes ) )
print( "The test set is composed by %d samples: %d (%.2f%%) with diabetes and %d (%.2f%%) without diabetes." % ( test_size, test_num_diabetes, test_perc_diabetes, test_num_no_diabetes, test_perc_no_diabetes ) )


# ### 3.2 Imputation of missing values

# As we have seen during the Exploratory Data Analysis, the dataset contains many missing values and so we have to deal with them somehow.
# 
# The two most typical solutions are:
# - removing from the dataset all the samples containing those values.
# - estimating and imputing those values from the other samples available in the dataset.
# 
# Given the small size of the dataset, I preferred not to remove any sample, which would have caused its size to reduce even further.
# 
# Instead, I decided to replace those values by imputing the median of the corresponding columns:

# In[ ]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer( missing_values=np.nan, strategy='median' )

X_train_imputed = imputer.fit_transform( X_train )
X_test_imputed = imputer.transform( X_test )


# ### 3.3 Standardization

# Standardizing data consists of centering the data and scaling each feature to unit variance:
# 
# ${ \displaystyle x_{ ij } = \frac{ x_{ ij } \space - \mu_j }{ \sigma_j } }$
# 
# This way, all the features are brought on the same scale:

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_normalized = sc.fit_transform( X_train_imputed )
X_test_normalized = sc.transform( X_test_imputed )

df_X_train_normalized = pd.DataFrame( X_train_normalized, columns=cols[ 0:8 ], index=y_train.index )
df_y_train_normalized = pd.DataFrame( y_train, columns = [ cols[ 8 ] ] )
df_train_normalized = df_X_train_normalized.join( df_y_train_normalized )


# ### 3.4 Principal Component Analysis

# Principal Component Analysis is an unsupervised machine learning algorithm for reducing the dimensionality of the dataset.
# 
# It works by projecting the original dataset into a lower-dimensional space. The basis of this space is composed by a set of Principal Components that represent the directions where there is the most variance in the data. All the Principal Components are also orthogonal each other, so the correlation among the reduced features is minimized, and so, as a consequence, the redundancy of the information.
# 
# The algorithm consists of the following steps:
# - standardizing the dataset (as we have done previously), which is represented by the matrix ${ \mathbf{ X } }$ of size ${ n \times d }$, where ${ n }$ is the number of samples and ${ d }$ is the original number of features.
# - computing the sample covariance matrix ${ \mathbf{ \Sigma } }$ of the data of size ${ d \times d }$.
# - performing the eigendecomposition of ${ \mathbf{ \Sigma } }$ in order to obtain the corresponding set of eigenvalues and eigenvectors.
# - building the projection matrix ${ \mathbf{ W } }$ of size ${ d \times l }$ containing on its columns the ${ l }$ eigenvectors of ${ \mathbf{ \Sigma } }$ corresponding to the eigenvalues of largest magnitude, where ${ l }$ is the desired number of features after the PCA transformation.
# - performing the PCA transformation in order to obtain the reduced dataset ${ \mathbf{ Z } = \mathbf{ X } \times \mathbf{ W } }$ of size ${ n \times l }$.
# 
# In order to optimal number of reduced features, let's first check the amount of variance explained by each of the Principal Components:

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA( whiten=True )
pca.fit( X_train_normalized )

pca_evr = pca.explained_variance_ratio_
pca_evr_cum = np.cumsum( pca_evr )

x = np.arange( 1, len( pca_evr ) + 1 )
y = np.linspace( 0.1, 1, 10 )

plt.bar( x, pca_evr, alpha=1, align='center', label='Individual' )
plt.step( x, pca_evr_cum, where='mid', label='Cumulative', color=sns.colors.xkcd_rgb.get( 'dusty orange' ) )
plt.ylabel( 'Explained Variance Ratio' )
plt.xlabel( 'Principal Components' )
plt.legend()
plt.xticks( x )
plt.yticks( y )
plt.show()


# We can see that the higher the number of Principal Components retained, the higher is the variance preserved, but it is also higher the number of features after applying PCA. So, we have to find a trade-off between the two.
# 
# Looking at the plot, I decided to retain the first 6 Principal Components so that the amount of variance preserved is about 90%:

# In[ ]:


pca = PCA( n_components=6 )

X_train_pca = pca.fit_transform( X_train_normalized )
X_test_pca = pca.transform( X_test_normalized )


# After the PCA transformation we expect the number of features to be reduced from 8 to 6:

# In[ ]:


print( "The training set transormed by PCA is composed by %d rows and %d columns." % ( X_train_pca.shape[ 0 ], X_train_pca.shape[ 1 ] ) )
print( "The test set transormed by PCA is composed by %d rows and %d columns." % ( X_test_pca.shape[ 0 ], X_test_pca.shape[ 1 ] ) )


# ### 3.5 SMOTE

# As we have seen previously, there is an imbalance between the number of samples belonging to the two classes. So, I will try an oversampling algorithm for balancing the dataset, in order to see if it could improve the classification's performances of the models.
# 
# The <b>S</b>ynthetic <b>M</b>inority <b>O</b>versampling <b>TE</b>chnique is an oversampling approach in which the minority class is oversampled by creating synthetic data samples. In particular, it works by taking each minority class sample and introducing synthetic data samples along the line segments joining any/all of the *k* minority class nearest neighbors.
# 
# Synthetic samples are generated in the following way:
# - Take the difference between the minority class sample under consideration and one if its *k* nearest neighbors. 
# - Multiply this difference by a random number between 0 and 1, and add it to the minority class sample under consideration. 
# - This causes the selection of a random point along the line segment between two data points where it is synthetized a new minority class sample. 
# 
# ###### The objective of this section is only the exploration of the effects of the SMOTE algorithm on the dataset and not building an oversampled training set to be used lately for fitting a model.
# _As we will see in the following section [4. Classification](#4.-Classification), in order to estimate the best set of hyperparameters for each classifier I will use Cross-Validation. <br> 
# Performing Cross-Validation using the oversampled training set would result in biased outcomes of the algorithm, because the folds used for validation would contain synthetic samples generated by the oversampling of the original samples contained in the folds used for training, resulting in a leakage of information. <br>
# The correct way of applying SMOTE using Cross-Validation, as we will see later, is to create a Pipeline so that the oversampling is performed only after the split and esclusively on the folds used for training. So, the oversampled training set that I am going to generate in this section, will never be used in the next chapters._
# 
# Let's check what are the effects of applying SMOTE on the training data:

# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE( random_state=RANDOM_STATE_SEED )
X_train_smote, y_train_smote = smote.fit_resample( X_train_normalized, y_train )

df_X_train_smote = pd.DataFrame( X_train_smote, columns=cols[ 0:8 ] )
df_y_train_smote = pd.DataFrame( y_train_smote, columns = [ cols[ 8 ] ] )
df_train_smote = df_X_train_smote.join( df_y_train_smote )


# Let's see how many samples belong to the two classes after the oversampling performed by SMOTE:

# In[ ]:


num_diabetes_smote = df_train_smote[ "Diabetes" ].sum()
num_no_diabetes_smote = df_train_smote.shape[ 0 ] - num_diabetes_smote
perc_diabetes_smote = num_diabetes_smote / df_train_smote.shape[ 0 ] * 100
perc_no_diabetes_smote = num_no_diabetes_smote / df_train_smote.shape[ 0 ] * 100

print( "There are %d (%.2f%%) people with diabetes and %d (%.2f%%) people without diabetes." % ( num_diabetes_smote, perc_diabetes_smote, num_no_diabetes_smote, perc_no_diabetes_smote ) )

def plot_diabetes_value_counts( normalize ):
    plt.grid( False )
    df_train_smote[ 'Diabetes' ].value_counts( normalize=normalize ).plot( kind="bar", grid=False, color=[ sns.color_palette()[ 0 ], sns.colors.xkcd_rgb.get( 'dusty orange' ) ] )
    plt.xticks( [ 0, 1 ], [ 'No', 'Yes' ], rotation=0 )
    plt.xlabel( "Diabetes" )
    
    if ( normalize == False ):
        plt.ylabel( "Count" )
    else:
        plt.ylabel( "Percentage" )    
        
    return
    
plt.subplot( 1, 2, 1 )
plot_diabetes_value_counts( False )
plt.subplot( 1, 2, 2 )
plot_diabetes_value_counts( True )
plt.tight_layout( True )
plt.show()


# As expected, the classes are now balanced.

# ***

# ## 4. Classification

# In this part, I will try several classifiers in order to evaluate which of them performs the best with our dataset.
# 
# I chose the following classifiers:
# - **k-Nearest Neighbors**
# - **Linear Discriminant Analysis**
# - **Support Vector Machine**
# - **Decision Tree**
# - **Random Forest**
# 
# For each classifier I will fit three different models: one using the original dataset, one using the dataset balanced by SMOTE and one using the dataset reduced with PCA.
# 
# In order to properly select the hyperparameters for each classifier I will use a **GridSearch 5-fold Cross-Validation** approach, which allows to estimate which are the hyperparameters that give the best generalized results in a more reliable way than just trying several hyperparameters' configurations using only the training set and a validation set.
# 
# The **5-fold Cross-Validation** consists of the following steps:
# - splitting the dataset into 5 folds of equal size.
# - building 5 different models using the 5 different possible combinations of 4 of the 5 folds as training data.
# - evaluating each model using the remaining fold that was not used to train that model as validation data and obtaining its score value.
# - averaging all the score values obtained in order to provide a good estimation of the generalization performance of the model.
# 
# The **GridSearch 5-fold Cross-Validation** consists of applying the 5-fold Cross-Validation algorithm explained before for each of the possible configurations of hyperparameters that we want to test. As a result, the mean validation score is obtained for each parameter setting and the one that gave the best results is chosen.
# 
# Given the class imbalance, I decided to use different validation scores depending on the type of dataset used: 
# - for the original dataset and the dataset reduced by PCA I selected the F1 score, which takes into account the class imbalance and the Precision-Recall trade-off.
# - for the dataset oversampled by SMOTE I selected the Accuracy because the samples of both classes are balanced by the algorithm.
# 
# Finally, after having fitted the models using the hypermaters' configurations suggested by the GridSearch 5-fold Cross-Validation, I will evaluate them by having them make predictions on the data samples of the test set.
# 
# In order to properly evaluate each model I will use several metrics, which are defined through the following values:
# - **TP**: the number of true positive samples that have been correctly classified as positives. In our case the positive samples are those with a positive diagnosis of diabetes.
# - **TN**: the number of true negatives samples that have been correctly classified as negatives. In our case the negative samples are those who have not been diagnosed with diabetes.
# - **FP**: the number of true negatives samples that have been wrongly classified as positives.
# - **FN**: the number of true positives samples that have been wrongly classified as negatives.
# 
# The metrics that I will use are the following:
# - **Accuracy**: ${ \displaystyle \frac{ \mathsf{ TP + TN } }{ \mathsf{ TP + TN + FP + FN } } }$
# 
# 
# - **Precision**: ${ \displaystyle \frac{ \mathsf{ TP } }{ \mathsf{ TP + FP } } }$
# 
# 
# - **Recall**: ${ \displaystyle \frac{ \mathsf{ TP } }{ \mathsf{ TP + FN } } }$
# 
# 
# - **F1 Score**: ${ \displaystyle \frac{ \mathsf{ 2TP } }{ \mathsf{ 2TP + FP + FN } } }$
# 
# 
# - **Confusion matrix**: it is a square matrix that visually reports the counts of the true positive, true negative, false positive, and false negative predictions of a classifier.
# 
# Each of those metrics provides different information about the performace of a classifier: Accuracy is usually a good metric when dealing with balanced datasets while the other ones are more suitable when there is a class skew or when there are differential misclassification costs, like in our case.
# 
# In fact, predicting an healthy patient as affected by Diabetes would result in further useless analysis and costs, but predicting a deseased one as healthy could result in serious consequences for the health of that person. So, I will give more importance at detecting as many deseased patients as possible (higher Recall score) at the expense of a lower classification accuracy (lower Precision score).
# 
# Lastly, after having chosen the best classifier among the three built in the previous steps, I will also plot its **learning curve**, which shows the training and the validation scores for several sizes of the training set, for detecting wheter the model suffers from high bias or high variance.

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# In the following I define some utility functions that I will use for each classifier.

# In[ ]:


def get_estimator_names( estimator_name ):
    estimator_name_smote = estimator_name + " (SMOTE)"
    estimator_name_pca = estimator_name + " (PCA)"
    
    return [ estimator_name, estimator_name_smote, estimator_name_pca ]


# In[ ]:


def print_grid_search_cross_validation_model_details( gscv_model, estimator_name, scoring ):
    print()
    print( estimator_name )
    print( "Best parameters: ", gscv_model.best_params_ )
    
    return


# In[ ]:


def grid_search_cv_fit( estimator, param_grid, X_train, scoring='f1' ):
    gscv = GridSearchCV( estimator=estimator, param_grid=param_grid, cv=5, n_jobs=-1, scoring=scoring )
    gscv.fit( X=X_train, y=y_train )
    
    return gscv


# In[ ]:


def grid_search_cv_fit_smote( estimator, param_grid, X_train, scoring='accuracy' ):
    pipeline_param_grid = {}

    try:
        for key in param_grid.keys():
            pipeline_param_grid[ "estimator__" + key ] = param_grid[ key ]
            
    except:
        pipeline_param_grid = []
        
        for d in param_grid:
            grid = {}
            
            for key in d.keys():
                grid[ "estimator__" + key ] = d[ key ]
                
            pipeline_param_grid.append( grid )
    
    
    smote = SMOTE( random_state=RANDOM_STATE_SEED, n_jobs=-1 )
    pipeline = Pipeline( [ ( 'smote', smote ), ( 'estimator', estimator ) ] )
    
    return grid_search_cv_fit( pipeline, pipeline_param_grid, X_train, scoring )


# In[ ]:


def grid_search_cross_validation( estimator, param_grid, estimator_names ):
    gscv = grid_search_cv_fit( estimator, param_grid, X_train_normalized )
    gscv_smote = grid_search_cv_fit_smote( estimator, param_grid, X_train_normalized )
    gscv_pca = grid_search_cv_fit( estimator, param_grid, X_train_pca )
    
    print_grid_search_cross_validation_model_details( gscv, estimator_names[ 0 ], 'f1' )
    print_grid_search_cross_validation_model_details( gscv_smote, estimator_names[ 1 ], 'accuracy' )
    print_grid_search_cross_validation_model_details( gscv_pca, estimator_names[ 2 ], 'f1' )
    
    return [ gscv, gscv_smote, gscv_pca ]


# In[ ]:


def print_confusion_matrix( confusion_matrix, estimator_name ):
    plt.grid( False )
    plt.title( estimator_name )
    sns.heatmap( confusion_matrix, cbar=False, annot=True, square=False, fmt='.0f', cmap=plt.cm.Blues, robust=True, linewidths=0, linecolor='black', vmin=0 )
    plt.xlabel( "Predicted labels" )
    plt.ylabel( "True labels" )
    
    return


# In[ ]:


def print_compared_cofusion_matrices( test_predictions, estimator_names ):
    confusion_matrix_ = confusion_matrix( y_test, test_predictions[ 0 ] )
    confusion_matrix_smote = confusion_matrix( y_test, test_predictions[ 1 ] )
    confusion_matrix_pca = confusion_matrix( y_test, test_predictions[ 2 ] )
    
    plt.figure( figsize=( 7.0, 2.8 ) )
    
    axs = plt.subplot( 1, 3, 1 )
    print_confusion_matrix( confusion_matrix_, estimator_names[ 0 ] )
    axs.set_xlabel( "Predicted labels" )
    axs.set_ylabel( "True labels" )
    plt.subplot( 1, 3, 2 )
    print_confusion_matrix( confusion_matrix_smote, estimator_names[ 1 ] )
    plt.subplot( 1, 3, 3 )
    print_confusion_matrix( confusion_matrix_pca, estimator_names[ 2 ] )
    
    plt.tight_layout( True )
    plt.show()
    
    return


# In[ ]:


def test_predictions( estimators ):
    predictions = estimators[ 0 ].predict( X_test_normalized )
    predictions_smote = estimators[ 1 ].predict( X_test_normalized )
    predictions_pca = estimators[ 2 ].predict( X_test_pca )
    
    return [ predictions, predictions_smote, predictions_pca ]


# In[ ]:


def evaluate_test_predictions( test_predictions, estimator_name ):
    test_f1 = f1_score( y_test, test_predictions )
    test_accuracy = accuracy_score( y_test, test_predictions )
    test_precision = precision_score( y_test, test_predictions )
    test_recall = recall_score( y_test, test_predictions )
    
    results = { 
        'F1' : [ test_f1 ],
        'Accuracy' : [ test_accuracy ], 
        'Precision' : [ test_precision ], 
        'Recall' : [ test_recall ]
    }
    
    df_results = pd.DataFrame( results, index=[ estimator_name ] )
    
    return df_results


# In[ ]:


def merge_and_sort_results( results ):
    df_results = pd.DataFrame( columns=results[ 0 ].columns )

    for result in results:
        df_results = df_results.append( result )
        
    df_results = df_results.sort_values( "F1", ascending=False )
    
    return df_results.round( 3 )


# In[ ]:


def evaluate_test_results( test_predictions, estimator_names ):
    df_results = evaluate_test_predictions( test_predictions[ 0 ], estimator_names[ 0 ] )
    df_smote_results = evaluate_test_predictions( test_predictions[ 1 ], estimator_names[ 1 ] )
    df_pca_results = evaluate_test_predictions( test_predictions[ 2 ], estimator_names[ 2 ] )

    df_overall_results = merge_and_sort_results( [ df_results, df_pca_results, df_smote_results ] )
    
    return df_overall_results


# In[ ]:


def evaluate_best_estimators_results( best_estimators ):
    df_results = []
    
    for estimator in best_estimators:
        if estimator[ 1 ].endswith( "(PCA)" ):
            test_predictions = estimator[ 0 ].predict( X_test_pca )
        else:
            test_predictions = estimator[ 0 ].predict( X_test_normalized )
            
        df_result = evaluate_test_predictions( test_predictions, estimator[ 1 ] )
        df_results.append( df_result )
        
    df_results = merge_and_sort_results( df_results )
    
    return df_results


# In[ ]:


def plot_learning_curve( estimator, X_train_, estimator_name, legend_location='best', scoring='f1', scoring_name='F1' ):
    train_sizes = np.linspace( 0.2, 1.0, 6 )
    train_size, train_scores, test_scores = learning_curve( estimator, X_train_, y_train, train_sizes=train_sizes, cv=5, n_jobs=-1, shuffle=True, random_state=RANDOM_STATE_SEED, scoring=scoring )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.title( "Learning curve: " + estimator_name )
    plt.plot( train_size, train_mean, label='Train', marker='o', markerfacecolor='white', markeredgewidth=2.0 )
    plt.fill_between(train_size,train_mean + train_std,train_mean - train_std, alpha=0.2 )
    plt.plot( train_size, test_mean, label='Validation', marker='o', markerfacecolor='white', markeredgewidth=2.0 )
    plt.fill_between(train_size,test_mean + test_std,test_mean - test_std, alpha=0.2 )
    
    plt.xlabel( 'Number of training samples' )
    plt.ylabel( scoring_name )
    
    plt.legend( loc=legend_location )
    
    plt.plot()


# I also initialized the list that will contain the best estimators for each kind of classifier that will be used for the final comparison to select the best among all of them:

# In[ ]:


best_estimators = []


# ### 4.1 k-Nearest Neighbors

# **k-Nearest Neighbors** is a simple Machine Learning algorithm for classification.
# 
# The learning phase consists only of storing the training set. 
# 
# Then, in order to predict the class for a new data point, the algorithm finds the closests data points to the data point that we want to classify in the training set, its nearest neighbors, and assigns to it the class to which belong the majority of them.
# 
# The number of nearest neighbors that the algorithm considers in order to predict a class is chosen a-priori by the user through the hyperparameter *k*.
# 
# Choosing the right value of *k* is crucial to find a good balance between overfitting and underfitting: 
# 
# - a too low value of *k* will result in very good performance when classifying samples of the training set, but it will perform worse when classifying samples that it has never seen during the training procedure.  This happens because the classifier is not able to generalize well and it is overfitting the training data. 
# - a too high value of *k* will result in a too simple model that is not able to discriminate properly the sample among the different classes. This causes underfitting.
# 
# The hyperparameters that I will test using the GridSearch Cross-Validation are the following:
# - `n_neighbors`: it is the value of *k*. It represents the number of neighbors that the algorithm considers in order to classify data points.
# - `weights`: wheter to consider all the neighbours equally or to give more weight to the nearest ones.

# #### 4.1.1 Model Selection

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_param_grid = {
    'n_neighbors' : [ 5, 9, 15, 21 ],
    'weights' : [ 'uniform', 'distance' ]
}

knn_estimator_names = get_estimator_names( "kNN" )
knn_best_estimators = grid_search_cross_validation( KNeighborsClassifier(), knn_param_grid, knn_estimator_names )


# #### 4.1.2 Model Evaluation

# In[ ]:


knn_test_predictions = test_predictions( knn_best_estimators )


# In[ ]:


print_compared_cofusion_matrices( knn_test_predictions, knn_estimator_names )


# In[ ]:


df_knn_overall_results = evaluate_test_results( knn_test_predictions, knn_estimator_names )
df_knn_overall_results


# From the test results we can see that the classifier trained with the oversampled dataset is quite different from the other two.
# 
# kNN trained with the original dataset and the one reduced with PCA provided similar results: they both scored an acceptable overall Accuracy and Precision scores. Their downside is that, with a very low Recall score, they miss over 40% of the people with positive diagnosis, which is not acceptable in our case.
# 
# Instead, kNN using SMOTE provides a lower overall Accuracy and a lower Precision score but, with the highest Recall score, it is the one that misses the lowest number of people with diabetes, which makes this classifier the most suitable for our purposes among the three.

# In[ ]:


knn_best_estimator = knn_best_estimators[ 1 ].best_estimator_
knn_best_estimator_name = knn_estimator_names[ 1 ]
best_estimators.append( [ knn_best_estimator, knn_best_estimator_name ] )


# Let's now take a look at its learning curve:

# In[ ]:


plot_learning_curve( knn_best_estimator, X_train_normalized, knn_best_estimator_name, 'lower right' )


# The learning curve shows that both the training score and the validation score increase with the number of samples of the training set but it seems that using more than 350 samples does not improve the performances of the classifier as both the scores remain almost steady.

# ### 4.2 Linear Discriminant Analysis

# **Linear Discriminant Analysis** is a probabilistic and generative model for classification. 
# 
# The algorithm consists of fitting class-conditional densities separately for each class and using
# the Bayes theorem to flip things around in order to obtain ${ p( Y | X ) }$:
# 
# ${ \displaystyle p( Y = k | X = x ) = \frac{ p( X = x | Y = k ) p( Y = k ) }{ p( X = x ) } = \frac{ f_k \pi_k }{ \sum_{ c }{ f_c \pi_c } } = p_k( x ) }$
# 
# where:
# - ${ \displaystyle f_k( x ) }$ is the class-conditional density of class ${ k }$ that, in Linear Discriminant Analysis, is a Gaussian distribution with the following equation: <br><br>
# ${ \displaystyle f_k( x ) = \frac{ 1 }{ ( 2 \pi )^{ p/2 } \space |\mathbf{\Sigma}|^{ 1/2 } } \space e^{ - \frac{1}{2} ( x - \mu_k )^T \space \mathbf{\Sigma}^{-1} \space ( x - \mu_k ) } }$ <br><br>
# where ${ \mu_k }$ is the mean vector of class ${ k }$ and ${ \mathbf{\Sigma} }$ is the shared covariance matrix among all the classes. <br><br>
# - ${ \displaystyle \pi_k }$ is the prior probability of a sample of belonging to class ${ k }$.
# 
# Then, in order to classify a data sample ${ x }$, we need to see for which class ${ p_k( x ) }$ is largest, which is equivalent to assigning ${ x }$ to the class with the largest discriminant score:
# 
# ${ \displaystyle \delta_k ( x ) = x^T \mathbf{\Sigma}^{ -1 } \mu_k - \frac{ 1 }{ 2 } \mu_k^T \mathbf{\Sigma}^{ -1 } \mu_k + \log{ \pi_k }  }$

# #### 4.2.1 Assumption Checking

# ###### In order to obtain meaningful results from this classifier it is necessary that the assumptions that it makes are verified.
# First, we have to check if the distributions of the two classes are normal. Since they are composed by many features, they have to be multivariate normal distributions, which is possible if and only if all of their components follow a normal distribution. 
# 
# Let's check if this assumption is verified by visually inspecting the histograms of the features separately for the two classes:

# In[ ]:


plt.figure( figsize=( 7.0, 5.0 ) )

for i in range( 8 ):
    plt.subplot( 3, 3, i + 1 )
    sns.distplot( df_train_normalized[ df_train_normalized.Diabetes == 0 ][ cols[ i ] ], bins=15 )
    sns.distplot( df_train_normalized[ df_train_normalized.Diabetes == 1 ][ cols[ i ] ], bins=15 )

plt.tight_layout( True )
plt.show()


# The histograms show that, although some feature seem to follow a normal distribution for both classes like BMI and BloodPressure, there are features that are clearly more skewed than normal like Age and DiabetesPedigreeFunction. This implies that the distributions of the two classes cannot be normal.
# 
# The fact that the normality assumption is not verified means that fitting this classifier on this dataset could result in an unstable model. So, before proceeding, let's check the behavior of the classifier during the training using the learning curve:

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

plot_learning_curve( LinearDiscriminantAnalysis( priors=[ ( train_perc_diabetes / 100 ), ( train_perc_no_diabetes / 100 ) ] ), X_train_normalized, "LDA" )


# The learning curve shows that both the train and validation scores do not have a clear trend as they increase and decrease several times as the training set size changes. This shows, as expected, that **the classifier is unstable** and so, as a consequence, we can conclude that __this model is not appropriate for our purposes__.

# ### 4.3 Support Vector Machine

# **Support Vector Machine** is a discriminative binary linear classifier.
# 
# In particular, it is a maximum margin linear classifier that chooses among all the possible separating hyperplanes the one that makes the biggest margin between the two classes. The data points nearest to the separating hyperplane are called Support Vectors.
# 
# If the data points are not linearly separable, they could be linearly separable in an higher-dimensional space. So, if we have a transformation ${ \displaystyle \phi : \mathbb{ R }^D \rightarrow \mathbb{ R }^M }$ , where ${ M > D }$ , then we can train a linear SVM in ${ \mathbb{ R }^M }$ and project back the decision boundary to ${ \mathbb{ R }^D }$ where it is non-linear. 
# 
# However, the training process of the SVMs uses the training data only for computing the inner products between all pairs of training observations, so it is not necessary to compute the transformation of the dataset to the higher-dimensional space. Instead, it is possible to replace all the inner products using kernel functions, which are functions that, given two vectors in ${ \displaystyle \mathbb{ R }^D }$, implicitly compute the dot product between them in an higher-dimensional space ${ \displaystyle \mathbb{ R }^M }$, without explicitly transforming the vectors to ${ \displaystyle \mathbb{ R }^M }$.
# 
# ${ \displaystyle \kappa( \vec{ x }_i, \vec{ x }_j ) = \langle \phi( \vec{ x }_i ), \phi( \vec{ x }_j ) \rangle }$
# 
# One of the most popular kernel functions for SVMs is the Radial Basis Function:
# 
# ${ \displaystyle \kappa( \vec{ x }_i, \vec{ x }_j ) = e^{ - \gamma || \space \vec{ x }_i - \vec{ x }_j \space ||^2 } }$
# 
# The hyperparameters that I will test using the GridSearch Cross-Validation are the following:
# - `kernel`: defines the kernel function to be used.
# - `C`: the regularization parameter of the Support Vector Classifier.
# - `gamma`: the parameter of the Radial Basis Kernel.
# - `class_weight`: wheter or not to take in account the class imbalance.

# #### 4.3.1 Model Selection

# In[ ]:


from sklearn.svm import SVC

svc_param_grid = [ 
    {
     'kernel': [ 'linear' ],
     'C': [ 0.001, 0.01, 0.1, 1, 10, 100 ],
     'class_weight': [ None, 'balanced' ] },
    {
     'kernel': [ 'rbf' ],
     'C': [ 0.001, 0.01, 0.1, 1, 10, 100 ],
     'gamma': [ 0.001, 0.01, 0.1, 1, 10, 100 ],
     'class_weight': [ None, 'balanced' ] } ]



svc_estimator_names = get_estimator_names( "SVM" )
svc_best_estimators = grid_search_cross_validation( SVC( random_state=RANDOM_STATE_SEED ), svc_param_grid, svc_estimator_names )


# #### 4.3.2 Model Evaluation

# In[ ]:


svc_test_predictions = test_predictions( svc_best_estimators )


# In[ ]:


print_compared_cofusion_matrices( svc_test_predictions, svc_estimator_names )


# In[ ]:


df_svc_overall_results = evaluate_test_results( svc_test_predictions, svc_estimator_names )
df_svc_overall_results


# The test results clearly show that the SVC trained on the original dataset outperformed the other two. It scored an high Recall, by detecting over 80% of the people with diabetes, still providing an acceptable Precision. It also scored the best overall Accuracy and the best F1 score, making it the best classifier among them.

# In[ ]:


svc_best_estimator = svc_best_estimators[ 0 ].best_estimator_
svc_best_estimator_name = svc_estimator_names[ 0 ]
best_estimators.append( [ svc_best_estimator, svc_best_estimator_name ] )


# Let's now take a look at its learning curve:

# In[ ]:


plot_learning_curve( svc_best_estimator, X_train_normalized, svc_best_estimator_name, 'lower right' )


# The learning curve shows that the training score slightly decreases as the training set size increases while the validation score improves significantly. This suggests that adding more data to the training set could actually improve the performances of the classifier.

# ### 4.4 Decision Tree

# The **Decision Tree** is a particular kind of classifier which is represented by a tree of finite depth. Every node of the tree specifies a test involving an attribute and every branch descending from that node matches one of the possible outcomes of the test.
# 
# Classifying an istance means performing a sequence of tests, starting with the root node and terminating with a leaf node, which represents a class.
# 
# Decision Trees are induced from a training set using a top-down approach, from the root to the leafs, by recursively binary spliting the predictor space. The attribute selected to perform each split is the one that partitions the training set into subsets as pure as possible.
# 
# The most popular measures of purity are:
# 
# - Entropy: ${ \displaystyle D = - \sum_{ k = 1 }^{ K }{ \hat{ p }_{ mk } \log{ \hat{ p }_{ mk } } } }$
# 
# - Gini: ${ \displaystyle G = \sum_{ k = 1 }^{ K }{ \hat{ p }_{ mk } ( 1 - \hat{ p }_{ mk } ) } }$
# 
# where ${ \hat{ p }_{ mk } }$ is the proportion of samples belonging to class ${ k }$ in the ${ m }$th region.
# 
# The hyperparameters that I will test with the GridSearch Cross-Validation are the following: 
# - `criterion`: the split criterion to be used. 
# - `max_depth`: the maximum depth of the Tree.
# - `class_weight`: wheter or not to take in account the class imbalance.

# #### 4.4.1 Model Selection

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decision_tree_param_grid = {
    'max_depth': [ 10, 15, 20, None ],
    'criterion' : [ 'gini', 'entropy' ],
    'class_weight': [ None, 'balanced' ]
}

decision_tree_estimator_names = get_estimator_names( "Decision Tree" )
decision_tree_best_estimators = grid_search_cross_validation( DecisionTreeClassifier( random_state=RANDOM_STATE_SEED ), decision_tree_param_grid, decision_tree_estimator_names )


# #### 4.4.2 Model Evaluation

# In[ ]:


decision_tree_test_predictions = test_predictions( decision_tree_best_estimators )


# In[ ]:


print_compared_cofusion_matrices( decision_tree_test_predictions, decision_tree_estimator_names )


# In[ ]:


df_decision_tree_overall_results = evaluate_test_results( decision_tree_test_predictions, decision_tree_estimator_names )
df_decision_tree_overall_results


# As we can see, the only classifier that scored acceptable results is the one trained with the original dataset, which is the best among the three.

# In[ ]:


decision_tree_best_estimator_name = decision_tree_estimator_names[ 0 ]
decision_tree_best_estimator = decision_tree_best_estimators[ 0 ].best_estimator_
best_estimators.append( [ decision_tree_best_estimator, decision_tree_best_estimator_name ] )


# Let's now take a look at its learning curve:

# In[ ]:


plot_learning_curve( decision_tree_best_estimator, X_train_normalized, decision_tree_best_estimator_name, 'center left' )


# We can see that the validation score slightly increases and the training score slightly decreases as the number of samples grows. However, the big gap that there is between the two measures, even when using the entire dataset, is a clear indicator that the Decision Tree is highly overfitting the training data.

# ### 4.5 Random Forest

# The **Random Forest** is an ensemble method specifically designed for Decision Tree classifiers. As an ensemble method, it generates several different weak learners and then it combines them in order to achieve an higher accuracy.
# 
# It consists of growing a forest of many different Decision Tree classifiers by introducing two sources of randomness:
# - each tree is trained using a different bootstrap sample of the original training data.
# - at each node of each tree, the best split is chosen not from all the attributes but from a random subset of them.
# 
# The final result of classification is given by majority voting over all the trees in the forest.
# 
# The hyperparameters that I will test with the GridSearch Cross-Validation are the following:
# - `criterion`: the split criterion to be used.
# - `n_estimators`: the number of Trees to be trained.
# - `max_depth`: the maximum depth of each Tree.
# - `class_weight`: wheter or not to take into account the class imbalance.

# #### 4.5.1 Model Selection

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest_param_grid = {
    'max_depth': [ 10, 15, 20, None ],
    'criterion': [ 'gini', 'entropy' ],
    'n_estimators': [ 25, 50, 100 ],
    'class_weight': [ None, 'balanced' ]
}

random_forest_estimator_names = get_estimator_names( "Random Forest" )
random_forest_best_estimators = grid_search_cross_validation( RandomForestClassifier( random_state=RANDOM_STATE_SEED ), random_forest_param_grid, random_forest_estimator_names )


# #### 4.5.2 Model Evaluation

# In[ ]:


random_forest_test_predictions = test_predictions( random_forest_best_estimators )


# In[ ]:


print_compared_cofusion_matrices( random_forest_test_predictions, random_forest_estimator_names )


# In[ ]:


df_random_forest_overall_results = evaluate_test_results( random_forest_test_predictions, random_forest_estimator_names )
df_random_forest_overall_results


# We can see that the Random Forest trained with the dataset balanced by SMOTE scored the highet Recall, F1 and overall Accuracy, which makes this one the best among the three.

# In[ ]:


random_forest_best_estimator_name = random_forest_estimator_names[ 1 ]
random_forest_best_estimator = random_forest_best_estimators[ 1 ].best_estimator_
best_estimators.append( [ random_forest_best_estimator, random_forest_best_estimator_name ] )


# Let's take a look at its learning curve:

# In[ ]:


plot_learning_curve( random_forest_best_estimator, X_train_normalized, random_forest_best_estimator_name, 'center left' )


# We can see that the Random Forest scores perfectly on the training set while its validation score remains pretty low. This indicates that the classifier is strongly overfitting the training data. From the curve we can also see that the validation score remains almost steady after 200 samples, which suggests that increasing the training set size would be probably not useful for improving its performances. 

# ***

# ## 5. Comparison

# In this section, I will compare the scores of the best estimators trained and selected previously in order to evaluate which of them provided the overall best performances.

# In[ ]:


df_results = evaluate_best_estimators_results( best_estimators )
df_results


# From the final results we can see that reducing the dataset using PCA did not improved the performances of any of the classifiers. On the other hand, balancing the dataset using SMOTE provided better results with the kNN and the Random Forest.
# 
# Among the classifiers, the Decision Tree is clearly the worst, which scored both a low Recall and a low Precision and, as a consequence, the worst F1 score.
# 
# The results obtained by the kNN with SMOTE are slightly better because of the higher Recall score. Still, with a Precision score of about 0.6, almost 40% of the people it classifies with diabetes are not affected by the desease.
# 
# The Random Forest is the one that provided the most balanced results between Precision and Recall and also an high overall Accuracy score. Still, its Recall score is not high enough for our purposes, which shows that this classifier misses over 30% of the positive diagnosis.
# 
# On the other hand, the SVM scored a pretty high Recall value with still an acceptable Precision. Its satisfactory trade-off between the two measures is also shown by the highest F1 score and the highest overall Accuracy, which makes this classifier the most suitable for our purposes.
# 
# In conclusion, the estimator that provided the most satisfactory results is the SVM trained with the original dataset with the following complete set of hyperparamters:

# In[ ]:


df_best_estimator = pd.DataFrame( svc_best_estimator.get_params(), index=[ 'SVM' ] )
df_best_estimator


# ***

# ## 6. Conclusions

# The goal of this notebook was to analyse the Pima Indians Diabetes dataset and to try to predict wheter or not a patient is affected by Diabetes given a set of medical parameters. After having preliminary explored the data, I prepared several versions of the dataset by applying some preprocessing techniques in order to see if they could actually improve the classifiers' performances. I then trained different models using some of the most widespread Machine Learning algorithms for classification and I selected among them the one that gave the most optimal test results for our purposes.
