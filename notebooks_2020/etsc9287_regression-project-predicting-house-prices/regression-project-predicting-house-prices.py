#!/usr/bin/env python
# coding: utf-8

# # Regression Project Notebook

# This is the notebook for my Applied Regression project.  All code for my models and visualizations that back up my report can be found here.
# 
# This is also my first machine learning notebook on Kaggle.  If you have any feedback, feel free to like and comment as I have much to learn! :)

# # 1. Importing Libraries and Data

# I will be using numpy and pandas to clean my data, feature engineer, and perform any other operations on my dataframes.  I will be using matplotlib and seaborn for visualizations and Scikit-Learn for modeling and preprocessing.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RANSACRegressor

import statsmodels.api as sm
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore')


# Now, the data.

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# We will be working with 80 features including house square footage, year built, quality rating, etc.  The target variable we are attempting to predict with a high level of accuracy is SalePrice.  There are 1460 total houses in the data, but we can see there exist missing values in some of the columns.

# # 2. Cleaning

# This section will involve dealing with the missing values and removing any unnecessary variables.   

# In[ ]:


def missing_props(df):
    missing_values = []
    for i in df.columns:
        missing_values.append(round(df[i].isnull().sum() / len(df), 3))
    missing_props = pd.DataFrame(list(zip(df.columns, missing_values)), columns = ["Var", "Prop_Missing"]).sort_values(by = "Prop_Missing", ascending = False)
    
    return missing_props[missing_props["Prop_Missing"] != 0]


# In[ ]:


table1 = missing_props(train)


# In[ ]:


table1


# The insignificant variables with many missing values will be removed:

# In[ ]:


train = train.drop(columns = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])


# The remaining variables with missing columns will be filled depending on their context.

# In[ ]:


#Lot Frontage is the amount of street connected to the property (ft).  NA values represent 
#houses with no connected street, probably in rural areas.

train["LotFrontage"] = train["LotFrontage"].fillna(0)


# In[ ]:


#All garage variables have the same amount of missing values, illustrating that these values 
#are within the same rows.  These houses have no garage.

train["GarageYrBlt"] = train["GarageYrBlt"].fillna("No Garage")
train["GarageCond"] = train["GarageCond"].fillna("No Garage")
train["GarageType"] = train["GarageType"].fillna("No Garage")
train["GarageFinish"] = train["GarageFinish"].fillna("No Garage")
train["GarageQual"] = train["GarageQual"].fillna("No Garage")


# In[ ]:


#All basement variables have the same amount of missing values, illustrating that these values 
#are within the same rows.  These houses have no basement.

train["BsmtFinType1"] = train["BsmtFinType1"].fillna("No Basement")
train["BsmtFinType2"] = train["BsmtFinType2"].fillna("No Basement")
train["BsmtExposure"] = train["BsmtExposure"].fillna("No Basement")
train["BsmtQual"] = train["BsmtQual"].fillna("No Basement")
train["BsmtCond"] = train["BsmtCond"].fillna("No Basement")


# In[ ]:


#These missing values represent houses with no masonry vaneers.

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
train["MasVnrType"] = train["MasVnrType"].fillna("None")


# In[ ]:


#Lastly, these missing values have no known electrical system.

train["Electrical"] = train["Electrical"].fillna("None")


# All missing values are now filled and the data is clean.

# # 3. Feature Engineering

# This section will involve creating new helpful variables from the ones we already have.

# In[ ]:


train.head()


# ### Total Inside Area of the House (Sq Ft.)

# This value represents total area of the house, not including outside area.

# In[ ]:


train["TotalInsideArea"] = train["TotalBsmtSF"] + train["GrLivArea"] + train["GarageArea"]


# ### Total Outside Area of the House (Sq Ft.)

# This value represents the total area of the property on the outside of the house, including any porches, pool, and deck. 

# In[ ]:


train["TotalOutsideArea"] = train["WoodDeckSF"] + train["OpenPorchSF"] + train["EnclosedPorch"] + train["3SsnPorch"] + train["ScreenPorch"] + train["PoolArea"]


# ### Pool?

# This is the type of variable that would serve best as a boolean because houses with any pool at all might be much different from houses without a pool regardless of pool size, which is already now taken into account in total outside area.

# In[ ]:


train["Pool"] = train["PoolArea"].apply(lambda x: "Yes" if x > 0 else "No")


# ### Dropping subsets of engineered variables

# If we include any of these variables in our model in addition to the engineered variables, we could risk overfitting to to multicollinearity.

# In[ ]:


train = train.drop(columns = ["TotalBsmtSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",
                              "EnclosedPorch", "PoolArea", "3SsnPorch", "ScreenPorch", "1stFlrSF", "2ndFlrSF",
                             "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "MasVnrArea"])


# # 4. Feature Selection

# As we have an enormous amount of features, it would be useful to pick ones we use in our model before we perform any data analysis or preprocessing.
# 
# Note: Usually, I would perform this step later in the process.  However, due to the large number of features that might lower the quality and speed of EDA/heatmaps, I will select features now.

# ## Selecting Continuous Features

# Before performing feature selection methods, let's take a look at our model summary with all the continuous variables.

# In[ ]:


import statsmodels.api as sm


# In[ ]:


def model_summary(df):
    
    x = df.drop(columns = ["SalePrice"]).select_dtypes(exclude=['object'])
    y = df["SalePrice"].values
    
    model_original = sm.OLS(y, x).fit()
    return model_original.summary()


# In[ ]:


model_summary(train)


# We can see the F-statistic is very high, as is adjusted R^2.  We will attempt to get these measures higher after selecting features.

# ### Pearson's Correlation

# Correlation heatmaps are an excellent way to visualize the relationships between all of the continuous variables in a dataset and decide on what's relevant based on what R^2 thresholds give us the best models.  The colorscale represents R values.

# In[ ]:


corr_matrix = train.corr()
pd.options.display.float_format = "{:,.2f}".format
plt.figure(figsize = (12,10))
sns.heatmap(corr_matrix)
plt.title("Correlation Matrix of All Continuous Variables (Target = SalePrice)")
plt.show()


# In[ ]:


var_corrs = pd.DataFrame((corr_matrix["SalePrice"] ** 2).sort_values(ascending = False))
var_corrs


# This table represents R^2 values instead of R values unlike the heatmap which is why no negative numbers are present.  The inside square footage variable correlates with sales price the strongest.
# 
# Now, a closer look at the most correlated variables.

# In[ ]:


def corr_matrix_filter(corr_matrix, r2): # <- We can experiment with different r^2 thresholds when we tune our model
    
    corr_matrix2 = corr_matrix[corr_matrix["SalePrice"] ** 2 >= r2]
    
    low_corrs = list(corr_matrix[corr_matrix["SalePrice"] ** 2 < r2].index)
    
    if r2 != 0:
        corr_matrix2 = corr_matrix2.drop(columns = low_corrs)

    return corr_matrix2


# In[ ]:


corr_matrix2 = corr_matrix_filter(corr_matrix, 0.1) #only includes variables with r^2 >= 0.1
pd.options.display.float_format = "{:,.2f}".format
plt.figure(figsize = (12,10))
sns.heatmap(corr_matrix2, annot = True)
plt.title("Filtered Correlation Matrix of Continuous Variables (SalePrice R^2 > 0.1)")
plt.savefig("corrmap1.png")
plt.show()


# After testing many r^2 values by running the entirety of this completed notebook, the threshold resulting in the best model (best balance between overfitting and underfitting) is 0.1.  These are the continous features we will use in our model (including target variable, sale price):  

# In[ ]:


pearson_selection_cont = list(corr_matrix2.index)
pearson_selection_cont


# Finally, we run a summary of the new model to see what has changed.

# In[ ]:


train_cont = train[pearson_selection_cont]
model_summary(train_cont)


# Interestingly, The R^2 value went down though the F-statistic more than doubled.

# ## Selecting Categorical Features

# In addition to the continuous features that were selected in the previous section, we must find out which categorical features best predict the sale price.  First, these variables will be hot one encoded.

# In[ ]:


cat_vars = train.select_dtypes(include = "object")


# In[ ]:


cat_vars_encode = pd.get_dummies(cat_vars)


# In[ ]:


cat_vars_encode["SalePrice"] = train["SalePrice"]


# In[ ]:


cat_vars_encode.head()


# ### Pearson's Correlation

# Now, the same correlation matrix process as before with our dataframe of binary encoded values.

# In[ ]:


corr_matrix_cat = cat_vars_encode.corr()


# In[ ]:


var_corrs_cat = pd.DataFrame((corr_matrix_cat["SalePrice"] ** 2).sort_values(ascending = False))
var_corrs_cat.head(22)


# Our quality variables seem to the strongest correlation with sale price.  After testing multiple thresholds, 0.1 seemed to work best for the categorical variables, as well.

# In[ ]:


corr_matrix_cat2 = corr_matrix_filter(corr_matrix_cat, 0.1)


# In[ ]:


plt.figure(figsize = (12,10))
sns.heatmap(corr_matrix_cat2)
plt.title("Filtered Correlation Matrix of Categorical Variables (SalePrice R^2 > 0.1)")
plt.savefig("corrmap2.png")
plt.show()


# All of these variables (the text before the "_") will be used in our model.  

# In[ ]:


relevant_cat_vars = list(corr_matrix_cat2.index)


# In[ ]:


remove_underscores = []
for i in relevant_cat_vars:
    remove_underscores.append(i.split("_", 1)[0]) #Splits the variables from their distinct values
pearson_selection_cat = list(set(remove_underscores)) #Removes repeats to get list of relevant variables.


# Now, here are the categorical variables that will be used:

# In[ ]:


pearson_selection_cat


# In[ ]:


pearson_selection_cat.remove('SalePrice') #The target variable, sale price, is already included in continous list


# In[ ]:


train_cat = train[pearson_selection_cat]


# ## Putting it Together

# Now, a dataframe of all variables with out target strength of relationship with sales price. 

# In[ ]:


train2 = pd.concat([train_cont, train_cat], axis = 1)


# In[ ]:


train2.head()


# In[ ]:


list(train2.columns)


# Note that categorical features will be encoded when we begin the modeling process.  But this dataframe is ready for EDA.

# # 5. Exploratory Data Analysis

# The process of EDA will allow us to derive insights from our selected variables and help us see how we will scale the features and target.

# ## Univariate Analysis

# Univariate analysis illustrates basic distributions of the variables of interest.  We will scale these distributions to be normal.

# ### Testing Normality With Histograms

# First, some basic histograms and bar plots.

# In[ ]:


def graph_cont(df, var):
    plt.hist(df[var])
    plt.xlabel(f"{var}")
    plt.ylabel("Count")
    plt.title(f"Distribution of {var}")
    plt.show()


# First, histograms of continuous variables.

# In[ ]:


[graph_cont(train_cont, i) for i in train_cont.columns]


# Note that sales price is the target variable.  Most of these distributions will have to be normalized when we begin scaling.

# ### Bar Plots

# Now, bar graphs of categorical variables.

# In[ ]:


def graph_disc(df, var):
    sns.countplot(df[var])
    plt.xlabel(f"{var}")
    plt.ylabel("Count")
    plt.title(f"Distribution of {var}")
    plt.xticks(rotation = 90)
    plt.show()


# In[ ]:


[graph_disc(train_cat, i) for i in train_cat.columns]


# An example of an insight is that most houses in this dataset are in the "NAmes" neighborhood (probably Northern Ames).  

# ## Bivariate Analysis

# Now, we will visualize how sales price varies among all of our variables.

# ### Testing Linearity With Scatterplots

# First, scatterplots with continuous variables

# In[ ]:


def scatter_plot(df, var):
    plt.scatter(df[var], df["SalePrice"])
    plt.xlabel(f"{var}")
    plt.ylabel("SalePrice")
    plt.title(f"{var} vs. SalePrice")
    plt.xticks(rotation = 90)
    plt.show()


# In[ ]:


[scatter_plot(train_cont, i) for i in train_cont.columns]


# All positive relationships that appear exponential.  These should become linear relationships after scaling.
# 
# There also exist some outliers we can remove.

# ### Boxplots

# Now, let's visualize how sales price varies among all categorical variables.

# In[ ]:


def box_plot(train2, df, var):
    df["SalePrice"] = train2["SalePrice"]
    sns.boxplot(df[var], df["SalePrice"])
    plt.xlabel(f"{var}")
    plt.ylabel("SalePrice")
    plt.title(f"{var} vs. SalePrice")
    plt.xticks(rotation = 90)
    plt.show()


# In[ ]:


[box_plot(train2, train_cat, i) for i in train_cat.columns if i != "SalePrice"]


# One example of an insight is that the "NridgHt" and "StoneBr" neighborhoods appear to be the most expensive.

# # 6. Scaling and Normalizing 

# The last step before creating our model is to normalize all of our data and remove any outliers.  In the EDA section, we already made some histograms and scatterplots to test normality and linearity, though we will now use more advanced techiniques.

# ## QQ-Plots

# In[ ]:


for i in list(train2.select_dtypes(exclude = "object").columns):
    sm.qqplot(train2[i])
    plt.title(f"QQ Plot for {i}")
    plt.show()


# OverallQual, FullBath, TotRmsAbvGrd, and GarageCars are the only distributions that are approximately normal.  Note that a couple are numerical discrete values.

# ## Normalizing and Removing Outliers

# Now, we will scale the necessary variables using different methods depending on the nature of the distribution.  I will also be removing clear outliers manually in this section but will use test a robust regressor later on to remove them by means of an algorithm.

# ### SalePrice (Target)

# Because the sales price distribution is only skewed by some extremely high values, we will scale it logarithmically.

# In[ ]:


train2["SalePrice"] = np.log1p(train2["SalePrice"])


# In[ ]:


graph_cont(train2, "SalePrice")


# ### TotalInsideArea

# We can also scale TotalInsideArea logarithimically as it's mostly normal with the exception of a few very large outliers.

# In[ ]:


train2["TotalInsideArea"] = train_cont["TotalInsideArea"]
train2["TotalInsideArea"] = np.log1p(train2["TotalInsideArea"])


# In[ ]:


graph_cont(train2, "TotalInsideArea")


# In[ ]:


scatter_plot(train2, "TotalInsideArea")


# This relationship is now nearly perfectly linear, but we can see that a few outliers are skewing the data.  

# In[ ]:


train2 = train2[(train2["SalePrice"] > 11) & (train2["TotalInsideArea"] < 9)]


# In[ ]:


scatter_plot(train2, "TotalInsideArea")


# The scatterplot is now without outliers.

# ### TotalOutsideArea

# Unlike TotalInsideArea, this engineered variable follows more of a log-normal distribution.  We can logarithmically transform but many of the values will stick to 0 since some houses have no outside area.  These will be removed.

# In[ ]:


train2["TotalOutsideArea"] = np.log1p(train2["TotalOutsideArea"])


# In[ ]:


#train2 = train2[train2["TotalOutsideArea"] != 0]


# In[ ]:


len(list(train[train["TotalOutsideArea"] == 0]))


# We only removed 64 observations by removing the 0's.

# In[ ]:


graph_cont(train2, "TotalOutsideArea")


# In[ ]:


scatter_plot(train2, "TotalOutsideArea")


# Again, some outliers exist but we have achieved linearity and normality after taking away the values of 0.

# ## Preparing Model

# Before modeling, all of our variables should be scaled down to unit variance.

# In[ ]:


train_set = pd.get_dummies(train2)


# In[ ]:


x_vals = train_set.drop(columns = ["SalePrice"])
y_val = train_set["SalePrice"].values.reshape(-1,1)


# Our final dataset ready to model:

# # 7. Modeling

# Finally, we can train and test our model with various linear regression techniques  Again, we can go back and play around with R^2 thresholds on continuous and categorial variables, as well as different methods of scaling, to see which features create the best models.
# 
# The models will be evaluated by RMSLE (Room Mean Squared Logarithmic Error).

# In[ ]:


import math

def RMSLE(predict, target):
    
    total = 0 
    
    for k in range(len(predict)):
        
        LPred= np.log1p(predict[k]+1)
        LTarg = np.log1p(target[k] + 1)
        
        if not (math.isnan(LPred)) and  not (math.isnan(LTarg)): 
            
            total = total + ((LPred-LTarg) **2)
        
    total = total / len(predict)  
    
    return np.sqrt(total)


# Finally, we fit the model, make predictions, and evaluate performance.

# In[ ]:


def create_model(x_vals, y_val, model_type, t):
    
    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_val, test_size = 0.2) #splitting into train and test
    
    model = model_type
    model.fit(x_train, y_train) #fitting the model
    
    y_train_pred = np.expm1(model.predict(x_train)) #predicting and converting back from log(SalePrice)
    y_test_pred = np.expm1(model.predict(x_test))
    y_train = np.expm1(y_train)
    y_test = np.expm1(y_test)
    
    if t == "test":
        return RMSLE(y_test_pred, y_test) #evaluating
    elif t == "train":
        return RMSLE(y_train_pred, y_train)


# The specifics of these modeling techniques will be elaborated on in the presentation and paper.

# ### Basic Linear Regression

# In[ ]:


print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, LinearRegression(), "train") for i in range(100)]), 4))
print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, LinearRegression(), "test") for i in range(100)]), 4))


# ### Ridge Regression

# In[ ]:


print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, Ridge(alpha = 0.01), "train") for i in range(100)]), 4))
print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, Ridge(alpha = 0.01), "test") for i in range(100)]), 4))


# ### LASSO

# In[ ]:


print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, Lasso(alpha = 0.01), "train") for i in range(100)]), 4))
print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, Lasso(alpha = 0.01), "test") for i in range(100)]), 4))


# ### Elastic Net

# In[ ]:


print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, ElasticNet(alpha = 0.01), "train") for i in range(100)]), 4))
print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, ElasticNet(alpha = 0.01), "test") for i in range(100)]), 4))


# ### RANSAC

# This algorithm punishes outliers based on the RANSAC algorithm.

# In[ ]:


print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, RANSACRegressor(), "train") for i in range(100)]), 4))
print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, RANSACRegressor(), "test") for i in range(100)]), 4))


# Ridge appears to be the most accurate model on average.  LASSO is the worst.

# # 8. Evaluating the Model

# Lastly, we will use use a learning curve to test the validity of our model.

# In[ ]:


train_sizes, train_scores, validation_scores = learning_curve(
estimator = Ridge(),
X = x_vals,
y = y_val, train_sizes = [1,10,50,100,300,600,900], cv = 5,
scoring = 'neg_mean_squared_error')


# In[ ]:


train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)


# In[ ]:


plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE')
plt.xlabel('Training set size')
plt.title('Learning curves for Linear regression model')
plt.legend()
plt.show()


# The two curves very closely converage, so the balance between over and underfitting is quite good.

# ## 9. Conclusion / Create Kaggle Submissions

# The sole purpose of this section is to create a file that can be submitted to the Kaggle leaderboard.  I must apply most the same steps from the rest of this notebook to the testing set from Kaggle that will be used for the leaderboard.

# In[ ]:


#cleaning and feature engineering

test["TotalBsmtSF"] = test["TotalBsmtSF"].fillna(0)
test["GrLivArea"] = test["GrLivArea"].fillna(0)
test["GarageArea"] = test["GarageArea"].fillna(0)
test["GarageCars"] = test["GarageCars"].fillna(0)
test["GarageYrBlt"] = test["GarageYrBlt"].fillna("No Garage")
test["GarageCond"] = test["GarageCond"].fillna("No Garage")
test["GarageType"] = test["GarageType"].fillna("No Garage")
test["GarageFinish"] = test["GarageFinish"].fillna("No Garage")
test["GarageQual"] = test["GarageQual"].fillna("No Garage")
test["BsmtFinType1"] = test["BsmtFinType1"].fillna("No Basement")
test["BsmtFinType2"] = test["BsmtFinType2"].fillna("No Basement")
test["BsmtExposure"] = train["BsmtExposure"].fillna("No Basement")
test["BsmtQual"] = test["BsmtQual"].fillna("No Basement")
test["BsmtCond"] = test["BsmtCond"].fillna("No Basement")
test["MasVnrArea"] = test["MasVnrArea"].fillna(0)
test["MasVnrType"] = test["MasVnrType"].fillna("None")
test["TotalInsideArea"] = test["TotalBsmtSF"] + test["GrLivArea"] + test["GarageArea"]
test["TotalOutsideArea"] = test["WoodDeckSF"] + test["OpenPorchSF"] + test["EnclosedPorch"] + test["3SsnPorch"] + test["ScreenPorch"] + test["PoolArea"]
train2_drop = train2.drop(columns = ["SalePrice"])


# In[ ]:


test.info()


# In[ ]:


#creating test and train

x_train = train_set.drop(columns = ["SalePrice"])
y_train = train_set["SalePrice"].values
x_test = test[train2_drop.columns] #accessing only the features I selected earlier in the notebook
x_test = pd.get_dummies(x_test)

#preprocessing

x_test["TotalInsideArea"] = np.log1p(x_test["TotalInsideArea"])
x_train = x_train[x_train["TotalInsideArea"] < 9]
x_test["TotalOutsideArea"] = np.log1p(x_test["TotalOutsideArea"])
x_train = x_train[x_train["TotalOutsideArea"] != 0]

#fitting the Ridge model (most accurate from earlier)
    
model = Ridge()
model.fit(x_train, y_train) #fitting the model
    
y_train_pred = np.expm1(model.predict(x_train)) #predicting and converting back from log(SalePrice)
y_test_pred = np.expm1(model.predict(x_test))


# In[ ]:


x_train.info()


# In[ ]:


y_train_pred


# In[ ]:


y_test_pred


# In[ ]:


preds = y_test_pred


# In[ ]:


ids = np.array(test["Id"])


# In[ ]:


submissions = pd.DataFrame({"Id":ids, "SalePrice":preds})


# In[ ]:


submissions.head()


# In[ ]:


submissions.to_csv("submission2.csv", index = False)

