#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#4E944F'>|</span> Introduction</b>
# Most people start working with data from exactly the wrong end. They begin with
# a data set, then apply their favorite tools and techniques to it. The result is narrow
# questions and shallow arguments remember <b> Why Before How </b> ,The secret is to have structure that you
# can think through, rather than working in a vacuum , <b> But how to have that structure  ?</b>
# -  Our first place to find structure is in creating the scope for a data problem
# There are four parts to a project scope. The four parts are the <b> context </b> of the
# project; the <b> needs </b> that the project is trying to meet; the <b> vision </b> of what success might
# look like; and finally what the <b> outcome </b> will be
# 
# ![](https://2382812164-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-M5Vxy9PzQFOZjhyK8YP%2F-MEXhP-_QL4GqNUVHV2U%2F-MEXpPcIlpFrvqgpPQrE%2Fimage.png?alt=media&token=1c4ad2a4-ca40-49c8-937e-adaa749d02ad)
# ### Context (emerge from understanding who we are working with and why they are doing what they are doing):
# the insurer wants to develop the best medical insurance products, plan a particular insurance outcome, or manage a big portfolios. For all these cases, the objective is to accurately predict insurance costs.
# 
# 
# #### Explanation of the variables of the dataset :
# -  age: age of primary beneficiary
# -  sex: insurance contractor gender, female, male
# 
# -  bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight(kg/m $^2$) using the ratio of height to weight, ideally 18.5 to 24.9
#  
# -  children: Number of children covered by health insurance / Number of dependents
# 
# -  smoker: Smoking
# 
# -  region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# 
# -  charges: Individual medical costs billed by health insurance
# 
# ### Needs (challenges  that could be fixed by intelligently using data):
# -  can we  boost up the financial performance of a medical insurer ?
# 
# ### Vision (where we are going and what it might look like to achieve our goal):
# -   train a ML regression model that generates the target column charges more accurately. Being a regression model problem, metrics such as the coefficient of determination and the mean squared error are used to evaluate the model.
# 
# ### Outcome (is focused on what will happen when we are "done"):
# -   medical insurer can approximate the medical costs of each user and develop more accurate pricing models
# 
# At the end of everything, the finished work will often be fairly simple.Because
# of all of the work done in thinking about context and need, vision,
# and thinking about outcomes, our work will be the right kind of simple. <b>Simple
# results are the most likely to get used.</b>

# # <b>2 <span style ='color:#83BD75'>|</span> Importing Libraries</b>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')
colors_nude = ['#E9EFC0','#B4E197','#83BD75','#4E944F'] #if wandring why greens ?to reinforce the emotion (prediction dollar$)
sns.palplot(sns.color_palette(colors_nude))
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)
from scipy import stats
from scipy.stats import norm, skew
from sklearn.metrics import  mean_squared_error ,r2_score , explained_variance_score
from time import time
from sklearn.model_selection import GridSearchCV


# # <b>3 <span style ='color:#4E944F'>|</span> Loading the data</b>

# In[ ]:


df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df['children'] =df['children'].astype('object')
df


#  Usually ,at that stage i do <b>Data cleaning</b> then <b>EDA </b> but  <b>Data cleaning</b> is the process of modifying data to fit the criteria for a specific problem, and you'll have no idea what you might want to change about a data set until you know what's in it.So  <b>EDA </b> first ,for more info 
# (https://www.kaggle.com/questions-and-answers/103089)
# 
# 

# # <b>4 <span style ='color:#4E944F'>|</span> Exploratory Data Analysis</b>

# I have always wondered how to explore data? do I need to do some random cool visualizations that no one understands except me, when I look at people's notebooks I tell myself that I can do that visualization but why particularly he did that? if you have these questions don't worry I will do my best to help you and remind myself
# -  <b> First </b>,The public-facing side of visualization—the polished graphics that you see are fine examples of data graphics at their best,
#     but what is the process to get to that final picture? There is an exploration phase
#     that most people never see, but it can lead to visualization that is a level above
#     the work of those who do not look closely at their data.
# -  <b>Second</b> ,you must struture the notebook as i told you above that will help to
#       - keeps us from doing the first things to cross our minds
#       - understand and get sence of what we want from data ,We don’t want unsurprising—we want knowledge.
#       - avoid  drowning in data
# - <b> Third </b> ,The common mistake is to form a visual first and get the data later. It should
#     be the other way around—data first and visualization follows
# - <b>Forth </b>, if you reach here great job !, you are almost ready to begain  your exploration adventure ,but just pause a second and ask yourself what you want to know from the data.Your answer doesn’t need to be complex or profound. Just make it less vague than, “I want to know what the data looks like.” The more specific you are the more direction you get.
# - <b>Fifth </b>,  I will assume that you now have an awesome question, and wondering what visualization methods should I use (that is normal), There are many different graphs and other types of visual displays of information but before telling you what is the best visualization ever !, remember that the value of the visualization is not how fancy and cool that is , <b> it's but about Simplify the information as much as possible so that if a non-technical person look at that visualization can understand the purpose </b> , so there is no best visualization ever ? yeah , just do what Simplify information.
# - <b>Sixth </b> , after answering your question , you give yourself a place to start,
#     and if you’re lucky, as you investigate, you’ll come up with more questions,
#     and then you dig into those. Coming up with and answering potential questions a user might have while you explore also provides focus and purpose,
#     and helps farther along in the design process when you make graphics for a
#     wider audience

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.1 | What Region Costs More ? </b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px

# In[ ]:


region_cost= df.groupby('region')['charges'].sum() * 1e-6
fig = plt.figure(figsize=(16,8))
plt.plot(region_cost ,'-o',c = colors_nude[-1])
plt.title('Region Costs In Million' ,size = 15)
plt.ylabel('In million')
plt.show()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.2 | Dose Old Costs More ? </b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px

# In[ ]:


fig = plt.figure(figsize=(16,8))
plt.scatter(df['age'] , df['charges'], cmap = 'summer' ,c = df['bmi'] ,alpha = 0.8 )
plt.xlabel('Age')
plt.ylabel('Charges')
plt.colorbar(label = 'BMI')
plt.title('Age and Charges',size = 15);


# In[ ]:


df[['charges' , 'age']].corr()


# #### Insight
# -  it seems that we have clustering data ( will do it at Feature Engineering ),but as that more age more charges 
# -  what it's like to have a bmi more than 50 ?
# -  what is the charges that is more than 50,000$ ?
# 

# ![](https://www.cdc.gov/healthyweight/images/assessing/bmi-adult-fb-600x315.jpg?_=07167)

# In[ ]:


df.query('bmi > 50 ')['age']


#   having a BMI of more than 50 and you're less than 25 years old, I guess those are outliers

# In[ ]:


df.drop(df.query('bmi > 50 ').index ,axis= 0 ,inplace=True)
df.drop(df.query('charges > 50000 ').index  ,axis= 0 ,inplace=True)


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.3 | Which Gender Costs More ? </b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px

# In[ ]:


fig = plt.figure(figsize=(16,8))
sns.histplot(data =df[df['sex'] =='male'] ,  x = 'charges' ,color = colors_nude[-1]  ,label = 'male' ,alpha = 0.6)
sns.histplot(data =df[df['sex'] =='female'] ,  x = 'charges',color = colors_nude[1] ,label = 'female' ,alpha = 0.6)
plt.title('Gender Costs',size = 15)
plt.legend()
plt.show()


# #### Insight
# -  it seems that men costs more than women
# 

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.4 | Does smoking Costs More ? </b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px

# In[ ]:


fig  ,ax= plt.subplots(figsize=(16,8))
sns.histplot(data =df[df['smoker'] =='yes'] ,  x = 'charges' ,color = colors_nude[-1]  ,label = 'smoker' ,alpha = 0.6)
sns.histplot(data =df[df['smoker'] =='no'] ,  x = 'charges',color = colors_nude[1] ,label = 'non-smoker' ,alpha = 0.6)
plt.title('smoking Costs',size = 15)
plt.axvline(37000, color="k", linestyle="--");
ax.annotate('here',size = 15, xy=(37000,20), xytext=(30000, 20),arrowprops=dict(arrowstyle="->"
                                                                              ,connectionstyle="angle3,angleA=0,angleB=-90"
                                                                             ,color = 'k'));
plt.legend()
plt.show()


# #### Insight
#   can you see that we have only charges more than 37,000$ for smoking people , so yes smoking affect the cost
# 

# # <b>5 <span style='color :#4E944F'>| </span>  Data Preprocessing</b>

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.1 | Basic data cleaning</b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px

# In[ ]:


df.nunique()


# In[ ]:


df.isna().sum()


# In[ ]:


df.drop_duplicates(inplace=True)
df.shape


#   No missing data, that is bad 

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.2 |  Assumptions of Regression
#  </b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px
#             
# 

# 
# * **Linearity ( Correct functional form )** 
# * **Homoscedasticity ( Constant Error Variance )( vs Heteroscedasticity )**
# * **Independence of Errors ( vs Autocorrelation )**
# * **Multivariate Normality ( Normality of Errors )**
# * **No or little Multicollinearity** 
# 
# Since we fit a linear model, we assume that the relationship is **linear**, and the errors, or residuals, are pure random fluctuations around the true line. We expect that the variability in the response(dependent) variable doesn't increase as the value of the predictor(independent) increases, which is the assumptions of equal variance, also known as **Homoscedasticity**. We also assume that the observations are independent of one another(**No Multicollinearity**), and a correlation between sequential observations or auto-correlation is not there.
# 
# Now, these assumptions are prone to happen altogether. In other words, if we see one of these assumptions in the dataset, it's more likely that we may come across with others mentioned above. Therefore, we can find and fix various assumptions with a few unique techniques.
# 
# So, **How do we check regression assumptions? We fit a regression line and look for the variability of the response data along the regression line.** Let's apply this to each one of them.
# 
# **Linearity(Correct functional form):** 
# Linear regression needs the relationship between each independent variable and the dependent variable to be linear. The linearity assumption can be tested with scatter plots. The following two examples depict two cases, where no or little linearity is present. 

# In[ ]:


fig, ax = plt.subplots(1,2,figsize = (16,8),sharey=True )

fig.subplots_adjust(hspace=0.01, wspace=0.05)

sns.regplot(x=df.age, y=df.charges, ax=ax[0] ,color=colors_nude[-1])

sns.regplot(x=df.bmi, y=df.charges, ax=ax[1],color=colors_nude[-1]);


#   what about residual (<b>Homoscedasticity</b>) ?

# In[ ]:


plt.subplots(figsize = (16,8))
sns.residplot(df.bmi, df.charges,color=colors_nude[-1]);


# Ideally, if the assumptions are met, the residuals will be randomly scattered around the centerline of zero with no apparent pattern. The residual will look like an unstructured cloud of points centered around zero. However, our residual plot is anything but an unstructured cloud of points. Even though it seems like there is a linear relationship between the response variable and predictor variable, the residual plot looks more like a funnel. The error plot shows that as **BMI** value increases, the variance also increases, which is the characteristics known as **Heteroscedasticity**. One way to fix this Heteroscedasticity is by using a transformation method like log-transformation or box-cox transformation. We will do that later.

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.3 | Target Variable </b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px
#             
# 

#   charges is the variable we need to predict. So let's do some analysis on this variable first.
# 
# 

# In[ ]:


sns.distplot(df['charges'] , fit=norm);

(mu, sigma) = norm.fit(df['charges'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(df['charges'], plot=plt)
plt.show()


#   The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.

# In[ ]:


df['charges'] = np.log1p(df['charges'])
sns.distplot(df['charges'] , fit=norm );

(mu, sigma) = norm.fit(df['charges'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(df['charges'], plot=plt )
plt.show()


#  Let's see post-transformed state of residual plots.

# In[ ]:


fig = plt.figure(figsize=(16,8))
sns.residplot(x = df.bmi, y = df.charges,color=colors_nude[-1]);


#   Not bad ,Almost an equal amount of variance across the zero lines

# <b> No or Little multicollinearity </b>: Multicollinearity is when there is a strong correlation between independent variables. Linear regression or multilinear regression requires independent variables to have little or no similar features.

# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr() ,annot=True,cmap='summer')


#  Little multicollinearity is acceptable  👍 

# 
# # <b>6 <span style='color :#4E944F'>| </span> Feature Engineering</b>

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.1 | Fixing Skewness </b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px
#             
# 

# In[ ]:


numeric_col = df.dtypes[df.dtypes != 'object'].index
skewed_col = df[numeric_col].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_col


#   more than 0.75 i will consider that skewed column ,so we don't have skewness

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#4E944F;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.2 | Creating Features </b></p>
# </div>
# <div style="color:white;display:fill;border-radius:8px
#             

# within this problem i want to share two method on creating feature 
# -  Generating all the possible combinations
# -  Polynomials
# 
# but first we need to split the data to check if these new variable doing a good job or not

# In[ ]:


from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
cat_col = df.select_dtypes(include = 'object').columns
df = pd.get_dummies(df , cat_col ,drop_first=True)
y =df.pop('charges')
X_train, X_test, y_train, y_test = train_test_split( df , y , test_size=0.2 , random_state=True)


# In[ ]:


from itertools import combinations

column_list = df.columns
interactions = list(combinations(column_list, 2))
interactions
interaction_dict = {}
for interaction in interactions:
    X_train_int = X_train.copy()
    X_train_int['int'] = X_train_int[interaction[0]] * X_train_int[interaction[1]]
    lr3 = LinearRegression()
    lr3.fit(X_train_int, y_train)
    interaction_dict[lr3.score(X_train_int, y_train)] = interaction
top_5 = sorted(interaction_dict.keys(), reverse = True)[:5]
for interaction in top_5:
    print(interaction_dict[interaction])


#   before making these new feature we should test R$^2$ to compare

# In[ ]:


lr2 = LinearRegression()
lr2.fit(X_train, y_train)
y_hat_train = lr2.predict(X_train)
print(f"R-Squared Score ': {r2_score(y_train, y_hat_train)}")


# In[ ]:


X_train['age_smoker_yes'] = X_train['age'] * X_train['smoker_yes']
X_train['bmi_smoker_yes'] = X_train['bmi'] * X_train['smoker_yes']
X_train['age_children_2'] = X_train['age'] * X_train['children_2']
X_train['region_southwest_smoker_yes'] = X_train['region_southwest'] *X_train ['smoker_yes']
X_train['sex_male_smoker_yes'] = X_train['sex_male']* X_train['smoker_yes']


# In[ ]:


lr2 = LinearRegression()
lr2.fit(X_train, y_train)
y_hat_train = lr2.predict(X_train)
print(f"R-Squared Score ': {r2_score(y_train, y_hat_train)}")


# better than what i thought

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
X_train_int = X_train[['age' , 'bmi']]
poly_dict = {}
for feature in X_train_int:
    for p in range(2, 5):
        X_train_poly = X_train_int.copy()
        X_train_poly['sq'] = X_train_poly[feature] ** p
        lr = LinearRegression()
        lr.fit(X_train_poly, y_train)
        poly_dict[lr.score(X_train_poly, y_train)] = [feature, p]
    
poly_dict[max(poly_dict.keys())]


# In[ ]:


X_train['age_2'] = X_train['age'] **2


# In[ ]:


lr2 = LinearRegression()
lr2.fit(X_train, y_train)
y_hat_train = lr2.predict(X_train)
print(f"R-Squared Score ': {r2_score(y_train, y_hat_train)}")


#  Accepatble result

# In[ ]:


df['age_smoker_yes'] = df['age'] * df['smoker_yes']
df['bmi_smoker_yes'] = df['bmi'] * df['smoker_yes']
df['age_children_2'] = df['age'] * df['children_2']
df['region_southwest_smoker_yes'] = df['region_southwest'] *df ['smoker_yes']
df['sex_male_smoker_yes'] = df['sex_male']* df['smoker_yes']
df['age_2'] = df['age'] **2

X_train, X_test, y_train, y_test = train_test_split( df , y , test_size=0.2 , random_state=True)


# for more information about these two method ,there is the article
#  (https://towardsdatascience.com/feature-engineering-combination-polynomial-features-3caa4c77a755)

# # <b>7 <span style='color :#4E944F'>| </span> Modeling</b>

# because I always have a problem memorizing these hyperparameters and go get some from any blog I will force myself and write a
# a small note of each parameter

# In[ ]:


from sklearn.linear_model import Lasso , Ridge ,ElasticNet
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor ,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


RD =Ridge()
RD_param_grid = {'alpha' : [  0.1, 0.12 , 1 ],
                 'solver':['svd']
                }
#alpha:controls how much you want to regularize the model
#kernel:trick makes it possible to get the same result as if you had added many polynomial features0
#gamma and C makes the bell-shaped curve narrower ,if your model is overfitting, you should reduce it
#degree:degree polynomial kernel
#coef0:how much the model is influenced by highdegree polynomials versus low-degree polynomials

gsRD = GridSearchCV(RD,param_grid = RD_param_grid, cv=5,scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)
gsRD.fit(X_train, y_train)
RD_best = gsRD.best_estimator_
RD_best


# In[ ]:


las =Lasso()
las_param_grid = {'alpha' : [  0.99 ,0.1, 0.12 , 1 ]
                }
#alpha:controls how much you want to regularize the model

gslas = GridSearchCV(las,param_grid = las_param_grid, cv=5,scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)
gslas.fit(X_train, y_train)
las_best = gslas.best_estimator_
las_best


# In[ ]:


ElN =ElasticNet()
ElN_param_grid = {'alpha' : [  0.99 ,0.1, 0.12 , 1 ],
                  'l1_ratio' :[ 0.0001, 0.01,0.05,0.4 ,0.5,0.6,0.8,0.99 ]}
#l1_ratio corresponds to the mix ratio r               
#When l1_ratio = 0 , Elastic Net is equivalent to Ridge Regression, and when l1_ratio = 1, it is equivalent to Lasso Regression
                  
gsElN = GridSearchCV(ElN,param_grid = ElN_param_grid, cv=5,scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)
gsElN.fit(X_train, y_train)
ElN_best = gsElN.best_estimator_
ElN_best


# In[ ]:


RFR =RandomForestRegressor()
RFR_param_grid = {"max_depth": [None],
                    "max_features": [1, 3, 10],
                    "min_samples_split": [2, 3, 10],
                    "min_samples_leaf": [1, 3, 10],
                    "max_leaf_nodes": [ 10 , 16 , 20],
                    "bootstrap": [False],
                    "n_estimators" :[100,300],
                    }
#max_depth :The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#min_samples_split:The minimum number of samples required to split an internal node
#min_samples_leaf:The minimum number of samples required to be at a leaf node
#max_leaf_nodes: max leaf for each tree if none means unlimited
#bootstrap:instances randomly sampled from the training set with replacement
#n_estimators: Decision Tree classifiers

gsRFR = GridSearchCV(RFR,param_grid = RFR_param_grid, cv=5,scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)
gsRFR.fit(X_train, y_train)
RFR_best = gsRFR.best_estimator_
RFR_best


# In[ ]:


GBR = GradientBoostingRegressor()
GBR_param_grid = {
                'n_estimators' : [100,200,300],
                'learning_rate': [0.1, 0.05, 0.01],
                'max_depth': [4, 8],
                'min_samples_leaf': [100,150],
                'max_features': [0.3, 0.1]
}
#learning_rate:scales the contribution of each tree

gsGBR = GridSearchCV(GBR,param_grid = GBR_param_grid, cv=5,scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)
gsGBR.fit(X_train, y_train)
GBR_best = gsGBR.best_estimator_
GBR_best


# In[ ]:


ADR_param_grid = {
                'n_estimators' : [100,200,300],
                'learning_rate': [0.1, 0.05, 0.01],
}
DTR = DecisionTreeRegressor()
ADR = AdaBoostRegressor(base_estimator=DTR)

#learning_rate:scales the contribution of each tree
gsADR = GridSearchCV(ADR,param_grid = ADR_param_grid, cv=5,scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)
gsADR.fit(X_train, y_train)
ADR_best = gsADR.best_estimator_
ADR_best


# In[ ]:


regressors = [ADR_best , GBR_best , RFR_best ,ElN_best ,las_best ,RD_best ]
for model in regressors:
    start = time()
    model.fit(X_train, y_train)
    train_time = time() - start
    start = time()
    y_pred = model.predict(X_test)
    predict_time = time()-start    
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print("\tExplained variance:", explained_variance_score(y_test, y_pred))
    print("\tMean absolute error:", mean_squared_error(y_test, y_pred))
    print("\tR2 score:", r2_score(y_test, y_pred))
    print()


# It seems that KernelRidge and GradientBoostingRegressor did a very good job 

# In this notebook I assumed a lot of things and the last thing is that you are a amazing person and gonna make upvote and make a 
# nice comment 😉
