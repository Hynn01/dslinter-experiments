#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.api import OLS
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The attributes are deﬁned as follows (taken from the UCI Machine Learning Repository1): 

# * CRIM: per capita crime rate by town
# * ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# * INDUS: proportion of non-retail business acres per town
# * CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# * NOX: nitric oxides concentration (parts per 10 million)
# * 20.2. Load the Dataset 124
# * RM: average number of rooms per dwelling
# * AGE: proportion of owner-occupied units built prior to 1940
# * DIS: weighted distances to ﬁve Boston employment centers
# * RAD: index of accessibility to radial highways
# * TAX: full-value property-tax rate per 10,000 dollars
# * PTRATIO: pupil-teacher ratio by town 12.
# * B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. 
# * LSTAT: perc lower status of the population
# * MEDV: Median value of owner-occupied homes in 1000 dollars
# * We can see that the input attributes have a mixture of units.
# 

# In[ ]:


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('../input/boston-house-prices/housing.csv', header=None, delimiter=r"\s+", names=column_names)


# In[ ]:


df.head(5)


# In[ ]:


df.describe()


# # We have several variables let's first try to fit a simple linear regression model using Lstat as predictor and MEDV are response

# * MEDV - (Median House Value)
# * LSTAT - (Percent Of Households with low socio-econmic status)

# Intuitively the higher the LSTAT the higher the MEDV (Beacuse if a considerable amount of population does not belong to a backward soci-economic backgroud , they might tend to live in costlier households)

# In[ ]:


X=df['LSTAT']
y=df['MEDV']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


X_train = np.array(X).reshape((len(X), 1))
y_train = np.array(y).reshape((len(y), 1))


# In[ ]:


reg = LinearRegression().fit(X_train, y_train)


# In[ ]:


print("Coefficient B1 is :" ,reg.coef_)
print("Intercept B0 is :" ,reg.intercept_)


# # Interpreting coefficients
# ### ** I'll be using the notation y= B0 + B1(X)**
# * Coefficient B1 is : [[-0.95004935]]
# 
# * Intercept B0 is : [34.55384088]
# 
# If the LSTAT changes by 1% (or rather 1 unit)then the value is value of households in the area is decreasing by 0.95 units (which is as per what we assumed it to be , the no. of poor households when increase , will eventually decrease the value of houses)
# 
# If the LSTAT is 0 then the  value is value of hosueholds in the area is 34.55 units

# In[ ]:


est = smf.ols('MEDV ~ LSTAT', df).fit()
est.summary().tables[1]


# **I have used df as a whole dataframe and not in terms of X_train and y_train**

# In[ ]:


print(sm.OLS(y_train, X_train).fit().summary())


# # Interpreting t-statistics and their respective p-values
# 
# 
# ## MY INTERPRETATION OF T-Statistic - How many standard deviations are we away from the 0
# 
#                                     NULL HYPOTHESES : B1=0
#                                     ALTERNATE HYPOTHESIS : B1 not = 0
# 
# **We can see that the t-statistics are sufficiently high in order to say that the B1 and B0 are significantly far away from the 0**
# 
# **Also the p - values of t-statistic are low enough (<0.05) in order to back up the fact that this has not taken place by chance**
# 
# # **SO WE REJECT NULL HYPOTHESIS AND SELECT THE ALTERNATE HYPOTHESIS**

# The 95% confidence interval is 
# 
#  [(B1-2*(S.E)) , (B1+2*(S.E))] and similar for the B0 

# In[ ]:


model_min = est.conf_int(alpha=0.05)[0]
model_max = est.conf_int(alpha=0.05)[1]

print(model_min)
print('')
print(model_max)


# **I.e 95% of the times 1 unit change in LSTAT would change the value of MEDV by at min -> (-1.028) and at max -> (-0.873)**

# # Training a Simple Linear Reg , LSTAT(X) VS MEDV(Y)

# In[ ]:


reg = LinearRegression().fit(X_train, y_train)


# **Return the coefficient of determination R^2 of the prediction.**
# 
# R^2 -> The LSTAT is able to explain about 54% of the variability in the MEDV column

# In[ ]:


reg.score(X_train,y_train)


# In[ ]:


reg.coef_


# # What if we want to see what would be the estimation for a random instance, let's say lstat=10 . Here we got 25.05 units of MEDV , if LSTAT =10

# In[ ]:


X_ex=np.array([[10]])
reg.predict(X_ex)


# In[ ]:


y_pred=reg.predict(X_train)


# # PLOTTING A REGRESSION LINE THROUGH THE DATA GIVES US INSIGHT ABOUT , HOW'S THE ACTUAL DISTRIBUTION OF THE DATA AND THE BEHAVIOUR OF THE ACTUAL FUNCTION F(x) 

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(X_train, y_train,  color='black',marker='*')
plt.plot(X_train, y_pred, color='blue', linewidth=3)


# ## # **INSIGHTS **
# 1. The data shows some linearity , wrt to the X
# 2. There not a very bad variation in the data like an outlier (but it's working abnormally at both ends (when x->0 and x->35)

# 

# 

# # FROM THIS POINT FORWARD I'LL PREPROCESS DATA WITH RESPECT TO SOME STATISTICAL TECHNIQUES , WE'LL PROBABLY SEE A LOT OF CHANGE IN RESULTS 

# # POTENTIAL PROBLEMS
# 
# * Non Linearity of the response-predictor relationships
# * Correlation of error terms
# * Non constant variance of error terms
# * Outliers
# * High Leverage Points
# * Colinearity
# 

# ## 1) Non Linearity of data 
# ### Creating a residual plot , so as to reconsider our approach to observe the distribution of data

# **The linear regression model assumes that there is a linear relationship b/w the predictors and the response . If the true relationship is far from linear , then all the insights that we get from data are questionable** - Gareth James
# 

# We can actually see that there is a trend that is somewhat linear but not perfectly linear , so let's try playing with the variable X , and see if we can get a graph , which is somewhat near to linear behaviour

# In[ ]:


plt.scatter(X_train, y_train,  color='black',marker='*')


# In[ ]:



fig, axs = plt.subplots(1, 4)
fig.set_figheight(5)
fig.set_figwidth(20)

X_train_log = np.log(1 + X_train)
X_train_sqr=X_train**2
X_train_log_sqr=X_train_log**(2)
# let's convert X -> log(x) , so that it get's closer to the linear behaviour
axs[0].set_title('LOG(X) vs Y')
axs[1].set_title('X^2 vs Y')
axs[2].set_title('LOG(X)^2 vs Y')
axs[3].set_title('LOG(X)^(1/2) vs Y')
axs[0].scatter(X_train_log, y_train,  color='black',marker='*')
axs[1].scatter(X_train_sqr,y_train,color='y')
axs[2].scatter(X_train_log_sqr,y_train,color='b')
axs[3].scatter(X_train_log**(1/2),y_train,color='r')

fig, axs = plt.subplots(1, 1)
fig.set_figheight(5)
fig.set_figwidth(20//4)
axs.set_title('X vs Y')
axs.scatter(X_train, y_train,  color='black',marker='*')


# ## By seeing the above 5 graphs , we come to the conclusion that graph 4  LOG(X)^(1/2) vs Y is the most linear in nature and we'll further use it to build the model

# In[ ]:


X_train=X_train_log**(1/2)


# # 2) CORRELATION B/W ERROR  TERMS 

# **The error terms are randomly distributed and show no pattern while we observe the residual plots , so we do not fix the data for anything here**

# # 3) NON CONSTANT VARIANCE OF ERROR TERMS

# There is one more assumption that we make while creatign linear models - "We assume the error terms have a constant variance". Going back to the same residual plot , we can rather see that , the residual vary by large and weird amounts at different intervals.

# **Heteroscedasticity - If the value of the variance of the error term  (y - y^) might increase or decrease with the variation in y , such actions lead to shrinkage of evaluation of responses at the high or low ranges**

# In[ ]:


import seaborn as sns
sns.residplot(x=X_train,y=y_train)


# # We can clearly see that the residual plot  has no shape , i.e it has linearity wrt the residuals for entire data , we could have seen a pattern if this wasn't the case . Nevertheless we'll try some other variations of the Y

# In[ ]:


sns.set(style="whitegrid")

# let's see if taking the log of the output helps
y_train_log = np.log(1 + y_train)
# Plot the residuals after fitting a linear model
sns.residplot(y=y_train_log, x=X_train, lowess=True, color="b")


# # It doesn't actually make data seem more linear and uncorrelated wrt to X

# In[ ]:


sns.residplot(y=y_train**(1/2),x= X_train, lowess=True, color="b")


# In[ ]:


y_train_log = np.log(1 + y_train)
y_train_log_sqrt=y_train_log**(1/2)
sns.residplot(y=(y_train_log_sqrt), x=X_train, lowess=True, color="b")


# # I tried buch of combinations , so as to reduce the effect of variance , but it does not work :(

# # 4)OUTLIER DETECTION 

# In[ ]:


plt.scatter(X_train, y_train,  color='black',marker='*')


# # For calculating outliers , generally we calc studentized residuals , which if >3 or <-3 , is considered an outlier

# In[ ]:


calc=est.outlier_test()
print(calc)


# **So here we can see the rows where the data lies outside the acceptable permit of studentized_intervals**
# 
# **Let's remove this data**

# In[ ]:



list1=[]
for i in range(len(calc)):
    if calc['student_resid'].iloc[i]>3 or calc['student_resid'].iloc[i]<-3:
        list1.append(i)
df=df.drop(list1,axis=0)


# In[ ]:


X=df['LSTAT']
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train = np.array(X).reshape((len(X), 1))
y_train = np.array(y).reshape((len(y), 1))
X_train_log = np.log(1 + X_train)
X_train=X_train_log**(1/2)
reg = LinearRegression().fit(X_train, y_train)
print("Coefficient B1 is :" ,reg.coef_)
print("Intercept B0 is :" ,reg.intercept_)


# In[ ]:


print(reg.score(X_train,y_train))


# # 5)LEVERAGE POINT DETECTION

# In[ ]:


plt.scatter(x=X_train,y=y_train)


# # We see that there are no leverage points present in this data , so for now i won't be removing anything from the dataframe , a leverage point is basically an outlier which lies on the X axis

# # 6)COLINEARITY - In simple linear regression there is no other variable to consider colinearity with , so we just pass on this part as well

# # NOW WE SEE HOW MUCH IMPACT HAVE OUR ACTION BROUGHT TO INITIAL OBSERVATIONS

# In[ ]:


X=df['LSTAT']
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train = np.array(X).reshape((len(X), 1))
y_train = np.array(y).reshape((len(y), 1))
X_train_log = np.log(1 + X_train)
X_train=X_train_log**(1/2)
reg = LinearRegression().fit(X_train, y_train)
print("Coefficient B1 is :" ,reg.coef_)
print("Intercept B0 is :" ,reg.intercept_)
print(reg.score(X_train,y_train))


# * **WE SEE THAT THE COEFFICIENT HAS CHANGED FROM -0.95 INITIALLY TO -0.40 , ALSO THE INTERCEPT HAS CHANGED FROM 35 TO 84 WHICH IS SIGNIFICANT CHANGE TBH, BUT WHILE TAKING INFERENCES WE MUST CONSIDER THE FACT THAT X=LOG(X)^(1/2)**
# * **WE CAN ALSO SEE THAT THERE IS A LOT OF CHANGE IN THE STATISTIC OF R^2 WHICH WENT FROM 54% TO 70% , WUHUUU WE JUST INCREASED EXPLANABILITY**

# # NOW WE GO AHEAD AND CALCULATE THE ACCURACY OF OUR MODEL , FROM HERE WE'LL BE MOVING TO MULTIPLE LINEAR REGRESSION AND COMPARE IT WITH THE SIMPLE LINEAR REGRESSION

# In[ ]:


X_test = np.array(X).reshape((len(X), 1))
y_test = np.array(y).reshape((len(y), 1))
X_test_log = np.log(1 + X_test)
X_test=X_test_log**(1/2)


# In[ ]:


print(reg.score(X_test,y_test))


# # The same accuracy is achieved for test set as well, which means we worked well :)

# # NOW LET'S GET STARTED WITH MULTIPLE LINEAR REGRESSION

# In[ ]:


df.head(5)


# # EXPLORATORY DATA ANALYSIS

# 1. CRIM - Per capita crime rate in the neighbourhood

# In[ ]:


sns.scatterplot(x=df['CRIM'],y=df['MEDV'],data=df)


# * **As one would predict the crime rate is very high from some houses where the value is very low , simplifying it we can say that  - if the neighbourhood has a low MEDV then they might/might not have per capita crime rate very very high**
# * **But it doesn't really ring a bell to have a per capital crime rate of 60-80 , that might just be some leverage points that we should rather skip**
# * Coefficient B1 is : CRIM -0.09262839331882253(In a multiple linear reg model ) which means that as MEDV increases by 1 unit the crime rate decreases by 9 times per capita

# ZN: proportion of residential land zoned for lots over 25,000 sq.ft.

# In[ ]:


sns.scatterplot(x=df['ZN'],y=df['MEDV'],data=df)


# * This variable probably makes no sense 
# * And it might be harmful to keep it in linear model , as this variable is clearly far away from linear "

# INDUS: proportion of non-retail business acres per town

# In[ ]:


sns.scatterplot(x=df['INDUS'],y=df['MEDV'],data=df)


# * This variable shows some beahviour and pattern which might be useful 
# * So , we won't be dropping this column

# In[ ]:


sns.scatterplot(x=df['CHAS'],y=df['MEDV'],data=df)


# * Usually categorical variable do not come encoded in the data , but here they are finely coded 
# * We'll be keeping the variable in the dataset

# In[ ]:


sns.scatterplot(x=df['NOX'],y=df['MEDV'],data=df)


# * This variable also shows some useful info that we can use for further analysis
# * We'll be usin the variable
# * Although might try some feature scaling on the variable

# In[ ]:


sns.scatterplot(x=df['RM'],y=df['MEDV'],data=df)


# * Seems like a very good variable and shows linear behaviour
# * So we'll probably not even require scaling for the variable

# In[ ]:


sns.scatterplot(x=df['AGE'],y=df['MEDV'],data=df)


# * Another useful variable , we'll be keeping in the 
# * Non-linear scaling could severly help the variable

# In[ ]:


sns.scatterplot(x=df['DIS'],y=df['MEDV'],data=df)


# * A good variable , which won't require any scaling as it's behaviour is very close to linear

# In[ ]:


sns.scatterplot(x=df['TAX'],y=df['MEDV'],data=df)


# * Might just code this variable to become a categorical variable  , where threshold =400

# In[ ]:


sns.scatterplot(x=df['PTRATIO'],y=df['MEDV'],data=df)


# * Doesn't show any linear behaviour , we might just drop it

# In[ ]:


sns.scatterplot(x=df['B'],y=df['MEDV'],data=df)


# * Doesn't look like a good variable 
# * Don't know how to handle this feature tbh

# In[ ]:


sns.scatterplot(x=df['LSTAT'],y=df['MEDV'],data=df)


# * We encoded this variable earlier as well , certainly did a good job after scaling

# # Let's train a general multiple regression model without working on the present state of the data

# In[ ]:


X=df.drop(columns=['MEDV'])
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
reg = LinearRegression().fit(X_train, y_train)
for i in range(len(X.columns)):
    print("Coefficient B1 is :" ,X.columns[i],reg.coef_[i])
    print('')
print("Intercept B0 is :" ,reg.intercept_)


# In[ ]:


X=df.drop(columns=['MEDV'])
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
reg = LinearRegression().fit(X_train, y_train)


# In[ ]:


y_pred=reg.predict(X_train)


# # NOW LET'S CALCULATE THE RSE OF THE FIT NORMALLY WITHOUT ANY CLEANING , SCALING , INTERACTION, NON LINEAR TRANSFORMATION ETC

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
num_data = X_train.shape[0]
mse = mean_squared_error(y_train,y_pred)
rse = math.sqrt(mse/(num_data-2))
mae=mean_absolute_error(y_train,y_pred)
print(rse,' is the residual standard error')
print(mae,' is the mean absolute error')


# In[ ]:


print(reg.score(X_train,y_train),' is  the R^2 statistic obtained for the multiple regression ')


# # After fitting the data using multiple linear regression , we get a R2 statistic of 77% which isn't great considering we got 70% in simple linear regression itself ,we'll work towards improving the model , from here forward.

# # We'll be using Non Linear Transformations , Interaction Terms, Colinearity Detection, Outlier Removal ,Non Linearity of Data , Non Constant Variance of Error Terms etc

# In[ ]:


print(sm.OLS(y_train, X_train).fit().summary())


# # We'll drop the columns where p>0.05 , indicating that there is no linear relationship bw the dropped variables and the output-
# 
# * INDUS 
# * NOX
# * RAD 
# * TAX
# * ZN
# 

# In[ ]:


df=df.drop(columns=['INDUS','NOX','RAD','TAX','ZN'])


# In[ ]:


X=df.drop(columns=['MEDV'])
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
reg = LinearRegression().fit(X_train, y_train)


# In[ ]:


y_pred=reg.predict(X_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
num_data = X_train.shape[0]
mse = mean_squared_error(y_train,y_pred)
rse = math.sqrt(mse/(num_data-2))
mae=mean_absolute_error(y_train,y_pred)
print(rse,' is the residual standard error')
print(mae,' is the mean absolute error')


# In[ ]:


print(reg.score(X_train,y_train),' is  the R^2 statistic obtained for the multiple regression ')


# # Although we dropped by 3% in our calcuation of R2 statistic , i don't mind dropping 5 columns even if there's a loss of 3%

# * Introducing Interaction Terms
# * Intoduction Non Transformations
# 

# # Introducing an interaction columns 
# 
# Let's try to introduce a columns which is dependant on LSTAT*AGE

# In[ ]:


df['LS*AGE']=df['LSTAT']*df['AGE']


# In[ ]:


X=df.drop(columns=['MEDV'])
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
reg = LinearRegression().fit(X_train, y_train)
y_pred=reg.predict(X_train)
num_data = X_train.shape[0]
mse = mean_squared_error(y_train,y_pred)
rse = math.sqrt(mse/(num_data-2))
mae=mean_absolute_error(y_train,y_pred)
print(rse,' is the residual standard error')
print(mae,' is the mean absolute error')


# In[ ]:


print(sm.OLS(y_train, X_train).fit().summary())


# In[ ]:



for col in X_train.columns:
    fig, axs = plt.subplots(1, 4)
    fig.set_figheight(5)
    fig.set_figwidth(20)
    var_train_log = np.log(1 + X_train[str(col)])
    var_train_sqr= X_train[str(col)]**2
    var_train_log_sqr=var_train_log**(2)
    # let's convert X -> log(x) , so that it get's closer to the linear behaviour
    print('This is for columns named',col)
    axs[0].set_title('LOG(X) vs Y')
    axs[1].set_title('X^2 vs Y')
    axs[2].set_title('LOG(X)^2 vs Y')
    axs[3].set_title('LOG(X)^(1/2) vs Y')
    axs[0].scatter(var_train_log, y_train,  color='black',marker='*')
    axs[1].scatter(var_train_sqr,y_train,color='y')
    axs[2].scatter(var_train_log_sqr,y_train,color='b')
    axs[3].scatter(var_train_log**(1/2),y_train,color='r')
    fig, axs = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(20//4)
    axs.set_title(str(col)+' vs Y')
    axs.scatter(X_train[str(col)], y_train,  color='black',marker='*')


# # FEATURE SCALING (NON LINEAR TRANSFORMATIONS FOR -CRIM , AGE, DIS,LSTAT)

# In[ ]:


df['CRIM']=df['CRIM']**2
df['AGE']=(np.log(1 + df['AGE']))**(1/2)
df['DIS']=(np.log(1 + df['DIS']))**(1/2)
df['LSTAT']=(np.log(1 + df['LSTAT']))**(1/2)


# In[ ]:


X=df.drop(columns=['MEDV'])
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
reg = LinearRegression().fit(X_train, y_train)
y_pred=reg.predict(X_train)
num_data = X_train.shape[0]
mse = mean_squared_error(y_train,y_pred)
rse = math.sqrt(mse/(num_data-2))
mae=mean_absolute_error(y_train,y_pred)
print(rse,' is the residual standard error')
print(mae,' is the mean absolute error')


# In[ ]:


y_pred=reg.predict(X_train)


# In[ ]:


print(reg.score(X_train,y_train),' is  the R^2 statistic obtained for the multiple regression ')


# # Although the R2 statistic of the graph has increased to 80% it's going to be very hard for me to explain such tremendous mathematically scaled graphs in terms of references :) 
# 

# # NON CONSTANT VARIANCE OF ERROR TERMS
# 
# 

# In[ ]:


sns.residplot(x=X['CRIM'],y=y)


# In[ ]:


sns.residplot(x=X['CRIM'],y=y**2)


# In[ ]:


y_log = np.log(1 + y)
sns.residplot(x=X['CRIM'],y=y_log)


# In[ ]:


df.columns


# In[ ]:


sns.residplot(x=X['RM'],y=y)


# In[ ]:


sns.residplot(x=X['RM'],y=y**2)


# In[ ]:


sns.residplot(x=X['RM'],y=y_log)


# In[ ]:


sns.residplot(x=X['DIS'],y=y)


# In[ ]:


sns.residplot(x=X['DIS'],y=y_log)


# In[ ]:


sns.residplot(x=X['DIS'],y=y**2)


# In[ ]:


#est= sm.OLS(y_train, X_train).fit()
#calc=est.outlier_test()
#print(calc)
#list1=[]
#for i in range(len(calc)):
#    if calc['student_resid'].iloc[i]>3 or calc['student_resid'].iloc[i]<-3:
#        list1.append(i)
#df=df.drop(list1,axis=0)


# In[ ]:


X=df.drop(columns=['MEDV'])
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
reg = LinearRegression().fit(X_train, y_train)
y_pred=reg.predict(X_train)
num_data = X_train.shape[0]
mse = mean_squared_error(y_train,y_pred)
rse = math.sqrt(mse/(num_data-2))
mae=mean_absolute_error(y_train,y_pred)
print(rse,' is the residual standard error')
print(mae,' is the mean absolute error')
y_pred=reg.predict(X_train)
print(reg.score(X_train,y_train),' is  the R^2 statistic obtained for the multiple regression ')


# # EXPERIMENTSSSSSSSSSS

# ## 1) ARE 3D PLOTS USEFUL ??
# 

# In[ ]:


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df['LSTAT'], df['MEDV'] , df['RM']  , c=df['LSTAT'], cmap='Greens');


# In[ ]:


def f(x, y):
    return (np.array(x *-0.3226  + y *-0.0035 ))

y=df['LSTAT']
x=df['RM']

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_ylabel('LSTAT')
ax.set_xlabel('RM')
ax.set_zlabel('MEDV');


# In[ ]:


y_pred_test=reg.predict(X_test)


# # Mean Absolute Error 
# * Average of the difference between the Original Values and the Predicted Values.
# * Do not gives any idea of the direction of the error i.e. whether we are under predicting the data or over predicting the data.
# * Smaller the MAE, better is the model.

# In[ ]:


from sklearn import metrics
print(metrics.mean_absolute_error(y_train, y_pred),'for training dataset')
print(metrics.mean_absolute_error(y_test, y_pred_test),'for test dataset')


# # Mean Squared Error 
# * Takes the average of the square of the difference between the original values and the predicted values.
# * As we take square of the error, the effect of larger errors(sometimes outliers) become more pronounced then smaller error. Model will be penalized more for making predictions that differ greatly from the corresponding actual value.

# In[ ]:


print(metrics.mean_squared_error(y_train, y_pred),'for training daatset')
print(metrics.mean_squared_error(y_test, y_pred_test),'for test dataset')


# # RMSE
# * Because the MSE is squared, its units do not match that of the original output. RMSE is the square root of MSE.
# * Since the MSE and RMSE both square the residual, they are similarly affected by outliers.

# In[ ]:


from math import sqrt
print(sqrt(metrics.mean_squared_error(y_train, y_pred)),'for training dataset')
print(sqrt(metrics.mean_squared_error(y_test, y_pred_test)),'for testing dataset')


# # R_squared 

# In[ ]:


print(metrics.r2_score(y_train, y_pred),'for training dataset')
print(metrics.r2_score(y_test, y_pred_test),'for test datset')


# In[ ]:


corr = df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')


# # INTERACTION TERMS
# 
# 1. **The use of interaction terms has been seen in the case of LSTAT WITH AGE , but that was a random interaction term , with no intuition behind it , we now further persue the topic in order to experiment with out regression model **

# * CRIM: per capita crime rate by town
# *  ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# * INDUS: proportion of non-retail business acres per town
# * CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# * NOX: nitric oxides concentration (parts per 10 million)
# * 20.2. Load the Dataset 124
# * RM: average number of rooms per dwelling
# * AGE: proportion of owner-occupied units built prior to 1940
# * DIS: weighted distances to ﬁve Boston employment centers
# * RAD: index of accessibility to radial highways
# * TAX: full-value property-tax rate per 10,000 dollars
# * PTRATIO: pupil-teacher ratio by town 12.
# * B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13.
# * LSTAT: perc lower status of the population
# * MEDV: Median value of owner-occupied homes in 1000 dollars
# We can see that the input attributes have a mixture of units.

# INTERACTION TERM  1) The PTRATIO and DIS might have the potential to be an interaction term, as they don't have much power by themselves but together in my opinion could be stronger variables , for eg if the pupil teacher ratio is high and weighted sitance is low , then the medv values might actually be very high

# In[ ]:


df['New']=df['DIS']*df['PTRATIO']


# In[ ]:


X=df.drop(columns=['MEDV'])
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
reg = LinearRegression().fit(X_train, y_train)
y_pred=reg.predict(X_train)
num_data = X_train.shape[0]
mse = mean_squared_error(y_train,y_pred)
rse = math.sqrt(mse/(num_data-2))
mae=mean_absolute_error(y_train,y_pred)
print(rse,' is the residual standard error')
print(mae,' is the mean absolute error')
y_pred=reg.predict(X_train)
print(reg.score(X_train,y_train),' is  the R^2 statistic obtained for the multiple regression ')


# # EXPERIMENT FAILED :( - The new interaction term does not add any significant value to the present model

# In[ ]:


df=df.drop(columns=['New'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# # Removing colinear variables

# In[ ]:


X_train.head()


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

X_train=calculate_vif_(X_train)


# In[ ]:


X_train


# In[ ]:


reg = LinearRegression().fit(X_train, y_train)
y_pred=reg.predict(X_train)
num_data = X_train.shape[0]
mse = mean_squared_error(y_train,y_pred)
rse = math.sqrt(mse/(num_data-2))
mae=mean_absolute_error(y_train,y_pred)
print(rse,' is the residual standard error')
print(mae,' is the mean absolute error')
y_pred=reg.predict(X_train)
print(reg.score(X_train,y_train),' is  the R^2 statistic obtained for the multiple regression ')


# # THAT DIDNT GO VERY WELL, THE R2 STATISTIC WENT FROM 80% TO 50% ON REMOVAL OF COLINEAR VARIBALE ACCORDING TO VIF 
# 
# # EXPERIMENT FAILED :(

# In[ ]:


X=df.drop(columns=['MEDV'])
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
reg = LinearRegression().fit(X_train, y_train)
y_pred=reg.predict(X_train)
num_data = X_train.shape[0]
mse = mean_squared_error(y_train,y_pred)
rse = math.sqrt(mse/(num_data-2))
mae=mean_absolute_error(y_train,y_pred)
print(rse,' is the residual standard error')
print(mae,' is the mean absolute error')
y_pred=reg.predict(X_train)
print(reg.score(X_train,y_train),' is  the R^2 statistic obtained for the multiple regression ')


# # ADVANCED VISUALIZATIONS 

# In[ ]:


df.columns


# In[ ]:


df.head(5)


# # Line Charts

# ## Let's visulaize LSTAT and LS * AGE wrt to the MEDV value

# In[ ]:


from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[ ]:


# Creating trace1
trace1 = go.Scatter(
                    x = df['B'],
                    y = df['PTRATIO'],
                    mode = "markers",
                    name = "value=MEDV",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df['MEDV'])
# Creating trace2


    
data = [trace1]
layout = dict(title = 'The percentage of black people in the city vs the pupil to teacher ratio',
              xaxis= dict(title= 'BLACKS%',ticklen= 5,zeroline= False),yaxis=dict(title='PTRAIO')
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# Creating trace1
trace1 = go.Scatter(
                    x = df['CRIM'],
                    y = df['B'],
                    mode = "markers",
                    name = "value=MEDV",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df['CRIM'])
# Creating trace2


    
data = [trace1]
layout = dict(title = 'The crime rate vs The percentage of black people in the city ',
              xaxis= dict(title= 'Crime Rate',ticklen= 5,zeroline= False),yaxis=dict(title='Blacks%')
             )
fig = dict(data = data, layout = layout)
iplot(fig)

