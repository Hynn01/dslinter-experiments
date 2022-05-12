#!/usr/bin/env python
# coding: utf-8

# # Titanic Solution Beginner Walkthrough (top 8%)
# 
# The purpose of this notebook is to show you how I have navigated through the Kaggle Titanic project by following an effective Data Science Workflow. To be specific, the project will tackle some of the core elements of a Data Science pipeline as follows: 
# - Exploratory Data Analysis
# - Data Visualization
# - Feature Selection
# - Feature Engineering
# - Machine Learning

# ## Data Science Workflow
# > I want to give credit to the author of this [notebook](https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook) for a very comprehensive Data Science guide and for introducing the **4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting**, which was the foundation of this workflow. I just thought, wouldn't it be better if there were more C's.
# - **Comprehend.** Understand the nature and relationships among each features in the datasets through analyzing and visualizing.
# - **Correlate.** Known as *'Feature Selection',* this approach aims to validate the strength of association across features with the appopriate statistical tools and metrics, and to select the features that are deemed relevant.
# - **Correct.** Identify and remedy the missing/null values. May consider imputing them for features that are deemed significant.  
# - **Create.** Known as *'Feature Engineering',* this approach attempts to create new features out of the existing ones which can make better predictions while also reducing noise in the number of features.
# - **Convert.** Perform the necessary adjustments and transformations to make the datasets normally distributed and fit for modelling.
# - **Combine.** Known as *'Ensemble Models',* this approach aims to combine multiple algorithms into one which leverages the strengths and compensates the weaknesses of the tested models.

# ## Background of the Problem
# The complete overview and description of the Kaggle competition be found [here](https://www.kaggle.com/c/titanic). Here are some of information we were provided with the link.
# - On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg.
# - Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# - While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

# In[ ]:


# data analysis
import pandas as pd
import numpy as np

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Acquire Training and Testing Data
# 
# The information regarding the features are explicitly presented [here](https://www.kaggle.com/competitions/titanic/data?select=train.csv) in detail.

# In[ ]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df_copy = test_df.copy()
df = [train_df, test_df]

test_df_copy.head()


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Exploratory Data Analysis
# This stage will focus on comprehending the nature and relationships of the features.

# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


len(test_df)/(len(train_df)+len(test_df))


# In[ ]:


train_df.describe()


# In[ ]:


#test_df.describe()


# #### Observations
# 
# **Distribution**
# - The train-test is split around 70/30, with testing set representing 31.93% of the combined sets.
# - The survival rate in context of the training set is at 38.38%, which is representative of the original survival rate of 32.46%, 1502 out of 2224.
# - Majority (around 75%) of the people didn't aboard with siblings/spouses
# - The distributions of SibSP and Parch are right skewed, since the means are greater than the medians.
# 
# **Data Types**
# - Numerical: *Age, Sibsp, Parch, Fare*
# - Categorical: *Survival, Sex, Pclass, Embarked*
# - Mixed/Alphanumeric: *Name, Ticket, Cabin*
# 
# **Assumptions**
# - **Continuous vs Nominal.** Children (people belonging to lower age brackets) are more likely to survive.
# - **Nominal vs Nominal.** Women are more likely to survive than men. 
# - **Ordinal vs Nominal.** The higher-echelon and the wealthier classes are more likely to survive.
# - **Continuous vs Nominal.** Individuals who travel with larger families have a lower likelihood of surviving.
# - **Nominal vs Nominal.** Those who are travel alone have a higher chance of surviving.
# - **Nominal vs Nominal.** Are the individuals' port of embarkation associated with their survival?

# # Analyzing the Numerical Features
# For the numerical variables, the seaborn **pairplot** will be helpful in presenting the pariwise relationships across each numerical variables. The diagonal plots are the main focus here as they are treated as distribution plots of the features. The rest is just being extra and for eye candy.

# In[ ]:


# Separate the training set into groups of numerical and categorical variables.
# Don't worry, the 'Survived' was only included in the numerical category so we can use it to classify the features when we create the pairplot
df_num = train_df[['Age', 'Survived', 'SibSp', 'Parch', 'Fare']]
df_cat = train_df[['Survived', 'Pclass', 'Sex', 'Embarked']]

# Classify by 'Survived'
sns.set_style('darkgrid')
sns.pairplot(df_num, hue='Survived', palette='Blues')


# In[ ]:


plt.figure(figsize=(24,4))
plt.subplot(1,4,1)
sns.histplot(data=train_df, x="Age", hue="Survived")

plt.subplot(1,4,2)
sns.histplot(data=train_df, x="SibSp", hue="Survived")

plt.subplot(1,4,3)
sns.histplot(data=train_df, x="Parch", hue="Survived")

plt.subplot(1,4,4)
sns.histplot(data=train_df, x="Fare", hue="Survived")


# **Age**
# - More normally distributed compared to the rest.
# - Majority of passengers fall in the 20-35 age bracket.
# - A large number of 20-30 yr olds didn't survive.
# - Infants (age<4) had one of the highest survival rates.
# - The oldest individual (age=80) survived.
# 
# **SibSp.**
# - Skewed to the right.
# - Those with 1-2 siblings/spouses were likely to survive.
# - Large number of passengers didn't have siblings/spouses with them.
# - An outlier, with over 8 siblings/spouses, didn't survive.
# 
# **Parch.**
# - Skewed to the right.
# - Large number of passengers didn't have parents/children with them.
# - Passengers without parents and children with them were more likely to die.
# 
# **Fare.**
# - Skewed to the right.
# - Majority of the passengers aboarded with cheaper fares.
# - Most passengers with cheaper fares (<50) didn't survive, while those who paid higher fares (>300) tend to survive.

# # Correlating the Numerical Variables

# In[ ]:


## Correlation Matrix
plt.subplots(figsize=(10,7))
sns.heatmap(df_num.corr(), cmap='Blues', annot=True, linewidths=2, annot_kws={"fontsize":15})


# #### Observations
# 
# - Parch and SibSp features a positively moderate correlation.
# - Parch and Fare has a positively weak correlation.
# - We noticed that 'Age' has a very weak correlation with our solution goal despite following a normal distribution.
# 
# #### Decisions
# - We can try normalizing the skewed distributions of 'SibSp', 'Parch', and 'Fare' to see if it improves the correlations.
# - Create categorical features 'AgeGroup' from existing 'Age'.
# - To address potential multicollinearity among our highly dependent input features 'SibSp' and 'Parch', let's combine them by multiplying both 'SibSp*Parch'.

# ## Creating 'SibSp*Parch' Feature

# In[ ]:


# Parch*SibSp
for dataset in df:
    dataset['SibSp*Parch'] = dataset['SibSp']*dataset['Parch']

plt.subplots(figsize=(10,7))
df_num = train_df[['Age', 'Survived', 'SibSp', 'Parch', 'SibSp*Parch', 'Fare']]
sns.heatmap(df_num.corr(), cmap='Blues', annot=True, linewidths=2, annot_kws={"fontsize":15})


# # Normalizing Numerical Features
# The **probability plot** or **quantile-quntile plot (QQplot)** allows us to plot our sample data against the quantiles of a normal distribution. This will serve as reference to see how our subsequent data transformations react to the curve, and enable us to select the best form of transformation which resulted to the best fit.
# 
# We can try use the following data transformation techniques:
# - Square Root
# - Cube Root
# - Logarithmic. **Log(x+1)** wil ensure the log transformation won't result in undefined values because our data contains 'zero' values, and log(0) returns undefined.

# In[ ]:


import scipy.stats as stats

# Defining the function to generate the distribution plot alongside QQplot
def QQplot(df, col):
    plt.figure(figsize = (10, 4))
    plt.subplot(1,2,1)
    sns.histplot(x=df[col].dropna(), kde=True)
    
    plt.subplot(1,2,2)
    stats.probplot(df[col].dropna(), dist="norm", plot=plt)


# #### Normalizing Age

# In[ ]:


QQplot(train_df, 'Age')


# #### Normalizing SibSp

# In[ ]:


QQplot(train_df, 'SibSp')


# In[ ]:


# Perform data transformations and generate QQplots for 'SibSp'

df_num = train_df[['Age', 'Survived', 'SibSp', 'Parch', 'Fare']]

df_num["SibSp_sqrt"] = df_num['SibSp']**(1./2)
QQplot(df_num, 'SibSp_sqrt')

df_num["SibSp_cbrt"] = df_num['SibSp']**(1./3)
QQplot(df_num, 'SibSp_cbrt')

df_num["SibSp_log(x+1)"] = np.log(df_num['SibSp'] + 1)
QQplot(df_num, 'SibSp_log(x+1)')


# #### Normalizing Parch

# In[ ]:


QQplot(train_df, 'Parch')


# In[ ]:


# Perform data transformations and generate QQplots
df_num["Parch_sqrt"] = df_num['Parch']**(1./2)
QQplot(df_num, 'Parch_sqrt')

df_num["Parch_cbrt"] = df_num['Parch']**(1./3)
QQplot(df_num, 'Parch_cbrt')

df_num["Parch_log(x+1)"] = np.log(df_num['Parch'] + 1)
QQplot(df_num, 'Parch_log(x+1)')


# #### Normalizing Fare

# In[ ]:


QQplot(train_df, 'Fare')


# In[ ]:


# Perform data transformations and generate QQplots for 'Fare'
tr_Fare = train_df[['Survived', 'Fare']]

tr_Fare["Fare_sqrt"] = tr_Fare['Fare']**(1./2)
QQplot(tr_Fare, 'Fare_sqrt')

tr_Fare["Fare_cbrt"] = tr_Fare['Fare']**(1./3)
QQplot(tr_Fare, 'Fare_cbrt')

tr_Fare["Fare_log(x+1)"] = np.log(tr_Fare['Fare'] + 1)
QQplot(tr_Fare, 'Fare_log(x+1)')


# #### Observations
# 
# - **Age.** Normally distributed. No need to transform.
# - **SibSp.** Still highly skewed. 
# - **Parch.** Still highly skewed.
# - **Fare.** The log(x+1) transformation yielded the best fit.
# 
# #### Decisions
# - Peform log(x+1) transformation on 'Fare' feature.
# - Create a new 'Family' feature which combines the numbers in 'SibSp' and 'Parch'.
# - Assess if 'Family' is normally distributed and correlated with the solution goal.

# ## Creating 'FamilySize' Feature

# In[ ]:


for dataset in df:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

QQplot(train_df, 'FamilySize')


# In[ ]:


# Perform data transformations and generate QQplots for 'Family'
df_num = train_df[['FamilySize']]

df_num["Family_sqrt"] = df_num['FamilySize']**(1./2)
QQplot(df_num, 'Family_sqrt')

df_num["Family_cbrt"] = df_num['FamilySize']**(1./3)
QQplot(df_num, 'Family_cbrt')

df_num["Family_log(x+1)"] = np.log(df_num['FamilySize'])
QQplot(df_num, 'Family_log(x+1)')


# In[ ]:


## Correlation Matrix for 'Fare' Transformations
plt.subplots(figsize=(10,7))
sns.heatmap(tr_Fare.corr(), cmap='Blues', annot=True, linewidths=2, annot_kws={"fontsize":15})


# In[ ]:


## Correlation Matrix for 'Family'
df_Family = train_df[['SibSp', 'Parch', 'FamilySize', 'Survived']]
plt.subplots(figsize=(10,7))
sns.heatmap(df_Family.corr(), cmap='Blues', annot=True, linewidths=2, annot_kws={"fontsize":15})


# #### Observations
# 
# - After transforming the 'Fare' feature, its correlation with solution goal improved from 0.26 to 0.33.
# - It appears that the 'Family' feature didn't improve the correlation, compared to the likes of 'SibSp' and 'Parch'. 
# - 'Age' has a very underwhelming correlation with the solution goal, only at -0..07.
# 
# #### Decisions
# - Let's create a new categorical feature called 'withFamily' given 'Family' where we set values to (0 = without family) and (1 = with family). Validate the correlation
# - We can drop the 'SibSp' and 'Parch' features as they are no longer relevant.
# - Create a categorical feature 'AgeGroup' out of 'Age' and see if the correlation improves.

# ## Creating 'AgeGroup' Feature from 'Age'

# In[ ]:


for dataset in df:
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
    dataset['AgeGroup'] = pd.cut(dataset['Age'], 8)

train_df[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean()


# In[ ]:


# Converting 'AgeGroup' into a Categorical Variable
for dataset in df:
    dataset.loc[dataset['Age'] <= 10, 'AgeGroupNum'] = 0
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'AgeGroupNum'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'AgeGroupNum'] = 2
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'AgeGroupNum'] = 3
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'AgeGroupNum'] = 4
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'AgeGroupNum'] = 5
    dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 70), 'AgeGroupNum'] = 6
    dataset.loc[(dataset['Age'] > 70), 'AgeGroupNum'] = 7

train_df['AgeGroupNum'].unique()


# ## Creating 'FareGroup' Feature from 'Fare'

# In[ ]:


# for dataset in df:
#     dataset['Fare'] = dataset['Fare'].fillna(dataset['Age'].median())
#     dataset['FareGroup'] = pd.cut(dataset['Fare'], 5)

# train_df[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean()


# ## Creating 'withFamily' Feature from 'Family'

# In[ ]:


for dataset in df:
    
    # We represent 1 if Person is with Family, 0 otherwise
    dataset['withFamily'] = 0
    dataset.loc[dataset['FamilySize'] > 1, 'withFamily'] = 1

train_df.head()


# # Analyzing the Categorical Features
# 
# The categorical variables, along with the features we created previously, that we want to analyze are as follows:
# - Pclass
# - Sex
# - Embarked
# - AgeGroupNum
# - withFamily
# 
# Let's use **countplots** to visualize the distribution of each classification with respect to survival, and **lineplots** to determine the corresponding survival rates (in % form). 

# In[ ]:


def Catplot(df, x, y):
    with sns.axes_style('darkgrid'):
        plt.figure(figsize = (12, 4))
        plt.subplot(1,2,1)
        sns.countplot( x=df[x].dropna(), hue=df[y], palette='Blues')
        
        plt.subplot(1,2,2)
        plt.ylim(0,1)
        sns.lineplot( x=df[x], y=df[y], data=df, ci=None, linewidth=2, marker="o")
        
Catplot(train_df, 'Sex', 'Survived')
Catplot(train_df, 'Pclass', 'Survived')
Catplot(train_df, 'Embarked', 'Survived')
Catplot(train_df, 'AgeGroupNum', 'Survived')
Catplot(train_df, 'withFamily', 'Survived')


# #### Observations
# 
# **Sex**
# - More males were on board.
# - Females had a higher survival rate than males did. No wonder Jack died over Rose.
# 
# **Pclass**
# - Most passengers were in Pclass 3. 
# - The significance of higher classes being correlated with higher survival rate is justified.
# - Pclass 1 is the only class with more survived passengers than dead passengers.
# 
# **Embarked**
# - A major chunk of the passengers embarked from Southampton, and the least from Queensland.
# - Highest survival rate were found for passengers that embarked from C.
# 
# **AgeGroupNum**
# - Most passengers belong to the 16-32 yr old group. 
# - Highest survival rate came from toddlers to teens group (<16).
# - Least survival rate came from senior age group (>64), followed by the adolescents to mid-age group (16-32).
# 
# **WithFamily**
# - Those who aboarded alone are likely to die than those with their families.
# > *"There is nothing stronger than family." - Dom* 
# 
# #### Decisions
# - Validate the orrelations of 'Sex', 'Pclass', 'Embarked', and 'withFamily' with solution goal.
# - Create a new categorical feature 'Title' by extracting the titles from the given names of the passengers. Validate correlation.
# - Create a new categorical feature 'Unit' by extracting the units (i.e. letters, numbers) from the cabin numbers. Validate correlation.

# # Analyzing the Alphanumeric Variables
# - Does a passenger's title relate to survival rates?
# - Does the cabin unit *(extract first letter from alphanumeric string)*  affect survival rates?

# In[ ]:


# Analyzing 'Name'

# Split the full name into a list by comma, then return the title by indexing the 2nd position [1]
# Split the name into a list by period, then return the title by indexing the 1st position [0]

train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


def Catplot_Title(df, x, y):
    with sns.axes_style('darkgrid'):
        plt.figure(figsize = (15, 5))
        plt.subplot(1,2,1)
        sns.countplot( x=df[x].dropna(), hue=df[y], palette='Blues')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        plt.xticks(rotation=45)
        sns.lineplot( x=df[x], y=df[y], data=df, ci=None, linewidth=2, marker="o")
    
Catplot_Title(train_df, 'Title', 'Survived')


# **Prefixes Meaning**
# 
# - **Rev.** Ministers of most Christian denominations; ordained clergymen since 17th century.
# - **Mlle and Miss.** Unmarried female; young lady.
# - **Mme.** Woman
# - **Master, Major, and Don.** Lord, master, or owner (of a household).
# - **Col.** Colonel; army officer of high rank.
# - **the Countess.** Wife or widow of a count.
# - **Capt.** could refer to the captain of the ship.
# - **Ms.** Any women regardless of marital status.
# - **Lady.** Princesses or daughters of royal blood.
# - **Jonkheer.** Female equivalent denoting the lowest rank within the nobility.
# 
# **Classify Titles**
# - Mlle and Ms > Miss
# - Mme > Mrs
# - Uncommon titles will be treated as a new category > Others

# In[ ]:


for dataset in df:
    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Returns the unique 'titles' from the dataset
Unique_titles = np.union1d(train_df['Title'].unique(), test_df['Title'].unique())
Unique_titles


# In[ ]:


for dataset in df:
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir','the Countess'], 'Others')
    
print(train_df['Title'].unique())
print(test_df['Title'].unique())


# In[ ]:


Catplot(train_df, 'Title', 'Survived')


# **Titles**
# 
# - It justifies the correlation between 'Sex' and 'Survived' as titles with 'Mr' and 'Master' tend to have lower survival rates than 'Mrs' and 'Miss'.
# - Also justifies the correlation between 'withFamily' and 'Survived' as titles with 'Mrs' tend to have higher survival rates than 'Miss'.
# - The 'Others' category compiles a very small sample of the given dataset. Despite these titles having characterized of nobile status, it appears that they were trivial at the time of the crisis.

# In[ ]:


# Analyzing 'Cabin'
for dataset in df:
    dataset['CabinUnit'] = train_df['Cabin'].apply(lambda x: str(x)[0])


# > Note that null values will be treated as a category, denoted as CabinUnit 'n'. 

# In[ ]:


pd.pivot_table(train_df, index='CabinUnit', columns='Survived', values='Name', aggfunc='count')


# In[ ]:


Catplot(train_df, 'CabinUnit', 'Survived')


# **Cabin Unit.**
# 
# - The abundance of missing values makes it difficult for 'CabinUnit' to be representative as a sample.
# - Deck/Cabin A was known as the Pomegrande, which is located at the superstructure of Titanic. The assumption that Cabin A would have a higher survival rate, however, isn't justified by our data. Hence, Cabin Unit column shall be removed.

# # Correlating the Categorical Features
# Assessing the strength of association (correlation) across variables is one way to conduct feature selection. This time, it wouldn't be ideal to use **Pearson's correlation matrix** we did for our numerical variables earlier because we are now dealing with categorical variables (both the predictors and response variables). Using the **[Chi-square test](https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223#:~:text=In%20feature%20selection%2C%20we%20aim,hypothesis%20of%20independence%20is%20incorrect.)** is the appropriate statistical method here.
# 
# Before we can conduct Chi-square tests, we must ensure that our categorical data are numerically encoded first using `LabelEncoder()`.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder()

train_df_copy = train_df.copy()

train_df_copy['Sex'] = label_encode.fit_transform(train_df_copy['Sex'])
train_df_copy['Embarked'] = label_encode.fit_transform(train_df_copy['Embarked'])
train_df_copy['Pclass'] = label_encode.fit_transform(train_df_copy['Pclass'])
train_df_copy['Title'] = label_encode.fit_transform(train_df_copy['Title'])

train_df_copy.head()


# In[ ]:


from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

# Split our dataset into x and y variables
x = train_df_copy[['Sex', 'Pclass', 'Embarked', 'withFamily', 'AgeGroupNum', 'Title']]
y = train_df_copy['Survived']
chi2_scores = chi2(x,y)

chi2_scores = pd.DataFrame(np.transpose(chi2_scores), index=['Sex', 'Pclass', 'Embarked', 'withFamily', 'AgeGroupNum', 'Title'], columns=['Chi2', 'p_value']).sort_values('p_value', ascending=True)
chi2_scores


# # Data Preprocessing
# - Drop null values from 'Embark' in datasets.
# - Dropping irrelevant column features.
# - Impute missing values in testing 'Fare' with median, because it is skewed.
# - Normalize 'Fare' through log transformation.
# - Perform one-hot encoding on our categorical data.
# - Scaling our numerical data.

# In[ ]:


# Remove 'Embark' null values
for dataset in df:
    dataset.dropna(subset=['Embarked'], inplace=True)
    
df = [train_df, test_df]

print(train_df.shape, test_df.shape)


# In[ ]:


# Impute missing values in 'Age' and 'Fare' with mean and median respectively
for dataset in df:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    
train_df.isnull().sum()


# In[ ]:


# Log transform 'Fare'
for dataset in df:
    dataset['Fare'] = np.log(dataset['Fare'] + 1)


# In[ ]:


# Dropping irrelevant column features
train_df.drop(['PassengerId','Age', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize', 'Cabin', 'CabinUnit', 'AgeGroup'], inplace=True, axis=1)
test_df.drop(['PassengerId', 'Age', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize', 'Cabin', 'CabinUnit', 'AgeGroup'], inplace=True, axis=1)
df = [train_df, test_df]


# In[ ]:


# Creating dummy indicator columns for categorical variables
train_df = pd.get_dummies(train_df, columns=['Sex', 'Pclass', 'Embarked', 'withFamily', 'AgeGroupNum', 'Title'])
test_df = pd.get_dummies(test_df, columns=['Sex', 'Pclass', 'Embarked', 'withFamily','AgeGroupNum', 'Title'])

train_df.head()


# # Splitting Training Data
# There is a need to split our training data into 2 subsets of training and testing data once more. Why is that? Note that the test.csv file provided is merely just a validation data for our competition submission, so it can't be treated as testing data which our current training data can learn from lest it would lead to overfitting our data.

# In[ ]:


from sklearn import model_selection
x1 = train_df.drop(['Survived'], axis=1)
y1 = train_df['Survived']
x1_train, x1_test, y1_train, y1_test = model_selection.train_test_split(x1, y1, random_state=42)

print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape)


# # Preliminary Model Generation
# Now that we have completed the data preparation, we can now begin training our model and predict our solution goal. Since we are working with a given dataset with a predetermined solution goal, we are running a form of machine learning algorithm known as supervised learning. With that, here are some of the few beginner models we can run and see how each of them perform based on accuracy.
# 
# If you want to understand the fundamentals of each ML algorithm, you may click on the hyperlinks below which will redirect you to very helpful and easy-to-folllow youtube video tutorials from *StatQuest with Josh Starmer.*
# 
# - [Logistic Regression](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjdkvWQ4qT3AhVEQd4KHUvNDPIQwqsBegQIFhAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DyIYKR4sgzI8&usg=AOvVaw3maZPWy-T2rEc4PFDM40af)
# - [Support Vector Machines](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjM0pq45KT3AhVcQfUHHVYJBysQwqsBegQIAhAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DefR1C6CvhmE&usg=AOvVaw1alnpuy6aMk4ogaK4NtmXy)
# - [K-Means Nearest Neighbors](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj60qDK5KT3AhUbAYgKHcyoDbAQwqsBegQIAhAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DHVXime0nQeI&usg=AOvVaw1h03i8dfC0gXYPU9lFRzJ_)
# - [Decision Tree](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi-t-rS5KT3AhWGAogKHbBfBGQQwqsBegQIAhAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D7VeUPuFGJHk&usg=AOvVaw2KBODG3Oh7AiSz-4h5wnMd)
# - [Random Forest](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiyptHZ5KT3AhWNEYgKHelnCqgQwqsBegQIAhAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DJ4Wdy0Wc_xQ&usg=AOvVaw0moI0sPTwd34hRAKxbDRVN)
# 
# Another important approach which is crucial here is to conduct cross validation. It is a useful technique to address overfitting as it evaluates models through a fixed number of folds k. In my case, I decided to do 10-fold cross-validation. In other words, I do 10 different subsets of sample from training set to arrive at my solutions, then get the mean of all the accuracy scores from these tests.

# In[ ]:


#Common Model Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Defining a list of Machine Learning Algorithms I will be running
MLA = [
    LogisticRegression(max_iter = 2000),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier()   
]

row_index = 0

# Setting up the table to compare the performances of each model
MLA_cols = ['Model', 'Train Accuracy Mean', 'Test Accuracy Mean', 'Fit Time']
MLA_compare = pd.DataFrame(columns = MLA_cols)

for model in MLA:
    MLA_compare.loc[row_index, 'Model'] = model.__class__.__name__
    cv_results = model_selection.cross_validate(model, x1_train, y1_train, cv=10, return_train_score=True)
    MLA_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()
    MLA_compare.loc[row_index, 'Fit Time'] = cv_results['fit_time'].mean()
    
    row_index+=1

MLA_compare.sort_values(by=['Test Accuracy Mean'], ascending=False, inplace=True)
MLA_compare


# In[ ]:


# Defining a list of Machine Learning Algorithms I will be running
MLA = [
    LogisticRegression(max_iter = 2000),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier()   
]

row_index = 0

# Setting up the table to compare the performances of each model
MLA_cols = ['Model', 'Train Accuracy Mean', 'Test Accuracy Mean', 'Fit Time']
MLA_compare = pd.DataFrame(columns = MLA_cols)

for model in MLA:
    MLA_compare.loc[row_index, 'Model'] = model.__class__.__name__
    cv_results = model_selection.cross_validate(model, x1_train, y1_train, cv=10, return_train_score=True)
    MLA_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()
    MLA_compare.loc[row_index, 'Fit Time'] = cv_results['fit_time'].mean()
    
    row_index+=1

MLA_compare.sort_values(by=['Test Accuracy Mean'], ascending=False, inplace=True)
MLA_compare


# # Model Optimization
# Another effective technique to improve the accuracy that I want to try out is optimizing the model by identifying the best parameters for our model. In my case, I will try optimizing my SVM as it performed the best out of the existing ones.
# 
# There are two parameters that I can optimize in SVM, **gamma** and C, and using GridSearchCV() allows us to test the possible combinations of these parameters.

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

param_grid = [{'C': [0.5, 1, 10, 100],
             'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001]}]

optimal = GridSearchCV(SVC(), param_grid, cv = 10, scoring = 'accuracy')
optimal.fit(x1_train, y1_train)
print(optimal.best_params_)


# In[ ]:


# Run the SVM model again with the optimal parameters
svm_optimal = SVC(random_state=42, C=1, gamma='scale')
cvs = cross_val_score(svm_optimal, x1_train, y1_train, cv=10)
cvs.mean()


# # Ensemble Learning
# The accuracy scores we determined earlier only reflect our training set, so these have the likelihood to overfit, which means our model may not necessarily perform well when we test it on our testing data. One approach in addressing this issue is to conduct ensmeble learning. It aims to improve the model by combining multiple algorithms and classifications in order to reduce the biases and address the weaknesses of using standalone models. There are actually several methods of ensemble learning, listed below are some:
# - Majority Voting
# - Bagging
# - Boosting
# - Gradient Boosting
# - Random Forests
# - Stacking
# 
# For now, I will only delve into using the Majority Voting Classifier first as a working example. From the name itself, a voting ensemble involves favoring the class label (i.e. 1=Survived, 0=Died) with the majority or the most votes as the prediction. They are two types of voting:
# - **Hard voting.** summing the votes of class labels from other models and selecting the class label with the most votes as the prediction.
# - **Soft voting.** summing the predicted probabilities of classes from other models and selecting the class label with largest sum probability as the prediction.

# In[ ]:


from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

# Defining the model algorithms for easier access
lr = LogisticRegression(max_iter = 2000)
svm = SVC()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()


# Creating an ensemble for Hard Voting Classifer for Top 3 models
Ensemble_HV = VotingClassifier(estimators =[('Support Vector Machines', svm),
                                      ('K-Means Nearest Neighbors', knn),
                                      ('Logistic Regression', lr)],
                         voting = 'hard')

# Compare previous models with Ensemble_HV
MLA = [
    LogisticRegression(max_iter = 2000),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    Ensemble_HV
]

row_index = 0

# Setting up the table to compare the performances of each model
MLA_cols = ['Model', 'Train Accuracy Mean', 'Test Accuracy Mean', 'Fit Time']
MLA_compare = pd.DataFrame(columns = MLA_cols)

for model in MLA:
    MLA_compare.loc[row_index, 'Model'] = model.__class__.__name__
    cv_results = model_selection.cross_validate(model, x1_train, y1_train, cv=10, return_train_score=True)
    MLA_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()
    MLA_compare.loc[row_index, 'Fit Time'] = cv_results['fit_time'].mean()
    
    row_index+=1

MLA_compare.sort_values(by=['Test Accuracy Mean'], ascending=False, inplace=True)
MLA_compare


# # Prediction Submissions
# We can now submit our predictions based on the models/ensemble models we tuned and see how our scores perform with other fellow Kaggle competitors. Here are the public scores I have gotten.

# In[ ]:


# Submitting predictions with standalone SVM

X_test = test_df

svm.fit(x1_train, y1_train)
Y_pred = svm.predict(X_test)
pred = pd.DataFrame({
    "PassengerId": test_df_copy["PassengerId"],
    "Survived": Y_pred
})
pred.to_csv('submission_svm.csv', index=False)


svm_optimal.fit(x1_train, y1_train)
Y_pred = svm_optimal.predict(X_test)
pred = pd.DataFrame({
    "PassengerId": test_df_copy["PassengerId"],
    "Survived": Y_pred
})
pred.to_csv('submission_svm_optimal.csv', index=False)


rf.fit(x1_train, y1_train)
Y_pred = rf.predict(X_test)
pred = pd.DataFrame({
    "PassengerId": test_df_copy["PassengerId"],
    "Survived": Y_pred
})
pred.to_csv('submission_rf.csv', index=False)


knn.fit(x1_train, y1_train)
Y_pred = knn.predict(X_test)
pred = pd.DataFrame({
    "PassengerId": test_df_copy["PassengerId"],
    "Survived": Y_pred
})
pred.to_csv('submission_knn.csv', index=False)


lr.fit(x1_train, y1_train)
Y_pred = lr.predict(X_test)
pred = pd.DataFrame({
    "PassengerId": test_df_copy["PassengerId"],
    "Survived": Y_pred
})
pred.to_csv('submission_lr.csv', index=False)


Ensemble_HV.fit(x1_train, y1_train)
Y_pred = Ensemble_HV.predict(X_test)
pred = pd.DataFrame({
    "PassengerId": test_df_copy["PassengerId"],
    "Survived": Y_pred
})
pred.to_csv('predictions_ensemble_hv.csv', index=False)


# ## Stay Safe and Happy Kaggling!
# Any form of feedback and advise are welcome. If you have any questions and clarifications regarding my code and work, feel free to ask them in the comments section and I will be happy to answer them.
# 
# ## My Other Works
# If you are interested, you can go to my Kaggle profile [HERE](https://www.kaggle.com/shilongzhuang) and browse through my other works and contributions. Just don't read my bio, I wouldn't if I were you.
# 
# ---
# # References
# Special thanks and credits to these awesome and comprehensively informative resources (notebooks) and guides created by talented professionals in the field. I highly recommend you go check them out especially for beginners like me.
# - [A Data Science Framework: To Achieve 99% Accuracy | Kaggle](https://www.kaggle.com/code/shilongzhuang/a-data-science-framework-to-achieve-99-accuracy/edit)
# - [Titanic Data Science Solutions | Kaggle](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions)
