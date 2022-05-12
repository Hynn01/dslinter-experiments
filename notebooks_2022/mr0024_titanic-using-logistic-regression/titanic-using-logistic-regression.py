#!/usr/bin/env python
# coding: utf-8

# # Loading libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


# In[ ]:


warnings.filterwarnings('ignore')
sns.set_theme()


# # Importing the dataset

# In[ ]:


orig_train = pd.read_csv("/kaggle/input/titanic/train.csv", sep=",")
orig_test = pd.read_csv("/kaggle/input/titanic/test.csv", sep=",")


# In[ ]:


train = orig_train.copy()
test = orig_test.copy()


# # A quick glance at the data

# In[ ]:


pd.concat([train, test], axis=0).drop(columns='Survived').info()


# In[ ]:


sns.histplot(x=train.Age, data=train, element="step", color="orange")


# In[ ]:


train.Embarked.value_counts()


# In[ ]:


sns.histplot(x=train.Fare, data=train, element="step", color="orange")


# In[ ]:


train.Fare.value_counts()


# In[ ]:


train.Pclass.value_counts()


# In[ ]:


train.Name.value_counts()


# In[ ]:


train.Name.str.extract(r'([A-Za-z]*\.)').value_counts()


# In[ ]:


train.Ticket.value_counts()


# In[ ]:


(train.SibSp + train.Parch).value_counts()


# # Reorganizing features
# 
# At first I will fill every null value and then reorganize Name, Cabin, SibSp and Parch in categorical variables.

# In[ ]:


def fix_df (df):
    df['Embarked'] = df.Embarked.fillna('S')
    df["Age"] = df.Age.fillna(df.Age.median())
    df["Fare"] = df.Fare.fillna(df.Fare.median())
    
    df['Cabin'] = df.Cabin.apply(lambda x : 'yes' if isinstance(x, str) else 'no')
    df['title_Name'] = df.Name.str.extract(r'([A-Za-z]*\.)')
    title_head = df.title_Name.value_counts() > 7
    df['title_Name'] = df.title_Name.apply(lambda x: x if title_head[x] == True  else 'Other')
    df['n_family'] = df.SibSp + df.Parch
    
    df.drop(columns=['Name', 'Parch', 'SibSp', 'Ticket'], inplace=True)
    
    return df


# In[ ]:


fix_df(train)


# In[ ]:


fix_df(test)


# # Plotting data

# In[ ]:


def select_type (df):
        cat = []
        num = []
        
        for i in df.columns:
            if (i == 'Survived') or (i == 'PassengerId') :
                continue
            elif df.dtypes[i] == 'float64':
                num.append(i)
            else:
                cat.append(i)
               
        return cat, num


# In[ ]:


cat, num = select_type(train)


# In[ ]:


cat


# In[ ]:


lcat = len(cat)
fig, axes = plt.subplots(round((lcat)/2), 2, figsize=(10, round(lcat/2)*5))

for ax, i in zip(axes.flat, cat):
    sns.countplot(ax=ax, x=train[i], hue=train.Survived, data=train)
    ax.set_title(i)  


# In[ ]:


train[['title_Name', 'Age']].query("title_Name == 'Master.'")


# In[ ]:


train.query("n_family > 4")


# Many passengers with large family have Pclass equal 3.

# In[ ]:


train.query("(n_family > 0) and (n_family < 5)").sample(30)


# In[ ]:


train.groupby(by=['n_family', 'Sex', 'Pclass', 'Survived'])['PassengerId'].count().head(50)


# In[ ]:


train[['title_Name', 'Age']].query("title_Name == 'Mrs.'")


# At first glance, plots of Sex and title_Name highlight a major percentage of survived among females. Pclass equals to 1, 2 and C-Embarked indicates a different trend than other variables: I will try to check it in the next plots. 

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.countplot(ax=axes[0,0], x=train.query("(Pclass == 1) | (Pclass == 2)").Sex, data=train)
sns.countplot(ax=axes[0,1], x=train.query("(Pclass == 1) | (Pclass == 2)").Embarked, data=train)
axes[0,1].set_title("Pclass 1 or 2")  
sns.histplot(ax=axes[0,2], x=train.query("(Pclass == 1) | (Pclass == 2)").Age, data=train, element="step", color="orange")
sns.countplot(ax=axes[1,0], x=train.query("Embarked == 'C'").Sex, data=train)
sns.countplot(ax=axes[1,1], x=train.query("Embarked == 'C'").Pclass, data=train)
axes[1,1].set_title("Embarked C")  
sns.histplot(ax=axes[1,2], x=train.query("Embarked == 'C'").Age, data=train, element="step", color="orange")


# Plotting Pclass only with values 1 and 2 doesn't show particular difference among other data. On the other hand, the plot of C-Embarked shows a strong presence of pclass values equal 1 that we found to have a connection with a higher probability of being survived. 

# In[ ]:


num


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(10, 6))
sns.boxplot(ax=axes[0], y=train.Age, x=train.Survived, data=train)
sns.violinplot(ax=axes[1], y=train.Age, x=train.Survived, data=train)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(ax=axes[0], x=train.Age, hue=train.Survived, data=train, multiple="dodge", bins=10)
sns.histplot(ax=axes[1], x=train.Age, data=train, bins=10, element="step", color="orange")


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(ax=axes[0], x=train.Age, hue=train.title_Name, data=train, multiple="dodge", bins=10)
sns.histplot(ax=axes[1], x=train.Age, hue=train.Sex, data=train, multiple="dodge", bins=10)


# The connection between survived, age and sex can be summarized by the title_Name.

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(10,6))
sns.boxplot(ax=axes[0], y=train.Fare, x=train.Survived, data=train)
sns.violinplot(ax=axes[1], y=train.Fare, x=train.Survived, data=train)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(ax=axes[0], x=train.Fare, hue=train.Survived, data=train, multiple="dodge", bins=10)
sns.histplot(ax=axes[1], x=train.Fare, data=train, bins=10, element="step", color="orange")


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,6))
sns.boxplot(ax=axes[0], y=train.Fare, x=train.Pclass, data=train)
sns.violinplot(ax=axes[1], y=train.Fare, x=train.Pclass, data=train)


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.scatterplot(ax=axes[0], x=train.query("Embarked == 'S' and Fare != 0").Age, y=train.Fare, hue=train.Survived, data=train, alpha=0.6)
sns.scatterplot(ax=axes[1], x=train.query("Embarked == 'C' and Fare != 0").Age, y=train.Fare, hue=train.Survived, data=train, alpha=0.6)
sns.scatterplot(ax=axes[2], x=train.query("Embarked == 'Q' and Fare != 0").Age, y=train.Fare, hue=train.Survived, data=train, alpha=0.6)


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.scatterplot(ax=axes[0], x=train.query("Pclass == 1 and Fare != 0").Age, y=train.Fare, hue=train.Survived, data=train, alpha=0.6)
sns.scatterplot(ax=axes[1], x=train.query("Pclass == 2 and Fare != 0").Age, y=train.Fare, hue=train.Survived, data=train, alpha=0.6)
sns.scatterplot(ax=axes[2], x=train.query("Pclass == 3 and Fare != 0").Age, y=train.Fare, hue=train.Survived, data=train, alpha=0.6)


# There seem to be no particular connections between Age and Fare.  

# # Transforming categorical variables into boolean

# I will drop Cabin since the ratio between survived and not is so different when we compare who had a filled data and who did not, and considering the initial amount of null values. I will also transform n_family in alone or not. 

# In[ ]:


train['alone'] = train.n_family.apply(lambda x: x if x == 0 else 1)
train.drop(columns='n_family', inplace=True)
test['alone'] = test.n_family.apply(lambda x: x if x == 0 else 1)
test.drop(columns='n_family', inplace=True)


# In[ ]:


cat, num = select_type(train)


# In[ ]:


def dum_cat(df):
    for i in cat:
        if i == 'Cabin':
            continue
        else:
            cls_dum = pd.get_dummies(df[i], prefix=i[0:3], drop_first=True)
            for x in cls_dum.columns:
                df[x] = cls_dum[x]
    
    return df   
    


# In[ ]:


dum_cat(train)
dum_cat(test)


# In[ ]:


train.drop(columns=cat, inplace=True)
test.drop(columns=cat, inplace=True)


# # Feature correlation

# In[ ]:


fig = plt.figure(figsize=(15,15))
corrMatrix = train.drop(columns='PassengerId').corr()
sns.heatmap(corrMatrix, annot=True, fmt='.2f', cmap="YlGnBu")


# # Logistic regression assumptions  

# ##### 1) Boolean outcome

# In[ ]:


sns.countplot(x=train['Survived'], data=train)


# ##### 2) Box-Tidwell Test
# 
# The relationship between independent variables and their logodds must be linear.

# In[ ]:


X_bt = train.query("(Fare != 0) and (Age != 0)")
y = X_bt['Survived']
X = X_bt.drop(columns=['Survived','PassengerId'])


# In[ ]:


X_ln = X[['Fare', 'Age']]
X_ln['Fare_ln'] = X_ln.Fare.apply(lambda x: x * np.log(x))
X_ln['Age_ln'] = X_ln.Age.apply(lambda x: x * np.log(x))
X_ln = sm.add_constant(X_ln, prepend=False)


# In[ ]:


X_ln.info()


# In[ ]:


ln_result = GLM(y, X_ln, family=families.Binomial()).fit()
round(ln_result.pvalues, 3)


# The p value for Fare is < 0,01. This means there is no linearity.

# In[ ]:


y = train['Survived']
X = train.drop(columns=['Survived', 'PassengerId'])
X = sm.add_constant(X, prepend=False)
result = GLM(y, X, family=families.Binomial()).fit()
pred = result.predict(X)
l_odds = np.log(pred / (1 - pred))


# In[ ]:


sns.scatterplot(x=X['Fare'].values, y=l_odds)


# Scatterplot seems to confirm the previous result.

# ##### 3) Multicollinearity
# 
# Independent variables should not be highly correlated with each other. I will use Variance inflation factor as a measure of multicollinearity. 

# In[ ]:


def calc_vif(df):
    vif = pd.Series([variance_inflation_factor(df.values, i) for i in range(df.shape[1])], index=df.columns)
    return(vif)


# In[ ]:


calc_vif(X)


# In[ ]:


result = GLM(y, X, family=families.Binomial()).fit()
round(result.pvalues, 3)


# In[ ]:


result = GLM(y, X.drop(columns='Sex_male'), family=families.Binomial()).fit()
round(result.pvalues, 3)


# In[ ]:


calc_vif(X.drop(columns=['Sex_male', 'Fare']))


# In[ ]:


result = GLM(y, X.drop(columns=['Sex_male', 'Fare']), family=families.Binomial()).fit()
round(result.pvalues, 3)


# In[ ]:


result = GLM(y, X.drop(columns=['Sex_male', 'Fare']), family=families.Binomial()).fit()
pred = result.predict(X.drop(columns=['Sex_male', 'Fare']))
l_odds = np.log(pred / (1 - pred))
sns.scatterplot(x=X['Age'].values, y=l_odds)


# In[ ]:


result = GLM(y, X['Age'], family=families.Binomial()).fit()
round(result.pvalues, 3)


# In[ ]:


result = GLM(y, X[['Age', 'tit_Miss.', 'tit_Mr.', 'tit_Mrs.', 'tit_Other', 'alo_1']], family=families.Binomial()).fit()
round(result.pvalues, 3)


# In[ ]:


result = GLM(y, X.drop(columns=['Sex_male', 'Fare', 'Age']), family=families.Binomial()).fit()
result.summary()


# In[ ]:


calc_vif(X.drop(columns=['Sex_male', 'Fare', 'Age']))


# Being tit_Mr. a categorical variable with more than three categories, and, as the proportion of cases in tit_Master, the reference category, is small, thus the VIF value results accettable.

# Age had some missing values and could be explained by the title name and Fare could be explained by Pclass, so I will drop highly correlated and redundant features.

# In[ ]:


train.drop(columns=['Fare', 'Sex_male', 'Age'], inplace=True)
test.drop(columns=['Fare', 'Sex_male', 'Age'], inplace=True)


# ##### 4) Outliers

# In[ ]:


y = train['Survived']
X = train.drop(columns=['Survived', 'PassengerId'])
X = sm.add_constant(X, prepend=False)
result = GLM(y, X, family=families.Binomial()).fit()
influence = result.get_influence()
inf = influence.summary_frame().sort_values(by='standard_resid', ascending=False).query("abs(standard_resid)>3")
inf.head(10)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(x=influence.summary_frame().index, y=influence.summary_frame().standard_resid)
plt.axhline(y=3, color='red')
plt.axhline(y=-3, color='red')
plt.axhline(y=0, color='blue')
fig.tight_layout(pad=2)
for i, txt in enumerate(inf.index.values):
    ax.annotate(txt, xy=(inf.index.values[i], inf.standard_resid.iloc[i]))


# In[ ]:


orig_train[orig_train.PassengerId.isin(train.iloc[inf.index.values,:].PassengerId)].sort_values(by='Age')


# In[ ]:


train.iloc[inf.index.values,:].shape[0]


# In[ ]:


train[(train['tit_Mr.']==1) & (train['Pcl_3']==1) & (train['Emb_S']==1)]


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
influence = result.get_influence(observed=False)
influence.plot_influence(size=60, ax=ax, plot_alpha=0.1, criterion="cooks")
fig.tight_layout(pad=1.0)


# In[ ]:


result.summary()


# In[ ]:


train1 = train.drop(index=inf.index.values)
y = train1['Survived']
X = train1.drop(columns=['Survived', 'PassengerId'])
X = sm.add_constant(X, prepend=False)
result = GLM(y, X, family=families.Binomial()).fit()
result.summary()


# I will drop all indexes having standard_resid exceeding 3. 

# In[ ]:


train.drop(index=inf.index.values, inplace=True)


# ##### 5) Observation indipendence 

# In[ ]:


y = train['Survived']
X = train.drop(columns=['Survived', 'PassengerId'])
X = sm.add_constant(X, prepend=False)
result = GLM(y, X, family=families.Binomial()).fit()


# In[ ]:


fig = plt.figure(figsize=(20,4))
sns.lineplot(x=X.index, y=result.resid_deviance)
plt.axhline(y=0, color='red');


# The absence of non-random pattern show that the independence of errors is satisfied. 

# ##### 6) Sample size

# In[ ]:


cat, num = select_type(train)


# In[ ]:


for i in cat:
    print(train[i].value_counts())


# # Building logistic model 

# In[ ]:


X = train.drop(columns=['PassengerId', 'Survived'])
y = train['Survived']


# I will use Gridsearch to tune hyperparameters.

# In[ ]:


pipeline = Pipeline([('lr', LogisticRegression(random_state=0))])

pars = {#'lr__C': [1e-15, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10, 15],
        'lr__C': np.arange(0.2,1.9, 0.1),
        'lr__solver' : ['liblinear'],
        'lr__penalty' : ['l1']}

lr_grid = GridSearchCV(
          pipeline, 
          pars, 
          scoring='f1_macro',
          cv=12).fit(X, y)
lr_grid.best_params_


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


pred_proba_df = pd.DataFrame(lr_grid.predict_proba(X_test))
#threshold_list = [0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65]
threshold_list = np.arange(0.51, 0.64, 0.01)
for i in threshold_list:
    print (f'\n For i = {i}')
    Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
    print('F_score: ', f1_score(y_test, Y_test_pred.iloc[:,1], average='macro'))
    print(confusion_matrix(y_test, Y_test_pred.iloc[:,1]))


# In[ ]:


y_prob = lr_grid.predict_proba(X_test)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(false_positive_rate, true_positive_rate, color='red',label = 'AUC1 = %0.3f' % auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
for i, j, txt in zip(false_positive_rate[::2], true_positive_rate[::2], thresholds[::2]):
    plt.annotate(np.round(txt,2), (i, j-0.04))


# In[ ]:


prob_y = np.array(pd.DataFrame(lr_grid.predict_proba(X_test)[:,1]).applymap(lambda x: 1 if x>0.56 else 0))


# In[ ]:


print(classification_report(y_test, prob_y))


# In[ ]:


fig = plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test, prob_y), annot=True, fmt='.2f', cmap="YlGnBu")


# # Final result

# In[ ]:


prob_y = np.array(pd.DataFrame(lr_grid.predict_proba(test.drop(columns='PassengerId'))[:,1]).applymap(lambda x: 1 if x>0.59 else 0))


# In[ ]:


pred = prob_y.squeeze()
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




