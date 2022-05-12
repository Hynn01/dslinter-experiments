#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data


# In[ ]:


data.info()


# The dataset contains 215 and 15 columns.

# In[ ]:


## check columnwise missing values
data.isnull().sum()


# Only Salary varaible has 67 missing values. As salary is post placement attribute and our aim is to find whether the student got placed or not.Therfore its better to drop it. Also column sl_no has no use in prediction.So drop both variables

# Drop salary ans sl_no

# In[ ]:


data.drop(['salary',"sl_no"], axis = 1,inplace=True)


# ## Data Exploration

# #### For Exploring the data first all categorical features are taken and analysed (univariate and bivariate analysis) and then continous features.

# In[ ]:


data.describe(include='object')


# #### Above table shows the description of categorical varaibles i,e count,unique,top and frequency of  levels.Lets start with univariate analysis of target variable(status).

# In[ ]:


#Import matplotlib and seaborn for data visualisation
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


fig, ax1 = plt.subplots(figsize=(12,5))
graph=sns.countplot(x='status',data=data,order=data.status.value_counts().index)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,
        data['status'].value_counts()[i],ha="center")
    i += 1


# #### We can see out of total (215) 148 students are placed and 67 not placed.

# ### Workex(Work Experience)

# In[ ]:


sns.countplot(data["workex"])


# #### No of students with workexperience are less.

# ### Bivariate analysis of categorical variables with respect to target variable

# In[ ]:


# Change target varaible to numeric(0 and1)
data["status"]=data.apply(lambda x: 1 if x.status=="Placed" else 0,axis=1)


# In[ ]:


# variable 'workex'
data.groupby("workex")["status"].mean()


# In[ ]:


sns.barplot(x='workex',y='status',data=data)


# From table and barplot it is clear that mean value of number of placed student with workex is more(.864) than no workex(.594).

# #### check variable 'gender'

# In[ ]:


data.groupby("gender")["status"].mean()


# #### Percentage of placed students in male category are slighly more than female.

# * #### ssc_b (board of education)

# In[ ]:


data.groupby("ssc_b")["status"].mean()


# #### ssc_b (board of education) either central or other does not make any high impact on placement

# In[ ]:


#### similarly check hsc_b(board of eduction in hsc)
data.groupby("hsc_b")["status"].mean()


# In[ ]:


#### hsc_s (hsc specialization)


# In[ ]:


data.groupby("hsc_s")["status"].mean()


# In[ ]:


sns.barplot(x='hsc_s',y='status',data=data)


# as we can see placement number commerec and science students are more than arts.infact commerce and science reponding same way. Therefore consider commerce and science as single level "com/sci".

# In[ ]:


data['hsc_s']=data['hsc_s'].apply(lambda x: "com/sci" if (x=="Commerce" or x=="Science") else "Arts")


# Now degree_t(Under Graduation(Degree type)- Field of degree education)

# In[ ]:


data.groupby("degree_t")["status"].mean()


# Again comm&mgmt ans sci&Tech responsing same ways with status ,so covert it too single

# In[ ]:


data['degree_t']=data['degree_t'].apply(lambda x: "Com/sci" if (x=="Comm&Mgmt" or x=="Sci&Tech") else "Others")


# #### "specialisation"(Post Graduation(MBA)- Specialization)

# In[ ]:


data.groupby("specialisation")["status"].mean()


# Mkt&Fin have high placement percentage than MktH

# ### Now numeric Variables

# In[ ]:


data.describe()


# In[ ]:


#### correlation with target
data.corr()


# from correlation table ssc_p is higly correlated with placement (.607)

# Heatmap 

# In[ ]:


plt.figure(figsize=(8,5))
sns.heatmap(data.corr(),
            vmin=-1,
            cmap='coolwarm',
            annot=True);


# Pairplot

# In[ ]:


sns.pairplot(data,hue='status')


# In[ ]:


##lets check ssc_p
sns.distplot(data["ssc_p"])


# ### Bivariate Analysis

# In[ ]:


sns.boxplot("status","ssc_p",data=data)


# In[ ]:


#### Boxplot clearly shows that higher mean ssc_p are in Placed status(1)


# In[ ]:


sns.factorplot(x='gender',y='ssc_p' , col='workex', data=data , hue='status' , kind = 'box', palette=['r','g'])


# In[ ]:


sns.factorplot(x='gender',y='ssc_p' , col='workex', data=data , hue='status' , kind = 'violin', palette=['r','g'])


# #### most of the females having ssc_p between 70-80 are placed(high density)

# 

# Now code the categorical varaibles to dummies for model building

# In[ ]:


data = pd.get_dummies( data,drop_first=True)


# ## Model Building

# #### For model building first we will make single decision tree classifier and then Random Forest.

# ### Train-Test split

# In[ ]:


X = data.drop('status', axis=1)
y = data['status']


# In[ ]:


from sklearn.model_selection import train_test_split
# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)


# ## Decision Tree classifier

# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train,y_train)


# #### Prediction and Evaluation

# In[ ]:


predictions = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# #### Our decision tree predicts 14 0's as correct zeros but 8 as wrong zeros . similarly 47  1's as correct 1's and 2 as wrong 1's which are actually zeros. Accuracy is 0.86.

# #### Now Area under curve

# In[ ]:


from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, predictions)
auc


# ![](http://)#### Area under curve is .797 which is okk.

# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))


# In[ ]:


print(classification_report(y_test,rfc_pred))


# #### Area under curve

# In[ ]:


auc = roc_auc_score(y_test, rfc_pred)
auc


# #### If we compare single decision tree and random forest algorithem, the accuracy of random forest is 0.90 and that of single decision tree is 0.86. Also area under the curve is high in random forest.

# 
