#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # read data

# In[ ]:


data = pd.read_csv('../input/diamonds/diamonds.csv')
data.sample(4)


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# In[ ]:


x = data.select_dtypes('object').columns
for i in x:
    print(i,': ',data[i].unique())


# In[ ]:


del data['Unnamed: 0']


# # visualization

# In[ ]:


from wordcloud import WordCloud
plt.figure(figsize =(15,30))
text = ' '.join(list(data['cut']))
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:


plt.figure(figsize =(15,30))
text = ' '.join(list(data['color']))
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:


from wordcloud import WordCloud
plt.figure(figsize =(15,30))
text = ' '.join(list(data['clarity']))
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:


corr = data.corr()
fig = px.imshow(corr, text_auto=True)
fig.show()


# In[ ]:


df = data[['depth','price','cut']]
 
sns.regplot(data=df, x="depth", y="price", fit_reg=False, marker="+", color="skyblue")

plt.show()


# In[ ]:


plt.figure(figsize=(14, 10))

sns.barplot(
    y="price", 
    x="clarity", 
    data=data, 
    estimator=sum, 
    ci=None, 
    color='#69b3a2');


# In[ ]:


plt.figure(figsize=(14, 10))

sns.barplot(
    y="price", 
    x="cut", 
    data=data, 
    estimator=sum, 
    ci=None, 
    color='#69b3a2');


# In[ ]:


plt.figure(figsize=(14, 10))

sns.barplot(
    y="price", 
    x="color", 
    data=data, 
    estimator=sum, 
    ci=None, 
    color='#69b3a2');


# In[ ]:


sns.boxplot( x=data["cut"], y=data["price"] )


# In[ ]:


sns.boxplot( x=data["color"], y=data["price"] )


# In[ ]:


sns.boxplot( x=data["clarity"], y=data["price"] )


# # Missing Data

# In[ ]:


data_num = data.select_dtypes(['float64','int64'])
data_obj =  data.select_dtypes('object')


# ## numerical data fill na

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
d=imputer.fit_transform(data_num)
data_num=pd.DataFrame(d,columns=data.select_dtypes(['float64','int64']).columns)


# ## Categorical data fill na

# In[ ]:


imp = SimpleImputer(strategy="most_frequent")
d2 = imp.fit_transform(data_obj)
data_obj = pd.DataFrame(d2,columns=data_obj.columns)
data_obj


# # Categorical data

# In[ ]:


data_obj_ord = data_obj[['cut','clarity']]
data_obj_no = data_obj[['color']]


# - ## Ordinal Data

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories = [['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], 
                                              ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']])
diamond_cat_encoded = ordinal_encoder.fit_transform(data_obj_ord)
data_obj_ord = pd.DataFrame(diamond_cat_encoded,columns=['cut','clarity'])


# - ## Nominal Data

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()

data_obj_no = cat_encoder.fit_transform(data_obj_no)
data_obj_no
data_obj_no = pd.DataFrame(data_obj_no.toarray(),dtype=np.float64,columns=['E', 'I', 'J', 'H', 'F' ,'G', 'D'])


# In[ ]:


data_processing = pd.concat([data_num,data_obj_ord,data_obj_no],axis=1)
data_processing.sample()


# # Outilers

# In[ ]:


from scipy import stats
import numpy as np
for i in [2,3,4,5,6]:
    print(f'===========================<<<<<<< {i} >>>>>>>>==============================')
    x = data_processing[(np.abs(stats.zscore(data_processing))<i).all(axis=1)]
    print('remove col : ',data_processing.shape[0]-x.shape[0])


# In[ ]:


x = data_processing.copy()
data_processing1 = x[(np.abs(stats.zscore(data_processing))<3).all(axis=1)]


# # Feature extraction

# In[ ]:


data_processing1['size'] = data_processing1['x']*data_processing1['y']*data_processing1['z'] 
data_processing1_copy = data_processing1.copy()
data_processing1.drop(['x','y','z'],inplace=True,axis=1)
data_processing1.sample(2)


# # Feature Scalling

# In[ ]:


from sklearn.preprocessing import StandardScaler

num_scaler=StandardScaler()
data_processing1_Scaler = pd.DataFrame(num_scaler.fit_transform(data_processing1[['carat','cut','clarity','depth','table','price','size']]),columns=['carat','cut','clarity','depth','table','price','size'],index=data_processing1.index)
color = data_processing1[[ 'E','I','J','H','F','G','D']]
data_processing1_after_scaler = pd.concat([data_processing1_Scaler,color],axis=1)


# In[ ]:


data_processing1_after_scaler.describe()


# In[ ]:


data_processing1_after_scaler.info()


# In[ ]:


sns.distplot(data_processing1_after_scaler['price'])


# In[ ]:


corr = data_processing1_after_scaler.corr()
corr['price'].sort_values()


# In[ ]:


fig = px.imshow(corr, text_auto=True)
fig.show()


# In[ ]:


data_processing1_after_scaler.to_csv(r'my_data.csv', index=False)


# # Welcome to my notebook
# ## This is the first part of a machine learning project for beginners, which is data preprocessing 
# 
# > I used a imputer although I didn't need it, but I applied it to the data so that the beginners know that there is a imputer in the sklearn library that can fill in the missing data 

# # The End
# > If you any questions or advice me please write in the comment

# # Upvote
