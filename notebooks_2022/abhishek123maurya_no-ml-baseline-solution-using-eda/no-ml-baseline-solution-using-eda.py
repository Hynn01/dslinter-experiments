#!/usr/bin/env python
# coding: utf-8

# # <p style="text-align:center;font-size:150%;font-family:Roboto;background-color:#a04070;border-radius:50px;font-weight:bold;margin-bottom:0">Autism Prediction</p>
# 
# <p style="font-family:Roboto;font-size:140%;color:#a04070;">In this Notebook, I had shown how we can achieve a baseline score of 78% with just exploratory data analysis. This can be used as our baseline score and a ML model will be considered best if it can build upon this.<p> 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/train.csv')
print(df.shape)
df.head()


# In[ ]:


sb.countplot(data=df, x='Class/ASD')
plt.show()


# <p style="font-family:Roboto;font-size:140%;color:#a04070;">We have highly skewed data with 80% of the negative examples.<p> 

# In[ ]:


def add_feature(data):
    data['sum_score'] = 0
    for col in df.loc[:,'A1_Score':'A10_Score'].columns:
        data['sum_score'] += data[col]
    
    return data

df = add_feature(df)
df.head(2)


# In[ ]:


plt.figure(figsize=(10,5))
sb.countplot(data=df, x='sum_score', hue='Class/ASD')
plt.show()


# <p style="font-family:Roboto;font-size:140%;color:#a04070;">This is one of the graph which shows a clear relation between sum_score and the target.</p>
# 
# <ul style="font-family:Roboto;font-size:140%;color:#a04070;">
# <li>If the value of sum_score is in between 1 to 5 then most probably that person does not have autism.</li>
# <li>And if the sum_score of the patient is high then the chances of that person having autism are quite high.</li></ul>
# 
# 
# <p style="font-family:Roboto;font-size:140%;color:#a04070;">This second point is not visible that clearly in the above graph because the given data is highly skewed and contains 80% of the examples for the negative class.</p>

# In[ ]:


df_test = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/test.csv')
print(df.shape)
df.head()


# In[ ]:


df_test = add_feature(df_test)


# In[ ]:


plt.figure(figsize=(10,10))
sb.countplot(data=df_test, x='sum_score')
plt.show()


# <p style="font-family:Roboto;font-size:140%;color:#a04070;">So, we will apply the obervation derived from the train data to the test data and predict the target.</p>

# In[ ]:


df_test['target'] = 0
ind = df_test[df_test['sum_score'] > 6].index
df_test.loc[ind, 'target'] = 1


# In[ ]:


ss = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/sample_submission.csv')
ss['Class/ASD'] = df_test['target'].values
ss['Class/ASD'].value_counts().plot.bar()


# <p style="font-family:Roboto;font-size:140%;color:#a04070;">If a model just predict one then it is 50% accurate means we have equal number of test points with target 0 and 1. That is why above graph gives a feel of the accuracy of this prediction.</p>

# In[ ]:


ss.to_csv("Submission.csv",index=False)
ss.head()


# <p style="font-family:Roboto;font-size:140%;color:#a04070;">I am working on a solution based on ML as soon as I will complete that I will post it.<br>
# Thank you for Reading.</p>
