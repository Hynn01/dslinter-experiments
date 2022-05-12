#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


dataset= ['Hey welcome to the study mart',
          'Hey I am Sharif',
          'I love Data Science',
          'Please send me friend me me request',
         ]


# In[ ]:


cv = CountVectorizer()
x=cv.fit_transform(dataset)


# In[ ]:


cv.get_feature_names()


# In[ ]:


x.toarray()


# In[ ]:





# In[ ]:





# In[ ]:




