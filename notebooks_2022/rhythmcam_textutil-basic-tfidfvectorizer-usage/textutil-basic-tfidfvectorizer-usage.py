#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


corpus = [
    'This is a book',
     'This book is language book.',
     'The book is Korean book',
     'That book is difficult',
 ]


# In[ ]:


len(corpus)


# In[ ]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)


# In[ ]:


vectorizer.get_feature_names_out()


# In[ ]:


X


# # index 

# In[ ]:


X.indices


# In[ ]:


len(X.indices)


# In[ ]:


X.has_sorted_indices


# ##### index pointer

# In[ ]:


X.indptr


# # data

# In[ ]:


X.data


# In[ ]:


len(X.data)


# In[ ]:


X.nnz


# In[ ]:


X.ndim


# In[ ]:


X.shape


# In[ ]:


X.maxprint


# In[ ]:


X.format


# In[ ]:


X.has_canonical_format

