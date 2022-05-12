#!/usr/bin/env python
# coding: utf-8

# # hola🙋‍♀️ este es un repaso del datacamp🥾👩‍💻

# # types🐢

# In[ ]:


a = type(11)
b = type(False)
c = type("ño")
d = type(11.1)

[a, b, c, d]


# > ## '🐤' + '🐤'?

# In[ ]:


a = 11 + 11
b = "pio" + "pio"

[a, b]


# ## 111 + '🐔'?

# In[ ]:


111 + "kikiriki" # ERROR


# # lists [🐘,🐬,🦘]

# In[ ]:


[False, 99, "ño"]


# In[ ]:


type([77, "ño"])


# # subsetting lists🐁

# In[ ]:


y = [11, 22, 33, 44, 55, 66]

y[0]


# In[ ]:


y[3]


# In[ ]:


y[-1]


# # list slicing🦢

# *inclusive -> [start:end] <- exclusive*

# In[ ]:


y = [1, 2, 3, 4, 99]

y[0:1]


# In[ ]:


y = [1, 2, 3, 4, 99]

y[0:2]


# In[ ]:


y = [1, 2, 3, 4, 99]

y[1:-1]


# In[ ]:


y = [1, 2, 3, 4, 99]

y[-3:-1]


# In[ ]:


y = [1, 2, 3, 4, 99]

y[:]


# In[ ]:


y = [1, 2, 3, 4, 99]

y[:3]


# In[ ]:


y = [1, 2, 3, 4, 99]

y[3:]

