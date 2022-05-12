#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)
# 
# 
# 
# 
# Approximately 3.5 billion searches are performed on Google daily, which means that approximately 40,000 searches are performed every second on Google. So Google search is a great use case for analyzing data based on search queries. With that in mind, we will perform Google search analysis.

# ### Google Search Analysis
# 
# Google doesn’t give much access to the data about daily search queries, but another application of google known as **Google Trends** can be used for Google search analysis. 
# 
# * **Google Trends** provides an API that can be used to analyze the daily searches on Google. This API is known as **pytrends**. We can easily install it in our systems by using the **pip command**; `pip install pytrends`.

# In[ ]:


pip install pytrends


# In[ ]:


import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
trends = TrendReq()


# Here we will be analyzing the Google search trends on the queries based on **Machine Learning**. So let’s create a DataFrame of the top 10 countries which search for **Machine Learning** on `Google`:

# In[ ]:


trends.build_payload(kw_list=["Machine Learning"])
data = trends.interest_by_region()
data = data.sort_values(by="Machine Learning", ascending=False)


# In[ ]:


data = data.head(10)
data


# According to the above results, the search queries based on **Machine learning** are mostly done in China. We can also visualize this data using a bar chart:

# In[ ]:


plt.style.use('fivethirtyeight')
# plt.figure(figsize = (20,15))
data.reset_index().plot(x="geoName", y="Machine Learning", 
                         kind="bar" , figsize = (20,15))
plt.show()


# As we all know that Machine Learning has been the focus of so many companies and students for the last 3-4 years, so let’s have a look at the trend of searches to see how the total search queries based on **Machine Learning** increased or decreased on Google:

# In[ ]:


data = TrendReq(hl='en-US', tz=360)
data.build_payload(kw_list=['Machine Learning'])
data = data.interest_over_time()


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 15))
data['Machine Learning'].plot()
plt.style.use('fivethirtyeight')
plt.title('Total Google Searches for Machine Learning', fontweight='bold')
plt.xlabel('Year' ,  fontweight='bold')
plt.ylabel('Total Count' , fontweight='bold')
plt.show()


# We can see that searches based on **machine learning** on `Google` started to increase in **2017** and the highest searches were done in **2022** till today. This is how we can analyze Google searches based on any keyword. A business can perform Google search analysis to understand what people are looking for on Google at any given time.

# Let us find a list of related queries for **machine learning** and return the `top` queries.

# In[ ]:


data = TrendReq(hl='en-US', tz=360)
data.build_payload(kw_list=['Machine Learning'])
data = data.related_queries()


# In[ ]:


data['Machine Learning']['top'].head(10)


# In[ ]:


data['Machine Learning']['top'].head(10).plot(x="query", y="value", 
                         kind="bar" , figsize=(20, 15))
plt.style.use('fivethirtyeight')
plt.title('Total Google Searches Related To Machine Learning', fontweight='bold')
plt.xlabel('Qurey' , fontweight='bold')
plt.ylabel('Total Count' , fontweight='bold')
plt.show()


# ### If you find my notebook Interested, please Upvote it.

# In[ ]:




