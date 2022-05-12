#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/stores-area-and-sales-data/Stores.csv')
df


# In[ ]:


attributs = ['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']

for attribut1 in attributs:
    for attribut2 in attributs:
        if attribut1 != attribut2:
            sns.jointplot(x=attribut1, y=attribut2, data=df,
                  kind="reg", truncate=False,
                  color="m", height=7)

