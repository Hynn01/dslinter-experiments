#!/usr/bin/env python
# coding: utf-8

# # Mergers & Acquisitions Data
# 

# ## Importing Modules

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Setting Theme
# ### - Colorblind - Friendly

# In[ ]:


sns.color_palette("colorblind")


# ## Exploratory Data Analysis (and data cleaning)

# ### Reading data file

# In[ ]:


data_df = pd.read_csv("../input/company-acquisitions-7-top-companies/acquisitions_update_2021.csv")


# ### Number of (Rows, Columns) in the data

# In[ ]:


data_df.shape


# ### What are the column names?

# In[ ]:


data_df.columns


# ### Looking at the first 5 rows

# In[ ]:


data_df.head(5)


# ### Replacing "-" with NaN values

# In[ ]:


data_cleaned = data_df.replace("-", np.nan)


# ### Recheck data

# In[ ]:


data_cleaned.head()


# ### Which columns contain missing/ NaN values?

# In[ ]:


data_cleaned.isna().any()


# In[ ]:


data_cleaned.Country.count()


# ### Type of Data for each column
# 

# In[ ]:


data_df.dtypes


# ### Convert Year and Price to "int" from "objects"

# In[ ]:


data_cleaned["Acquisition Year"] = pd.to_numeric(
    data_cleaned["Acquisition Year"])
data_cleaned['Acquisition Price'] = pd.to_numeric(
    data_cleaned['Acquisition Price'], errors='coerce')


# ### Finally; some brief overview of the data

# In[ ]:


data_cleaned.info()


# ### What's the range of numerical values?

# In[ ]:


data_cleaned.describe()


# ### "month" column data integrity check

# In[ ]:


data_cleaned["Acquisition Month"].unique()


# ## Data Visualization and Statistical Analysis

# ### Acquisitions through the years

# In[ ]:


plt.figure(figsize=(20, 10))
sns.set_style("ticks")
plt.title("Acquisitions over the years", fontsize=30, fontweight="bold")
plt.xlabel("Acquisition Year", fontsize=25)
plt.ylabel("Companies Acquired", fontsize=25)
sns.histplot(data=data_cleaned, x="Acquisition Year", kde=True, bins=25,
             color="navy")
sns.despine(offset=15, trim=False)


# ### Is there a particular time of the year when companies are acquired?

# In[ ]:


data_cleaned['Acquisition Month'] = pd.Categorical(data_cleaned['Acquisition Month'], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                                                                                       'Sep', 'Oct', 'Nov',
                                                                                       'Dec'])

plt.figure(figsize=(10, 5))
sns.set_style("ticks")
plt.title("Do Companies Prefer A particular month? ",
          fontsize=30, fontweight="bold")
plt.xlabel("Acquisition Month", fontsize=25)
plt.ylabel("Companies Acquired (%)", fontsize=25)
sns.histplot(data=data_cleaned, x="Acquisition Month", hue="Acquisition Month", stat="percent",
             legend=False, palette="colorblind")
sns.despine(offset=15, trim=False)


# ### Most Acquisitions by comapnies

# In[ ]:


companies = data_cleaned["Parent Company"].value_counts().reset_index()
companies = companies.rename(
    {"index": "Company", "Parent Company": "Acquired Companies"}, axis=1)
companies


# In[ ]:


plt.figure(figsize=(20, 10))
sns.set_style("ticks")
plt.title("Most Companies Acquired", fontsize=30, fontweight="bold")
plt.xlabel("Acquisition Year", fontsize=25)
plt.ylabel("Companies Acquired", fontsize=25)
sns.barplot(data=companies, x="Company",
            y="Acquired Companies", palette="colorblind")
sns.despine(offset=15, trim=True)


# ### Was there a Company Acquired Multiple times?
# 

# In[ ]:


data_cleaned["Acquired Company"].value_counts().head()


# ### Most common types of businesses acquired

# In[ ]:


business_types = data_cleaned["Business"].value_counts().reset_index()
business_types = business_types.rename(
    {"index": "Business", "Business": "Number of Companies"}, axis=1)
business_types = business_types.head(25)
business_types


# In[ ]:



plt.figure(figsize=(20, 10))
sns.set_style("ticks")
plt.title("Most Common Types of Businesses Acquired",
          fontsize=30, fontweight="bold")
plt.xlabel("No. of Companies", fontsize=25)
plt.ylabel("Type of Business", fontsize=25)
sns.barplot(data=business_types, x="Number of Companies",
            y="Business", palette="colorblind")
sns.despine(offset=20, trim=True)


# ### Does the "Category" variable have any significance?
# 

# In[ ]:


data_cleaned.Category.value_counts()


# ### What do acquired companies end up as most?

# In[ ]:


new_products = data_cleaned["Derived Products"].value_counts().reset_index()
new_products = new_products.rename(
    {"index": "Derived Product", "Derived Products": "Number of Companies"}, axis=1)
new_products = new_products.head(25)
new_products


# In[ ]:



plt.figure(figsize=(20, 10))
sns.set_style("ticks")
plt.title("Most Common Products Derived from Acquisitions",
          fontsize=30, fontweight="bold")
plt.xlabel("No. of Companies", fontsize=25)
plt.ylabel("Type of Business", fontsize=25)
sns.barplot(data=new_products, y="Derived Product",
            x="Number of Companies", palette="colorblind")
sns.despine(offset=20, trim=True)


# ### Where do most acquisitions happen?

# In[ ]:


countries = data_cleaned["Country"].value_counts().reset_index()
countries = countries.rename(
    {"index": "Country", "Country": "Number of Companies"}, axis=1)
countries = countries.head(25)
countries


# In[ ]:



plt.figure(figsize=(20, 10))
sns.set_style("ticks")
plt.title("Countries with most acquisitions",
          fontsize=30, fontweight="bold")
sns.barplot(data=countries, y="Country",
            x="Number of Companies", palette="colorblind")

plt.xlabel("No. of Companies", fontsize=25)
plt.ylabel("", fontsize=25)
sns.despine(offset=20, trim=True)


# ### Whats the average cost of acquiring a company?

# In[ ]:


plt.figure(figsize=(15, 5))
sns.set_style("ticks")
plt.title("Average Price of Acquisition", fontsize=30, fontweight="bold")
plt.xlabel("Acquisition Price (in 10 billion)", fontsize=25)
plt.ylabel("Companies Acquired", fontsize=25)
sns.histplot(data=data_cleaned, x="Acquisition Price", kde=True, bins=150,
             color="navy")
plt.xlim(0, 5e10)

sns.despine(offset=15, trim=False)


# ### Top 10 most expensive acquisitions

# In[ ]:


most_valuable = data_cleaned.sort_values(
    "Acquisition Price", ascending=False).head(10)
most_valuable


# ### Most expensive companies and who acquired them

# In[ ]:



plt.figure(figsize=(20, 10))
sns.set_style("ticks")
plt.title("Most Expensive Acquisitions",
          fontsize=30, fontweight="bold")
ax = sns.barplot(data=most_valuable, x="Acquired Company",
                 y="Acquisition Price", palette="colorblind")
for bar, label in zip(ax.patches, most_valuable['Parent Company']):
    x = bar.get_x()
    width = bar.get_width()
    height = bar.get_height()
    ax.text(x+width/2., height + 0.1e10, label,
            ha="center", fontsize=15, fontweight="bold")

plt.ylabel("Price (in $10 billion)", fontsize=25)
plt.xlabel("")
ax.tick_params(axis="x", labelsize=12, rotation=90)
sns.despine(offset=20, trim=True)


# ### What does each company spend on average on an acquisition?

# In[ ]:


average_spend = data_cleaned.groupby(['Parent Company']).mean(
).reset_index().sort_values("Acquisition Price", ascending=False)
average_spend


# In[ ]:



plt.figure(figsize=(20, 10))
sns.set_style("ticks")
plt.title("Average Acquisition Cost for companies",
          fontsize=30, fontweight="bold")
sns.barplot(data=average_spend, y="Parent Company",
            x="Acquisition Price", palette="colorblind")

plt.xlabel("Cost (in Billions)", fontsize=25)
plt.ylabel("", fontsize=25)
sns.despine(offset=20, trim=True)


# ### Has the price of acquiring a company changed over the years?

# In[ ]:



plt.figure(figsize=(20, 10))
sns.set_style("ticks")

plt.title("Acquisition Price over the years",
          fontsize=30, fontweight="bold")
sns.lineplot(data=data_cleaned, x='Acquisition Year',
             y='Acquisition Price')

sns.scatterplot(data=data_cleaned, x='Acquisition Year',
             y='Acquisition Price', palette="colorblind")
plt.ylabel("Cost (in Billions)", fontsize=25)
plt.xlabel("Year", fontsize=25)
sns.despine(offset=20, trim=False)


# ### Rate of Acquisition

# #### Number of years company has actively acquired other companies

# In[ ]:


years_active = data_cleaned.groupby(
    ['Parent Company', 'Acquisition Year']).count().reset_index()
years_active = years_active.groupby(['Parent Company']).count()
years_active


# #### Number of Companies Acquired

# In[ ]:


total_acquisitions = data_cleaned.groupby('Parent Company').count()
total_acquisitions


# ### Acquisition Rate for each company per year

# In[ ]:



acquisition_rate = pd.DataFrame(
    {'years': years_active['Acquisition Year'], 'acquired companies': total_acquisitions['Acquired Company']})
acquisition_rate = acquisition_rate['acquired companies']/acquisition_rate['years']
acquisition_rate = acquisition_rate.reset_index()

acquisition_rate= acquisition_rate.rename(
    {"Parent Comapny": "Company", 0: "Average Acquisitions"}, axis=1).sort_values("Average Acquisitions", ascending= False)
acquisition_rate


# In[ ]:



plt.figure(figsize=(20, 10))
sns.set_style("ticks")
plt.title("Average Acquisitions each year",
          fontsize=30, fontweight="bold")
sns.barplot(data=acquisition_rate, y="Parent Company",
            x="Average Acquisitions", palette="colorblind")

plt.xlabel("Companies per year", fontsize=25)
plt.ylabel("", fontsize=25)
sns.despine(offset=20, trim=True)

