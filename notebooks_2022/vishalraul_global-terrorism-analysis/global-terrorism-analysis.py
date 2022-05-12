#!/usr/bin/env python
# coding: utf-8

# ## <b> The Global Terrorism Database (GTD) is an open-source database including information on terrorist attacks around the world from 1970 through 2017. The GTD includes systematic data on domestic as well as international terrorist incidents that have occurred during this time period and now includes more than 180,000 attacks. The database is maintained by researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism (START), headquartered at the University of Maryland.</b>
# 
# # <b> Explore and analyze the data to discover key findings pertaining to terrorist activities. </b>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#import the Gobal Terrorism dataset
data = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv',encoding="latin-1")


# In[ ]:


#show all columns in the dataset
pd.set_option('display.max_columns',None)


# In[ ]:


data.head(5)


# In[ ]:


#number of rows and columns present in dataset
data.shape


# In[ ]:


pd.set_option('display.max_rows',None)


# In[ ]:


#Attribute infromation
data.info()


# In[ ]:


#data types
data.dtypes


# In[ ]:


#null values and its count
data.isna().sum()


# In[ ]:


#percentage of Null values
data.isna().sum()*100/len(data)


# In[ ]:


#the columns heading not make any sense,so we will rename it for better understanding.
data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'State','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','success':'Success','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)


# In[ ]:


#dataset columns
list(data.columns)


# In[ ]:


'''
Global Terrorism dataset having 135 columns and some of columns have most 99% null values, 
thats not contribute in the analysis so we removed the null values and keep only neccessary columns from the dataset.
'''
columns_to_keep = ['Year','Month','Day','Country','State','Region','latitude','longitude','AttackType','Success','Target','Killed','Wounded','Summary','Group','Target_type',
                   'Weapon_type','Motive']

data = data[columns_to_keep]


# In[ ]:


#new extracted dataset
data.head()


# In[ ]:


#Size of new dataset
data.shape


# In[ ]:


#Remove eventid
#data=data.drop('eventid', axis=1, inplace=False)


# In[ ]:


#details about dataset
data.describe()


# In[ ]:


#correlation
data.corr()


# In[ ]:


#Correlation Analysis
plt.figure(figsize=(15,10))

sns.heatmap(np.round(data.corr(),4), annot=True, cmap = 'viridis')
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)


# # **Data Visualization**

# **1.Terrorist Activities in Each Year**

# In[ ]:


plt.figure(figsize=(25,12))
sns.countplot(data['Year'], color='y', edgecolor='k')
plt.xlabel('Attack Year',fontweight='bold')
plt.ylabel('Number Of Attack',fontweight='bold')
plt.title('Number OF Terror Attack Each Year', fontweight='bold',fontsize=20)
plt.xticks(rotation = 90)


# 
# 
# ## **In Year 2014 and 2015 has a larger Number of Terrorist Activities**
# 
# 
# 
# 
# 

# **2.Number of Success Attacks Each Year**

# In[ ]:


plt.figure(figsize=(25,12))
sns.countplot(x=data['Year'], hue='Success', data=data, edgecolor = 'k')
plt.xlabel('Attack Year',fontweight='bold')
plt.ylabel('Number Of Attack',fontweight='bold')
plt.title('Number OF Terror Attack Each Year', fontweight='bold',fontsize=20)
plt.xticks(rotation = 90)
plt.legend(title='Attacks', loc='upper left', labels=['Success Attack', 'Unsuccess Attack'], fontsize=12)


# ## **In Year 2014 having large number of Terrorist Activities but in 2016 has Most successful Attacks done by terrorist.**

# **3.Terrorist Activities by Region In Each Year**

# In[ ]:



pd.crosstab(data['Year'], data['Region']).plot(kind='area', figsize=(25,12))
plt.title('Terrorist Activities by Region in each Year')
plt.xlabel('Attack Year')
plt.ylabel('Number of Attacks')
plt.show()


# ##**In 'Middle East & North Africa' Region having Most attacks.**

# **4.Type of Attack**

# In[ ]:


plt.figure(figsize=(18,8))
sns.countplot(data['AttackType'], order=data['AttackType'].value_counts().index, edgecolor='k',palette='rocket')
plt.xlabel('Attack Type',fontweight='bold')
plt.ylabel('Number Of Attack',fontweight='bold')
plt.title('Type OF Attack', fontweight='bold',fontsize=20)
plt.xticks(rotation = 60)


# ##**Maximum Number Of Attack are From Bombing/Explosion and Armed Assault**

# **5.Type of Target**

# In[ ]:


plt.figure(figsize=(18,8))
sns.countplot(data['Target_type'], order=data['Target_type'].value_counts().index, edgecolor='k',palette='hot')
plt.xlabel('Target type',fontweight='bold')
plt.ylabel('Number Of Attack',fontweight='bold')
plt.title('Target type', fontweight='bold',fontsize=20)
plt.xticks(rotation = 90)


# ##**The Main Target of Terrorist is Private Citizens&Property and Military.**

# **6.Most Active Terrorrist Groups**

# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(y=data['Group'].value_counts()[1:10].index,x=data['Group'].value_counts()[1:10].values,
           palette='copper',edgecolor='k')
plt.xlabel('Number of Attack',fontweight='bold',fontsize=14)
plt.ylabel('Name of Terrorist Group', fontweight='bold',fontsize=14)
plt.title('Most Active Terrorist Organizations', fontweight='bold',fontsize=18)
plt.show()


# ##**Taliban and ISIL are Most Active Terrorist Organisation**

# **7.Total  Number of Attack in Each Contry  and Region**(Top10)

# In[ ]:


#plt.figure(figsize=(15,10))
#axes=plt.subplots(nrows=1, ncols=2)
fig,axes = plt.subplots(figsize=(16,11),nrows=1,ncols=2)
sns.barplot(y=data['Country'].value_counts()[0:10].index,x=data['Country'].value_counts()[0:10].values,palette='hot', ax=axes[0],edgecolor='k' )
axes[0].set_title('Terrorist Attack per Country',fontweight='bold',fontsize=18)
axes[0].set_xlabel('Number of Attack',fontweight='bold',fontsize=14)
axes[0].set_ylabel('Country', fontweight='bold',fontsize=14)


sns.barplot(y=data['Region'].value_counts()[1:10].index,x=data['Region'].value_counts()[1:10].values,palette='magma',ax=axes[1],edgecolor='k')
axes[1].set_xlabel('Number of Attack',fontweight='bold',fontsize=14)
axes[1].set_ylabel('Region', fontweight='bold',fontsize=14)
axes[1].set_title('Terrorist Attack per Region', fontweight='bold',fontsize=18)
fig.tight_layout()
plt.show()


# ##**Maximum Terrorist Activities are in Iraq.**

# * **Number People Killed during the attack**
# 

# In[ ]:


plt.figure(figsize=(25,12))
plt.scatter(data['Year'], data['Killed'], color='b',edgecolor='k')
plt.xlabel('Year',fontweight='bold',fontsize=14)
plt.ylabel('Number Of Killed',fontweight='bold',fontsize=14)
plt.title('Number People Killed during the attack',fontweight='bold',fontsize=18)
plt.show()


# ##**Maximum people died in terrorist attacks in 2014**

# * **Number People Wounded during the attack**

# In[ ]:


plt.figure(figsize=(25,12))
plt.scatter(data['Year'], data['Wounded'], color='b',edgecolor='k')
plt.xlabel('Year',fontweight='bold',fontsize=14)
plt.ylabel('Number Of Wounded',fontweight='bold',fontsize=14)
plt.title('Number People Wounded during the attack',fontweight='bold',fontsize=18)
plt.show()


# Conclusion
# Iraq ranked first on the global terrorism for their terrorist activity followed by Pakistan, Afganistan, India, and so on
# 
# Most targeted areas are private citizens and property, military, police, and so on.
# 
# Global terror attack deaths rose sharply starting year 2011
# 
# In conclusion with the ranking, Iraq suffered from most terrorist attacks in 2014, with the most deaths in that year 

# **Conclusion**
# 
# 
# *   **Iraq ranked first on global terrorist activity followed by Pakistan, Afghanistan then India and so on.**
# *   **Most Targeted Areas are Private Citizens&Property, Military,Police and so on.**
# *   **Global Terror attack rise sharply from 2011 and Maximum Attacks are in 2014 and Maximum people where died in 2014.**
# *   **Taliban and ISIL are the most active terrorist Groups.**
