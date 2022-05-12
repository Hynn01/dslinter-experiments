#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno 
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#settings :
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width',500)
pd.set_option('display.max_colwidth',500)


# <p style="background-color:lightblue ;color :black;,font-size:15em; border-radius: 20px; text-align: center-right; padding: 20px ; display: inline-block;" >
#     <b>Data_Analytics_final_task_03_5_2022</b></p>
# 

# ![title](https://cdn.pixabay.com/photo/2017/01/12/02/21/melbourne-1973533_960_720.jpg)

# ## About Dataset
# This data was scraped from publicly available results posted every week from Domain.com.au by **Tony Pino**. 
# 
# The dataset includes Address, Type of Real estate, Suburb, Method of Selling, Rooms, Price, Real Estate Agent, Date of Sale and distance from C.B.D.
# 
# Original Dataset : https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market?select=Melbourne_housing_FULL.csv

# ## Create a price report and information about the houses of Melbourne.
# 
# ### Task Description 
# 
# Create a report with information as if you were buying a property in
# Melbourne as an investment.
# Or someone who wanted to invest in Melbourne homes.
# 
# Content:
# 
# Introduction (overview of the problem)
# Body (plots, numbers, information)
# 
# Conclusion
# 
# Evaluation criteria:
# Organization
# The appearance of the report
# Note: if you choose to return as pdf, doc or excel. Remember to send the code
# as well. Put your name and date on the report.

# ### Understanding the Column information
# 

# In[ ]:


data=pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')
data.head(2)


# In[ ]:


data.columns


# In[ ]:


data_info= pd.DataFrame({"Column_names":
                         ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
                           'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 'Regionname', 'Propertycount'],
                        
    
                        "Description":["""Suburb""","""Address""","""Number of rooms""",
                                       """Type 
                                       br - bedroom(s);
                                       h - house,cottage,villa, semi,terrace;
                                       u - unit, duplex;
                                       t - townhouse;
                                       dev site - development site; 
                                       o res - other residential.""",
                                    """Price in Australian dollars""",
                        """Method: 
                        S - property sold;
                        SP - property sold prior;
                        PI - property passed in;
                        PN - sold prior not disclosed;
                        SN - sold not disclosed;
                        NB - no bid;
                        VB - vendor bid;
                        W - withdrawn prior to auction;
                        SA - sold after auction;
                        SS - sold after auction price not disclosed.
                        N/A - price or highest bid not available.""",
                                       
                        """Real Estate Agent""","""Date sold""","""Distance from CBD in Kilometres""","""Postcode""","""Scraped # of Bedrooms (from different source)""",
                        """ Number of Bathrooms""","""Number of carspots""","""Land Size in Metres""", """Building Size in Metres""",
                        """Year the house was built""","""Governing council for the area""","""Lattitude""", """Longtitude"""
                        ,"""General Region (West, North West, North, North east …etc)""",
                        """Number of properties that exist in the suburb."""          
            
        ]})
data_info.replace({ '\n' : ' '}, regex=True, inplace=True)
data_info


# ## Exploring the dataset

# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.shape


# ### Describing the statical information

# In[ ]:


data.describe()


# In[ ]:


plt.figure(figsize = (10,6))
sns.heatmap(data.corr(), cmap = 'YlGnBu', linewidth = 1, annot = True, annot_kws = {'size':9})
plt.title('Variable Correlation')


# ### Correlation Insights
# 
# * According to this dataset ,Price and number of rooms,bedroom,bathroom are positively  corelated with each other .
# 

# ## Checking for missing values

# In[ ]:


print("-"*30)
print("Missing values of each column")
print("-"*30)
print(data.isna().sum())
print("-"*30)
print("Total missing values:",data.isna().sum().sum())


# In[ ]:


msno.bar(data)


# ## Handling Missing Values

# ### What to do with the missing values?
# 
# Now that we can clearly  identified the missing values in our data, 7610 from our target column and it's about 20% of the dataset .From my perspective,missing values under 10%  can be easily solved by deletion or dropping them as we do not lose so much  information .However ,I will be using imputaion method to fill missing values
# for price column as it's important column for this dataset.
# 
# From the below box plot, we can see that there are outliers in price column.If outliers are present ,median is  better choise comparing to mean .
# 
# 
# 
# 
# ### Replacing values for each column are imputed by :
# 
# *     Price        : Median
# *     Postcode     : Median
# *     Distance     : Mean
# *     Bathroom     : 1
# *     Bedroom2     : 1
# *     Car          : mean
# *     Landsize     : median
# *     BuildingArea : mean
# *     YearBuilt    : mean
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


plt.figure(figsize = (4,8))
sns.boxplot(y = data.Price)


# In[ ]:


data['Price'].fillna(data['Price'].median(), inplace = True)


# In[ ]:


data['Distance'].fillna(data['Distance'].mean(), inplace = True)
data['Postcode'].fillna(data['Postcode'].median(), inplace = True)


# note: I am assuming every house have minimum 1 bedroom and bathroom .
# Instead of dropping them I will be fill with ones

# In[ ]:


data.Bedroom2.value_counts()


# In[ ]:


data.Bathroom.value_counts()


# In[ ]:


data['Bathroom'].fillna(1, inplace= True)
data['Bedroom2'].fillna(1, inplace= True)


# Before I fill a certain missing values for Car column which is a number of carspots available .
# I would like to explore each type of houses available for carspots.

# ## Type of houses available in Melbourne Area according to this dataset

# In[ ]:


data.Type.value_counts(dropna=False).to_frame()


# In[ ]:


fig = px.pie(data.Type.value_counts(dropna=False), values='Type', names=['h','u','t'],
             title='Type of the houses available in Melbourne Area',width=600)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# ## Most frequent available car parking spaces for each type of houses

# In[ ]:


# h - house
type_h=data.loc[(data['Type'] == 'h') ,['Type','Car']]
type_h.head(5)
type_h.Car.value_counts()[:2]


# In[ ]:


#u - unit, duplex;
type_u=data.loc[(data['Type'] == 'u') ,['Type','Car']]
type_u.Car.value_counts()[:2]


# In[ ]:


#t - townhouse
type_t=data.loc[(data['Type'] == 't') ,['Type','Car']]
type_t.Car.value_counts()[:2]


# As we can see above , houses and townhouse has usually 2 car parking spcaces and unit has only one parking space available and some are available for 2 spots.

# In[ ]:


data['Car'].fillna(data['Car'].mean(), inplace = True)
data['Landsize'].fillna(data['Landsize'].median(), inplace = True)
data['BuildingArea'].fillna(data['BuildingArea'].mean(), inplace = True)
data['YearBuilt'].fillna(data['YearBuilt'].mean(), inplace = True)


# In[ ]:


data['CouncilArea'].value_counts(dropna=False)


# ### Checking missing values for councilArea

# In[ ]:


data.loc[(data['CouncilArea'].isna()) ,['Postcode','CouncilArea','Regionname']]


# * ####  Earlier ,I have filled missing postcode that has been imputed by median and I will be filling CouncilArea and Regionname based on Postcode.
# 

# In[ ]:


#Selecting the dataframe that match postcode with missing values
ps_3011=data.loc[(data['Postcode']==3011 )]
print("Council Area based on postcode 3011 is ",ps_3011.CouncilArea.value_counts()[:1])
print("Region name based on postcode 3011 is ",ps_3011.Regionname.value_counts()[:1])


# In[ ]:


ps_3124=data.loc[(data['Postcode']==3124 )]
print("Council Area based on postcode 3011 is ",ps_3124.CouncilArea.value_counts()[:1])
print("Region name based on postcode 3011 is ",ps_3124.Regionname.value_counts()[:1])


# In[ ]:


ps_3103=data.loc[(data['Postcode']==3103 )]
print("Council Area based on postcode 3011 is ",ps_3103.CouncilArea.value_counts()[:1])
print("Region name based on postcode 3011 is ",ps_3103.Regionname.value_counts()[:1])


# ### Replacing missing values of ConcilArea  based on Postcode

# In[ ]:


data.loc[(data['CouncilArea'].isna()) & (data['Postcode']==3011) , 'CouncilArea'] = "Maribyrnong City Council"
data.loc[(data['CouncilArea'].isna()) & (data['Postcode']==3124) , 'CouncilArea'] = "Boroondara City Council"
data.loc[(data['CouncilArea'].isna()) & (data['Postcode']==3103) , 'CouncilArea'] = "Boroondara City Council"


# ### Replacing missing values of Regionname  based on Postcode

# In[ ]:


data.loc[(data['Regionname'].isna()) & (data['Postcode']==3011) , 'Regionname'] = "Western Metropolitan"
data.loc[(data['Regionname'].isna()) & (data['Postcode']==3124) , 'Regionname'] = "Southern Metropolitan"
data.loc[(data['Regionname'].isna()) & (data['Postcode']==3103) , 'Regionname'] = "Southern Metropolitan"


# In[ ]:


data.sample(3)


# ### Forward Fill For Lattitude and Longtitude
# This method will fill the nan values with the last valid observation down the column .

# In[ ]:


data['Lattitude'].fillna(method = 'ffill', inplace = True)
data['Longtitude'].fillna(method = 'ffill', inplace = True)


# In[ ]:


data.loc[(data['Propertycount'].isna()) ,['Landsize','Price','Type']]


# 

# In[ ]:


data.iloc[18521:18525]


# In[ ]:


data.iloc[26886:26890]


# In[ ]:


data['Propertycount'].fillna(method = 'ffill', inplace = True)


# In[ ]:


print("-"*30)
print("Missing values of each column")
print("-"*30)
print(data.isna().sum())
print("-"*30)
print("Total missing values:",data.isna().sum().sum())


# In[ ]:


msno.bar(data)


# In[ ]:


data.sample(2)


# ## All possible Categorical Columns  to Dummy Columns by using the get_dummies() from Pandas

# In[ ]:



data_ = data.copy()

data_ = pd.get_dummies(data_,
                     columns = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG','CouncilArea', 'Regionname'])
  
data_.head()


# ##  Linear Regression Model
# ### Split the data into Train, Test set

# In[ ]:


X = data_.drop(['Price','Address','Date'],axis=1) # Features
y=data_['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)#70 % training and 30% testing


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# 
# ### Predicting  test set

# In[ ]:


y_pred = model.predict(X_test)


# ## Evaluating our model with RMSE 

# In[ ]:


print(np.sqrt(mean_squared_error(y_test,y_pred)))


# ## Comparison between Actual and Predicted

# In[ ]:



df= pd.DataFrame({'Actual_Price': np.round(y_test), 
                   'Predicted_Price': np.round(y_pred),
                 })
df['difference']=df.apply(lambda x: x.Predicted_Price-x.Actual_Price,axis=1)
df.head(10)


# <p style="background-color:lightblue ;color :black;,font-size:15em; border-radius: 20px; text-align: center-right; padding: 20px ; display: inline-block;" >
#     <b>      Thanks for checking out my notebook , Upvote this if you enjoy ! 😀</b></p>
