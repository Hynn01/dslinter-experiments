#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

data = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")[
    ["country", "points", "price", "province", "taster_name", "variety", "winery"]
]


# **At the moment we are studying means of points and prices.**

# In[ ]:


wineries = data[["winery", "points", "price"]].groupby(by="winery").mean()
print( "Coeffitient of Pirson: "+str(wineries["points"].corr(wineries["price"])) )


# In[ ]:


points = wineries["points"]
prices = wineries["price"]
plt.plot(points, prices, "r.")
plt.title("Points and prices in wineries")
plt.xlabel("Points")
plt.ylabel("Prices")


# In[ ]:


priceForPoint = wineries.groupby(by="points").mean()
sns.boxplot(priceForPoint)


# In[ ]:


priceForPoint = priceForPoint["price"]
qR = priceForPoint.quantile(0.75)-priceForPoint.quantile(0.25)
delValues = []
for i in priceForPoint.index:
    if( priceForPoint[i]<priceForPoint.quantile(0.25)-1.5*qR or priceForPoint[i]>priceForPoint.quantile(0.75)+1.5*qR ):
        delValues.append(i)
priceForPoint = priceForPoint.drop( labels=delValues )
delValues = []
for i in priceForPoint.index:
    if( priceForPoint[i]<priceForPoint.quantile(0.25)-1.5*qR or priceForPoint[i]>priceForPoint.quantile(0.75)+1.5*qR ):
        delValues.append(i)
priceForPoint = priceForPoint.drop( labels=delValues )
delValues = []
for i in priceForPoint.index:
    if( priceForPoint[i]<priceForPoint.quantile(0.25)-1.5*qR or priceForPoint[i]>priceForPoint.quantile(0.75)+1.5*qR ):
        delValues.append(i)
priceForPoint = priceForPoint.drop( labels=delValues )

sns.boxplot(priceForPoint)


# In[ ]:


plt.plot( priceForPoint.index, priceForPoint, '.r' )


# In[ ]:


print( "Coeffitient of correlation: "+str( priceForPoint.corr(pd.Series(priceForPoint.index)) ) )


# Now we see countries and numbers wineries from these countries.

# In[ ]:


countries = data[ ["country", "winery"] ].groupby(by="country").count().sort_values(by="winery")[::-1]
print(countries)


# In[ ]:


print( "Number of countries: "+str(len(countries)) )


# In[ ]:


sns.boxplot(countries)


# Here we see how many types of grapes used every winery.

# In[ ]:


num_grapes = data[ ["winery", "variety"] ].groupby(by="winery").count()
max_num = num_grapes["variety"].max()
min_num = num_grapes["variety"].min()
k = 0

#print max number of grapes
print( "Max number: "+str(max_num) )
#print number of wineries which have max number of types of grapes
for i in range(len(num_grapes)):
    if( num_grapes["variety"][i]==max_num ):
        k+=1
print("This number of types of grapes use "+str(k)+" winery(ies).")

#print min number of grapes
print( "Min number: "+str(min_num) )
#print number of wineries which have max number of types of grapes
k = 0
for i in range(len(num_grapes)):
    if( num_grapes["variety"][i]==min_num ):
        k+=1
print("This number of types of grapes use "+str(k)+" winery(ies).")


# **Go to see average mark of taster.**

# In[ ]:


tasters = data[ ["taster_name", "points"] ].groupby(by="taster_name").mean()[::-1]
print(tasters)


# Do to see a standart deviation of points.

# In[ ]:


print( data[ ["taster_name", "points"] ].groupby(by="taster_name").std()[::-1] )

