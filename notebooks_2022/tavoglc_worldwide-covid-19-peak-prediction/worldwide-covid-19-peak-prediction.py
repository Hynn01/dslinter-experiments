#!/usr/bin/env python
# coding: utf-8

# # Correlation between day lenght and pandemic peaks across the globe
# 
# In a series of notebooks, I shared some ideas about the use of VAEs to understand the internal structure within the SARS Cov2 sequence. And that structure was related to day length.  This particular time scale can be useful to synchronize the different pandemic curves to get a better understanding of the dynamics. 

# # Code

# ## Packages 

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


# ## Auxiliary functions

# In[ ]:


def GetDayLenght(J,lat):
    #CERES model  Ecological Modelling 80 (1995) 87-95
    phi = 0.4093*np.sin(0.0172*(J-82.2))
    coef = (-np.sin(np.pi*lat/180)*np.sin(phi)-0.1047)/(np.cos(np.pi*lat/180)*np.cos(phi))
    ha =7.639*np.arccos(np.max([-0.87,coef]))
    return ha


# In[ ]:


def GetHalfWidth(xdta, ydta,tdata):
    
    max_y2 = max(ydta)/2
    xs = [x for x,y in zip(xdta,ydta) if y > max_y2]
    
    minval,maxval = np.max(xs), np.min(xs)
    mint,maxt = tdata[minval],tdata[maxval]
    
    return np.abs(maxval-minval),np.abs(maxt-mint)

def ProcessSeries(data,fact=0.25):
    
    feats = []
    series = data['cases'].sum()
    b, a = signal.butter(3, fact)
    y0 = signal.filtfilt(b, a, np.array(series))
    index = series.index
    
    halfy0 = y0[0:int(len(y0)/2)]
    halfi0 = index[0:int(len(y0)/2)]
    
    halfy1 = y0[int(len(y0)/2)::]
    halfi1 = index[int(len(y0)/2)::]
    
    widthlg0,widtht0 = GetHalfWidth(halfi0,halfy0,data['dayofyear'].mean())
    widthlg1,widtht1 = GetHalfWidth(halfi1,halfy1,data['dayofyear'].mean())
    
    feats.append(data['dayofyear'].mean()[halfi0[np.argmax(halfy0)]])
    feats.append(data['dayofyear'].mean()[halfi1[np.argmax(halfy1)]])
    
    feats.append(widtht0)
    feats.append(widtht1)
    
    feats.append(halfi0[np.argmax(halfy0)])
    feats.append(halfi1[np.argmax(halfy1)])
    
    feats.append(widthlg0)
    feats.append(widthlg1)
    
    return feats


def GetFeatures(df):
    
    dta1 = df.groupby('lengthofday')
    feats = ProcessSeries(dta1,fact=0.1) 
    
    return feats


# In[ ]:


def MakePanel(data,size):

    fig,axs = plt.subplots(4,4,figsize=size)

    axs[0,0].scatter(data['long'],data['lat'],c=data['maxdaywave01'])
    axs[0,0].set_title('First Wave Peak Day')
    axs[0,1].scatter(data['long'],data['lat'],c=data['maxlgwave01'])
    axs[0,1].set_title('First Wave Day length at Peak Day')
    axs[0,2].scatter(data['long'],data['lat'],c=data['maxdaywave02'])
    axs[0,2].set_title('Second Wave Peak Day')
    axs[0,3].scatter(data['long'],data['lat'],c=data['maxlgwave02'])
    axs[0,3].set_title('Second Wave Day length at Peak Day')

    axs[1,0].hist(data['maxdaywave01'],bins=50)
    axs[1,1].hist(data['maxlgwave01'],bins=50)
    axs[1,2].hist(data['maxdaywave02'],bins=50)
    axs[1,3].hist(data['maxlgwave02'],bins=50)

    axs[2,0].scatter(data['long'],data['lat'],c=data['widthdaywave01'])
    axs[2,0].set_title('First Wave Half Width in days')
    axs[2,1].scatter(data['long'],data['lat'],c=data['widthlgwave01'])
    axs[2,1].set_title('First Wave Half Width in Hours')
    axs[2,2].scatter(data['long'],data['lat'],c=data['widthdaywave02'])
    axs[2,2].set_title('Second Wave Half Width in days')
    axs[2,3].scatter(data['long'],data['lat'],c=data['widthlgwave02'])
    axs[2,3].set_title('Second Wave Half Width in Hours')

    axs[3,0].hist(data['widthdaywave01'],bins=50)
    axs[3,1].hist(data['widthlgwave01'],bins=50)
    axs[3,2].hist(data['widthdaywave02'],bins=50)
    axs[3,3].hist(data['widthlgwave02'],bins=50)


# # Data

# In[ ]:


covMX = pd.read_csv('../input/covid19-data-from-mexico/covidmex.csv')
covMX['cases'] = covMX['covidt']


# In[ ]:


vals = []
uniqueqrys = set(covMX['qry'])

for val in uniqueqrys:
    mx = covMX[covMX['qry']==val]
    geo = mx[['lat','long']].mean()
    localData = geo.tolist()
    if len(mx)>1500:
        try:
            vals.append(localData+GetFeatures(mx))
        except ValueError:
            pass

headers = ['lat','long','maxdaywave01','maxdaywave02','widthdaywave01','widthdaywave02','maxlgwave01','maxlgwave02','widthlgwave01','widthlgwave02']
dataMX = np.array(vals)
dataMX = pd.DataFrame(dataMX,columns=headers)


# # Curves Syncronization
# 
# Let's use Mexico's pandemic cases as an example. Total cases through time show the different waves. While joining all the data by day length shows only two waves And pandemic cases always start to rise at the time with the lowest day length. 

# In[ ]:


fig,axs = plt.subplots(1,2,figsize=(20,6))

covMX.groupby(['FECHA_INGRESO'])['covidt'].sum().plot(ax=axs[0],label='Cases')
axs[0].set_title('Epidemic Curve by day')
axs[0].legend(loc=1)
covMX.groupby(['lengthofday'])['covidt'].sum().plot(ax=axs[1],label='Cases')
axs[1].set_title('Epidemic Curve by day length')
axs[1].legend(loc=1)


# This specific pattern is easier to observe when the pandemic curves are plotted per specific geographic location 

# In[ ]:


fig,axs = plt.subplots(3,3,figsize=(12,6))

axs = axs.ravel()

for axl in axs:
    disc = 10
    while disc < 10000:
        cqy = np.random.choice(list(uniqueqrys))
        mx = covMX[covMX['qry']==cqy]
        disc = len(mx)
    mx.groupby(['lengthofday'])['covidt'].sum().plot(ax=axl,label='Cases')
    axl.set_ylabel('Cases')
    axl.set_title('Epidemic Curve by day length')


# ## Mexico
# 
# Calculating the pandemic peak and the half-width duration of the epidemic curves per each geographic location yields an ordered pattern. This analysis relies on the assumption that there are always two waves per year. There might be some caveats with this assumption and half-width might be not properly estimated. However, a particular order can be obtained under different geographical locations. 

# In[ ]:


mapsize = (30,14)

MakePanel(dataMX,mapsize)


# ## USA

# In[ ]:


usfull = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/covid_us_county.csv')
covUSA = usfull.copy().query('lat < 55 & lat > 25 & long < -60 & long > -130')
covALSK = usfull.copy().query('lat < 80 & lat > 55 & long < -120 & long > -170')


# ### Mainland USA

# In[ ]:


covUSA['date'] = pd.to_datetime(covUSA['date'],format='%Y-%m-%d')
covUSA['dayofyear'] = covUSA['date'].dt.dayofyear
covUSA['lengthofday'] = [GetDayLenght(val,sal) for val,sal in zip(covUSA['dayofyear'],covUSA['lat'])]

locUSA = [[est,mun] for est,mun in zip(covUSA['lat'],covUSA['long'])]
uniqueLocUSA = [list(x) for x in set(tuple(x) for x in locUSA)]

covUSA['qry'] = ["lat==" +str(val)+ " & long==" +str(sal) for val,sal in zip(covUSA['lat'],covUSA['long'])]


# In[ ]:


qryToDataUSA = {}

for val in uniqueLocUSA:
    
    qry = "lat==" +str(val[0])+ " & long==" +str(val[1])
    dta = covUSA.query(qry)
    geo = dta[['lat','long']].mean()
    localData = geo.tolist()
    
    qryToDataUSA[qry] = localData


# In[ ]:


vals = []

for val in qryToDataUSA.keys():
    mx = covUSA[covUSA['qry']==val]
    if len(mx)>350:
        try:
            vals.append(qryToDataUSA[val]+GetFeatures(mx))
        except ValueError:
            pass

headers = ['lat','long','maxdaywave01','maxdaywave02','widthdaywave01','widthdaywave02','maxlgwave01','maxlgwave02','widthlgwave01','widthlgwave02']
dataUSA = np.array(vals)
dataUSA = pd.DataFrame(dataUSA,columns=headers)


# In[ ]:


mapsize = (30,15)

MakePanel(dataUSA,mapsize)


# ### Alaska USA

# In[ ]:


covALSK['date'] = pd.to_datetime(covALSK['date'],format='%Y-%m-%d')
covALSK['dayofyear'] = covALSK['date'].dt.dayofyear
covALSK['lengthofday'] = [GetDayLenght(val,sal) for val,sal in zip(covALSK['dayofyear'],covALSK['lat'])]

locALSK = [[est,mun] for est,mun in zip(covALSK['lat'],covALSK['long'])]
uniqueLocALSK = [list(x) for x in set(tuple(x) for x in locALSK)]

covALSK['qry'] = ["lat==" +str(val)+ " & long==" +str(sal) for val,sal in zip(covALSK['lat'],covALSK['long'])]


# In[ ]:


qryToDataALSK = {}

for val in uniqueLocALSK:
    
    qry = "lat==" +str(val[0])+ " & long==" +str(val[1])
    dta = covALSK.query(qry)
    geo = dta[['lat','long']].mean()
    localData = geo.tolist()
    
    qryToDataALSK[qry] = localData


# In[ ]:


vals = []

for val in qryToDataALSK.keys():
    mx = covALSK[covALSK['qry']==val]
    if len(mx)>350:
        try:
            vals.append(qryToDataALSK[val]+GetFeatures(mx))
        except ValueError:
            pass

headers = ['lat','long','maxdaywave01','maxdaywave02','widthdaywave01','widthdaywave02','maxlgwave01','maxlgwave02','widthlgwave01','widthlgwave02']
dataALSK = np.array(vals)
dataALSK = pd.DataFrame(dataALSK,columns=headers)


# In[ ]:


mapsize = (30,14)

MakePanel(dataALSK,mapsize)


# ## Italy

# In[ ]:


covITA = pd.read_csv('../input/covid19-in-italy/covid19_italy_province.csv')
covITA['Date'] = pd.to_datetime(covITA['Date'],format='%Y-%m-%d')
covITA['dayofyear'] = covITA['Date'].dt.dayofyear
covITA = covITA.dropna()

covITA['lengthofday'] = [GetDayLenght(val,sal) for val,sal in zip(covITA['dayofyear'],covITA['Latitude'])]

locITA = [[est,mun] for est,mun in zip(covITA['Latitude'],covITA['Longitude'])]
uniqueLocITA = [list(x) for x in set(tuple(x) for x in locITA)]

covITA['qry'] = ["Latitude==" +str(val)+ " & Longitude==" +str(sal) for val,sal in zip(covITA['Latitude'],covITA['Longitude'])]
covITA['cases'] = covITA['TotalPositiveCases']


# In[ ]:


qryToDataITA = {}

for val in uniqueLocITA:
    
    qry = "Latitude==" +str(val[0])+ " & Longitude==" +str(val[1])
    dta = covITA.query(qry)
    geo = dta[['Latitude','Longitude']].mean()
    localData = geo.tolist()
    
    qryToDataITA[qry] = localData


# In[ ]:


vals = []

for val in qryToDataITA.keys():
    mx = covITA[covITA['qry']==val]
    if len(mx)>30:
        try:
            vals.append(qryToDataITA[val]+GetFeatures(mx))
        except ValueError:
            pass

headers = ['lat','long','maxdaywave01','maxdaywave02','widthdaywave01','widthdaywave02','maxlgwave01','maxlgwave02','widthlgwave01','widthlgwave02']

dataITA = np.array(vals)

dataITA = pd.DataFrame(dataITA,columns=headers)


# In[ ]:


MakePanel(dataITA,mapsize)


# ## Brazil

# In[ ]:


covBRA = pd.read_csv('../input/corona-virus-brazil/brazil_covid19_cities.csv')
covBRA['date'] = pd.to_datetime(covBRA['date'],format='%Y-%m-%d')
covBRA['dayofyear'] = covBRA['date'].dt.dayofyear
covBRA = covBRA.dropna()

cityData = pd.read_csv('../input/corona-virus-brazil/brazil_cities_coordinates.csv')
cityToGeo = {}

for val,lat,long in zip(cityData['city_code'],cityData['lat'],cityData['long']):
    cityToGeo[int(''.join(str(val)[0:-1]))] = [lat,long] 

covBRA['lengthofday'] = [GetDayLenght(val,cityToGeo[sal][0]) for val,sal in zip(covBRA['dayofyear'],covBRA['code'])]


# In[ ]:


vals = []

for val in cityToGeo.keys():
    mx = covBRA[covBRA['code']==val]
    if len(mx)>30:
        try:
            vals.append(cityToGeo[val]+GetFeatures(mx))
        except ValueError:
            pass

headers = ['lat','long','maxdaywave01','maxdaywave02','widthdaywave01','widthdaywave02','maxlgwave01','maxlgwave02','widthlgwave01','widthlgwave02']

dataBRA = np.array(vals)

dataBRA = pd.DataFrame(dataBRA,columns=headers)


# In[ ]:


MakePanel(dataBRA,mapsize)


# ## Pakistan

# In[ ]:


covPAK = pd.read_csv('../input/pakistan-covid19-dataset/combined_report.csv')
covPAK['date'] = pd.to_datetime(covPAK['Last_Update'],format='%Y-%m-%d')
covPAK['dayofyear'] = covPAK['date'].dt.dayofyear

covPAK['lengthofday'] = [GetDayLenght(val,sal) for val,sal in zip(covPAK['dayofyear'],covPAK['Lat'])]

locPAK = [[est,mun] for est,mun in zip(covPAK['Lat'],covPAK['Long_'])]
uniqueLocPAK = [list(x) for x in set(tuple(x) for x in locPAK)]

covPAK['qry'] = ["Lat==" +str(val)+ " & Long_==" +str(sal) for val,sal in zip(covPAK['Lat'],covPAK['Long_'])]
covPAK['cases'] = covPAK['Confirmed']


# In[ ]:


qryToDataPAK = {}

for val in uniqueLocPAK:
    
    qry = "Lat==" +str(val[0])+ " & Long_==" +str(val[1])
    dta = covPAK.query(qry)
    geo = dta[['Lat','Long_']].mean()
    localData = geo.tolist()
    
    qryToDataPAK[qry] = localData


# In[ ]:


vals = []

for val in qryToDataPAK.keys():
    mx = covPAK[covPAK['qry']==val]
    if len(mx)>30:
        try:
            vals.append(qryToDataPAK[val]+GetFeatures(mx))
        except ValueError:
            pass

headers = ['lat','long','maxdaywave01','maxdaywave02','widthdaywave01','widthdaywave02','maxlgwave01','maxlgwave02','widthlgwave01','widthlgwave02']

dataPAK = np.array(vals)

dataPAK = pd.DataFrame(dataPAK,columns=headers)


# In[ ]:


MakePanel(dataPAK,mapsize)


# # Peak day lenght and latitude
# 
# Clear changes in peak day length as the pandemic moves through the latitude of different geographical locations. Scatter plots show a linear relationship between day length at the pandemic peak. However, is not the only factor involved. 

# In[ ]:


fig,axs = plt.subplots(2,2,figsize=(15,15))

axs[0,0].plot(dataMX['lat'],dataMX['maxlgwave01'],'bo')
axs[0,1].plot(dataITA['lat'],dataITA['maxlgwave01'],'bo')
axs[1,0].plot(dataUSA['lat'],dataUSA['maxlgwave01'],'bo')
axs[1,1].plot(dataBRA['lat'],dataBRA['maxlgwave01'],'bo')


# In[ ]:


fig,axs = plt.subplots(2,2,figsize=(15,15))

axs[0,0].plot(dataMX['lat'],dataMX['maxlgwave02'],'bo')
axs[0,1].plot(dataITA['lat'],dataITA['maxlgwave02'],'bo')
axs[1,0].plot(dataUSA['lat'],dataUSA['maxlgwave02'],'bo')
axs[1,1].plot(dataBRA['lat'],dataBRA['maxlgwave02'],'bo')


# In[ ]:


Xdata = np.vstack([np.array(dataMX[['lat','long']]),np.array(dataITA[['lat','long']]),np.array(dataUSA[['lat','long']]),np.array(dataBRA[['lat','long']])])
Ydata0 = np.array(dataMX['maxlgwave01'].tolist()+dataITA['maxlgwave01'].tolist()+dataUSA['maxlgwave01'].tolist()+dataBRA['maxlgwave01'].tolist())
Ydata1 = np.array(dataMX['maxlgwave02'].tolist()+dataITA['maxlgwave02'].tolist()+dataUSA['maxlgwave02'].tolist()+dataBRA['maxlgwave02'].tolist())


# In[ ]:


Folds = KFold(n_splits=10,shuffle=True)

models0 = []
perf0 = []

for trainIndex, testIndex in Folds.split(Ydata0):
    Xtrain, Xtest = Xdata[trainIndex], Xdata[testIndex]
    Ytrain, Ytest = Ydata0[trainIndex], Ydata0[testIndex]
    regr = RandomForestRegressor(random_state=0)
    regr.fit(Xtrain, Ytrain)
    models0.append(regr)
    perf0.append(regr.score(Xtest,Ytest))

plt.plot(perf0)
best0 = models0[np.argmin(perf0)]


# In[ ]:


Folds = KFold(n_splits=10,shuffle=True)

models1 = []
perf1 = []

for trainIndex, testIndex in Folds.split(Ydata1):
    Xtrain, Xtest = Xdata[trainIndex], Xdata[testIndex]
    Ytrain, Ytest = Ydata1[trainIndex], Ydata1[testIndex]
    regr = RandomForestRegressor(random_state=0)
    regr.fit(Xtrain, Ytrain)
    models1.append(regr)
    perf1.append(regr.score(Xtest,Ytest))
    

plt.plot(perf1)
best1 = models1[np.argmin(perf1)]


# # World predictions
# 
# Complete world modeling of peak prediction appears to be in reach, yet more data is needed to get a more accurate view of viral spread. 

# In[ ]:


world = pd.read_csv('../input/world-cities-database/worldcitiespop.csv')


# In[ ]:


plt.figure(figsize=(25,12))
colors0 = best0.predict(np.array(world[['Latitude','Longitude']]))
plt.scatter(world['Longitude'],world['Latitude'],c=colors0,alpha=0.1)


# In[ ]:


plt.figure(figsize=(25,12))
colors1 = best1.predict(np.array(world[['Latitude','Longitude']]))
plt.scatter(world['Longitude'],world['Latitude'],c=colors1,alpha=0.1)

