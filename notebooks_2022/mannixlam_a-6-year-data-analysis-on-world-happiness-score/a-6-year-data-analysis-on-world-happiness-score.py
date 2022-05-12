#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import iplot
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn; seaborn.set()
import scipy.stats


# # Data pre-processing

# In[ ]:


wh2021 = pd.read_csv('../input/world-happiness-report-2021/DataForFigure2.1WHR2021C2.csv',decimal=',')
wh2019 = pd.read_csv("../input/world-happiness/2019.csv")
wh2018 = pd.read_csv("../input/world-happiness/2018.csv")
wh2017 = pd.read_csv("../input/world-happiness/2017.csv")
wh2016 = pd.read_csv("../input/world-happiness/2016.csv")
wh2015 = pd.read_csv("../input/world-happiness/2015.csv")


# In[ ]:


wh2015.shape


# In[ ]:


wh2016.shape


# In[ ]:


wh2017.shape


# In[ ]:


wh2018.shape


# In[ ]:


wh2019.shape


# In[ ]:


wh2021.shape


# In[ ]:


wh2021 = wh2021.drop(['Regional indicator','Standard error of ladder score','upperwhisker','lowerwhisker','Social support','Ladder score in Dystopia','Explained by: Log GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy','Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption','Dystopia + residual'],axis = 1)
wh2021.columns = ["Country or region","Score","GDP per capita",
                 "Healthy life expectancy","Freedom to make life choices",
                 "Generosity","Perceptions of corruption"]


# In[ ]:


wh2017 = wh2017.drop(["Whisker.high","Whisker.low","Dystopia.Residual"], axis = 1)

wh2017.columns = ["Country or region","Overall rank","Score","GDP per capita",'Social support',
                 "Healthy life expectancy","Freedom to make life choices",
                 "Generosity","Perceptions of corruption"]


# In[ ]:


wh2016 = wh2016.drop(['Region','Lower Confidence Interval','Upper Confidence Interval', 'Dystopia Residual'],axis = 1)

wh2016.columns = ["Country or region","Overall rank","Score","GDP per capita",'Social support',
                 "Healthy life expectancy","Freedom to make life choices",
                 "Perceptions of corruption","Generosity"]


# In[ ]:


wh2015 = wh2015.drop(["Region",'Standard Error','Dystopia Residual'],axis=1)

wh2015.columns = ["Country or region","Overall rank","Score","GDP per capita",'Social support',
                 "Healthy life expectancy","Freedom to make life choices",
                 "Perceptions of corruption","Generosity"]


# In[ ]:


wh2021['year'] = 2020
wh2019["year"] = 2019
wh2018["year"] = 2018
wh2017["year"] = 2017
wh2016["year"] = 2016
wh2015["year"] = 2015


# In[ ]:


wh2021.isnull().sum()


# In[ ]:


tol = pd.concat([wh2019,wh2018,wh2017,wh2016,wh2015])
tol.isnull().sum()


# In[ ]:


# One missing value on the column "Perceptions of corruption"
tol.loc[tol.isnull().any(axis=1),:]


# In[ ]:


tol = tol.dropna()


# In[ ]:


tol.isnull().sum()  #No any more missing value


# In[ ]:


tol.duplicated().value_counts()


# In[ ]:


tol.dtypes


# ## Data visualization & analytics

# # Part 1

# In[ ]:


fig1 = px.scatter(tol, x = "GDP per capita", y = 'Score', facet_row="year", color = "year", trendline= "ols", title = 'Score vs GDP per Capita (Each Year)')
fig1.show()


# In[ ]:


figwhat = px.scatter(tol, x = "Social support", y = 'Score', facet_row="year", color = "year", trendline= "ols", title = 'Score vs Social support (Each Year)')
figwhat.show()


# In[ ]:


fig2 = px.scatter(tol, x = "Healthy life expectancy", y = 'Score', facet_row="year", color = "year", trendline= "ols", title = 'Score vs Healthy life expectancy (Each Year)')
fig2.show()


# In[ ]:


fig3 = px.scatter(tol, x = "Freedom to make life choices", y = 'Score', facet_row="year", color = "year", trendline= "ols", title = 'Score vs Freedom to make life choices (Each Year)')
fig3.show()


# In[ ]:


fig4 = px.scatter(tol, x = "Generosity", y = 'Score', facet_row="year", color = "year", trendline= "ols", title = 'Score vs Generosity (Each Year)')
fig4.show()


# In[ ]:


fig5 = px.scatter(tol, x = "Perceptions of corruption", y = 'Score', facet_row="year", color = "year", trendline= "ols", title = 'Score vs Perceptions of corruption (Each Year)')
fig5.show()


# In[ ]:


#seaborn.pairplot(tol,kind='reg',diag_kind = 'kde',palette='husl',hue='year')


# In[ ]:


def r_p(x,y):
    corr = ""
    if r > 0:
        if r > 0.5:
            if p < 0.05:
                corr = " which show they have a significant relationship with strong positive correlation."
            else:
                corr = " which show they have a strong positive correlation but not with a significant relationship."
        else:
            corr = " which show they have a significant difference with weak positive correlation."
    elif r < 0 :
        if r < -0.5 :
            if p < 0.05:
                corr = " which show they have a significant relationship with strong negative correlation."
            else:
                corr = " which show they have a strong negative correlation but not with a significant relationship."
        else:
            corr = " which show they have a significant difference with weak negative correlation."
    print("r value = ",round(r,3),"p value = ",round(p,3))
    print("")
    print(corr)


# In[ ]:


r, p = scipy.stats.pearsonr(wh2015["GDP per capita"],wh2015["Score"])
print('\33[95m'"2015")
r_p(r,p)


# In[ ]:


r, p = scipy.stats.pearsonr(wh2016["GDP per capita"],wh2016["Score"])
print('\33[92m'"2016")
r_p(r,p)


# In[ ]:


# and so on...


# In[ ]:


# Conclude: All graph have show a significant relationship 
#           with strong positive correlation
#           when calculating the relation of 
#           Score and GDP

# So: GDP and Happiness Score have a significant relationship 
#     with strong positive correlation


# In[ ]:


df = tol.copy()
del df['year']


# In[ ]:


seaborn.heatmap(df.corr(), annot=True).set_title('Correlation')
plt.show()


# In[ ]:


corr = df.corr()
dum = corr[['Score']]
seaborn.heatmap(dum,annot=True).set_title('Correlation (Score and Factors)')


# In[ ]:


corr = df.corr()**2
dum = corr[['Score']]
seaborn.heatmap(dum,annot=True).set_title('Coefficient of Determination (Score and Factors)')


# In[ ]:


dum = dum.drop(['Overall rank','Score'])


# In[ ]:


fig = px.line_polar(dum, r = 'Score', theta = dum.index,line_close=True,title = 'How the factors can explained the Happiness Score')
fig.show()


# In[ ]:


dum.index.name = 'Factors'
dum = dum.sort_values('Score', ascending=False)
fig = px.bar(dum,x = 'Score', y = dum.index,color = dum.index,
             orientation='h',color_discrete_sequence=px.colors.cyclical.HSV, width = 850
             ,title = 'Ranking (Most Influential -> Least Influential)')
fig.show()


# # Part 2

# # Influence of Covid-19 to Happiness Score 

# In[ ]:


dummytol = pd.concat([wh2021,wh2019,wh2018,wh2017,wh2016,wh2015])


# In[ ]:


wh2021['Score']


# In[ ]:


mean = {'Mean':[wh2021['Score'].mean(),wh2019['Score'].mean(),wh2018['Score'].mean(),
        wh2017['Score'].mean(),wh2016['Score'].mean(),wh2015['Score'].mean()]}

me_df = pd.DataFrame(mean,columns = ['Mean'],index = dummytol.year.unique())
me_df['Median'] = [wh2021['Score'].median(),wh2019['Score'].median(),wh2018['Score'].median(),
                     wh2017['Score'].median(),wh2016['Score'].median(),wh2015['Score'].median()]

me_df['year'] = me_df.index


# In[ ]:


plt.plot('year','Mean',data = me_df, marker = '.',color = 'blue',linewidth = 1)
plt.plot('year','Median',data = me_df, marker = 'o',color = 'black',linewidth = 2)
plt.title('The Change of Happiness Score in Mean and Median (2015-2020)')
plt.legend()
plt.show()


# In[ ]:


figint = px.box(dummytol, x = "year", y = 'Score', color = "year",color_discrete_sequence = px.colors.qualitative.Light24, title = 'Distribution of Happiness Score in various years')
figint.show()


# In[ ]:


data = [wh2021['Score'],wh2019['Score'],wh2018['Score'],wh2017['Score']
       ,wh2016['Score'],wh2015['Score']]
labels = ['2020','2019','2018','2017','2016','2015']

figinf = ff.create_distplot(data, labels, show_hist=True,show_curve=True,colors=px.colors.qualitative.Light24)
figinf.update_layout(title_text='Distribution of Happiness Score in various years')
figinf.show()


# In[ ]:


wh2021['Score'].mean()


# In[ ]:


wh2021['Score'].std()


# In[ ]:


tol['Score'].mean()


# In[ ]:


tol['Score'].std()


# In[ ]:


#from statistics import NormalDist

#print("% of overlapped Area = ")
#NormalDist(mu=wh2021['Score'].mean(), sigma=wh2021['Score'].mean()).overlap(NormalDist(mu=tol['Score'].mean(), sigma=tol['Score'].std()))


# In[ ]:


#from statistics import NormalDist

#print("% of overlapped Area = ")
#print(NormalDist(mu=5.532831544044034, sigma=1.0739225755800548).overlap(NormalDist(mu=5.3790179029986716, sigma=1.1274564601550128)))


# In[ ]:


wh2021dum = pd.read_csv('../input/world-happiness-report-2021/DataForFigure2.1WHR2021C2.csv',decimal=',')


# In[ ]:


wh2021dum.describe() # explained by:...


# In[ ]:


tol.describe() #2015-2019


# # Introduction (supplement work)

# In[ ]:


figla = px.scatter(dummytol, x = "Country or region", y = 'Score', color = "year", title = 'Distribution of Happiness Score in various years')
figla.show()


# In[ ]:


df2015 = wh2015.iloc[:10,:]
df2016 = wh2016.iloc[:10,:]
df2017 = wh2017.iloc[:10,:]
df2018 = wh2018.iloc[:10,:]
df2019 = wh2019.iloc[:10,:]


# Creating trace1
trace1 = go.Scatter(x = df2015['Country or region'],
                    y = df2015['Score'],
                    mode = "lines+markers",
                    name = "2015",
                    marker = dict(color = 'red'),
                    text= df2015['Country or region'])

# Creating trace2
trace2 = go.Scatter(x = df2015['Country or region'],
                    y = df2016['Score'],
                    mode = "lines+markers",
                    name = "2016",
                    marker = dict(color = 'blue'),
                    text= df2015['Country or region'])

# Creating trace3
trace3 = go.Scatter(x = df2015['Country or region'],
                    y = df2017['Score'],
                    mode = "lines+markers",
                    name = "2017",
                    marker = dict(color = 'green'),
                    text= df2015['Country or region'])

# Creating trace4
trace4 = go.Scatter(x = df2015['Country or region'],
                    y = df2018['Score'],
                    mode = "lines+markers",
                    name = "2018",
                    marker = dict(color = 'black'),
                    text= df2015['Country or region'])

# Creating trace5
trace5 = go.Scatter(x = df2015['Country or region'],
                    y = df2019['Score'],
                    mode = "lines+markers",
                    name = "2019",
                    marker = dict(color = 'pink'),
                    text= df2015['Country or region'])

data = [trace1, trace2, trace3, trace4, trace5]
layout = dict(title = 'Happiness Score of top 10 Countries from 2015 to 2019',
              xaxis= dict(title= 'Countries',ticklen= 5,),
              yaxis= dict(title= 'Happiness Score',ticklen= 5,),
              hovermode="x unified")
             
figla = dict(data = data, layout = layout)
iplot(figla)


# In[ ]:


dumtop = pd.concat([wh2019.head(10),wh2018.head(10),wh2017.head(10),wh2016.head(10),wh2015.head(10)])
dumtop['Country or region'].unique()


# In[ ]:


dumtop5 = pd.concat([wh2019.head(5),wh2018.head(5),wh2017.head(5),wh2016.head(5),wh2015.head(5)])
dumtop5['Country or region'].unique()


# In[ ]:


dumtop3 = pd.concat([wh2019.head(3),wh2018.head(3),wh2017.head(3),wh2016.head(3),wh2015.head(3)])
dumtop3['Country or region'].unique()


# In[ ]:


dumtop3['Country or region'].value_counts()


# #### Take the country that had the most frequency appear in top 3 be our final top 3

# In[ ]:


fintop3 = pd.concat([dumtop.loc[dumtop['Country or region'] == 'Denmark'],
                     dumtop.loc[dumtop['Country or region'] == 'Norway'],
                    dumtop.loc[dumtop['Country or region'] == 'Iceland']])


# In[ ]:


# Source from: The World Happiness Report

Hk = pd.DataFrame({'Country or region':['Hong Kong','Hong Kong','Hong Kong','Hong Kong','Hong Kong'],
                           'Score': [5.43,5.43,5.47,5.46,5.47],
                           'year': [2019,2018,2017,2016,2015]})


# In[ ]:


fintop3HK = pd.concat([fintop3,Hk])


# In[ ]:


figun = px.line(fintop3HK,x = 'year',y = 'Score',color = 'Country or region',
               title = 'Happiness Score of top 3 Countries vs Hong Kong from 2015 to 19',
               color_discrete_map={'Denmark':'black','Norway':'orange','Iceland':'red','Hong Kong':'magenta'})
figun.update_traces(mode='markers+lines')
figun.show()


# In[ ]:


print("This is the end!!!")

