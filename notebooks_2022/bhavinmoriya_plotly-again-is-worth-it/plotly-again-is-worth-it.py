#!/usr/bin/env python
# coding: utf-8

# We are going to learn using Plotly to create interactive and awesome looking plots. And it is extremely easy and super intuitive to produce it. So nothing to lose :) Thanks for being here :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = px.data.gapminder(datetimes=True)
df


# # Bar, Box, Histogram, Scatter and other plots

# In[ ]:


df.query('year == 2007')[:10].sort_values(by='lifeExp')


# In[ ]:


fig = px.histogram(df.query('year == 2007'), x='continent', y='pop', color='continent', hover_name='country')
fig.show()
#fig.add_hline(12420476	+ 31889923)


# In[ ]:


fig = px.bar(df.query('year == 2007'), x='continent', y='pop', color='lifeExp', hover_name='country', )
fig.show()


# In[ ]:


df.query('year == 2007')[df.query('year == 2007').continent == 'Oceania']


# In[ ]:


df7 = df.query('year == 2007')
df7


# In[ ]:


px.bar(df7, x='continent', y ='pop', hover_name='country', color='country')


# So we are assining color to a person based on the country he belongs to. Well, this concept would fit better with sunburst.

# In[ ]:


px.sunburst(df7, path=['continent', 'country'], values='pop', hover_name='country', color='lifeExp', height=900)


# # Kind of similar to sunburst is treemap.

# In[ ]:


px.treemap(df7, path=['continent', 'country'], values='pop', hover_name='country', color='lifeExp', height=900)


# In[ ]:


px.choropleth(df7, locations='iso_alpha', hover_name='country', hover_data=df.columns, color='lifeExp', height=900, 
              projection='natural earth')


# In[ ]:


px.scatter(df7, x='gdpPercap', y='lifeExp', log_x=True, size='pop', size_max=65, color='continent', hover_data=df7.columns,
          hover_name='country', title='Life Expectancy vs GDP', template='plotly_dark',
          labels=dict(gdpPercap="GDP", lifeExp='Life Expectancy')).update_xaxes(tickprefix='$')


# In[ ]:


df.info()


# # A **year** column seem to be causing error which I could not figure out (would appreciate any help). To settle the proble I had created **Year** column.

# In[ ]:


df['Year'] = df.year.dt.year
df


# In[ ]:


px.scatter(df, x='gdpPercap', y='lifeExp', log_x=True, size='pop', size_max=65, color='continent', hover_data=df.columns,
          hover_name='country', animation_frame='Year', animation_group='country', range_y=[20,100], height=900)


# If you do not want animation, you could instead create yearwise scatter plots.

# In[ ]:


px.scatter(df, x='gdpPercap', y='lifeExp', log_x=True, size='pop', size_max=65, color='continent', hover_data=df.columns,
          hover_name='country', facet_col='Year', facet_col_wrap=4)


# In[ ]:


px.line(df[df.year == '2007'], y='lifeExp')


# In[ ]:


px.box(df, x='continent', y='lifeExp', color='continent')


# In[ ]:


px.violin(df, x='continent', y='lifeExp', box=True, color='continent')


# In[ ]:


px.line(df.query("continent == ['Asia'] and pop > 102e6"), 'year', 'pop', color='country')


# In[ ]:


new = df.groupby('continent').mean()
display(new)
px.bar(new, x=new.index, y='lifeExp')


# In[ ]:


px.box(new, y='lifeExp')


# In[ ]:


px.bar(df, y='continent', x='pop')


# In[ ]:


px.histogram(df, 'pop')


# In[ ]:


px.scatter(df, x='gdpPercap', y='lifeExp', color='continent')


# # Adding more complexity

# In[ ]:


px.scatter(df, x='gdpPercap', y='lifeExp', color='continent',
          marginal_x='histogram', marginal_y='box', trendline='ols')


# In[ ]:


px.scatter(df, x='gdpPercap', y='lifeExp', color='continent', hover_name='country')


# In[ ]:


px.scatter(df, x='gdpPercap', y='lifeExp', color='continent', hover_name='country', log_x=True)


# In[ ]:


df.continent.unique()


# In[ ]:


px.scatter(df.query('continent == "Europe"'), x='gdpPercap', y='lifeExp', hover_name='country', size='pop')


# In[ ]:


px.scatter(df.query('year < 2010'), x='gdpPercap', y='lifeExp', hover_name='country', size='pop', facet_col='continent')


# In[ ]:


df


# In[ ]:


df_year = df.groupby('year').mean()
df_year


# In[ ]:


df_year['Year'] = df_year.index.year
df_year['Country'] = df.country.unique()[:len(df_year)]
df_year


# In[ ]:


px.scatter(df_year, size='lifeExp', y='pop', x='gdpPercap', animation_frame='Year', animation_group='Country',
          hover_name='Country', size_max=45, color='Country',
          range_x=[100,13e3], range_y=[0,5e7])


# Thanks to, https://www.youtube.com/watch?v=_b2KXL0wHQg for an awesome video.

# In[ ]:


px.pie(df, values='lifeExp', names='continent')


# In[ ]:


px.pie(df, values='lifeExp', names='continent', hole=.2)


# In[ ]:


import pandas as pd

bird = pd.read_csv('../input/bird-window-collison/bird-window-collision-death.csv')
bird


# In[ ]:


fig = px.pie(bird, values='Deaths', names='Bldg #', color='Side')
fig.update_traces(textinfo='label+percent', insidetextfont=dict(color='white'))
fig.update_layout(legend=dict(itemclick=False))
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




