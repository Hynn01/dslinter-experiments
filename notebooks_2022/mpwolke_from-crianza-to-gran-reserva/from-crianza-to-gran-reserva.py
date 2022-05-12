#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import plotly
plotly.offline.init_notebook_mode(connected=True)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: red;"><b style="color:yellow;">From Crianza to Gran Reserva</b></h1></center>
# 
# By Madeline Puckette
# 
# "The fortress known as Rioja Wine is built on a foundation of Tempranillo. This grape is hugely popular around the world but its true homeland is on the Iberian Peninsula."
# 
# "Rioja has a strict set of rules regarding its classification, and these produce a beautiful result: a big wine with high tannin that pairs up well with meat, delivering structure and fruitiness."
# 
# "Rioja Wine enjoys a broad selection of flavors depending on how its made and aged. Across all versions you can expect a big flavor wine with high tannins – younger wines have zippy fruit, while the longer aged wines pick up oak flavors from the barrel."
# 
# https://winefolly.com/deep-dive/rioja-wine-from-crianza-to-gran-reserva/

# In[ ]:


df = pd.read_csv("/kaggle/input/spanish-wine-quality-dataset/wines_SPA.csv", delimiter=',', encoding='ISO-8859-2')
pd.set_option('display.max_columns', None)
df.head()


# In[ ]:


df.isnull().sum()


# <h1><span class="label label-default" style="background-color:red;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:yellow; padding:10px">Young Wine</span></h1><br>
# 
# By Madeline Puckette
# 
# "Wines in their first or second year, which keep their primary freshness and fruitiness."
# 
# "Rioja used to be called “vin joven” which literally means “young wine.” Now when a wine is labeled Rioja you can assume it’s a base-model Tempranillo."
# 
# "These wines don’t have the tannin structure or oak flavors that are common in the higher end wines."
# 
# "What they don’t have in structure they make up for in zippy fruit."
# 
# https://winefolly.com/deep-dive/rioja-wine-from-crianza-to-gran-reserva/

# In[ ]:


df["wine"].value_counts()


# <h1><span class="label label-default" style="background-color:red;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:yellow; padding:10px">Crianza</span></h1><br>
# 
# By Madeline Puckette
# 
# "A minimum of one year in casks and a few months in the bottle. For white wines, the minimum cask aging period is 6 months."
# 
# "Crianza is perhaps the most accessible level of Rioja wines, especially since most can be found for less than 15 euros."
# 
# "At the Crianza level, the wines are most commonly aged in used oak, so the oak flavors are not as strong. The goal of Crianza is a high-quality daily drinking wine."
# 
# "It’s not too rich, but with Tempranillo’s natural high tannin it has quite a bit more body than Merlot."
# 
# https://winefolly.com/deep-dive/rioja-wine-from-crianza-to-gran-reserva/

# In[ ]:


df["region"].value_counts()


# ![](https://media.winefolly.com/Rioja-Wine-Classification-aging-reserva-crianza-winefolly.png)https://winefolly.com/deep-dive/rioja-wine-gets-a-new-classification-system/

# <h1><span class="label label-default" style="background-color:red;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:yello; padding:10px">Reserva</span></h1><br>
# 
# By Madeline Puckette
# 
# "Selected Red wines of the best vintages with an excellent potential that have been aged for a minimum of 3 years, with at least one year in casks."
# 
# "This is where Rioja tastes serious. At the Reserva level, winemakers often age their wines longer than the minimum and select better grapes. Many Rioja wine enthusiasts swear by Reserva level because they are a medium between super fruity Crianza and oakey-bottle-aged Gran Reserva."
# 
# https://winefolly.com/deep-dive/rioja-wine-from-crianza-to-gran-reserva/

# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'red',
                      height =2000,
                      width = 2000
                     ).generate(str(df["wine"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Spanish Wines Classification")
plt.show()


# In[ ]:


df["price"].value_counts()


# <h1><span class="label label-default" style="background-color:red;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:yellow; padding:10px">Gran Reserva</span></h1><br>
# 
# By Madeline Puckette
# 
# "Selected wines from exceptional vintages which have spent at least 2 years in oak casks and 3 years in the bottle. For white wines, the minimum aging period is 4 years, with at least one year in casks."
# 
# "What’s interesting about Gran Reserva is that most winemakers select the best grapes for this level and age them for as long as the wine needs. This means most of the new release Gran Reservas are around 10 years old or older when you first see them available."
# 
# https://winefolly.com/deep-dive/rioja-wine-from-crianza-to-gran-reserva/

# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'yellow',
                      height =2000,
                      width = 2000
                     ).generate(str(df["region"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Spanish Wines Regions")
plt.show()


# In[ ]:


df["type"].value_counts()


# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'red',
                      height =2000,
                      width = 2000
                     ).generate(str(df["type"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Spanish Wines Types")
plt.show()


# In[ ]:


#Code by Siti K https://www.kaggle.com/khotijahs1/2020-indonesia-university-rank/comments

#The Best 20 Wineries by Rating
top_winery = df.sort_values(by='rating', ascending=False)[:20]
figure = plt.figure(figsize=(10,6))
sns.barplot(y=top_winery.winery, x=top_winery.rating)
plt.xticks()
plt.xlabel('Ratings')
plt.ylabel('Wineries')
plt.title('Top 20 Wineries by Rating')
plt.show()


# In[ ]:


#Codes by Pooja Jain https://www.kaggle.com/jainpooja/av-guided-hackathon-predict-youtube-likes/notebook

text_cols = ['winery', 'wine', 'region', 'type']

from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(stopwords = set(list(STOPWORDS) + ['|']),colormap='YlOrRd', background_color="Red", random_state = 42)
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = [ax for axes_row in axes for ax in axes_row]

for i, c in enumerate(text_cols):
  op = wc.generate(str(df[c]))
  _ = axes[i].imshow(op)
  _ = axes[i].set_title(c.upper(), fontsize=24)
  _ = axes[i].axis('off')

#_ = fig.delaxes(axes[3])


# In[ ]:


#Code by Siti K https://www.kaggle.com/khotijahs1/2020-indonesia-university-rank/comments

#The Best 20 Wineries by Rating
top_wine = df.sort_values(by='rating', ascending=False)[:20]
figure = plt.figure(figsize=(10,6))
sns.barplot(y=top_wine.wine, x=top_wine.rating)
plt.xticks()
plt.xlabel('Ratings')
plt.ylabel('Wines')
plt.title('Top 20 Wines by Rating')
plt.show()


# #Vega Sicilia Winery
# 
# Congrats Unico wine for all your Medals

# ![](https://i0.wp.com/cervejasevinhos.com/cv/wp-content/uploads/2020/08/Unico-Valbuena.jpg?resize=750%2C471&ssl=1)cervejasevinhos.com

# In[ ]:


vega = df[(df['winery']=='Vega Sicilia')].reset_index(drop=True)
vega.head()


# In[ ]:


import plotly.express as px

fig = px.scatter(vega, x="year", y="price", color="num_reviews", title='Vega Sicilia Prices and Reviews')
fig.show()


# In[ ]:


fig = px.scatter(df, x="type", y="price", color="num_reviews", title='Wine Types Prices and Reviews')
fig.show()


# In[ ]:


fig = px.scatter(df, x="type", y="body", color="acidity", title='Wine Types Body and Acidity')
fig.show()


# In[ ]:


fig = px.parallel_categories(df, color="rating", title= 'Spanish Wine Ratings', color_continuous_scale=px.colors.sequential.OrRd)
fig.show()


# In[ ]:


fig = px.line(vega, x="year", y="rating", color_discrete_sequence=['darksalmon'], 
              title="Vega Sicilia Ratings Yearly")
fig.show()


# In[ ]:


fig = px.bar(df, 
             x='type', y='body', color_discrete_sequence=['crimson'],
             title='Spanish Wines Ratings', text='rating')
fig.show()


# In[ ]:


fig = px.bar(df, 
             x='type', y='price', color_discrete_sequence=['crimson'],
             title='Spanish Wines Reviews and Price', text='num_reviews')
fig.show()


# #Can you see any similarity between Wine and Kagglers?
# 
# "Young wines have zippy fruit. What they don’t have in structure they make up for in freshness."
# 
# "Crianza is the most accessible level. At the Crianza level, flavors are not as strong."
# 
# Reserva: "Selected Red wines of the best vintages with an excellent potential that have been aged for a minimum of 3 years, with at least one year in casks."
# 
# Gran Reserva: "Selected wines from exceptional vintages which have spent at least 2 years in oak casks and 3 years in the bottle. 
# 
# "What’s interesting about Gran Reserva is that most winemakers select the best grapes for this level and age them for as long as the wine needs. This means most of the new release Gran Reservas are around 10 years old or older when you first see them available."
# 
# 
# #In summary: Crianza are the most accessible level and Gran Reservas are around 10 years old or older when you first see them available."
