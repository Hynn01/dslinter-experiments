#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn; seaborn.set()


# In[ ]:


PlayStore = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")


# In[ ]:


PlayStore.head()


# In[ ]:


print("PlayStore shape is ",PlayStore.shape)


# In[ ]:


missing_data = pd.concat([PlayStore.isnull().sum().sort_values(ascending=False),
                          ((PlayStore.isnull().sum()/10841)*100)], 
                         axis=1, keys=['Total', 'Percent'])
missing_data.head()


# # Drop all rows which contains NA/Null value
# 

# In[ ]:


##Drop all rows which contains NA/Null value
PlayStore.dropna(inplace= True)
print('Number of apps in the dataset : ' , len(PlayStore))


# In[ ]:


#PlayStore = PlayStore.set_index(PlayStore['App'])
PlayStore['Last Updated'] = pd.to_datetime(PlayStore['Last Updated'])


# # Convert data type 
# 

# In[ ]:


PlayStore['Installs'] = PlayStore['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
PlayStore['Installs'] = PlayStore['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
PlayStore['Installs'] = PlayStore['Installs'].apply(lambda x: int(x))
print(PlayStore.dtypes['Installs'])


# In[ ]:


PlayStore['Price'] = PlayStore['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
PlayStore['Price'] = PlayStore['Price'].apply(lambda x: float(x))


# ## Convert 'Size' to MB
# ## PlayStore['Price'] contains 'M', 'k' character and 'Varies with device' string
# ## And it should be a floating point data type 

# In[ ]:


PlayStore['Size'] = PlayStore['Size'].apply(lambda x: str(x).replace('M',' ') if 'M' in str(x) else x)
PlayStore['Size'] = PlayStore['Size'].apply(lambda x: str(x).replace(',' , ' ') if ',' in str(x) else x)
PlayStore['Size'] = PlayStore['Size'].apply(lambda x: str(x).replace('Varies with device' , 'NaN') 
                                            if 'Varies with device' in str(x) else x)
PlayStore['Size'] = PlayStore['Size'].apply(lambda x: float(str(x).replace('k' , ' '))/1000 if 'k' in str(x) else x)
PlayStore['Size'] = PlayStore['Size'].apply(lambda x: float(x))


# ## Convert PlayStore['Reviews'] to int data types
# 

# In[ ]:


PlayStore['Reviews'] = PlayStore['Reviews'].apply(lambda x: int(x))


# In[ ]:


PlayStore.info()


# In[ ]:


print("Before drop PlayStore['Size'] null value shape",PlayStore.shape)
PlayStore.dropna(inplace= True)
print("After drop PlayStore['Size'] null value shape",PlayStore.shape)

temp = (7723-10841)/10841 *100
print("We have drop",abs(temp), "% data")


# # Data Analysis

# In[ ]:


seaborn.pairplot(PlayStore,kind='reg',diag_kind = 'kde',palette='husl',hue='Type')


# # Now let take a look of Free App and Paid Apps' percentage
# 

# In[ ]:



sizes = PlayStore['Type'].value_counts(sort = True)
labels = PlayStore['Type'].value_counts(sort = True).index
colors = ['deeppink','mediumorchid']
explode = (0,0.2)
pie, ax = plt.subplots(figsize=[10,6])
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.2f%%',  startangle=180,)
plt.axis='equal'
plt.legend(loc = 'best')
plt.title('Percent of Two types of App in store',size = 20)


# # First the number of Free App is **absolutely overwhelming** with Paid App 
# ## And ***why*** would it happen ?

# In[ ]:


PlayStore.describe().apply(lambda s: s.apply('{0:.5f}'.format))


# # As the "Reviews" and "Installs" columns' S.D is large so we take log on both columns

# In[ ]:


# We can see the data S.D are very large especially 'Reviews' and 'Installs'
# We may need to use log scale to visualize the data set
x = PlayStore['Rating']
y = PlayStore['Size']
z = np.log(PlayStore['Installs'])
p = np.log(PlayStore['Reviews'])
t = PlayStore['Type']
price = PlayStore['Price']

LogScarlePD = pd.DataFrame(list(zip(x, p, y , z, t, price)), 
                        columns=['Rating','Reviews', 'Size', 'Installs', 'Type', 'Price'])
#p = seaborn.pairplot(LogScarlePD, kind='reg',hue='Type', palette="husl",markers=['o','D'])


# In[ ]:


Free_App = LogScarlePD[LogScarlePD['Type']=="Free"].copy()
Paid_App = LogScarlePD[LogScarlePD['Type']=="Paid"].copy()


# In[ ]:


plt.figure(figsize=(15,6))
z = seaborn.boxplot(x="Installs", y="Type", data=LogScarlePD,orient='h')
plt.title('Free VS Paid on Installation',size = 20)


# ## In the log scarle, **the lowest 25% installation** of the Free App are nearly highest 75% Paid Apps' installation
# > ### Which means obviously Free App are installed more than Paid App

# In[ ]:


mask = np.triu(np.ones_like(LogScarlePD.corr()))
seaborn.heatmap(LogScarlePD.corr(),annot=True,mask =mask)
plt.title('The Overall Market Correlation')


# 
# # **"Price"** does not have much correlation whith others therefore when **"Price"** increases **Rating** would not also increase.  
# 

# In[ ]:


plt.figure(figsize=(15,6))
z = seaborn.boxplot(x="Rating",y="Type", data=LogScarlePD,orient='h')
plt.title('Rating',size = 20)


# In[ ]:


LogScarlePD.loc[ LogScarlePD['Price'] == 0, 'PriceBand'] = '1 Free'
LogScarlePD.loc[(LogScarlePD['Price'] > 0) & (LogScarlePD['Price'] <= 7.99), 'PriceBand'] = '2 Cheap'
LogScarlePD.loc[(LogScarlePD['Price'] > 7.99) & (LogScarlePD['Price'] <= 15.99), 'PriceBand']   = '3 Normal'
LogScarlePD.loc[(LogScarlePD['Price'] > 15.99) & (LogScarlePD['Price'] <= 29.99), 'PriceBand']   = '4 expensive'
LogScarlePD.loc[(LogScarlePD['Price'] > 29.99), 'PriceBand']  = '5 Too Expensive'

plt.figure(figsize=(15,6))

z = seaborn.boxplot(x="PriceBand", y="Rating", data=LogScarlePD)
plt.title('Rating VS PriceBand',size = 20)


# ## The result is quite similar to the previous heat map 
# ## Correlation between "Price" and "Rating" is -0.021
# ## Here we simply divide all the app to 5 boundaries, from Cheap to Expensive boundaries app gets better Rating than Free App **unless** it prices **too Expensive**

# ## Paid app has higher rating than Free app it may because users need to pay to fee and they would prefer a higher quality or best suitable app for them. 
# 

# In[ ]:


Category_label = PlayStore['Category'].value_counts(sort = True)[:10]
Category_list = list(Category_label.index)
print(Category_list)
FAMILY = pd.DataFrame(data = PlayStore[PlayStore['Category']=='FAMILY'])
GAME = pd.DataFrame(data = PlayStore[PlayStore['Category']=='GAME'])
TOOLS = pd.DataFrame(data = PlayStore[PlayStore['Category']=='TOOLS'])
MEDICAL = pd.DataFrame(data = PlayStore[PlayStore['Category']=='MEDICAL'])
LIFESTYLE = pd.DataFrame(data = PlayStore[PlayStore['Category']=='LIFESTYLE'])
PERSONALIZATION = pd.DataFrame(data = PlayStore[PlayStore['Category']=='PERSONALIZATION'])
FINANCE = pd.DataFrame(data = PlayStore[PlayStore['Category']=='FINANCE'])
SPORTS = pd.DataFrame(data = PlayStore[PlayStore['Category']=='SPORTS'])
BUSINESS = pd.DataFrame(data = PlayStore[PlayStore['Category']=='BUSINESS'])
PHOTOGRAPHY = pd.DataFrame(data = PlayStore[PlayStore['Category']=='PHOTOGRAPHY'])
top10 = pd.concat([FAMILY,GAME,TOOLS,MEDICAL,LIFESTYLE,PERSONALIZATION,FINANCE,SPORTS,BUSINESS,PHOTOGRAPHY])
g = seaborn.FacetGrid(top10,col='Category',hue='Type',col_wrap=2,height=4, aspect=3)
g.map(seaborn.histplot, "Rating")
g.add_legend()
g.set(xticks=[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])


# # Paid App are obviously less in the "LIFESTYLE","Finance","Business", "Sports" and "Photography" category
# # But they are more often on "FAMILY", "GAME", "TOOLS", "MEDICAL", "PERSONALIZATION"
# # It is surprisingly, on "Personalization" Category the number of Paid App get high rating is more than Free App.
# 

# In[ ]:


plt.figure(figsize=(15,6))
z = seaborn.boxplot(x="Size", y="Type", data=LogScarlePD,orient='h')
plt.title('Free VS Paid on Apps\' Size',size = 20)


# # In general the Apps' Size between Free App and Paid App are **no different**
# # Firms who make Paid App need to pay the same effort as Free App but result in lower installation  

# ---------------------------------------------------------------------------------------------------------------------------
# # Here will use another CSV file "googleplaystore_user_reviews.csv"

# In[ ]:


seaborn.displot(data= LogScarlePD, x = 'Reviews', hue ='Type',kind='kde')


# # Paid App has less reviews than Paid App

# In[ ]:


review_df = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")


# In[ ]:


PlayStore_cp = PlayStore.copy()


# In[ ]:


merged_df = pd.merge(PlayStore_cp, review_df, on ="App", how = "inner")
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])


# In[ ]:


fig, ax = plt.subplots(figsize=(17,13))
ax = seaborn.boxplot(x='Type', y='Sentiment_Polarity', data=merged_df)
title = ax.set_title('Sentiment Polarity Distribution',fontsize =15)


# # Obviously, in general Paid App have **less unsatisfied** than Free App from their sentiment Polarity

# In[ ]:


from wordcloud import WordCloud
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


wc = WordCloud(background_color="white", max_words=200, colormap="viridis")
stop = stopwords.words('english')


# In[ ]:


stop = stop + ['app', 'APP' ,'ap', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',
            'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher']

merged_df['Translated_Review'] = merged_df['Translated_Review'].apply(lambda x: " ".join(x for x in str(x).split(' ') if x not in stop))
merged_df.Translated_Review = merged_df.Translated_Review.apply(lambda x: x if 'app' not in x.split(' ') else np.nan)
merged_df.dropna(subset=['Translated_Review'], inplace=True)


# In[ ]:


free = merged_df.loc[merged_df.Type=='Free']['Translated_Review'].apply(lambda x: '' if x=='nan' else x)
wc.generate(''.join(str(free)))
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.show()


# # Negative word: ads, excessive, hate, Bad, Kid, Boring

# In[ ]:


paid = merged_df.loc[merged_df.Type=='Paid']['Translated_Review'].apply(lambda x: '' if x=='nan' else x)
wc.generate(''.join(str(paid)))
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.show()


# # To conclude Why the nunber of Pree App is greater than the nunber of Paid App
# > ## 1. **"Price"** does not have much correlation whith others therefore when **"Price"** increases **Rating** would also be increased.  
# > ## 2. In general the Apps' Size between Free App and Paid App are **not different**
# >> ## Firms who make Paid App need to pay the same effort as Free App but result in lower installation  
# >> ## The Q1 of installation of Free App is nearly equal to Q3 of installation of Paid App
# > ## 3. Free App can have put Ads inside their apps for addictional Ads income 
# # So, Free App can **probably earn more profit** than Paid App by additional Ads and greater installtion. And it is **incentive** for firms to make Free App rather than Paid App
# # However, there is another interesting finding about Paid App.
# > ## 1. In Rating the average rating of Paid App and Free App, Paid App has perfromed slightly better than Free App
# > ## 2. In Sentiment Polarity Distribution, many Free Apps are recorded in extremely low result but Paid App only has one outside the lowest. 
# >> ## And the median of Free App is behind Paid Apps 
# > ## 3. In WordCloud shows that Free Apps have more negative words such as Ads, excessive, hate, Bad, Kid, Boring, but Paid Apps seem not have negative words.
# >> ## We can draw one possible situtation is that many firms produce boring free app to deceive users downloading the app, and watching Ads.
# >> ## So, firms can earn a substantial amount of money 
# # Therefore, Firms who makes Paid App are more likely to focus on a niche market targeting a small group of user on the whole Google Play App Store

# In[ ]:




