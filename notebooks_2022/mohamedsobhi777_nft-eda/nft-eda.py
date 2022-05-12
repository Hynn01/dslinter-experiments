#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset = pd.read_csv("../input/nft-art-dataset/dataset/dataset.csv")
dataset.head()

list(dataset.columns)
dataset.head()


# **Initial EDA**
# 
# 
# 
# Take a look at the images and try out possible image processing technique
# 
# 
# Boilerplate

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
from spectral import *
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pal = sns.color_palette()
sns.set_style("whitegrid")


# Files

# In[ ]:


print('# File sizes')
folder = '../input/nft-art-dataset/dataset/'

for f in os.listdir(folder):
    if not os.path.isdir(folder + f):
        print(f.ljust(30) + str(round(os.path.getsize(folder + f) / 1000000, 2)) + 'MB')
    else:
        sizes = [os.path.getsize(folder+f+'/'+x)/1000000 for x in os.listdir(folder + f)]
        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))


# In[ ]:


train_df = pd.read_csv(folder + "dataset.csv")
train_df.head()
train_df = dataset [ dataset['type'] == 'PHOTO' ] [["name", "path"]]
train_df


# **View some JPEGs**

# In[ ]:


import cv2

new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
i = 0
st = 200
for f, l in train_df[st:st+9].values:
    file = folder + "image/" + l.split('/')[-1]
    img = cv2.imread(file)    
    up_width = 600
    up_height = 400
    up_points = (up_width, up_height)

    img2 = cv2.resize(img, up_points, interpolation = cv2.INTER_LINEAR)
    
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[i // 3, i % 3].set_title('{}'.format(f))
#     ax[i // 4, i % 4].show()
#     print('../input/train-tif/{}.tif'.format(f))
    i += 1


# **Basic Information**

# In[ ]:



total = len(dataset)
photos = len(dataset[ dataset['type'] == 'PHOTO' ])
gifs = len(dataset[ dataset['type'] == 'GIF' ])
videos = len(dataset[ dataset['type'] == 'VIDEO' ])


print(f"There are {total} records in the dataset in total, in the form of gifs, images, and videos.")
print(f"There are {photos} photos in the dataset.")
print(f"There are {gifs} gifs in the dataset.")
print(f"There are {videos} videos in the dataset")
print("\n")

total_creators = len(dataset['creator'].unique())
photo_creators = len(dataset[ dataset['type'] == 'PHOTO']['creator'].unique())
gif_creators = len(dataset[ dataset['type'] == 'GIF']['creator'].unique())
video_creators = len(dataset[ dataset['type'] == 'VIDEO']['creator'].unique())

print(f"The dataset contains the works of {total_creators} unique artists in total.")
print(f"The photos artwork is made by {photo_creators} unique artists in total.")
print(f"The gifs artwork is made by {gif_creators} unique artists in total.")
print(f"The videos artwork is made by {video_creators} unique artists in total.")
print("\n")

total_art_series = total_creators = len(dataset['art_series'].unique())
print(f"There are {total_art_series} art serieses in total")


# # Dataset Content

# In[ ]:



counts = dataset['type'].value_counts()
#define data
data = list(counts.values)
labels = list(counts.index)

#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:5]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title("Types of Artwork")
plt.show()


# # The likes counter

# In[ ]:


likes = dataset["likes"].value_counts()
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")

for i in range(0, 10 ):
    print(f"There are {list(likes.values)[i]} art piece(s) that has {list(likes.index)[i]} likes.")

ax = sns.barplot(x=list(likes.index), y=list(likes.values) )


# **Prices distribution**

# In[ ]:


prices = dataset[['price','year']]
sns.displot(prices.price.values, bins = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000]).set(title="Prices in Hive, 1 Hive = 0.9359 USD")


# **Price Statistics**

# In[ ]:


print(f"The average price of all art pieces is { '{0:.1f}'.format(prices.price.mean()) } coins")
print(f"The median price of all art pieces is { '{0:.1f}'.format(prices.price.median()) } coins")
print(f"The standard deviation of the prices is { '{0:.1f}'.format(prices.price.std()) } coins")
print("\n\n")

sum,cnt = 0, 0 
lst = []

for year in range(2017, 2022):
    min_price = prices[ prices['year'] == year ].price.min()
    max_price = prices[ prices['year'] == year ].price.max()
    
    art_count = len(prices[ prices['year'] == year ])
    sorted_prices = sorted(prices[(prices['year'] == year)].price)

    low_1 = sorted_prices[ int(art_count*0.01) ]
    top_1 = sorted_prices[ int(art_count*0.99) ]
        
    data_without_outliers = prices[ (prices['year'] == year) & (prices['price'] > low_1) & (prices['price'] < top_1) ]
    sum += data_without_outliers.price.sum()
    cnt += len(data_without_outliers)
    lst = lst + list(data_without_outliers.price)
    
    print(f"Number of art pieces in {year} is { len(data_without_outliers) }")
    print(f"The average price of art pieces in {year} is { '{0:.1f}'.format(data_without_outliers.price.mean()) }")
    print(f"The median price of art pieces in {year} is { '{0:.1f}'.format(data_without_outliers.price.median()) }")
    print("\n")

print(f"In total there are {len(lst)} piece(s), with average {'{0:.1f}'.format(np.mean(lst))}, and median {'{0:.1f}'.format(np.median(lst))}")


# **The year the artpieces were creater**

# The Kernel Density Estimate (KDE) shows that most of the art pieces were created between 2018 and 2021

# In[ ]:


year_dataset = dataset[ (dataset['year'] >= 2017) & (dataset['year'] <= 2021) ]
# sns.kdeplot(data=year_dataset, x="year")


year_freq = year_dataset['year'].value_counts()

ax = sns.barplot(x=list(year_freq.index), y=list(year_freq.values) )

