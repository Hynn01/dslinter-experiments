#!/usr/bin/env python
# coding: utf-8

# # Data exploration on tweets about Messi and Ronaldo, by Ibrahim SEROUIS 💻

# # What to expect 🤔
# 
# In this Notebook, we'll explore the contents of the ["_final" version of the tweets collected here](https://www.kaggle.com/datasets/ibrahimserouis99/twitter-sentiment-analysis-and-word-embeddings). 
# 
# We are going to generate : 
# - **A wordcloud** representing the frequency of words 
# - **A stacked area chart** representing the total engagement by day 
# - **A bar chart** showcasing the most prevalent countries
# - **A bar chart** showcasing the contents length
# - **A pie chart** representing the most prevalent sources

# # Libraries

# In[ ]:


get_ipython().system('pip install --user wordcloud')


# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS
from PIL import Image


# # Load the files and setup 🛠

# ## Load

# In[ ]:


dataset_ronaldo = pd.read_csv("../input/twitter-sentiment-analysis-and-word-embeddings/ronaldo_final.csv", encoding="utf-8")
dataset_messi = pd.read_csv("../input/twitter-sentiment-analysis-and-word-embeddings/messi_final.csv", encoding="utf-8")


# ## Display samples

# ### Ronaldo

# In[ ]:


dataset_ronaldo.head(3)


# ### Messi

# In[ ]:


dataset_messi.head(3)


# ## Create the subsets 

# ### Messi subsets

# In[ ]:


messi_negatives = dataset_messi[dataset_messi["label"]=="Negative"]
messi_positives = dataset_messi[dataset_messi["label"]=="Positive"]

print(f"Messi\n----------------------------------------------------\n Negative tweets : {len(messi_negatives)} || Positive tweets: {len(messi_positives)}")


# ### Ronaldo subsets

# In[ ]:


ronaldo_negatives = dataset_ronaldo[dataset_ronaldo["label"]=="Negative"]
ronaldo_positives = dataset_ronaldo[dataset_ronaldo["label"]=="Positive"]

print(f"Ronaldo\n----------------------------------------------------\n Negative tweets : {len(ronaldo_negatives)} || Positive tweets: {len(ronaldo_positives)}")


# # Data exploration 🔍 

# ## Contents length 📄

# In[ ]:


# Create the plot
plt.figure(figsize=(14,5))

# Ronaldo subplot (1 lines, 2 columns, first column)
plt.subplot(1,2,1)
# Assign each tweet to its length : convert to an array the calculate the length
dataset_ronaldo["content"].str.split().map(lambda tweet: len(tweet)).hist(
    color="#00c979"
)
# Set the title 
plt.title("Content length : Ronaldo")

# Messi subplot (1 lines, 2 columns, second column)
plt.subplot(1,2,2)
# Assign each tweet to its length : convert to an array the calculate the length
dataset_messi["content"].str.split().map(lambda tweet: len(tweet)).hist(
    color="#00c979"
)
# Set the title 
plt.title("Content length : Messi")

# Display the results 
plt.show()


# ## Most used words : Word Cloud ☁️

# In[ ]:


# Get all the positive tweets as a single corpus 
messi_positives_text = " ".join(tweet for tweet in messi_positives.content)
ronaldo_positives_text = " ".join(tweet for tweet in ronaldo_positives.content)
# Get all the negative tweets as a single corpus 
messi_negatives_text = " ".join(tweet for tweet in messi_negatives.content)
ronaldo_negatives_text = " ".join(tweet for tweet in ronaldo_negatives.content)


# ### Setup the wordcloud
# 
# Most important parameters : 
# 
# - **Mask :** an 8-bit image array which will decide the "shape" of the word cloud. We'll use the one in the input files. 
# - **Stopwords :** a set of words that are filtered out of our analysis. The word cloud package already comes with the most usual stopwords. 

# In[ ]:


# Setup the mask 

# Load the image mask and convert to 8-bits (L = 8-bit pixels, black and white image)
mask = Image.open("../input/twitter-sentiment-analysis-and-word-embeddings/Visualization/cloud.png").convert("L")
# Create an array from the mask
mask = np.array(mask)


# ###### Create the wordclouds
# 
# Most important parameters : 
#  - **Mask :** the one we've set right above
#  - **Stopwords** : we're going to use the ones from the package and their respective names (Messi, CR7...). However, feel free to add more to the set of stopwords. 
#  - **Minimum word length :** Minimum number of letters a word must have to be included

# In[ ]:


# Add some stopwords for tweets about Messi
stopwords_messi = set(STOPWORDS)
stopwords_messi.add("messi")
# Add some stopwords for tweets about Ronaldo
stopwords_ronaldo = set(STOPWORDS)
stopwords_ronaldo.add("cr7")
stopwords_ronaldo.add("ronaldo")
stopwords_ronaldo.add("cristiano")


# In[ ]:


# Negative tweets about Messi
wordcloud_messi_negatives = WordCloud(stopwords=stopwords_messi,min_word_length=2,
                                      mask=mask,background_color="white",width=500, 
                                      contour_width=2, contour_color="black").generate_from_text(messi_negatives_text)
# Positive tweets about Messi
wordcloud_messi_positives = WordCloud(stopwords=stopwords_messi,min_word_length=2,
                                      mask=mask,background_color="white",width=500, 
                                      contour_width=2, contour_color="black").generate_from_text(messi_positives_text)
# Negative tweets about Ronaldo
wordcloud_ronaldo_negatives = WordCloud(stopwords=stopwords_ronaldo,min_word_length=2,
                                      mask=mask,background_color="white",width=500, 
                                      contour_width=2, contour_color="black").generate_from_text(ronaldo_negatives_text)

# Positive tweets about Ronaldo
wordcloud_ronaldo_positives = WordCloud(stopwords=stopwords_ronaldo,min_word_length=2,
                                      mask=mask,background_color="white",width=500, 
                                      contour_width=2, contour_color="black").generate_from_text(ronaldo_positives_text)


# In[ ]:


# Set the figure titles
titles = ["Negative words used against Messi", "Positive words used towards Messi", 
          "Negative words used against Ronaldo", "Positive words used towards Ronaldo"]

# Set the wordclouds list
wordclouds = [wordcloud_messi_negatives, wordcloud_messi_positives, 
             wordcloud_ronaldo_negatives, wordcloud_ronaldo_positives]


# ### Results

# In[ ]:


# Create the figure with 4 columns 
figure, axes = plt.subplots(nrows=1, ncols=4)
# Set the figure size
figure.set_size_inches(w=20, h=18)
# Increase the resolution 
plt.gcf().set_dpi(400)

# Loop through the axis and figures
for i in range(4):
    # Remove the axis (coordinates around the plot area)
    axes[i].axis("off")
    # Set the title accordingly
    axes[i].set_title(titles[i], fontweight="bold")
    # Show the corresponding wordcloud 
    axes[i].imshow(wordclouds[i])


# ## Engagements ❤️🔁

# ### Drop unncessary columns

# In[ ]:


stats_messi = dataset_messi.drop(columns=["tweet_id", "author_id", "content", "label", "lang", "prediction", "geo"])
stats_ronaldo = dataset_ronaldo.drop(columns=["tweet_id", "author_id", "content", "label", "lang", "prediction", "geo"])

stats_messi.head()


#  ### Convert the dates to a more convenient datetime format (yy-mm-dd)

# In[ ]:


# Convert to a date time object then format 
stats_messi["date"] = pd.to_datetime(stats_messi.date).dt.strftime("%y-%m-%d")
stats_ronaldo["date"] = pd.to_datetime(stats_ronaldo.date).dt.strftime("%y-%m-%d")
# Display some results
stats_messi.head()


# ### Engagements by day : Messi

# In[ ]:


# Plot stats by day
figure = stats_messi.groupby(by="date").sum().plot(
        kind="area", 
        figsize=(12,5), 
        title="Engagements by day : Messi\n\nRetweets, Retweets+Likes, Retweets+Likes+Quotes",
        grid=True, 
        cmap="tab20c"
)
# Set the background color
figure.set_facecolor("#ebebf3")
# Prevent matplotlib from showing exponential axis values (1e6, 1e5...)
plt.ticklabel_format(style="plain", axis="y")
# Display the stats
plt.show()


# ### Engagements by day : Ronaldo

# In[ ]:


# Plot stats by day
figure = stats_ronaldo.groupby(by="date").sum().plot(
        kind="area", 
        figsize=(12,5), 
        title="Engagements by day : Ronaldo\n\nRetweets, Retweets+Likes, Retweets+Likes+Quotes",
        grid=True, 
        stacked=True, 
        cmap="tab20c"
)
# Set the background color
figure.set_facecolor("#ebebf3")
# Prevent matplotlib from showing exponential axis values (1e6, 1e5...)
plt.ticklabel_format(style="plain", axis="y")
# Display the stats
plt.show()


# ## Data by country 🌍

# In[ ]:


# Drop null locations 
messi = dataset_messi.dropna(subset=["geo"])
ronaldo = dataset_ronaldo.dropna(subset=["geo"])
# Get the subsets containing contents with countries available 
messi_with_countries = messi[messi["geo"]!="-1"]
ronaldo_with_countries = ronaldo[ronaldo["geo"]!="-1"]
# Display some results 
ronaldo_with_countries.head(3)


# In[ ]:


# Get the top countries
top_countries_messi = messi_with_countries.geo.value_counts().to_dict()
top_countries_ronaldo = ronaldo_with_countries.geo.value_counts().to_dict()


# ### Top countries : Messi 

# In[ ]:


# Get the data
keys_messi = list(top_countries_messi.keys())
values_messi = list(top_countries_messi.values())

# Get the color map
color_map = plt.get_cmap("Paired")

# Setup the figure
plt.figure(figsize=(10,5), dpi=300)
# Set the title 
plt.title("Tweets by country : Messi\n\n *(When the location is available)")
# Setup the plot
plt.barh(keys_messi, values_messi, color=color_map.colors)
# Display results
plt.show()


# ### Top countries : Ronaldo

# In[ ]:


# Get the data
keys_ronaldo = list(top_countries_ronaldo.keys())
values_ronaldo = list(top_countries_ronaldo.values())

# Get the color map
color_map = plt.get_cmap("Paired")

# Setup the figure
plt.figure(figsize=(12,5), dpi=450)
# Set the title 
plt.title("Tweets by country : Ronaldo\n\n *(When the location is available)")
# Setup the plot
plt.barh(keys_ronaldo, values_ronaldo, color=color_map.colors)
# Display results
plt.show()


# ## Most popular sources 💻📱🖥

# In[ ]:


# Get the top sources
top_sources_messi = dataset_messi.source.value_counts().to_dict()
top_sources_ronaldo = dataset_ronaldo.source.value_counts().to_dict()


# In[ ]:


# Get the data
keys_messi = list(top_sources_messi.keys())
values_messi = list(top_sources_messi.values())


# Setup the figure
plt.figure(figsize=(12,5), dpi=105)
# Set the colors 
colors = ["#ff5959","#5ffac9","#7dff7d", "#faed5f"]
# Set the title 
plt.title("Top 4 sources : Messi")
# Setup the plot
plt.pie(
    values_messi[0:4], 
    labels=keys_messi[0:4], 
    autopct="%1.1f%%", 
    shadow=True, 
    colors=colors
)
# Draw the hollow part of the pie
centre_circle = plt.Circle(xy=(0,0),radius=0.80,facecolor="white")
# Get the current figure 
current_figure = plt.gcf()
# Get the current axis and add the hollow part to the figure
current_figure.gca().add_artist(centre_circle)
# Display results
plt.show()


# ### Many thanks to [this Medium article](https://medium.com/@kvnamipara/a-better-visualisation-of-pie-charts-by-matplotlib-935b7667d77f) for the tips. (^_^)

# In[ ]:


# Get the data
keys_ronaldo = list(top_sources_ronaldo.keys())
values_ronaldo = list(top_sources_ronaldo.values())

# Setup the figure
plt.figure(figsize=(12,5), dpi=105)
# Set the colors 
colors = ["#ff5959","#5ffac9","#7dff7d", "#faed5f"]
# Set the title 
plt.title("Top 4 sources : Ronaldo")
# Setup the plot
plt.pie(
    values_ronaldo[0:4], 
    labels=keys_ronaldo[0:4], 
    autopct="%1.1f%%", 
    shadow=True, 
    colors=colors
)
# Draw the hollow part of the pie
centre_circle = plt.Circle(xy=(0,0),radius=0.80,facecolor="white")
# Get the current figure 
current_figure = plt.gcf()
# Get the current axis and add the circle
current_figure.gca().add_artist(centre_circle)
# Display results
plt.show()


# # Thank you for your time 😄
