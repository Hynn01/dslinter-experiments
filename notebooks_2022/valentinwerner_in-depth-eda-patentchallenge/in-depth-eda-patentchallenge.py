#!/usr/bin/env python
# coding: utf-8

# Dear Reader, please note that this notebook was written in Jupyter Notebook and as such some features (particularly graphs) may not work on kaggle - I have marked the respective parts and generally recommend downloading and opening in Jupyter Notebook to be able to follow my thoughts behind some graphs 

# In[ ]:


import warnings
warnings.filterwarnings("ignore") #can get annoying and visually distracting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import seaborn as sns

import pyLDAvis
import pyLDAvis.gensim #for kaggle
#import pyLDAvis.gensim_models #for Jupyter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim import corpora
import en_core_web_sm
import re
import spacy

from wordcloud import WordCloud


# In[ ]:


#Loading data directly in kaggle
df = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")

print(f"train data shape: {df.shape}; test data shape: {test.shape}")
#notably, the test data very short and contains no output feature


# Breakdown of features:
# 1. ID: unique identifier  - won't be used
# 2. anchor: first phrase
# 3. target: second phrase
# 4. context: CPC Classification Number - scoring similarity within these groups (https://en.wikipedia.org/wiki/Cooperative_Patent_Classification)
# 5. score: similarity score = outcome variable
#       * 1.0 = very close; 0.75 = close; 0.5 synonyms with different meaning; 0.25 = somewhat related; 0.0 = unrelated
#      

# # Goal:
# predict the score as value of similarity between anchor and target within each context
# 
# 
# -> While we want to score the similarity between anchor and target, the context can heavily impact this similarity! 
# 
# In result, all columns of the data set (except ID) need to be explored

# # Preprocessing

# In[ ]:


df.head()


# In[ ]:


#Are all IDs unique identifiers? (because you never know)
len(np.unique(df.id)), df.shape[0]
#the length of unique values matches the train shape; there are no duplicates in the dataset


# In[ ]:


#unique values per feature (not including ID)
vals = [len(np.unique(df.anchor)), len(np.unique(df.target)), len(np.unique(df.context))]
sns.barplot(x = ["anchor", "target", "context"], y = vals)
#notably, although anchor and target are heavily related by meaning, the unique values vary greatly. 
#However, ~7000 target values seem to be identical, given that there are 36473 unique entries in the df.


# ## Feature: Anchor

# In[ ]:


df.anchor.value_counts(), df.anchor.value_counts().reset_index().describe()


# In[ ]:


#tip: double clicking the plot will increase readability.
sns.set(font_scale = 0.5)
fig, ax =plt.subplots(figsize = (65,30))
sns.countplot(x = df.anchor, order = df.anchor.value_counts().index, ax = ax, color = "b")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
ax.axhline(df.anchor.value_counts().reset_index().describe().loc["25%"][0], color = "r", label = "25% percentile")
ax.axhline(df.anchor.value_counts().reset_index().describe().loc["50%"][0], color = "orange", label = "50% percentile")
ax.axhline(df.anchor.value_counts().reset_index().describe().loc["75%"][0], color = "r", label = "75% percentile")
plt.title("Counts of Anchors", fontsize = 40)
plt.legend(fontsize=40)
#there are many values that are above the 3rd quartile and below the first quartile


# In[ ]:


sns.set(font_scale = 0.6)
symbols = []
for i in df.anchor:
    symbols.append(len(i))

sns.countplot(x = symbols, color = "b")
plt.title("Number of letters in Anchors");
#the number of symbols in the anchor are normally distributed


# In[ ]:


word_count = []
for i in df.anchor:
    word_count.append(len(i.split()))

sns.countplot(x = word_count, color = "b")
plt.title("Number of words in Anchors")
#the anchors contain 1-5 words; most of them contain 2 or 3 words


# ## Target

# In[ ]:


df.target.value_counts(), df.target.value_counts().reset_index().describe()


# In[ ]:


#Checking numbers in anchor feature
#Code from: https://www.kaggle.com/code/remekkinas/eda-and-feature-engineering/notebook

pattern = '[0-9]'
mask = df['anchor'].str.contains(pattern, na=False)
df['nun_anchor'] = mask
df[mask]['anchor'].value_counts()
#5 anchors contain numbers
#generally these names are rather cryptic


# In[ ]:


df[df.anchor == "conh2"]
#there is a lot of domain knowledge necessary here


# In[ ]:


sns.set(font_scale = 0.4)
symbols = []
for i in df.target:
    symbols.append(len(i))

sns.countplot(x = symbols, color = "b")
plt.title("Number of letters in Anchors");
#the number of symbols in the target are (beautifully) normally distributed


# In[ ]:


sns.set(font_scale = 0.75)
word_count = []
for i in df.target:
    word_count.append(len(i.split()))

sns.countplot(x = word_count, color = "b")
plt.title("Number of words in Anchors");
#the targets contain 1-15 words; most of them contain 1 to 3 words


# ## Context

# In[ ]:


#Dropping the int of the context to cluster on general category (called gen_cat)
df["gen_cat"] = 0
for index in df.index:
    df["gen_cat"].iloc[index] = df.context.iloc[index][0]


# In[ ]:


df.context.value_counts(), df.context.value_counts().reset_index().describe()


# In[ ]:


#Checking numbers in target feature
#Code from: https://www.kaggle.com/code/remekkinas/eda-and-feature-engineering/notebook

pattern = '[0-9]'
mask = df['target'].str.contains(pattern, na=False)
df['num_target'] = mask
df[mask]['target'].value_counts()
#there are more values in target containing numbers, but they are always less frequent.


# In[ ]:


df[df.target == "h2o product"]
#this should have a higher score in my opinion.
#0.5 implies synonyms without the same meaning, I disagree on this score :)


# In[ ]:


#tip: double clicking the plot will increase readability.
sns.set(font_scale = 1.5)
fig, ax =plt.subplots(figsize = (65,30))
sns.countplot(x = df.context, order = df.context.value_counts().index, ax = ax, color = "b")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
ax.axhline(df.context.value_counts().reset_index().describe().loc["25%"][0], color = "r", linewidth = 3, label = "25% percentile")
ax.axhline(df.context.value_counts().reset_index().describe().loc["50%"][0], color = "orange",linewidth = 3, label = "50% percentile")
ax.axhline(df.context.value_counts().reset_index().describe().loc["75%"][0], color = "r", linewidth = 3, label = "75% percentile")
plt.title("Counts of Context", fontsize = 40)
plt.legend(fontsize=40)
#there are several values for context which heavily outweigh most other values


# In[ ]:


#tip: double clicking the plot will increase readability.
sns.set(font_scale = 1.5)
fig, ax =plt.subplots(figsize = (25,10))
sns.countplot(x = df.gen_cat, order = df.gen_cat.value_counts().index, ax = ax, color = "b")
ax.set_xticklabels(ax.get_xticklabels());
ax.axhline(df.gen_cat.value_counts().reset_index().describe().loc["25%"][0], color = "r", linewidth = 3, label = "25% percentile")
ax.axhline(df.gen_cat.value_counts().reset_index().describe().loc["50%"][0], color = "orange",linewidth = 3, label = "50% percentile")
ax.axhline(df.gen_cat.value_counts().reset_index().describe().loc["75%"][0], color = "r", linewidth = 3, label = "75% percentile")
plt.title("Counts of general Categories", fontsize = 25)
plt.legend(fontsize=15)
#unlike the individual contexts, the general contexts are more balanced
#However, there is only little context for the general categories E & D


# In[ ]:


#since there are many more anchors in the anchor-count plot than in the context-count plot, we know that some contexts
#have multiple anchors; at the same time: multiple contexts can also have the same anchor!
print(df[df.anchor == "activating position"].context.nunique(), df[df.anchor == "activating position"].gen_cat.nunique())
df[df.anchor == "activating position"]
#this example shows that some anchors are shared among contexts 
#(in this case 3 different contexts in 3 different general categories)


# In[ ]:


#How many unique contexts are given in train?
np.unique(df.context), f"{len(np.unique(df.context))} unique values"


# In[ ]:


#How many unique contexts are given in test?
np.unique(test.context), f"{len(np.unique(test.context))} unique values"
#notably, there are many context values given in the training data, which are not contained in the test data
#However, this does not mean, that the final kaggle resut will not contain the missing 77 values!


# In[ ]:


#Closer look at the contexts which only have a few entries
df[df.context == "F26"]
#it will probably be hard to train models on this little data.
#is there a way to arbitrarily increase the combinations for these contexts?


# In[ ]:


#Closer look at the contexts which only have a few entries
df[df.context == "A62"]
#some of these word combinations seem wildly different.
#also, some of these word combinations seem again ambigiously placed: 
#matel phase -> metal of material = 0.5
#metal phase -> metal material = 0.25


# In[ ]:


list(df["gen_cat"].unique())
#we would expect B, E, F, G and H to be close to another! (just from general domains)


#     A: Human Necessities
#     B: Operations and Transport
#     C: Chemistry and Metallurgy
#     D: Textiles
#     E: Fixed Constructions
#     F: Mechanical Engineering
#     G: Physics
#     H: Electricity
#     Y: Emerging Cross-Sectional Technologies

# In[ ]:


#Wordcloud per (general) context (most frequent words per context)
wc_a = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(target for target in df[df.gen_cat == "A"].target))
wc_b = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(target for target in df[df.gen_cat == "B"].target))
wc_c = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(target for target in df[df.gen_cat == "C"].target))
wc_d = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(target for target in df[df.gen_cat == "D"].target))
wc_e = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(target for target in df[df.gen_cat == "E"].target))
wc_f = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(target for target in df[df.gen_cat == "F"].target))
wc_g = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(target for target in df[df.gen_cat == "G"].target))
wc_h = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(target for target in df[df.gen_cat == "H"].target))


# In[ ]:


#Show the wordclouds
fig = plt.figure(figsize = (40,40))
ims = [[wc_a, "Wordcloud: Context A"],
       [wc_b, "Wordcloud: Context B"],
       [wc_c, "Wordcloud: Context C"],
       [wc_d, "Wordcloud: Context D"],
       [wc_e, "Wordcloud: Context E"],
       [wc_f, "Wordcloud: Context F"],
       [wc_g, "Wordcloud: Context G"],
       [wc_h, "Wordcloud: Context H"]]

for a, b in enumerate(ims):
    fig.add_subplot(4,2, a+1)
    plt.imshow(b[0], interpolation='bilinear')
    plt.title(b[1], fontsize = 30)
    plt.axis("off")
    
#Double-clicking may increase readability :) 
#Lets quickly look at the first things we can notice:
    #Looking at the wordcloud, we see the word "device" being common in context A, B, E, G, H
    #Context B and D both have the word "layer" as common occurence
    #Context B, D, E and F all have the word "water" as common occurence
    #Context A, B, E and F all have the word "member" as common occurence
    #Context B and C both have the word "metal" as common occurence
#In result, none of the wordclouds are fully disconnected from the others
    #C seems "the most disconnected"


# In[ ]:


#Lengths of target per context
df["target_length"] = 0
for i in df.index:
    df.target_length.iloc[i] = len(df.target.iloc[i].split())


# In[ ]:


sns.boxplot(x = "target_length", y = "gen_cat", data = df, color = "b")
plt.xticks([1,2,3,4,5, 10, 15]);
#most context categories are in the area of 2-3 words for target
#C has the relative-most longest targets
# C and D have the relative-most shortest targets


# In[ ]:


fig = plt.figure(figsize = (15,60))
sns.boxplot(x = "target_length", y = "context", data = df, hue = "gen_cat")
#interestingly some contexts (such as C07 and C08) are very short but also have the strongest outliers
#we can see that the sub categories' context-length are often similar within their categories


# In[ ]:


#Looking at these word lengths, lets have a look at the scores they receive
#(because maybe they have a terrible score just because of the lengths)
df[df.target_length >= 6].head(25)


# In[ ]:


df[df.target_length >= 6].boxplot(column = "score", by = "target_length")
#it seems like longer targets will not be able to receive full score


# In[ ]:


df[(df.target_length >= 6) & (df.score == 1)]
#the only case of a perfect score has a very long anchor itself (so its only 2 words longer)


# In[ ]:


#Maybe instead of looking at absolute lengths, we should look at relative lengths compared to the anchor
df["length_diff"] = 0
for i in df.index:
    df.length_diff.iloc[i] = df["target_length"].iloc[i] - len(df.anchor.iloc[i].split())


# In[ ]:


df.boxplot(column = "score", by = "length_diff")
#it seems like a length difference of more than 3 and lower than -2 will not allow a perfect score
#while it seems that the target being way shorter than the anchor is generally bad for score
#the target being longer than the anchor seems to generally have a positive impact

#these findings need to be looked at with some respect, though, given that there are only few data points, on which this data is based on
# Accordingly, this may be completely different for unknown test data


# ## Score

# In[ ]:


sns.set(font_scale = 1)
sns.boxplot(x = df.score)


# In[ ]:


sns.histplot(x = df.score, bins = 5)
plt.xticks([0.0, 0.25, np.mean(df.score), 0.5, 0.75, 1.0]);
plt.axvline(np.mean(df.score), color = "red", label = "mean")
plt.legend()
plt.title("Hitsogramm of Score");


# In[ ]:


#Which entries have a score of 1?
df[df.score == 1].head(15)
#it seems like patents with the same anchor and target have sometimes different context (B65 & G06; A41 & B23)


# In[ ]:


#How many are there per context group?
context_counts = df[df.score == 1].groupby("context").id.count().reset_index().sort_values("id", ascending = False)
context_counts


# In[ ]:


#context groups where there is only one patent with the score 1.0
list(context_counts[context_counts.id == 1].context)


# In[ ]:


df[df.context == "A22"].head(20)
#maybe turning word groups into syllables will help in prediction
#alternatively, it probably makes sense to reduce key words in to their parts for abbreviations
#such as electromagnectic -> electro magnetic -> em  


# In[ ]:


#Creating this dataframe for a stacked barchart is tidious but mostly copy-paste
scores_plot = df[df.score == 0].groupby(["context"]).id.count().reset_index()
scores_plot.columns = ["context","count_score_0"]
scores_plot = scores_plot.merge(df[df.score == 0.25].groupby(["context"]).id.count().reset_index(), on = "context")
scores_plot = scores_plot.merge(df[df.score == 0.50].groupby(["context"]).id.count().reset_index(), on = "context")
scores_plot = scores_plot.merge(df[df.score == 0.75].groupby(["context"]).id.count().reset_index(), on = "context")
scores_plot = scores_plot.merge(df[df.score == 1].groupby(["context"]).id.count().reset_index(), on = "context")
scores_plot = scores_plot.merge(df.groupby("context").id.count().reset_index(), on = "context")
scores_plot.columns = ["context", "count: score 0.0", "count: score 0.25", "count: score 0.50", "count: score 0.75", "count: score 1.0", "overall"]
scores_plot = scores_plot.sort_values("overall", ascending = False).set_index("context")
scores_plot.drop(columns = ["overall"], inplace = True)

#Creating the stacked barchart for scores
fig, ax =plt.subplots(figsize = (65,30))
scores_plot.plot(kind = "bar", stacked = True, ax = ax)
plt.legend(fontsize = 40)
#This plot underlines how rare perfect scores are and how very common 0.25 and 0.5 are as score.


# In[ ]:


perfect_scores = df[df.score == 1].groupby("context").id.count().reset_index().sort_values("id", ascending = False)

#tip: double clicking the plot will increase readability.
sns.set(font_scale = 1.5)
fig, ax =plt.subplots(figsize = (65,30))
sns.barplot(x = "context", y ="id", data = perfect_scores, ax = ax, color = "b")
sns.barplot(x = "context", y ="id", data = perfect_scores, ax = ax, color = "b")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
ax.axhline(perfect_scores.describe().loc["25%"][0], color = "r", linewidth = 3, label = "25% percentile")
ax.axhline(perfect_scores.describe().loc["50%"][0], color = "orange",linewidth = 3, label = "50% percentile")
ax.axhline(perfect_scores.describe().loc["75%"][0], color = "r", linewidth = 3, label = "75% percentile")
plt.title("Counts of perfect scores per context", fontsize = 40)
plt.ylabel("count")
plt.legend(fontsize=40);
#again, some contexts are heavily outweighing the other contexts
#However, the order of perfect scores is not identical to the order of overall counts per context


# In[ ]:


#Which entries have a score of 0?
df[df.score == 0].head(25)
#some of these seem unjustified scored low: abatement- rent abatement; abatement- tax abatement


# In[ ]:


df[df.score == 0.75].head(25)
#stopwords matter! (last two lines)


# ## Similiarities
# Further explore on the ideas that were first shown in the wordclouds

# In[ ]:


#This thing will take a hot minute but will help for word clouds and clustering
nlp = en_core_web_sm.load()
#Lemmatize the data 
data_lem = []
for i in list(df.target): 
    lemma = nlp(i)
    data_lem.append(" ".join([word.lemma_ for word in lemma]))


# In[ ]:


#Create dictionary and bag of words from the data
tokens = [[word for word in data.split()] for data in data_lem]
dictionary = corpora.Dictionary(tokens)
doc_term_matrix = [dictionary.doc2bow(patent) for patent in tokens]


# In[ ]:


#Initiate the gensim LDA model for pyLDAvis (also will take a short while)
LDA = gensim.models.ldamodel.LdaModel
ldamodel = LDA(corpus = doc_term_matrix,
               id2word = dictionary,
               num_topics = len(list(df["gen_cat"].unique())), 
               #it might make sense to explore how many ACTUALLY different topics there are based on the targets (probably less than 8)
               random_state = 0,
               chunksize = 2000,
               passes = 50, 
               iterations = 100)


# In[ ]:


#check coherence (high = good) and perplexity (low = good)
from gensim.models import CoherenceModel
coherence_model = CoherenceModel(model = ldamodel, texts = tokens, dictionary = dictionary, coherence = "c_v")
ldamodel.log_perplexity(doc_term_matrix, total_docs = df.shape[0]), coherence_model.get_coherence()


# In[ ]:


#Looks a lot better on white background ;)
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary) #for Kaggle
#vis = pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary) for Jupyter
vis

#we can see that stopwords are often in the most salient terms. 
#However, since the targets are very short, it doesnt make sense to remove them, since they sometimes reduce overall score


# In[ ]:


#Now, lets replicate the results with sk-learn (which shows the cluster less "beautiful")
#Sklearn is a great alternative, because we can see how the groups are actually located

#Vectorize data
idf = TfidfVectorizer(min_df = 0.001) 
#0.001 will reduce computing time (a lot) and increase variance ratio on the first 3 PCs
text_idf = idf.fit_transform(df.target).toarray()
y = list(df["gen_cat"])


# In[ ]:


#Fit classifier (may take a while)
clf = LinearDiscriminantAnalysis()
X_r2 = clf.fit(text_idf, y).transform(text_idf)


# In[ ]:


#the first 3 components explain 70% of variance
clf.explained_variance_ratio_


# In[ ]:


map_col = {"A":"blue",
          "B":"green",
          "C":"black",
          "D":"red",
          "E":"yellow",
          "F":"purple",
          "G":"brown",
          "H":"orange"}
df["colours"] = df["gen_cat"].map(map_col)
df.head()


# In[ ]:


#this plot was created to be opened in jupyter notebook (to have an interactive 3D Chart and being able to see the clusters better)
#%matplotlib notebook #activate this in jupyter
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection = '3d')


x = X_r2[:,0]
y = X_r2[:,1]
z = X_r2[:,2]

ax.scatter(x,y,z, c = df["colours"], marker = ".")

col_a = mpatches.Patch(color='blue', label='A / Human Necessities')
col_b = mpatches.Patch(color='green', label='B / Operations and Transport')
col_c = mpatches.Patch(color='black', label='C / Chemistry and Metallurgy')
col_d = mpatches.Patch(color='red', label='D / Textiles')
col_e = mpatches.Patch(color='yellow', label='E / Fixed Constructs')
col_f = mpatches.Patch(color='purple', label='F / Mechanical Engineering')
col_g = mpatches.Patch(color='brown', label='G / Physics')
col_h = mpatches.Patch(color='orange', label='H / Electricity')
handles=[col_a, col_b, col_c, col_d, col_e, col_f, col_g, col_h]
plt.legend(handles=handles, loc = "upper right", fontsize = 8);

#we would expect B, E, F, G and H to be close to another! (green, yellow, purple, brown, orange); just by topic names
#others: black, blue, red
#However, we can see that only black is clustered apart (and still some outliers fall into other clusters)
#interestingly, purple seems somewhat separated as well.


# ## Summarizing EDA & Preprocessing:
#  - Some anchors are shared among several contexts (and general categories
#  - Most context contain several different anchors
#  - Some contexts are heavily outweighing others in overall occurence (heavily right-skewed)
#  - In general, the contexts of the categories D & E are under-represented in the data
#  - The proportion of scores are more or less similar around all contexts
#  - Most of the contexts in which we want to predict the scores are similar in regards to words used and words lengths
#  
#  
#  - It does not make sense to remove stop words or short words, since they actually impact the score ("accept information -> accept this information" = 0.75)
#  - Abbreviations are a thing in the dataset (e.g., Electromagnetic = em; Water = h2o) -> It might makes sense to find a model for domain specific abbreviations (also for possibly unknown categories & abbreviations in the test set)
#      -> BUT: abreviations also penalize score!
#  - Synonyms are often not as heavily penalized as abreviations - a good synonym finder will be helpful
#      -> generally, the penalization of synonyms seems to be sometimes weird (e.g., absorbant properties and absorbant characteristics is a perfect match at one point (id: 621b048d70aa8867) but absorption characteristics an inperfect match (0.75) at another point (id: e6f92889099fd908)) -> maybe lemmatization will mess up these relationships were they are considered "inperfect" because there are two small misallignments
#  - Train Data is relatively small (given we also want validation), so we will need a pre-trained model (although it would be nice to train it ourselves, given we are acting in domain specific environments)

# # Modeling

# TODO:
# Next steps:
# - similarities (maybe wordnet; maybe more domain specific?, gotta read into it)
# - abbreviations 
# - generally, find pretrained model
# - just for fun: maybe score can be easily algorithmed (e.g., abbreviation = -0.25; using very similar word instead = -0; using two very similar words instead = -0.25; etc.; if target is way shorter than anchor = -0.25 etc.)

# In[ ]:




