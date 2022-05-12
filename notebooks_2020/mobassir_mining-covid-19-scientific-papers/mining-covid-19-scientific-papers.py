#!/usr/bin/env python
# coding: utf-8

# # Before we begin, let's start with a "MUST WATCH" <br>video : 

# In[ ]:


from IPython.display import HTML

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/qfdrNHqlNEk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# Apology : Due to kaggle problem,somehow my kernels contents +(few graphs) getting cut off after kernel commit,i hope that kaggle will solve this problem quickly, i asked for help here : https://www.kaggle.com/product-feedback/137723

# # Introduction

# The flu, the common cold, allergies, and COVID-19 — the disease associated with the new coronavirus — have similar symptoms, but the coronavirus has been far deadlier.Overlapping symptoms include a sore throat, fatigue, and a dry cough. That can make it challenging for doctors to diagnose COVID-19.People with COVID-19 don't typically have runny noses or sneeze a lot.The coronavirus primarily affects the lungs and commonly causes a fever, a dry cough, and shortness of breath.

# **[Here are the symptoms associated with COVID-19 and how they compare with symptoms of the common cold, the flu, and allergies](https://www.businessinsider.com/coronavirus-symptoms-compared-to-flu-common-cold-and-allergies-2020-3):**
# 
# 
# 

# <img src="https://i.insider.com/5e6a58e684159f61963287a2?width=1000&format=jpeg&auto=webp" width="600px" align="left"> 

# <img src="https://ichef.bbci.co.uk/news/640/cpsprodpb/16F8F/production/_111059049_corona_virus_symptoms_short_v4_640-nc.png" width="600px" align="left"> 
# 

# As a NLP beginner, In this notebook i will try to apply various TextAnalytics techniques on [COVID-19 Open Research Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
# 

# # Research Goal
# 
# Main goal of this research is to analyze the data and find **Risk Factors of COVID-19**

# **I will gradually update this kernel**

#  <h1 align="left" style="color:green;">
# If you find this kernel interesting, please drop an  <br><font color="red">UPVOTE</font>. It motivates me to produce more quality <br>contents 🤗 
# </h1> 

# **IMPORTS**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nltk.stem import WordNetLemmatizer
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
n = 1000

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install  tensorflow-gpu==1.15.0')


# In[ ]:


get_ipython().system('pip install bert-tensorflow')


# In[ ]:


get_ipython().system('wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip')
    


# In[ ]:


get_ipython().system('unzip cased_L-12_H-768_A-12.zip')


# In[ ]:


import tensorflow as tf
tf.VERSION


# In[ ]:



import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
import tensorflow as tf
import numpy as np
import itertools


# # NLTK Stop words

#  <h5 align="left" style="color:purple;">
# The process of converting data to something a computer can understand is referred to as pre-processing. One of the major forms of pre-processing is to filter out useless data. In natural language processing, useless words (data), are referred to as stop words.
# </h5>
# 
# ![](http://) [What are Stop words?](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)
# 
# <h5 align="left" style="color:blue;">
#  <font color="red">Stop Words:</font>  A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query.
# </h5> 
# 
# 

#  <img src="https://www.geeksforgeeks.org/wp-content/uploads/Stop-word-removal-using-NLTK.png" width="600px" align="left"> 

# In[ ]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['background', 'methods', 'introduction', 'conclusions', 'results', 
                   'purpose', 'materials', 'discussions','methodology','result analysis'])


# list of files given for [COVID-19 Open Research Dataset challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) 

# In[ ]:


os.listdir('../input/CORD-19-research-challenge/')


# let's read the readme file first

# In[ ]:


with open('../input/CORD-19-research-challenge/metadata.readme', 'r') as f:
    data = f.read()
    print(data)


# In[ ]:


biorxiv_dir = '../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))


# **From the cell above we can see there are 885  json files inside biorvix directory, the structure is likely too complex to directly perform analysis. we will use clean and updated dataset prepared by @xhlulu in this kernel [CORD-19: EDA, parse JSON and generate clean CSV🧹](https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv). Thanks to the author  (｡●́‿●̀｡) **

# The cell below shows that the updated datasets are in CSV format now - biorxiv_clean.csv, clean_comm_use.csv, clean_noncomm_use.csv and clean_pmc.csv

# In[ ]:


os.listdir( '/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv')


# **Reading The updated clean CSV files**

# In[ ]:


biorxiv_clean = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')
clean_comm_use = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv')
clean_noncomm_use =  pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv')
clean_pmc =  pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv')


# In[ ]:


biorxiv_clean.head(2)


# In[ ]:


clean_comm_use.head(2)


# In[ ]:


clean_noncomm_use.head(2)


# In[ ]:


clean_pmc.head(2)


# first text of biorxiv_clean dataset

# In[ ]:


biorxiv_clean.text[0]


# # Part 1 :  Working with biorxiv

# # biorxiv_clean papers Abstract - frequent words (400 sample)

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

stopwords = set(STOPWORDS)
#https://www.kaggle.com/gpreda/cord-19-solution-toolbox

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=30, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[ ]:


show_wordcloud(biorxiv_clean['abstract'], title = 'biorxiv_clean - papers Abstract - frequent words (400 sample)')


# # Convert abstract to list

# In[ ]:



df = biorxiv_clean
df = df.abstract.dropna()
data = df.values.tolist()


# # Find similar research papers using universalsentenceencoderlarge4

# In[ ]:


get_ipython().run_cell_magic('time', '', '#ref : https://gist.github.com/gaurav5430/8d7810495ec3f914ffb151458f352c60\n\n\'\'\'import tensorflow_hub as hub\nfrom sklearn.metrics.pairwise import cosine_similarity\ndef prepare_similarity(vectors):\n    similarity=cosine_similarity(vectors)\n    return similarity\n\ndef get_top_similar(sentence, sentence_list, similarity_matrix, topN):\n    # find the index of sentence in list\n    index = sentence_list.index(sentence)\n    # get the corresponding row in similarity matrix\n    similarity_row = np.array(similarity_matrix[index, :])\n    # get the indices of top similar\n    indices = similarity_row.argsort()[-topN:][::-1]\n    return [sentence_list[i] for i in indices]\n\n\nmeta=pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")\nmodule_url = "../input/universalsentenceencoderlarge4" \nembed = hub.load(module_url)\n\n\n# Creating an empty Dataframe with column names only\nsimsentence = pd.DataFrame()\n\ntitles=meta[\'title\'].fillna("Unknown")\nembed_vectors=embed(titles[:5000].values)[\'outputs\'].numpy()\nsentence_list=titles.values.tolist()\nfor i in range(5):\n\n    sentences=titles.iloc[i]\n    #print(">>>>>>>>>>>>Using title Find similar research papers for :",sentences, "<<<<<<<<<<<<")\n\n    similarity_matrix=prepare_similarity(embed_vectors)\n    similar=get_top_similar(sentences,sentence_list,similarity_matrix,6)\n    for sentence in similar:\n        #print(sentence)\n        simsentence = simsentence.append({\'sentence\': sentences, \'similar\': sentence}, ignore_index=True)\n        #print("\\n") \'\'\'')


# **The Cell above Finds similar research papers for given sentence,if you observe carefully,you can see i have stored all the sentences and corresponding similar sentences in a pandas dataframe called simsentence.let's check that file below**

# **IMPORTANT NOTE : I've commented out the codes above that finds similar papers to save time.i have executed that code in previous versions of this kernel to get related papers and save them as csv format**

# # Now let's save this new dataframe as csv file for possible further research

# In[ ]:


#simsentence.to_csv('simsentence.csv', index=False)


# **I have downloaded simsentence.csv from version 1's output of this kernel and now using that csv file in cell below for little analysis **

# In[ ]:


simsentence = pd.read_csv('../input/simsentence/simsentence.csv')


# **The function below converts sentences to words using gensim**

# In[ ]:



def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  

data_words = list(sent_to_words(data))

print(data_words[:1])


# # Build the bigram and trigram models using gensim

#  <h5 align="left" style="color:blue;">
#  <font color="red"> Question-1 : [What is a bigram and a trigram?](https://www.quora.com/What-is-a-bigram-and-a-trigram-layman-explanation-please) </font> <br>
#  
#  Answer : Start with a unigram. If you put all of the words in some sentence into a box, and choose one single word randomely, it is called a unigram. A unigram is just one single word. But a bigram is a word pair. The bigrams within a sentence are all possible word pairs formed from neighboring words in the sentence. It is easier to look at an example. The bigrams in the sentence I  really love Quora are I really, and really love, and love Quora. That is a total of 3 word pairs, or bigrams. The same goes for trigrams, or triplets. This sentence would contain only two trigrams, which are I really love, and really love quora.
# </h5> 

# <h5 align="left" style="color:green;">
#  <font color="red"> Question-2 : What is Gensim? </font> <br>
#  
#  Answer :  It is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
# </h5> 

# In[ ]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=20) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=20)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[1]]])


# # Define functions for stopwords, bigrams, trigrams and lemmatization

# <h5 align="left" style="color:green;">
#  <font color="blue"> Question-1 : [What is difference between stemming and lemmatization?](https://www.quora.com/What-is-difference-between-stemming-and-lemmatization) </font> <br><br>
#  
#  
#  Answer :  <br><br> <font color="red"> Stemming -  </font>  Stemming is a process of reducing words to its root form even if the root has no dictionary meaning. For eg: beautiful and beautifully will be stemmed to beauti which has no meaning in English dictionary. <br> <br>
# 
# 
#  <font color="purple"> Lemmatisation - </font> Lemmatisation is a process of reducing words into their lemma or dictionary. It takes into account the meaning of the word in the sentence. For eg: beautiful and beautifully are lemmatised to beautiful and beautifully respectively without changing the meaning of the words. But, good, better and best are lemmatised to good since all the words have similar meaning.
# </h5> 

# In[ ]:



#https://github.com/cjriggio/classifying_medical_innovation
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[ ]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[1])


# In[ ]:



print(data_lemmatized[:1])


# # Create Dictionary,Corpus and Document Frequency

# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


# # Human readable format of corpus (term-frequency)

# In[ ]:



[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# # Build LDA model

# 
#  <font align="left" color="red"> Topic Modeling -  </font> In recent years, huge amount of data (mostly unstructured) is growing. It is difficult to extract relevant and desired information from it. At the document level, the most useful ways to understand text by its topics. The statistical process of learning and extracting these topics from huge amount of documents is called topic modeling.In Text Mining Topic Modeling is a technique to extract the hidden topics from huge amount of text.
# 
# There are so many algorithms to do topic modeling. Latent Dirichlet Allocation (LDA) is one of those popular algorithms for topic modeling. 
# 
# 

# # Let’s see where topic modeling fit in machine learning spectrum.
# 
# 
#  

# <img src="https://2.bp.blogspot.com/-V_WqO4x3MEQ/XGWhK55_S3I/AAAAAAAABoY/riaPM64fUcovdV543zphBLwYPe3MHgGBwCLcBGAs/s1600/image002.png" width = "600px" align="left"> 

# 
#  <font color="purple"> How Latent Dirichlet Allocation (LDA) Works?</font> 
#  
# 
# 

# <img src="https://2.bp.blogspot.com/-UO8E6wws1Go/XGWgbLTPJnI/AAAAAAAABoQ/tGuBrjfJZ1UGmUQ112ZCv3gAu3Tg0O1FACLcBGAs/s1600/image001-min.png" width = "600px" align="left"> 
# 

# In[ ]:



lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True
                                        )


# # Print the Keyword in the n topics

# In[ ]:



#pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# # Compute Perplexity and Coherence Score

# *perplexity is a measurement of how well a probability distribution or probability model predicts a sample where The coherence score is for assessing the quality of the learned topics. *
# 
# 
# 

# In[ ]:



print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# # Visualize the topics

# In[ ]:


get_ipython().run_cell_magic('time', '', 'pyLDAvis.enable_notebook()\nvis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)\nvis')


# Save Topics as html format

# In[ ]:


pyLDAvis.save_html(vis, './lda4topics_v2.html')


# In[ ]:


optimal_model = lda_model

model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=20))


# In[ ]:


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# # Wordcloud of Top N words in each topic

# In[ ]:



from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=100,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False, 
                               num_words=30)

fig, axes = plt.subplots(5, 1, figsize=(10,20), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=500)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i + 1), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# # Sentence Coloring of N Sentences

# In[ ]:



from matplotlib.patches import Rectangle

def sentences_chart(lda_model=lda_model, corpus=corpus, start = 0, end = 12):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end-start, 1, figsize=(22, 10))      
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 1:
            #i = i+1
            corp_cur = corp[i-1] 
            topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
            word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
            ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                    fontsize=15, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=700)
                    word_pos += 0.009 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)       

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=20, x = 0.2, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()

sentences_chart()


# In[ ]:


corp = corpus[0:13]
corp_cur = corp[13-1] 
topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics] 
word_dominanttopic


# # Sentence Coloring of N Sentences

# In[ ]:



def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)


# # Topic Distribution Plot

# In[ ]:


from matplotlib.ticker import FuncFormatter

# Plot

fig, ax1  = plt.subplots(1, figsize=(10, 10))

# Topic Distribution by Dominant Topics
ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x + 1)+ ':\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
ax1.set_ylabel('Number of Documents')
ax1.set_ylim(0, 2000)

plt.show()


# In[ ]:


df_dominant_topic_in_each_doc


# In[ ]:


fig, ax2  = plt.subplots(1, figsize=(10, 10))
# Topic Distribution by Topic Weights
ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
ax2.xaxis.set_major_formatter(tick_formatter)
ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

plt.show()


# # Get topic weights and dominant topics

# In[ ]:



from bokeh.models import HoverTool
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 5
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])


plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=800, plot_height=600)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)


# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint


# In[ ]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

    # # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # # Run in terminal: python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

    # # Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:2])


# In[ ]:


vectorizer = CountVectorizer(analyzer='word', min_df=10,                        # minimum reqd occurences of a word 
                              stop_words='english',             # remove stop words
                              lowercase=True,                   # convert all words to lowercase
                              token_pattern='[a-zA-Z0-9]{3,}'  # num chars > 3
                              # max_features=50000,             # max number of uniq words
                             )
data_vectorized = vectorizer.fit_transform(data_lemmatized)


# # Define Search Param for GridSearch

# In[ ]:


get_ipython().run_cell_magic('time', '', "search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}\n\n# Init the Model\nlda = LatentDirichletAllocation(n_jobs=-1)\n\n# Init Grid Search Class\nmodel = GridSearchCV(lda, param_grid=search_params)\n\n\n# Do the Grid Search\nmodel.fit(data_vectorized)")


# # Best Model

# In[ ]:



best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))


# In[ ]:


categories = list(df.Dominant_Topic.unique())
categories


# # Part 2 : Working with clean_comm_use.csv

# In[ ]:


df1 = clean_comm_use
df1 = df1.dropna()


# # creating n-gram and fetching to HashingVectorizer to get feature vector X

# In[ ]:


#taking from https://www.kaggle.com/maksimeren/covid-19-literature-clustering
words = []
for ii in range(0,len(df1)):
    words.append(str(df1.iloc[ii]['text']).split(" "))
    
    
n_gram_all = []

for word in words:
    # get n-grams for the instance
    n_gram = []
    for i in range(len(word)-2+1):
        n_gram.append("".join(word[i:i+2]))
    n_gram_all.append(n_gram)


# In[ ]:



from sklearn.feature_extraction.text import HashingVectorizer

# hash vectorizer instance
hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)

# features matrix X
X = hvec.fit_transform(n_gram_all)


# In[ ]:


X.shape


# In[ ]:


from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")


# **K-means clustering for 15 cluster**

# In[ ]:


from sklearn.cluster import KMeans

k = 15 
kmeans = KMeans(n_clusters=k, n_jobs=4, verbose= k)
y_pred = kmeans.fit_predict(X_train)


# In[ ]:


y_pred.shape


# In[ ]:


y_train = y_pred
y_test = kmeans.predict(X_test)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.manifold import TSNE\n\ntsne = TSNE(verbose=1)\nX_embedded = tsne.fit_transform(X_train)')


# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(10,10)})

# colors
palette = sns.color_palette("bright", 1)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)

plt.title("t-SNE Covid-19 Articles")
# plt.savefig("plots/t-sne_covid19.png")
plt.show()


# # Visualizing clusters

# In[ ]:


X_embedded[:,1].shape


# In[ ]:


# sns settings
sns.set(rc={'figure.figsize':(10,10)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered")
# plt.savefig("plots/t-sne_covid19_label.png")
plt.show()


# # Part 3 : working with clean_noncomm_use.csv

# in this section
# * Text Summarization Approaches
# * Understanding the TextRank Algorithm
# * Understanding the Problem Statement
# * Implementation of the TextRank Algorithm o

# In[ ]:


type(clean_noncomm_use.abstract.dropna().tolist())


# gensim.summarization offers TextRank summarization

# In[ ]:



from gensim.summarization.summarizer import summarize
summarize(clean_noncomm_use.abstract.dropna().to_string())


# Have you come across the mobile app inshorts? It’s an innovative news app that converts news articles into a 60-word summary. And that is exactly what we are going to learn in this section — **Automatic Text Summarization.**

# In[ ]:


import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re


# #  Understanding the Text Rank Algorithm

# *All the informations below are taken from this beautiful article [An Introduction to Text Summarization using the TextRank Algorithm (with Python implementation)](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/)*

# Before getting started with the TextRank algorithm, there’s another algorithm which we should become familiar with – the **PageRank algorithm**. In fact, this actually inspired TextRank! PageRank is used primarily for ranking web pages in online search results. Let’s quickly understand the basics of this algorithm with the help of an example.

# **PageRank Algorithm**

# <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/10/Pagerank11.png" width="600px" align="left"> 

# Suppose we have 4 web pages — w1, w2, w3, and w4. These pages contain links pointing to one another. Some pages might have no link – these are called dangling pages.

# <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/10/webpages.png" width="600px" align="left"> 

# * Web page w1 has links directing to w2 and w4
# * w2 has links for w3 and w1
# * w4 has links only for the web page w1
# * w3 has no links and hence it will be called a dangling page

# In order to rank these pages, we would have to compute a score called the PageRank score. This score is the probability of a user visiting that page.
# 
# To capture the probabilities of users navigating from one page to another, we will create a square matrix M, having n rows and n columns, where n is the number of web pages.

# <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/10/m_matrix.png" width="600px" align="left"> 

# Each element of this matrix denotes the probability of a user transitioning from one web page to another. For example, the highlighted cell below contains the probability of transition from w1 to w2.

# <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/10/transition_probability.png" width="600px" align="left"> 

# The initialization of the probabilities is explained in the steps below: 
# 
# * Probability of going from page i to j, i.e., M[ i ][ j ], is initialized with 1/(number of unique links in web page wi)
# * If there is no link between the page i and j, then the probability will be initialized with 0
# * If a user has landed on a dangling page, then it is assumed that he is equally likely to transition to any page. Hence, M[ i ][ j ] will be initialized with 1/(number of web pages)
# 
# Hence, in our case, the matrix M will be initialized as follows:

# <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/10/final_matrix.png" width="600px" align="left"> 

# Finally, the values in this matrix will be updated in an iterative fashion to arrive at the web page rankings.

# # TextRank Algorithm

# * In place of web pages, we use sentences
# * Similarity between any two sentences is used as an equivalent to the web page transition probability
# * The similarity scores are stored in a square matrix, similar to the matrix M used for PageRank

# **TextRank is an extractive and unsupervised text summarization technique. Let’s take a look at the flow of the TextRank algorithm that we will be following:**

# <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/10/block_3.png" width="600px" align="left"> 

# 1. The first step would be to concatenate all the text contained in the articles
# 2. Then split the text into individual sentences
# 3. In the next step, we will find vector representation (word embeddings) for each and every sentence
# 4. Similarities between sentence vectors are then calculated and stored in a matrix
# 5. The similarity matrix is then converted into a graph, with sentences as vertices and similarity scores as edges, for sentence rank calculation
# 6. Finally, a certain number of top-ranked sentences form the final summary

# **We will apply the TextRank algorithm on this dataset of  articles with the aim of creating a nice and concise summary.**

# Now the next step is to break the text into individual sentences. We will use the sent_tokenize( ) function of the nltk library to do this.

# In[ ]:


#ref : https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/
from nltk.tokenize import sent_tokenize
sentences = []
for s in clean_noncomm_use.abstract.dropna():
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list


# In[ ]:


sentences[:3]


# # Download GloVe Word Embeddings

# [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings are vector representation of words. These word embeddings will be used to create vectors for our sentences. We could have also used the Bag-of-Words or TF-IDF approaches to create features for our sentences, but these methods ignore the order of the words (and the number of features is usually pretty large).
# 
# We will be using the pre-trained Wikipedia 2014 + Gigaword 5 GloVe vectors. Heads up – the size of these word embeddings is 822 MB.

# In[ ]:


get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
get_ipython().system('unzip glove*.zip')


# Let’s extract the words embeddings or word vectors.

# In[ ]:


# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


# In[ ]:


len(word_embeddings)


# We now have word vectors for 400,000 different terms stored in the dictionary – ‘word_embeddings’.

# In[ ]:


# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


# # function to remove stopwords

# In[ ]:



def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# In[ ]:


# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


# We will use clean_sentences to create vectors for sentences in our data with the help of the GloVe word vectors.

# # Vector Representation of Sentences

# In[ ]:


# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


# Now, let’s create vectors for our sentences. We will first fetch vectors (each of size 100 elements) for the constituent words in a sentence and then take mean/average of those vectors to arrive at a consolidated vector for the sentence.

# In[ ]:


sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)


# # Similarity Matrix preparation

# The next step is to find similarities between the sentences, and we will use the cosine similarity approach for this challenge. Let’s create an empty similarity matrix for this task and populate it with cosine similarities of the sentences.
# 
# Let’s first define a zero matrix of dimensions (n * n).  We will initialize this matrix with cosine similarity scores of the sentences. Here, n is the number of sentences.

# In[ ]:


# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])


# We will use Cosine Similarity to compute the similarity between a pair of sentences And initialize the matrix with cosine similarity scores.

# > NOTE : For all sentences, the cell below takes a lot of time,i waited more than  8 hours for this kernel to finish commit but the cell below doesn't finish execution for len(sentences) within 9 hours(kaggle gpu limit),so to save time and computation power i will use 1000 instead

# In[ ]:


import torch
torch.cuda.is_available()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics.pairwise import cosine_similarity\nfor i in range(1000):\n    for j in range(1000):\n        if i != j:\n            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]\n            #print(sim_mat[i][j])')


# # Applying PageRank Algorithm

# Before proceeding further, let’s convert the similarity matrix sim_mat into a graph. The nodes of this graph will represent the sentences and the edges will represent the similarity scores between the sentences. On this graph, we will apply the PageRank algorithm to arrive at the sentence rankings.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import networkx as nx\n\nnx_graph = nx.from_numpy_array(sim_mat)\nscores = nx.pagerank(nx_graph)')


# # Summary Extraction : Extract top 50 sentences as the summary

# Finally, it’s time to extract the top N sentences based on their rankings for summary generation.

# In[ ]:


ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
for i in range(50):
    print(ranked_sentences[i][1])
    print('\n\n')
  


# # Part 4 : Analyzing clean_pmc.csv file

# in this section, let's try bert for Topic Modeling

# Ref : https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb

# In[ ]:


clean_pmc.title


# In[ ]:


import os
os.listdir('cased_L-12_H-768_A-12')


# In[ ]:


BERT_VOCAB = 'cased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'cased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = 'cased_L-12_H-768_A-12/bert_config.json'   


# * generate ngrams.
# * Vectorize string inputs using bert attention 
# * Topic modeling Using bert for 10 topics
# 
# 

# In[ ]:


def generate_ngram(seq, ngram = (1, 3)):
    g = []
    for i in range(ngram[0], ngram[-1] + 1):
        g.extend(list(ngrams_generator(seq, i)))
    return g

def _pad_sequence(
    sequence,
    n,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
):
    sequence = iter(sequence)
    if pad_left:
        sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams_generator(
    sequence,
    n,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
):
    """
    generate ngrams.

    Parameters
    ----------
    sequence : list of str
        list of tokenize words.
    n : int
        ngram size

    Returns
    -------
    ngram: list
    """
    sequence = _pad_sequence(
        sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
    )

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def merge_wordpiece_tokens(paired_tokens, weighted = True):
    new_paired_tokens = []
    n_tokens = len(paired_tokens)

    i = 0

    while i < n_tokens:
        current_token, current_weight = paired_tokens[i]
        if current_token.startswith('##'):
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while current_token.startswith('##'):
                merged_token = merged_token + current_token.replace('##', '')
                merged_weight.append(current_weight)
                i = i + 1
                current_token, current_weight = paired_tokens[i]
            merged_weight = np.mean(merged_weight)
            new_paired_tokens.append((merged_token, merged_weight))

        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1

    words = [
        i[0]
        for i in new_paired_tokens
        if i[0] not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    weights = [
        i[1]
        for i in new_paired_tokens
        if i[0] not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    if weighted:
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    return list(zip(words, weights))

def _extract_attention_weights(num_layers, tf_graph):
    attns = [
        {
            'layer_%s'
            % i: tf_graph.get_tensor_by_name(
                'bert/encoder/layer_%s/attention/self/Softmax:0' % i
            )
        }
        for i in range(num_layers)
    ]

    return attns

def padding_sequence(seq, maxlen, padding = 'post', pad_int = 0):
    padded_seqs = []
    for s in seq:
        if padding == 'post':
            padded_seqs.append(s + [pad_int] * (maxlen - len(s)))
        if padding == 'pre':
            padded_seqs.append([pad_int] * (maxlen - len(s)) + s)
    return padded_seqs


def bert_tokenization(tokenizer, texts, cls = '[CLS]', sep = '[SEP]'):

    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    for text in texts:
        tokens_a = tokenizer.tokenize(text)
        tokens = [cls] + tokens_a + [sep]
        segment_id = [0] * len(tokens)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        s_tokens.append(tokens)

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen)
    input_masks = padding_sequence(input_masks, maxlen)
    segment_ids = padding_sequence(segment_ids, maxlen)

    return input_ids, input_masks, segment_ids, s_tokens

class _Model:
    def __init__(self, bert_config, tokenizer):
        _graph = tf.Graph()
        with _graph.as_default():
            self.X = tf.placeholder(tf.int32, [None, None])
            self._tokenizer = tokenizer

            self.model = modeling.BertModel(
                config = bert_config,
                is_training = False,
                input_ids = self.X,
                use_one_hot_embeddings = False,
            )
            self.logits = self.model.get_pooled_output()
            self._sess = tf.InteractiveSession()
            self._sess.run(tf.global_variables_initializer())
            var_lists = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert'
            )
            self._saver = tf.train.Saver(var_list = var_lists)
            attns = _extract_attention_weights(
                bert_config.num_hidden_layers, tf.get_default_graph()
            )
            self.attns = attns

    def vectorize(self, strings):

        """
        Vectorize string inputs using bert attention.

        Parameters
        ----------
        strings : str / list of str

        Returns
        -------
        array: vectorized strings
        """

        if isinstance(strings, list):
            if not isinstance(strings[0], str):
                raise ValueError('input must be a list of strings or a string')
        else:
            if not isinstance(strings, str):
                raise ValueError('input must be a list of strings or a string')
        if isinstance(strings, str):
            strings = [strings]

        batch_x, _, _, _ = bert_tokenization(self._tokenizer, strings)
        return self._sess.run(self.logits, feed_dict = {self.X: batch_x})

    def attention(self, strings, method = 'last', **kwargs):
        """
        Get attention string inputs from bert attention.

        Parameters
        ----------
        strings : str / list of str
        method : str, optional (default='last')
            Attention layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.

        Returns
        -------
        array: attention
        """

        if isinstance(strings, list):
            if not isinstance(strings[0], str):
                raise ValueError('input must be a list of strings or a string')
        else:
            if not isinstance(strings, str):
                raise ValueError('input must be a list of strings or a string')
        if isinstance(strings, str):
            strings = [strings]

        method = method.lower()
        if method not in ['last', 'first', 'mean']:
            raise Exception(
                "method not supported, only support 'last', 'first' and 'mean'"
            )

        batch_x, _, _, s_tokens = bert_tokenization(self._tokenizer, strings)
        maxlen = max([len(s) for s in s_tokens])
        s_tokens = padding_sequence(s_tokens, maxlen, pad_int = '[SEP]')
        attentions = self._sess.run(self.attns, feed_dict = {self.X: batch_x})
        if method == 'first':
            cls_attn = list(attentions[0].values())[0][:, :, 0, :]

        if method == 'last':
            cls_attn = list(attentions[-1].values())[0][:, :, 0, :]

        if method == 'mean':
            combined_attentions = []
            for a in attentions:
                combined_attentions.append(list(a.values())[0])
            cls_attn = np.mean(combined_attentions, axis = 0).mean(axis = 2)

        cls_attn = np.mean(cls_attn, axis = 1)
        total_weights = np.sum(cls_attn, axis = -1, keepdims = True)
        attn = cls_attn / total_weights
        output = []
        for i in range(attn.shape[0]):
            output.append(
                merge_wordpiece_tokens(list(zip(s_tokens[i], attn[i])))
            )
        return output


# In[ ]:


tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=False)
bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
model = _Model(bert_config, tokenizer)


# example 1(vectorize) :

# In[ ]:


v = model.vectorize(['hello nice to meet u', 'so long sucker'])
v


# example 2(attention) :

# In[ ]:


model.attention(['hello nice to meet u', 'so long sucker'])


# In[ ]:


batch_size = 10
ngram = (1, 3)
n_topics = 10


# taking all the titles from clean_pmc.csv 

# In[ ]:



df = clean_pmc
df = df.title.dropna()
negative = df.values.tolist()
negative[0]


# In[ ]:


negative = negative[:100]


# In[ ]:


from sklearn.cluster import KMeans
from tqdm import tqdm

rows, attentions = [], []
for i in (range (len(negative))):
          #index = min(i + batch_size, len(negative))
          rows.append(model.vectorize(negative[i]))
          attentions.extend(model.attention(negative[i]))
    


# In[ ]:


stopwords = stop_words


# In[ ]:


concat = np.concatenate(rows, axis = 0)
kmeans = KMeans(n_clusters = n_topics, random_state = 0).fit(concat)
labels = kmeans.labels_

overall, filtered_a = [], []
for a in attentions:
    #print(a)
    f = [i for i in a if i[0] not in stopwords]
    overall.extend(f)
    filtered_a.append(f)

o_ngram = generate_ngram(overall, ngram)
features = []
for i in o_ngram:
    #print(i)
    features.append(' '.join([w[0] for w in i]))
features = list(set(features))

components = np.zeros((n_topics, len(features)))
print(n_topics)
#print(features)
for no, i in enumerate(labels):
    if (no + 1) % 500 == 0: 
        print('processed %d'%(no + 1))
    f = generate_ngram(filtered_a[no], ngram)
    for w in f:
        word = ' '.join([r[0] for r in w])
        score = np.mean([r[1] for r in w])
        if word in features:
            components[i, features.index(word)] += score


# In[ ]:


def print_topics_modelling(
    topics, feature_names, sorting, n_words = 20, return_df = True
):
    if return_df:
        try:
            import pandas as pd
        except:
            raise Exception(
                'pandas not installed. Please install it and try again or set `return_df = False`'
            )
    df = {}
    for i in range(topics):
        words = []
        for k in range(n_words):
            words.append(feature_names[sorting[i, k]])
        df['topic %d' % (i)] = words
    if return_df:
        return pd.DataFrame.from_dict(df)
    else:
        return df


# In[ ]:


print_topics_modelling(
    10,
    feature_names = np.array(features),
    sorting = np.argsort(components)[:, ::-1],
    n_words = 10,
    return_df = True,
)


# # Summarization Task using Bart 

# In October 2019, teams from Google and Facebook published new transformer papers: T5 and BART. Both papers achieved better downstream performance on generation tasks, like abstractive summarization and dialogue, with two changes:
# 
# * add a causal decoder to BERT's bidirectional encoder architecture
# * replace BERT's fill-in-the blank cloze task with a more complicated mix of pretraining tasks.
# 
# > (BART) can be seen as generalizing Bert (due to the bidirectional encoder) and GPT2 (with the left to right decoder). - bart authors
# 
# Bert is pretrained to try to predict masked tokens, and uses the whole sequence to get enough info to make a good guess. This is good for tasks where the prediction at position i is allowed to utilize information from positions after i, but less useful for tasks, like text generation, where the prediction for position i can only depend on previously generated words.
# 
# In code, the idea of "what information can be used use when predicting the token at position i" is controlled by an argument called attention_mask1. A value of 1 in the attention mask means that the model can use information for the column's word when predicting the row's word.
# 
# Here is Bert's "Fully-visible"2 attention_mask:
# 
# <img src="https://sshleifer.github.io/blog_v2/images/copied_from_nb/diagram_bert_v5.png" width="600px" align="left"> 
# 

# 
# GPT2, meanwhile, is pretrained to predict the next word using a causal mask, and is more effective for generation tasks, but less effective on downstream tasks where the whole input yields information for the output.
# 
# Here is the attention_mask for GPT2:
# 
# <img src="https://sshleifer.github.io/blog_v2/images/copied_from_nb/diagram_bartpost_gpt2.jpg" width="600px" align="left"> 
# 
# 

# The prediction for "eating", only utilizes previous words: "<BOS> I love".
#     
# 
# for more information please check this reference link : [Introducing BART](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html)

# ref : https://github.com/renatoviolin/Bart_T5-summarization/blob/master/app.py

# In[ ]:


get_ipython().system('pip install -U transformers')
get_ipython().system('pip install -U torch')


# In[ ]:


import torch
import os
import json
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


# In[ ]:


BART_PATH = 'bart-large'
bart_model = BartForConditionalGeneration.from_pretrained(BART_PATH, output_past=True)


# In[ ]:


bart_tokenizer = BartTokenizer.from_pretrained(BART_PATH)


# In[ ]:


def bart_summarize(input_text, num_beams=4, num_words=80):
    #input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = bart_model.generate(input_tokenized,
                                      num_beams=int(num_beams),
                                      no_repeat_ngram_size=3,
                                      length_penalty=2.0,
                                      min_length=100,
                                      max_length=int(num_words),
                                      early_stopping=True)
    output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


# In[ ]:



df = clean_pmc
df = df.abstract.dropna()
abstracts = df.values.tolist()

len(abstracts)


# summarizing first 20 papers

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i in range(20):\n    try:\n        print(\'paper  \',i + 1, " : \\n" )\n        print(bart_summarize(abstracts[i]))\n        print(\'............................................................................\\n\\n\\n\\n\')\n    except:\n        print(\'paper \',i+1 ," has LONG ABSTRACT\\n\\n")')


# # Summarization Task using T5 model

# In[ ]:



df1 = biorxiv_clean
df1 = df1.abstract.dropna()
df1abstracts = df.values.tolist()

len(df1abstracts)


# In[ ]:


T5_PATH = 't5-base'
t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH, output_past=True)
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def t5_summarize(input_text, num_beams=4, num_words=80):
    #input_text = str(input_text).replace('\n', '')
    input_text = ' '.join(input_text.split())
    input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
    summary_task = torch.tensor([[21603, 10]]).to(device)
    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
    summary_ids = t5_model.generate(input_tokenized,
                                    num_beams=int(num_beams),
                                    no_repeat_ngram_size=3,
                                    length_penalty=2.0,
                                    min_length=30,
                                    max_length=int(num_words),
                                    early_stopping=True)
    output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


# summariing first 20 papers abstract of bioarvix

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i in range(20):\n    try:\n        print(\'BioArvix paper  \',i + 1, " : \\n" )\n        print(t5_summarize(df1abstracts[i]))\n        print(\'............................................................................\\n\\n\\n\\n\')\n    except:\n        print(\'paper \',i+1 ," has LONG ABSTRACT\\n\\n")')


# In[ ]:


simsentence.head(5)


# In[ ]:


get_ipython().system('pip install sentence-transformers')
"""
This is a simple application for sentence embeddings: semantic search
We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.
This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
# taken from : https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py
from sentence_transformers import SentenceTransformer
import scipy.spatial

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = simsentence.similar.tolist()
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = ['Range of incubation periods for the disease in humans', 'antiviral covid-19 success treatment','virus detected from animals?', 'risk of fatality among symptomatic hospitalized patients']
query_embeddings = embedder.encode(queries)

# Find the closest  sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences from similar")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))


# In[ ]:


"""
This is a simple application for sentence embeddings: semantic search
We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.
This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
# taken from : https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py
from sentence_transformers import SentenceTransformer
import scipy.spatial

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = df.values.tolist()
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = ['Range of incubation periods for the disease in humans','risk factors of covid-19','cure for covid-19', 'antiviral covid-19 success treatment','Does smoking or pre-existing pulmonary disease increase risk of COVID-19?', 'risk of fatality among symptomatic hospitalized patients']
query_embeddings = embedder.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))


# # Part 5 : Covid Paper Browser
# 
# **Browse Covid-19 & SARS-CoV-2 Scientific Papers with Transformers 🦠 📖**
# 
# i will  use  model : bert-base-nli-stsb-mean-tokens
# 
# more details can be found here : [COVID-19 Bert Literature Search Engine](https://towardsdatascience.com/covid-19-bert-literature-search-engine-4d06cdac08bd)
# 
# # ref : [Browse Covid-19](https://github.com/gsarti/covid-papers-browser)

# In[ ]:


pip install transformers


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import os\nimport tqdm\nimport textwrap\nimport json\nimport prettytable\nimport logging\nimport pickle\nimport warnings\nwarnings.simplefilter(\'ignore\')\n\nfrom  transformers import *\nimport pandas as pd\nimport scipy\nfrom sentence_transformers import SentenceTransformer\n\nCOVID_BROWSER_ASCII = """\n================================================================================\n  _____           _     _      __  ___    ____                                  \n / ____|         (_)   | |    /_ |/ _ \\  |  _ \\                                 \n| |     _____   ___  __| | ___ | | (_) | | |_) |_ __ _____      _____  ___ _ __ \n| |    / _ \\ \\ / / |/ _` ||___|| |\\__, | |  _ <| \'__/ _ \\ \\ /\\ / / __|/ _ \\ \'__|\n| |___| (_) \\ V /| | (_| |     | |  / /  | |_) | | | (_) \\ V  V /\\__ \\  __/ |   \n \\_____\\___/ \\_/ |_|\\__,_|     |_| /_/   |____/|_|  \\___/ \\_/\\_/ |___/\\___|_|   \n=================================================================================\n"""\n\nCOVID_BROWSER_INTRO = """\nThis demo uses a state-of-the-art language model trained on scientific papers to\nsearch passages matching user-defined queries inside the COVID-19 Open Research\nDataset. Ask something like \'Is smoking a risk factor for Covid-19?\' to retrieve\nrelevant abstracts.\\n\n"""\n\nBIORXIV_PATH = \'/kaggle/input/CORD-19-research-challenge//biorxiv_medrxiv/biorxiv_medrxiv/\'\nCOMM_USE_PATH = \'/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/\'\nNONCOMM_USE_PATH = \'/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/\'\nMETADATA_PATH = \'/kaggle/input/CORD-19-research-challenge/metadata.csv\'\n\nDATA_PATH = \'/kaggle/input/CORD-19-research-challenge/\'\nMODELS_PATH = \'models\'\nMODEL_NAME = \'scibert-nli\'\nCORPUS_PATH = os.path.join(DATA_PATH, \'corpus.pkl\')\nMODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)\nEMBEDDINGS_PATH = os.path.join(DATA_PATH, f\'{MODEL_NAME}-embeddings.pkl\')\n\n\ndef load_json_files(dirname):\n    filenames = [file for file in os.listdir(dirname) if file.endswith(\'.json\')]\n    raw_files = []\n\n    for filename in tqdm(filenames):\n        filename = dirname + filename\n        file = json.load(open(filename, \'rb\'))\n        raw_files.append(file)\n    print(\'Loaded\', len(raw_files), \'files from\', dirname)\n    return raw_files\n\n\ndef create_corpus_from_json(files):\n    corpus = []\n    for file in tqdm(files):\n        for item in file[\'abstract\']:\n            corpus.append(item[\'text\'])\n        for item in file[\'body_text\']:\n            corpus.append(item[\'text\'])\n    print(\'Corpus size\', len(corpus))\n    return corpus\n\n\ndef cache_corpus(mode=\'CSV\'):\n    corpus = []\n    if mode == \'CSV\':\n        df = pd.read_csv(METADATA_PATH)\n        corpus = [a for a in df[\'abstract\'] if type(a) == str and a != "Unknown"]\n        print(\'Corpus size\', len(corpus))\n    elif mode == \'JSON\':\n        biorxiv_files = load_json_files(BIORXIV_PATH)\n        comm_use_files = load_json_files(COMM_USE_PATH)\n        noncomm_use_files = load_json_files(NONCOMM_USE_PATH)\n        corpus = create_corpus_from_json(biorxiv_files + comm_use_files + noncomm_use_files)\n    else:\n        raise AttributeError(\'Mode should be either CSV or JSON\')\n    \'\'\'with open(CORPUS_PATH, \'wb\') as file:\n        pickle.dump(corpus, file)\'\'\'\n    return corpus\n\n\ndef ask_question(query, model, corpus, corpus_embed, top_k=5):\n    """\n    Adapted from https://www.kaggle.com/dattaraj/risks-of-covid-19-ai-driven-q-a\n    """\n    queries = [query]\n    query_embeds = model.encode(queries, show_progress_bar=False)\n    for query, query_embed in zip(queries, query_embeds):\n        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]\n        distances = zip(range(len(distances)), distances)\n        distances = sorted(distances, key=lambda x: x[1])\n        results = []\n        for count, (idx, distance) in enumerate(distances[0:top_k]):\n            results.append([count + 1, corpus[idx].strip(), round(1 - distance, 4)])\n    return results\n\n\ndef show_answers(results):\n    table = prettytable.PrettyTable(\n        [\'Rank\', \'Abstract\', \'Score\']\n    )\n    for res in results:\n        rank = res[0]\n        text = res[1]\n        text = textwrap.fill(text, width=75)\n        text = text + \'\\n\\n\'\n        score = res[2]\n        table.add_row([\n            rank,\n            text,\n            score\n        ])\n    print(\'\\n\')\n    print(str(table))\n    print(\'\\n\')\n\nif __name__ == \'__main__\':\n    os.system(\'cls\' if os.name == \'nt\' else \'clear\')\n    print(COVID_BROWSER_ASCII)\n    print(COVID_BROWSER_INTRO)\n    if not os.path.exists(CORPUS_PATH):\n        print("Caching the corpus for future use...")\n        corpus = cache_corpus()\n    else:\n        print("Loading the corpus from", CORPUS_PATH, \'...\')\n        with open(CORPUS_PATH, \'rb\') as corpus_pt:\n            corpus = pickle.load(corpus_pt)\n\n    model =  SentenceTransformer(\'bert-base-nli-stsb-mean-tokens\')\n\n    if not os.path.exists(EMBEDDINGS_PATH):\n        print("Computing and caching model embeddings for future use...")\n        embeddings = model.encode(corpus, show_progress_bar=True)\n        \'\'\'with open(EMBEDDINGS_PATH, \'wb\') as file:\n            pickle.dump(embeddings, file)\'\'\'\n    else:\n        print("Loading model embeddings from", EMBEDDINGS_PATH, \'...\')\n        with open(EMBEDDINGS_PATH, \'rb\') as file:\n            embeddings = pickle.load(file)\n\n    ')


# In[ ]:


questions = ['Is smoking a risk factor for Covid-19?','What has been published about medical care?','Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities','risk for  Neonates and pregnant women?','Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.']
for i in range(len(questions)):
        query = questions[i]
        print(f'Query {i+1} : {query}\n\n')
        results = ask_question(query, model, corpus, embeddings)
        show_answers(results)


# <h1 style="color:purple;">
# To be continued..................... 👨‍⚕️ 
# </h1>

# <h1 align="left" style="color:red;">
# If you find this kernel interesting, please drop an<br>  <font color="blue">UPVOTE</font>. It motivates me to produce more quality contents 🤗
# </h1>

# In[ ]:




