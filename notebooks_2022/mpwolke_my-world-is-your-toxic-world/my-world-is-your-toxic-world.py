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


# In[ ]:


df = pd.read_csv("/kaggle/input/jigsaw-specialized-rater-pools-dataset/specialized_rater_pools_data.csv", delimiter=',', encoding='utf8')
pd.set_option('display.max_columns', None)
df.head()


# #Columns in the data are the following:
# 
# id: The id of the comment from the CivilComments data.
# 
# unique_contributor_id: A pseudonymized id for the annotator.
# 
# identity_attack: The annotator's score for the "identity_attack" category. This is a value of -1 (identity attack), 0 (unsure), or 1 (not an - identity attack).
# 
# insult: The annotator's score for the "insult" category. This is a value of -1 (insult), 0 (unsure), or 1 (not an insult).
# 
# obscene: The annotator's score for the "obscene" (profanity) category. This is a value of -1 (profanity), 0 (unsure), or 1 (not profanity).
# 
# threat: The annotator's score for the "threat" category. This is a value of -1 (threat), 0 (unsure), or 1 (not a threat).
# toxic_score: The annotator's score for the "toxicity" category. This is a value of -2 (very toxic), -1 (toxic), 0 (unsure), or 1 (not toxic).
# 
# comment_text: The text of the comment.
# 
# rater_group: The rater group the annotator was a part of. This is a value of "African American", "LGBTQ", or "Control".
# 
# https://www.kaggle.com/datasets/google/jigsaw-specialized-rater-pools-dataset

# #My World Is your World - Bee Gees
# 
# <iframe width="707" height="530" src="https://www.youtube.com/embed/793texA5NhM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
# 
# https://www.youtube.com/watch?v=793texA5NhM

# #Whenever I searched My world is your World (1972) Bee Gees appeared some Bieber Bull shit. Damm Toxic Search!

# In[ ]:


df.isnull().sum()


# #It's Rater (one having a specified rating or class) or Hater?

# In[ ]:


df["rater_group"].value_counts()


# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'Red',
                      height =2000,
                      width = 2000
                     ).generate(str(df["rater_group"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Rater Group")
plt.show()


# In[ ]:


df["insult"].value_counts()


# In[ ]:


#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

df["insult"].value_counts().plot.bar(color=['blue', '#f5005a', '#7FFF00'], title='Insults')
plt.figure(figsize=(8,4));


# In[ ]:


df["obscene"].value_counts()


# In[ ]:


#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

df["obscene"].value_counts().plot.barh(color=['#FF7F50', '#008B8B', '#9932CC'], title='Obscenities')
plt.figure(figsize=(8,4));


# In[ ]:


#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

df["threat"].value_counts().plot.barh(color=['#B22222', '#4B0082', '#20B2AA'], title='Threats')
plt.figure(figsize=(8,4));


# In[ ]:


#Code by MD. Jafril Alam Shihab https://www.kaggle.com/code/mdjafrilalamshihab/eda-of-russia-ukraine-war

df.boxplot(by='toxic_score')
plt.xticks(rotation=45);


# In[ ]:


#Code by Siti K https://www.kaggle.com/khotijahs1/2020-indonesia-university-rank/comments

#Toxicity by Toxic Score
toxicity = df.sort_values(by='toxic_score', ascending=False)[:20]
figure = plt.figure(figsize=(10,6))
sns.barplot(y=toxicity.rater_group, x=toxicity.toxic_score)
plt.xticks()
plt.xlabel('Toxic Score')
plt.ylabel('Rater Group')
plt.title('Rater Groups by Toxic Score')
plt.show()


# In[ ]:


#382499th row, 8th column 

df.iloc[382499, 7]


# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTCr_68__2LdbOO9h5urKpN-xh_wG1G-Vbz0Q&usqp=CAU)independent.co.uk

# In[ ]:


#2nd row, 8th column 

df.iloc[2, 7]


# In[ ]:


#15th row, 8th column 

df.iloc[15, 7]


# In[ ]:


#11th row, 8th column 

df.iloc[11, 7]


# In[ ]:


#22nd row, 8th column 

df.iloc[22, 7]


# In[ ]:


#382477th row, 8th column 

df.iloc[382477, 7]


# In[ ]:


#382482nd row, 8th column 

df.iloc[382482, 7]


# In[ ]:


#35th row, 8th column 

df.iloc[35, 7]


# In[ ]:


#44th row, 8th column 

df.iloc[44, 7]


# #Post with the most comments

# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

# post with the most comments

df[df['toxic_score'] == df['toxic_score'].max()]['unique_contributor_id'].iloc[0]


# #Cleaning functions

# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

def remove_line_breaks(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text

#remove punctuation
def remove_punctuation(text):
    re_replacements = re.compile("__[A-Z]+__")  # such as __NAME__, __LINK__
    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
    '''Escape all the characters in pattern except ASCII letters and numbers'''
    tokens = word_tokenize(text)
    tokens_zero_punctuation = []
    for token in tokens:
        if not re_replacements.match(token):
            token = re_punctuation.sub(" ", token)
        tokens_zero_punctuation.append(token)
    return ' '.join(tokens_zero_punctuation)

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def lowercase(text):
    text_low = [token.lower() for token in word_tokenize(text)]
    return ' '.join(text_low)

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in word_tokens if word not in stop])
    return text

#remobe one character words
def remove_one_character_words(text):
    '''Remove words from dataset that contain only 1 character'''
    text_high_use = [token for token in word_tokenize(text) if len(token)>1]      
    return ' '.join(text_high_use)   
    
#%%
# Stemming with 'Snowball stemmer" package
def stem(text):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    text_stemmed = [stemmer.stem(token) for token in word_tokenize(text)]        
    return ' '.join(text_stemmed)

def lemma(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = nltk.word_tokenize(text)
    text_lemma = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokens])       
    return ' '.join(text_lemma)


#break sentences to individual word list
def sentence_word(text):
    word_tokens = nltk.word_tokenize(text)
    return word_tokens
#break paragraphs to sentence token 
def paragraph_sentence(text):
    sent_token = nltk.sent_tokenize(text)
    return sent_token    


def tokenize(text):
    """Return a list of words in a text."""
    return re.findall(r'\w+', text)

def remove_numbers(text):
    no_nums = re.sub(r'\d+', '', text)
    return ''.join(no_nums)



def clean_text(text):
    _steps = [
    remove_line_breaks,
    remove_one_character_words,
    remove_special_characters,
    lowercase,
    remove_punctuation,
    remove_stopwords,
    stem,
    remove_numbers
]
    for step in _steps:
        text=step(text)
    return text   
#%%


# In[ ]:


#https://stackoverflow.com/questions/55557004/getting-attributeerror-float-object-has-no-attribute-replace-error-while
#To avoid with tqdm AttributeError: 'float' object has no attribute

df["comment_text"] = df["comment_text"].astype(str)
df["comment_text"] = [x.replace(':',' ') for x in df["comment_text"]]


# In[ ]:


words = df["comment_text"].values


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

ls = []

for i in words:
    ls.append(str(i))


# In[ ]:


ls[:5]


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

most_toxic = df.sort_values('toxic_score', ascending =False)[['id', 'toxic_score']].head(12)

most_toxic['toxic_score1'] = most_toxic['toxic_score']/1000


# In[ ]:



#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

plt.figure(figsize = (8,8))

sns.barplot(data = df, y = 'id', x = 'toxic_score', color = 'c')
plt.xticks(fontsize=27, rotation=0)
plt.yticks(fontsize=31, rotation=0)
plt.xlabel('Toxicity', fontsize = 21)
plt.ylabel('Count')
plt.title('Most toxic posts', fontsize = 30);


# In[ ]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk


# In[ ]:


stemmer = SnowballStemmer('english')


# In[ ]:


nltk.download('wordnet')


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[ ]:


df['comment_text'].iloc[35]


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

doc_sample = df['comment_text'].iloc[1]
print('original document: ')

words = []

for word in doc_sample.split(' '):
    words.append(word)
    
    
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


# In[ ]:


df['comment_text'] = df['comment_text'].astype(str)


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

words = []

for i in df['comment_text']:
        words.append(i.split(' '))


# #Create the dictionary
# 
# Every unique word in comment_text

# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

dictionary = gensim.corpora.Dictionary(words)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[ ]:


# Filter out tokens in the dictionary by their frequency.

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# #Create Corpus -> term document frequency
# 
# doc2bow() simply counts the number of occurrences of each distinct word, converts the word to its integer word ID and returns the result as a sparse vector.

# In[ ]:


bow_corpus = [dictionary.doc2bow(doc) for doc in words]
bow_corpus[4310]


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

bow_doc_4310 = bow_corpus[4310]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))


# #Bigotry appeared 2 times?

# TF/IDF

# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

lda_model = gensim.models.LdaMulticore(bow_corpus,
                                       num_topics=10,
                                       id2word=dictionary,
                                       passes=2,
                                       workers=2)


# #Show the output of the model

# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf,
                                             num_topics=10,
                                             id2word=dictionary,
                                             passes=2,
                                             workers=4)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

unseen_document = 'It’s nice to be surrounded by people who share your models" - Do mpwolke has models?'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


processed_text = df['comment_text']


# #Create TF/IDF again

# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(processed_text)
print(tfidf.shape)
print('\n')
#print(vectorizer.get_feature_names())


# In[ ]:


y = df['comment_text']


# In[ ]:


X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(tfidf, y, test_size=0.2, random_state=42)


# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train_tf,y_train_tf)
# predict the labels on validation dataset
predictions_NB_tf = Naive.predict(X_test_tf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy -> ",accuracy_score(predictions_NB_tf, y_test_tf)*100)
print(classification_report(predictions_NB_tf,y_test_tf))


# #Your notebook tried to allocate more memory than is available. It has restarted. That's the most toxic in Kaggle!

# In[ ]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

#Save for the next since in subreddit.nsfw  ALL is False

#logmodel = LogisticRegression()
#logmodel.fit(X_train_tf, y_train_tf)

#predictions_LR_tf = logmodel.predict(X_test_tf)

#print("LR Accuracy -> ",accuracy_score(predictions_LR_tf, y_test_tf)*100)
#print(classification_report(predictions_LR_tf,y_test_tf))


# #Acknowledgements:
# 
# Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction
# 
# MD. Jafril Alam Shihab https://www.kaggle.com/code/mdjafrilalamshihab/eda-of-russia-ukraine-war
# 
# Siti K https://www.kaggle.com/khotijahs1/2020-indonesia-university-rank/comments
# 
# Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

# #My world is Your World  (Bee Gees Not Bieber. Dammit!)
# 
# "My world is our world"
# 
# "And this world is your world"
# 
# "And your world is my world"
# 
# "And my world is your world is mine"
# 
# 
# "Don't shed a tear for me"
# 
# "That's not your style"
# 
# "If you're not here by me"
# 
# "Then it's Still worth while"  (That's my toxic version)
# 
# Music/Song by Robin Gibb and Barry Gibb
# 
# https://www.letras.mus.br/bee-gees/1304328/traducao.html

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcToMSbyuqx8J-HNx4RynBPypt_oaXq7KFEzmA&usqp=CAU)chordify.net
