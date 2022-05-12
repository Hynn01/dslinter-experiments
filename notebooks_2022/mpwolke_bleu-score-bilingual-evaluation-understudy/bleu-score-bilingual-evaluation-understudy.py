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


# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: #8FBC8F;"><b style="color:black;">BLEU Score(Bilingual Evaluation Understudy)</b></h1></center>
# 
# "BLEU (Bilingual Evaluation Understudy) is a measurement of the differences between an automatic translation and one or more human-created reference translations of the same source sentence."
# 
# Scoring process
# 
# "The BLEU algorithm compares consecutive phrases of the automatic translation with the consecutive phrases it finds in the reference translation, and counts the number of matches, in a weighted fashion. These matches are position independent. A higher match degree indicates a higher degree of similarity with the reference translation, and higher score. Intelligibility and grammatical correctness are not taken into account."
# 
# https://docs.microsoft.com/en-us/azure/cognitive-services/translator/custom-translator/what-is-bleu-score

# ![](https://www.journaldev.com/wp-content/uploads/2020/12/bleu_score.png)journaldev.com

# In[ ]:


df = pd.read_csv('../input/machine-translation-mbr-with-neural-metrics/mbr_neural_metrics/de-en/newstest2021/human_eval/mqm_mbr_deen.tsv', sep='\t', error_bad_lines=False,warn_bad_lines=False)
df.tail()


# #Sentence BLEU Score
# 
# "NLTK provides the sentence_bleu() function for evaluating a candidate sentence against one or more reference sentences."
# 
# "The reference sentences must be provided as a list of sentences where each reference is a list of tokens. The candidate sentence is provided as a list of tokens. For example:"
# 
# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'new', 'season', 'starts'], ['new' 'season', 'starts']]
candidate = ['the', 'new', 'season', 'starts']
score = sentence_bleu(reference, candidate)
print(score)


# #Comparing German with English

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

from nltk.translate.bleu_score import sentence_bleu
reference = [['Die', 'Corona', 'krise', 'hat', 'verhindert'], ['Corona', 'krise', 'hat', 'verhindert']]
candidate = ['The', 'Corona', 'crisis', 'has', 'prevented']
score = sentence_bleu(reference, candidate)
print(score)


# #German with German

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

from nltk.translate.bleu_score import sentence_bleu
reference = [['Mit', 'viel', 'Verzögerung', 'startet'], ['Mit','Verzögerung', 'startet']]
candidate = ['Mit ', 'fast', 'einem', 'halben','Jahr' 'Verzögerung', 'startet']
score = sentence_bleu(reference, candidate)
print(score)


# #Corpus BLEU Score
# 
# "NLTK also provides a function called corpus_bleu() for calculating the BLEU score for multiple sentences such as a paragraph or a document."
# 
# "The references must be specified as a list of documents where each document is a list of references and each alternative reference is a list of tokens, e.g. a list of lists of lists of tokens. The candidate documents must be specified as a list where each document is a list of tokens, e.g. a list of lists of tokens."
# 
# "This is a little confusing; here is an example of two references for one document."
# 
# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# two references for one document
from nltk.translate.bleu_score import corpus_bleu
references = [[['the', 'main', 'focus', 'was', 'to', 'get'], ['main', 'focus', 'was', 'get']]]
candidates = [['the', 'main', 'focus', 'was', 'to', 'get']]
score = corpus_bleu(references, candidates)
print(score)


# #Bilingual Evaluation Understudy Score
# 
# By Jason Brownlee Last Updated on December 19, 2019 (A Gentle Introduction to Calculating the BLEU Score for Text in Python)
# 
# "The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric for evaluating a generated sentence to a reference sentence."
# 
# "A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0."
# 
# "The score was developed for evaluating the predictions made by automatic machine translation systems. It is not perfect, but does offer 5 compelling benefits:"
# 
# It is quick and inexpensive to calculate.
# 
# It is easy to understand.
# 
# It is language independent.
# 
# It correlates highly with human evaluation.
# 
# It has been widely adopted.
# 
# The BLEU score was proposed by Kishore Papineni, et al. in their 2002 paper “BLEU: a Method for Automatic Evaluation of Machine Translation“.
# 
# "The approach works by counting matching n-grams in the candidate translation to n-grams in the reference text, where 1-gram or unigram would be each token and a bigram comparison would be each word pair. The comparison is made regardless of word order."
# 
# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# two references for one document
from nltk.translate.bleu_score import corpus_bleu
references = [[['Sie', 'sind', 'so', 'klein'], ['Sie', 'sind', 'klein']]]
candidates = [['Sie', 'sind', 'so', 'winzig']]
score = corpus_bleu(references, candidates)
print(score)


# #Individual N-Gram Scores
# 
# "An individual N-gram score is the evaluation of just matching grams of a specific order, such as single words (1-gram) or word pairs (2-gram or bigram)."
# 
# "The weights are specified as a tuple where each index refers to the gram order. To calculate the BLEU score only for 1-gram matches, you can specify a weight of 1 for 1-gram and 0 for 2, 3 and 4 (1, 0, 0, 0). For example:"
# 
# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['Ihre', 'Haut', 'ist', 'manchmal', 'so', 'zart']]
candidate = ['Ihre', 'Haut', 'ist', 'zart']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)


# #Decoding and Diversity in Machine Translation
# 
# Authors: Nicholas Roberts, Davis Liang, Graham Neubig, Zachary C. Lipton
# 
# "Neural Machine Translation (NMT) systems are typically evaluated using automated metrics that assess the agreement between generated translations and ground truth candidates. To improve systems with respect to these metrics, NLP researchers employ a variety of heuristic techniques, including searching for the conditional mode (vs. sampling) and incorporating various training heuristics (e.g., label smoothing)."
# 
# "While search strategies significantly improve BLEU score, they yield deterministic outputs that lack the diversity of human translations. Moreover, search can amplify socially problematic biases in the data, as has been observed in machine translation of gender pronouns. This makes human-level BLEU a misleading benchmark; modern MT systems cannot approach human-level BLEU while simultaneously maintaining
# human-level translation diversity."
# 
# "In this paper, the authors characterized distributional differences between generated and real translations, examining the cost in diversity paid for the BLEU scores enjoyed by NMT. Moreover, their study implicated search as a salient source of known bias when translating gender pronouns."
# 
# https://arxiv.org/pdf/2011.13477.pdf

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# n-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['Ihre', 'Haut', 'ist', 'manchmal', 'so', 'zart']]
candidate = ['Ihre', 'Haut', 'ist', 'zart']
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))


# #Cumulative N-Gram Scores
# 
# "Cumulative scores refer to the calculation of individual n-gram scores at all orders from 1 to n and weighting them by calculating the weighted geometric mean."
# 
# "By default, the sentence_bleu() and corpus_bleu() scores calculate the cumulative 4-gram BLEU score, also called BLEU-4."
# 
# "The weights for the BLEU-4 are 1/4 (25%) or 0.25 for each of the 1-gram, 2-gram, 3-gram and 4-gram scores. For example:"
# 
# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# 4-gram cumulative BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['Und', 'sie', 'sind', 'so', 'winzig']]
candidate = ['Sie', 'sind','winzig']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)


# "The cumulative and individual 1-gram BLEU use the same weights, e.g. (1, 0, 0, 0). The 2-gram weights assign a 50% to each of 1-gram and 2-gram and the 3-gram weights are 33% for each of the 1, 2 and 3-gram scores."
# 
# "Let’s make this concrete by calculating the cumulative scores for BLEU-1, BLEU-2, BLEU-3 and BLEU-4:"
# 
# #Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
reference = [['Und', 'sie', 'sind', 'so', 'winzig']]
candidate = ['Sie', 'sind','winzig']
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))


# #A perfect score. A perfect match

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# prefect match
from nltk.translate.bleu_score import sentence_bleu
reference = [['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']]
candidate = ['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']
score = sentence_bleu(reference, candidate)
print(score)


# #Let’s change one word

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# one word different
from nltk.translate.bleu_score import sentence_bleu
reference = [['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']]
candidate = ['Sie', 'sind', 'so', 'winzig', 'und', 'so', 'früh', 'geboren']
score = sentence_bleu(reference, candidate)
print(score)


# #Changing two words

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# two words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']]
candidate = ['Sie', 'sind', 'so', 'winzig', 'und', 'so', 'spät', 'geboren']
score = sentence_bleu(reference, candidate)
print(score)


# #All words are different in the candidate

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# all words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']]
candidate = ['Ihre', 'Haut', 'ist', 'manchmal', 'zart']
score = sentence_bleu(reference, candidate)
print(score)


# #A candidate that has fewer words than the reference

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# shorter candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']]
candidate = ['Ihre', 'Haut', 'ist', 'manchmal', 'so', 'zart']
score = sentence_bleu(reference, candidate)
print(score)


# #Making the candidate two words longer than the reference

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# longer candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['Sie', 'sind', 'klein', 'und', 'früh', 'geboren']]
candidate = ['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']
score = sentence_bleu(reference, candidate)
print(score)


# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# very short
from nltk.translate.bleu_score import sentence_bleu
reference = [['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']]
candidate = ['Sie', 'sind', 'klein']
score = sentence_bleu(reference, candidate)
print(score)


# #Calculating the Bilingual Evaluation Understudy (BLEU) score
# 
# Cloistered Monkey 2021-02-11 19:51 
# 
# "The author will be implementing a popular metric for evaluating the quality of machine-translated text: the BLEU score proposed by Kishore Papineni, et al. In their 2002 paper "BLEU: a Method for Automatic Evaluation of Machine Translation", the BLEU score works by comparing "candidate" text to one or more "reference" translations. The result is better the closer the score is to 1. "
# 
# https://necromuralist.github.io/Neurotic-Networking/posts/nlp/bleu-score/

# In[ ]:


#Code by https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# very very short
from nltk.translate.bleu_score import sentence_bleu
reference = [['Sie', 'sind', 'so', 'klein', 'und', 'so', 'früh', 'geboren']]
candidate = ['Sie']
score = sentence_bleu(reference, candidate)
print(score)


# #Acknowledgement:
# 
# A Gentle Introduction to Calculating the BLEU Score for Text in Python - By Jason Brownlee on November 20, 2017 
# 
#  https://machinelearningmastery.com/calculate-bleu-score-for-text-python/  
