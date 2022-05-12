#!/usr/bin/env python
# coding: utf-8

# # Introducing a Self-Organizing Scientific Research Understanding System (SOSRUS)
# 
# 
# **Scenario:** A wordlwide pandemic has just broken out and you are an epedemiologist, medical researcher etc. and  The Whitehouse is looking to you for answers about what we know about drugs that may be an effective treatment?
# 
# **Problem:** The medical research is being published fast and furious and it is nearly impossible to stay on top of all the topics of all the documents published daily, let alone try to have a deep understanding of what all the documents may hold regarding a key to a promising drug.
# 
# **Current approach:** Currently, the approach to this problem would be to put all these documents into a database such as Google Scholar, Pubmed etc., and then begin searching for drugs that may show promise using keywords and trial and error. The user would learn something from each search and write down new keywords, drug names and concepts - slowly putting together a picture of drugs that may show promise.  All of this information would have to be distilled down by the user into a digestable and shareable format of everything they learned about effective drugs.
# 
# **Solution:** We propose a new approach to this problem called SOSRUS - Self-Organizing Scientific Research Understanding System.  In this notebook, we will demonstrate how the system, given just the word "drug", will analyze all the documents relating to the term drug and create a language model to automatically understand and uncover important topics about drugs and expand on them, ultimately providing an organized and concise output that summarizies all the information about drugs into an easy to read format.
# 
# **Goal:** To avoid a hand coded rules system and have the system actually learn concepts on its own starting with one word or short phrase.  The system, on its own, should expand, learn and present relevant information and supporting documents in a helpful and understandable manner for human understanding.
# 
# # Step 1: Import Python packages and full-text documents:
# Here the system will import all python packages needed to create itself.  The system will also import the data from the meta CSV file into a Pandas dataframe and filter out the documents that are not relevant to COVID-19 COV-2 etc.
# 
# Then the full text versions of the filtered documents will be loaded from the JSON files into the correpsonding rows of the dataframe.  If there is a full text, that is saved in the abstract column, if there is not full text version, the abstract is used for those papers.
# 
# Finally in this step, the system will focus only on documents that contain the word drug.

# In[ ]:


import spacy
nlp = spacy.load('en_core_web_lg')
import numpy as np
import pandas as pd
import re
import os
import json
from pprint import pprint
from copy import deepcopy
import math
import torch
get_ipython().system('pip install rake-nltk')
from rake_nltk import Rake
from nltk.corpus import stopwords
from rake_nltk import Metric, Rake
from IPython.display import display, Markdown, Latex, HTML

get_ipython().system('pip install bert-extractive-summarizer')
from summarizer import Summarizer
model = Summarizer()


# In[ ]:


print ('python packages imported')

# keep only documents with covid -cov-2 and cov2
def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    frames=[dfa,dfb,dfc,dfd]
    df = pd.concat(frames)
    df=df.drop_duplicates(subset='title', keep="first")
    return df

# load the meta data from the CSV file using 3 columns (abstract, title, authors),
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time','sha','full_text_file'])
print ('total documents ',df.shape)
#drop duplicates
#df=df.drop_duplicates()
#drop NANs 
df=df.fillna('no data provided')
df = df.drop_duplicates(subset='title', keep="first")
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
#show 5 lines of the new dataframe
df=search_focus(df)
df = df[df['publish_time'].str.contains('2020')]
print ('COVID-19 focused documents ',df.shape)
df.head()

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body


for index, row in df.iterrows():
    if ';' not in row['sha'] and os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['full_text_file']+'/'+row['full_text_file']+'/pdf_json/'+row['sha']+'.json')==True:
        with open('/kaggle/input/CORD-19-research-challenge/'+row['full_text_file']+'/'+row['full_text_file']+'/pdf_json/'+row['sha']+'.json') as json_file:
            data = json.load(json_file)
            body=format_body(data['body_text'])
            keyword_list=['TB','incidence','age']
            #print (body)
            body=body.replace("\n", " ")

            df.loc[index, 'abstract'] = body.lower()

df=df.drop(['full_text_file'], axis=1)
df=df.drop(['sha'], axis=1)
df.head()


# add full text back after testing


# In[ ]:


import functools
def search_focus_shape(df,focus):
    df1=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in focus))]
    return df1

focus_term='drug'
df1=search_focus_shape(df,focus_term)
print ('focus term: ',focus_term)
print ('# focused papers',df1.shape)


# # Step 2: Word proximity / pseudo bi-gram language model
# In this step, the system analyzes all the documents containing the word "drug" and extracts words in close proximity.  The theory here is that proximity of words helps ensure relevance and importance to "drug" and proximity also carries some weight relating to the symantic understanding. [Read about proximity search](https://en.wikipedia.org/wiki/Proximity_search_(text) After words in close proximity to "drug" are extracted from the documents, NLTK rapid automatic keyword extraction (RAKE) [Read about RAKE](https://csurfer.github.io/rake-nltk/_build/html/advanced.html) is used to extract keywords and produce a list of bi-grams that we use as a pseudo bi-gram language model to get better understanding of key terms used when discussing drugs in the CORD19 corpus. [Read about N-gram language models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
# 
# The system then presents an alphabetical list of keywords that increase the understanding about how drugs are discussed in the corpus.

# In[ ]:


r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,min_length=2, max_length=2) # Uses stopwords for english from NLTK, and all puntuation characters.Please note that "hello" is not included in the list of stopwords.

def extract_data(text,word):
    extract=''
    if word in text:
        #text = re.sub(r'[^\w\s]','',text)
        res = [i.start() for i in re.finditer(word, text)]
        after=text[res[0]:res[0]+15]
        before=text[res[0]-15:res[0]]
        raw = before+after
        parts=raw.split()
        parts = parts[1:-1]
        extract= ' '.join(parts)
        extract=extract.replace('drugs','drug')
    return extract
text=''
for index, row in df1.iterrows():
    extracted=extract_data(row['abstract'],focus_term)
    if extracted!='':
        text=text+' '+extracted
a=r.extract_keywords_from_text(text)
term_list=r.get_ranked_phrases()
term_list = sorted(term_list, key=str.lower)
#c=r.get_ranked_phrases_with_scores()
print(term_list)
#print(c)
print('___________________')


# # Step 3 - The system analyzes all documents at sentence level using the RAKE key words
# In this step, the list of keywords extracted with RAKE are now used to search the "drug" papers and return relevant excerpts at sentence level.  These excerpts and the corresponding papers are organized by publish date and presented in tables organized alphabeticlly for each RAKE keyword / topic.  This makes it easy for a researcher to review a sea of documents and  to quickly drill down on the topics that seem to have promise.

# In[ ]:


# custom sentence score
def score_sentence_prob(search,sentence,focus):
    final_score=0
    keywords=search.split()
    sent_parts=sentence.split()
    word_match=0
    missing=0
    for word in keywords:
        word_count=sent_parts.count(word)
        word_match=word_match+word_count
        if word_count==0:
            missing=missing+1
    percent = 1-(missing/len(keywords))
    final_score=abs((word_match/len(sent_parts)) * percent)
    if missing==0:
        final_score=final_score+.05
    if focus in sentence:
        final_score=final_score+.05
    return final_score

def score_docs(df,focus,search):
    df_results = pd.DataFrame(columns=['date','study','link','excerpt','score'])
    df1=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in focus))]
    master_text=''
    for index, row in df1.iterrows():
        pub_sentence=''
        sentences=row['abstract'].split('.')
        hi_score=0
        for sentence in sentences:
            if len(sentence)>75 and search in sentence:
                rel_score=score_sentence_prob(search,sentence,focus)
                #rel_score=score_sentence(search,sentence)
                if rel_score>.05:
                    #print (sentence,rel_score)
                    pub_sentence=pub_sentence+' '+sentence+' '+str(round(rel_score,2))
                    if rel_score>hi_score:
                        hi_score=rel_score
                    master_text=master_text+' '+pub_sentence
        if pub_sentence!='':
            #print (row['abstract'])
            #print ('------------------')
            link=row['doi']
            linka='https://doi.org/'+link
            to_append = [row['publish_time'],row['title'],linka,pub_sentence,hi_score]
            df_length = len(df_results)
            df_results.loc[df_length] = to_append
    df_results=df_results.sort_values(by=['date'], ascending=False)

    return df_results

def prepare_summary_answer(text,model):
    #model = pipeline(task="summarization")
    return model(text)


for term in term_list:
    if 'drug' in term and any(map(str.isdigit, term))==False and ')' not in term:
        #master_text=score_docs(df,focus_term,term)
        df_results=score_docs(df,focus_term,term)
        if df_results.empty==False:
            #print (term)
            #summarize questions
            #summary_answer=prepare_summary_answer(master_text,model)
            display(Markdown('# '+term))
            #display(HTML('<h4> Summarized Answer: </h4><i>'+summary_answer+'</i>'))
            df_table_show=HTML(df_results.to_html(escape=False,index=False))
            display(df_table_show)

