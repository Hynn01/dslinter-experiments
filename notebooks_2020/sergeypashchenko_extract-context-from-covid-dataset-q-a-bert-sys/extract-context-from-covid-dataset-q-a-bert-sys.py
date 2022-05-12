#!/usr/bin/env python
# coding: utf-8

# # **Problem statement:**
# Development of a system for the automatic search for answers and the formation of the context of documents
# 
# # **Problems:**
# The main problem of information retrieval is that the developed system should not only find sentences with similar words that appear in the request, but also take into account the context.
# 
# # **Solution:**
# To solve the Question-Answer problem, a pre-trained model of the BERT neural network on the squad dataset was used, and to obtain the main context of many articles, the TF-IDF algorithm with the KMeans clusterizer was used to visualize the data.

# # **STEP 1.** Read the data on json format from biorxiv, comm_use_subset, noncomm_use_subset.
# We form text documents that store the text of the papers, annotation, information about the authors and the title
# 
# If the paper does not contain a title, abstract or authors, the field is replaced with the value None

# In[ ]:


import os
import json
import time, sys
import re

MainDirectory = "../input/CORD-19-research-challenge/" # Path to coronavirus dataset
FoldersPath = ["biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/", "comm_use_subset/comm_use_subset/pdf_json/",
               "noncomm_use_subset/noncomm_use_subset/pdf_json/"]

Titles = []
Authors = []
Abstracts = []
Texts = []

for CurrentDirectory in FoldersPath:
    print("Current Directory")
    print(CurrentDirectory)
   
    Directory = MainDirectory + CurrentDirectory
    Files = os.listdir(Directory)
    print("Number files in directory")
    print(len(Files))
    
    print("Status Preprocessing, %:")
    Counter = 0
    for CurrentFileName in Files:
        
        FileStream = open(Directory + str(CurrentFileName), "r")
        CurrentFile = json.load(FileStream)
        FileStream.close()
        
        sys.stdout.write(str(Counter/len(Files) * 100))
        sys.stdout.flush()
        sys.stdout.write("\r")
        Counter += 1

        Author = ""
        AllKeys = CurrentFile.keys()
        AllKeys = list(AllKeys)
        MetaDataKeys = CurrentFile[AllKeys[1]]

        Title = MetaDataKeys["title"]

        if len(MetaDataKeys["authors"]) > 0:
            Author += MetaDataKeys["authors"][0]["first"] + " " + MetaDataKeys["authors"][0]["last"]
        else:
            Author = "None"

        if len(CurrentFile[AllKeys[2]]) > 0:
            Abstract = CurrentFile[AllKeys[2]][0]["text"]
        else:
            Abstract = "None"

        if len(CurrentFile[AllKeys[3]]) > 0:
            Text = CurrentFile[AllKeys[3]][0]["text"]
        else:
            Text = "None"


        if len(Title) > 0:
            Titles.append(Title)
        else:
            Titles.append("None")

        if len(Author) > 0:
            Authors.append(Author)
        else:
            Authors.append("None")

        if len(Abstract) > 0:
            Abstracts.append(Abstract)
        else:
            Abstracts.append("None")

        if len(Text) > 0:
            Texts.append(Text)
        else:
            Texts.append("None")
print("\n")


NewTexts = []
for CurrentPaper in Texts:
    Result = re.split("\s[a-z]+\s", CurrentPaper)
    if len(Result) > 20:
        NewTexts.append(CurrentPaper)
        
Texts = NewTexts
print("Number Papers")
print(len(Texts))


# # **Attention!!! The code block discussed below will take about 40 minutes to complete on a full dataset.**
# 
# # **STEP 2.** Retrieving the context from the dataset.
# * The first step is to clean the dataset from the "garbage". Delete all characters that are not letters of the alphabet or numbers.
# 
# * We will form a text dataframe that includes the main text of the paper. Using the spacy library we bring all verbs to the correct form and all nouns to the singular. This approach allows you to reduce the sentences space of proposals and better understand the context.
# * To get the area of text representations the word bag approach was used with the countVectorizer library implemented in sklearn. However, this approach did not give satisfactory results, since the classes turned out to be significantly unbalanced. This means that the algorithm did a poor job.
# * The TF-IDF frequency coding approach implemented in the sklearn Tfidfvectorizer library, which together with KMeans showed much better results, was also used. Clasters turned out to be more balanced than using the CountVectorizer approach.
# * For clustering, we used the KMeans method with 10 clusters.
# * To form a context, the most frequently encountered words that have high weight in the presentation area were selected from the resulting cluster.
# * PCA, t-SNE, NMF algorithms are used to reduce the feature space

# In[ ]:


import spacy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
import pylab
from mpl_toolkits.mplot3d import Axes3D
import time, sys

class DocumentsClusterisation:
    def DataPreprocessing(self):
        print("Begin Data Preprocessing Method")

        self.TextDF = pd.DataFrame()
        self.TextDF["Text"] = Texts
        self.TextDF["Text"] = self.TextDF["Text"].str.replace(r"\W+", " ") #Replace all non alphabetic symbols
        self.TextDF["Text"] = self.TextDF["Text"].str.replace(r"[0-9]+", " ") #Replace all numbers
        self.TextDF["Text"] = self.TextDF["Text"].str.replace(r"\s\w\s", " ") #Replace all single words
        self.TextDF["Text"] = self.TextDF["Text"].str.replace(r"\s+", " ") #Replace long spaces to single space 
        self.TextDF["Text"] = self.TextDF["Text"].str.lower()

        self.Regular = r"[a-z]+"
        for index, CurrentDocument in enumerate(self.TextDF["Text"].tolist()):
            NewCurrentDocument = ""
            AllWords = re.findall(self.Regular, CurrentDocument)
            for CurrentWord in AllWords:
                NewCurrentDocument += CurrentWord + " "
            self.TextDF["Text"].iloc[index] = NewCurrentDocument

        #Use spacy to delete stopwords and correct nouns and verbs: can`t ~ can not, dogs ~ dog
        self.NLP = spacy.load("en_core_web_sm") 

        Counter = 0
        self.TextList = self.TextDF["Text"].tolist()
        print("Preprocessing Papers. Status %:")
        for CurrentDocument in self.TextList:
        
            sys.stdout.write(str(Counter/len(self.TextList) * 100))
            sys.stdout.flush()
            sys.stdout.write("\r")
            
            CurrentText = ""
            Document = self.NLP(CurrentDocument)
            for CurrentToken in Document:
                if CurrentToken.dep_ != "punct" and CurrentToken.is_stop != True:
                    CurrentText += CurrentToken.lemma_ + " "

            self.TextDF["Text"].iloc[Counter] = CurrentText
            Counter += 1

        self.Text = self.TextDF["Text"].tolist()

        print("End Data Preprocessing Method")

    #Use Bag of words
    def CountVectorizerTokenizator(self):
        print("Begin CountVectorizer Model")
        self.CountVectorizerModel = CountVectorizer(stop_words="english")
        self.Features = self.CountVectorizerModel.fit_transform(self.Text)
        print("Shape ")
        print(self.Features.shape)
        self.FeaturesExtended = self.Features
        self.FeaturesExtended = self.FeaturesExtended.toarray()
        print("End CountVectorizer Model")

    def TFIDFTokenizator(self):
        print("Begin TFIDF Model")
        self.TfidfVectorizerModel = TfidfVectorizer(stop_words="english")
        self.Features = self.TfidfVectorizerModel.fit_transform(self.Text)
        self.Vocabulary = self.TfidfVectorizerModel.vocabulary_
        self.VocabularyKeys = self.Vocabulary.keys()
        self.VocabularyItems = []

        self.FeaturesExtended = self.Features
        self.FeaturesExtended = self.FeaturesExtended.toarray()

        for i in self.VocabularyKeys:
            self.VocabularyItems.append(self.Vocabulary[i])

        self.VocabularyKeys = list(self.VocabularyKeys)
        print("End TFIDF Model")

    #10 clusters for 10 tasks 
    def ClusteringKMeans(self):
        print("Begin KMeans Model")
        self.KMeansModel = KMeans(n_clusters=10)
        self.KMeansModel.fit(self.Features)
        self.Lables = self.KMeansModel.labels_

        self.CountLables = []
        self.UniqueLables = np.unique(self.Lables)
        print("Number Unique Items per Cluster")
        for i in self.UniqueLables:
            Counter = 0
            for j in self.Lables:
                if i==j:
                    Counter += 1
            self.CountLables.append(Counter)
            print("Lable " + str(i) + " - " + str(Counter))

        #Main KeyWords in cluster
        self.ClusterDescription = []
        self.MostFreqValues = []
        print("Current Cluster:")
        for indexI, i in enumerate(self.UniqueLables):
            
            sys.stdout.write(str(indexI))
            sys.stdout.flush()
            sys.stdout.write("\r")
            
            self.CurrentClusterDescription = []

            self.GroupedFeatures = np.zeros((self.CountLables[indexI],self.FeaturesExtended.shape[1]), dtype="float32")
            Counter = 0
            self.Positions = []
            for indexJ, j in enumerate(self.Lables):
                if i == j:
                    self.GroupedFeatures[Counter] = self.FeaturesExtended[indexJ]
                    Counter += 1

            for k in range(10):
                self.PositionMaxValue = np.unravel_index(np.argmax(self.GroupedFeatures, axis=None), self.GroupedFeatures.shape)
                self.PositionMaxValue = list(self.PositionMaxValue)
                self.Positions.append(self.PositionMaxValue)
                self.GroupedFeatures[self.PositionMaxValue[0]][self.PositionMaxValue[1]] = 0

            Sum = []

            for k in range(self.GroupedFeatures.shape[1]):
                Counter = 0
                for l in range(self.GroupedFeatures.shape[0]):
                    if self.GroupedFeatures[l][k] != 0:
                        Counter += 1

                Sum.append(Counter)

            Sum = np.array(Sum)

            #Most frequently words in cluster
            self.MostFreq = []
            CurrentMostFreqValues = []
            for k in range(10):
                ArgMax = np.argmax(Sum)
                CurrrentValue = Sum[ArgMax]
                CurrentMostFreqValues.append(CurrrentValue)
                self.MostFreq.append(ArgMax)
                Sum[ArgMax] = 0

            for CurrentPosition in self.MostFreq:
                for indexJ, CurrentItem in enumerate(self.VocabularyItems):
                    if CurrentPosition == CurrentItem:
                        self.CurrentClusterDescription.append(self.VocabularyKeys[indexJ])
                        
            

            self.ClusterDescription.append(self.CurrentClusterDescription)
            self.MostFreqValues.append(CurrentMostFreqValues)

        for id, CurrentDescription in enumerate(self.ClusterDescription):
          print("Cluster " + str(id) + " KeyWords")
          print(CurrentDescription)
          Fig, Axis = plt.subplots(figsize=(25,10))
          plt.rc('xtick', labelsize=20)
          plt.rc('ytick', labelsize=20)
          plt.rc('axes', titlesize=20)
          plt.rc('axes', labelsize=20)
          plt.rc('legend', fontsize=20)
          plt.rc('figure', titlesize=20)
          plt.rc('font', size=20)
          Axis.grid()
          Axis.bar(CurrentDescription,self.MostFreqValues[id])
          Axis.set_xlabel("Most frequently words")
          Axis.set_ylabel("Frequency")
          Title = "Most frequently words " + str(id) + " Cluster"
          plt.title(Title)
          plt.show()
            
            
        
        print("\n")
        print("End KMeans Model")

    def PCADecomposition2D(self):
        print("PCA Decomposition 2D")
        self.PCAModel = PCA(n_components=2)
        self.Result = self.PCAModel.fit_transform(self.FeaturesExtended)
        print(self.Result)

    def PCADecomposition3D(self):
        print("PCA Decomposition 3D")
        self.PCAModel = PCA(n_components=3)
        self.Result = self.PCAModel.fit_transform(self.FeaturesExtended)
        print(self.Result)

    def TSNEDecomposition(self):
        print("t-SNE Decomposition 2D")
        self.TSNEModel =TSNE(n_components=2, n_jobs=4)
        self.Result = self.TSNEModel.fit_transform(self.FeaturesExtended)

        print(self.Result)

    def NMFDecomposition(self):
        print("NMF Decomposition")
        print(self.FeaturesExtended.shape)
        self.NMFModel = NMF(n_components = 2)
        self.Result = self.NMFModel.fit_transform(self.FeaturesExtended)
        H = self.NMFModel.components_
        H = np.array(H)
        print(self.Result)


    def SimilarityDocuments(self):
        self.NLP = spacy.load("en_core_web_sm")
        PatternDocuments = ["transmission, incubation, environmental, stability",
                            "COVID-19 risk factors"]


        AllIndexDocuments = []
        for CurrentPatternDocument in PatternDocuments:
            SimilarityDocuments = []
            CurrentPatternDocument = self.NLP(CurrentPatternDocument)
            for IndexDocument, CurrentDoucumentI in enumerate(self.Text):
                CurrentDoucumentI = self.NLP(CurrentDoucumentI)
                Similarity = CurrentDoucumentI.similarity(CurrentPatternDocument)
                print(Similarity)
                if Similarity> 0.6:
                    SimilarityDocuments.append(IndexDocument)
            AllIndexDocuments.append(SimilarityDocuments)

        print("Counter")
        for i in range(len(AllIndexDocuments) - 1):
            Counter = 0
            FirstIndexes = AllIndexDocuments[i]
            SecondIndexes = AllIndexDocuments[i + 1]
            for j in FirstIndexes:
                if j in SecondIndexes:
                    Counter += 1
            print(Counter/len(AllIndexDocuments[0]))
            print(Counter / len(AllIndexDocuments[1]))

        Counter = 0


    def Visualisation2D(self):
        print("Visualisation")
        CplorList = ["red", "blue", "black", "orange", "green", "yellow", "cyan","magenta", "lime","violet"]
        CurrentText = [i for i in range(1,11)]
        Fig, Axis = plt.subplots(figsize=(25,10))
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=18)
        plt.rc('legend', fontsize=18)
        plt.rc('figure', titlesize=30)
        plt.rc('font', size=18)
        for indexI, i in enumerate(self.UniqueLables):
            CurrentX = []
            CurrentY = []

            for indexJ, j in enumerate(self.Lables):
                if i == j:
                    CurrentX.append(self.Result[indexJ,0])
                    CurrentY.append(self.Result[indexJ,1])

            Axis.plot(CurrentX, CurrentY, marker="o", linewidth=0, color=CplorList[indexI], label ="Cluster " + str(CurrentText[indexI]))
        Axis.grid()
        Axis.set_xlabel("Component 1")
        Axis.set_ylabel("Component 2")
        plt.title("PCE Decomposition")
        Axis.legend()
        plt.show()

    def Visualisation3D(self):
        CplorList = ["red", "blue", "black", "orange", "green", "yellow", "cyan","magenta", "lime","violet"]
        CurrentText = [i for i in range(1,11)]
        Fig = pylab.figure()
        Axes = Axes3D(Fig)
        for indexI, i in enumerate(self.UniqueLables):
            CurrentX = []
            CurrentY = []
            CurrentZ = []

            for indexJ, j in enumerate(self.Lables):
                if i == j:
                    CurrentX.append(self.Result[indexJ,0])
                    CurrentY.append(self.Result[indexJ,1])
                    CurrentZ.append(self.Result[indexJ,2])
            Axes.scatter(CurrentX, CurrentY, CurrentZ, color=CplorList[indexI])

        pylab.show()



DocumentsClassterisationObj = DocumentsClusterisation()
DocumentsClassterisationObj.DataPreprocessing()
#Features = CountVectorizerTokenizator(Text=Text)
DocumentsClassterisationObj.TFIDFTokenizator()
#DocumentsClassterisationObj.SimilarityDocuments()
DocumentsClassterisationObj.ClusteringKMeans()
DocumentsClassterisationObj.PCADecomposition2D()
DocumentsClassterisationObj.Visualisation2D()
#DocumentsClassterisationObj.PCADecomposition3D()
#DocumentsClassterisationObj.NMFDecomposition()
#DocumentsClassterisationObj.Visualisation()
#DocumentsClassterisationObj.Visualisation3D()


# # **Using the obtained keywords one can formulate the context of the clusters.**
# 
# * **Cluster 1 key words:** influenza, virus, pandemic, human, infection, cause, health, disease, respiratory, case
# 
# From these tokens, it can be assumed that the papers of this class relate to the causes of transmission of the disease between people and the respiratory syndrome.
# * **Cluster 2 key words:** respiratory, infection, cause, virus, child, tract, disease, acute, human, viral
# 
# From the data of tokens, it can be assumed that the papers of this class are related to the acute course of the disease in children, in particular diseases of the gastrointestinal tract
# 
# * **Cluster 3 key words:** cov, respiratory, syndrome, coronavirus, east, mers, human, middle, infection, severe
# 
# From these tokens, it can be assumed that the papers of this class relate to those about the place of occurrence of an infectious disease
# 
# * **Cluster 4 key words:** cell, response, infection, immune, virus, viral, protein, include, host, receptor
# 
# From these tokens, it can be assumed that the papers of this class relate to the structure of the coronavirus, its biological features and, possibly, the transmission method.
# 
# * **Cluster 5 key words:** china, case, coronavirus, wuhan, novel, report, december, disease, sars, health
# 
# From these tokens, it can be assumed that the papers of this class relate to topics about literature, possibly by Chinese scientists
# 
# * **Cluster 6 key words:** al, et, virus, disease, human, infection, cause, include, viral, cell
# 
# From these tokens, it can be assumed that the papers of this class relate to the topic of changing human cells, the functioning of the virus inside the cells. It can be seen that in this situation the words al, et. This suggests that perhaps a more thorough, clean text is required.
# 
# * **Cluster 7 key words:** pedv, virus, diarrhea, porcine, epidemic, pig, piglet, cause, swine, mortality
# 
# From these tokens, it can be assumed that the papers of this class relate to those on the methods of transmission of the virus through animals, in particular through pigs.
# 
# * **Cluster 8 key words:** protein, virus, rna, viral, genome, cell, strand, include, single, family
# 
# From these tokens, it can be assumed that the papers of this class relate to those on the methods of transmission of the virus through animals, in particular through pigs.
# 
# * **Cluster 9 key words:** disease, infection, include, study, virus, cause, high, human, patient, result
# 
# From these tokens, it can be assumed that the papers of this class relate to those about the study of ways to treat a person
# 
# * **Cluster 10 key words:** disease, virus, health, human, infection, infectious, cause, outbreak, public, include
# 
# From these tokens, it can be assumed that the papers of this cluster relate to topics on the methods of spreading and outbreaks of viral infection in society.

# # **Step 3. Filtering the dataset**
# To use the Question-Answer system, it is necessary to filter out articles whose context is not related to coronavirus

# In[ ]:


# Run if you need to filter covid dataset
print("Filter Dataset")
import re
import copy
import numpy as np
KeyWords = ["corona", "cov", "sars", "cord", "covid-19"]
HelpText = copy.copy(Texts)


for index, CurrentText in enumerate(HelpText):
  CurrentText = CurrentText.lower()
  HelpText[index] = CurrentText

GeneralResult = []
for Word in KeyWords:
  ResultList = []
  for CurrentText in HelpText:
    Result = re.findall(Word,CurrentText)
    ResultList.append(Result)
  GeneralResult.append(ResultList)

Indexes = []
for i in range(len(GeneralResult[0])):
  Sum = 0
  for j in range(len(KeyWords)):
    Sum += len(GeneralResult[j][i])
  if Sum != 0:
    Indexes.append(i)

Texts = np.array(Texts)
Abstracts = np.array(Abstracts)
Titles = np.array(Titles)
Authors = np.array(Authors)

Texts = Texts[Indexes]
Abstracts = Abstracts[Indexes]
Titles = Titles[Indexes]
Authors = Authors[Indexes]


# In[ ]:


print("Length Text, Abstract, Authors, Titles")
print(len(Texts))
print(len(Abstracts))
print(len(Authors))
print(len(Titles))


# # **Step 4. Using a BERT-based Question and Answer System**

# In[ ]:


get_ipython().system('pip install cdqa')
import pandas as pd
from ast import literal_eval
from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline
import numpy as np


# In[ ]:


download_model(model='bert-squad_1.1', dir='./models')


# In[ ]:


import spacy

import en_core_web_lg

#Download large english words model
print("Download large model")
NLP = en_core_web_lg.load()


# In[ ]:


# CDQA dataframe format. It is important to make dataframe which consist of columns
#'date', 'title', 'category', 'link', 'abstract', 'paragraphs'

DF = pd.DataFrame(columns=['date', 'title', 'category', 'link', 'abstract', 'paragraphs'])

for i in range(len(Texts)):
  CurrentText = [Texts[i]]
  CurrentList = ["None", Titles[i], "None", "None", Abstracts[i], CurrentText]
  CurrentList = np.array(CurrentList)
  CurrentList = CurrentList.reshape(1, CurrentList.shape[0])

  CurrentList = pd.DataFrame(data = CurrentList, columns=['date', 'title', 'category', 'link', 'abstract', 'paragraphs'])

  DF = pd.concat([DF, CurrentList], ignore_index=True)

print(DF.shape)


# In[ ]:


DF = filter_paragraphs(DF)


# In[ ]:


CDQAPipeline = QAPipeline(reader='models/bert_qa.joblib')


# In[ ]:


CDQAPipeline.fit_retriever(df=DF)
Question = 'How long does covid live outside the human body'
QuestionList = ["What is the incubation period for covid", "How long does a covid live on different surfaces?",
               "How long has a person been ill with a covid?", "What covid transmission routes are known",
               "How not to get infected with covid"]

for CurrentQuestion in QuestionList:
    Predict = CDQAPipeline.predict(query=CurrentQuestion)
    print("Question: " + CurrentQuestion)
    print("Answer: " + Predict[0])
    print("Title: " + Predict[1])
    print("Current Paper: " + Predict[2])
    print("\n")


# # **Step 5. BERT embedding. Designing a vector representation of papers using t5 google transformer**

# # **To analyze the content of papers, a new Google neural network is used. To install, you must run the following commands.**

# In[ ]:


get_ipython().system('pip install pytorch-pretrained-bert')
get_ipython().system('pip install transformers==2.8.0')
get_ipython().system('pip install torch==1.4.0')


# # **The BERT neural network is used to obtain a vector representation**

# In[ ]:


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

Papers = Texts

import re, sys
import numpy as np

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

print("Current Paper ID")
PaperEmbedding = []

AllPaperWords = []
AllPaperWordsEmbedding = []

for CurrentPaperID, CurrentPaper in enumerate(Papers):

  sys.stdout.write(str(CurrentPaperID * 100/ len(Papers)))
  sys.stdout.flush()
  sys.stdout.write("\r")
  
  SentenceEmbeddingList = []
  PaperSentences = re.split("\.\s",CurrentPaper)

  CurrentPaperWords = []
  CurrentPaperWordsEmbedding = []

  for CurrentSentence in PaperSentences:
    CurrentSentence = "[CLS] " + CurrentSentence + " [SEP]"
    TokenizedText = tokenizer.tokenize(CurrentSentence)

    if len(TokenizedText) < 510:

        IndexedTokens = tokenizer.convert_tokens_to_ids(TokenizedText)
        SegmentsId = [1] * len(TokenizedText)
        TokensTensor = torch.tensor([IndexedTokens])
        SegmentsTensor = torch.tensor([SegmentsId])
        with torch.no_grad():
          EncodedLayer, _ = model(TokensTensor, SegmentsTensor)

        TokenEmbedding = torch.stack(EncodedLayer, dim=0)
        TokenEmbedding.size()
        TokenEmbedding = torch.squeeze(TokenEmbedding, dim=1)
        TokenEmbedding.size()
        TokenEmbedding = TokenEmbedding.permute(1,0,2)
        TokenEmbedding.size()
        TokensVectors = []
        for CurrentToken in TokenEmbedding:
          SumVectors = torch.sum(CurrentToken[-4:], dim=0)
          TokensVectors.append(SumVectors)


        for CurrentToken in TokenizedText:
          CurrentPaperWords.append(CurrentToken)

        for CurrentWordVector in TokensVectors:
          CurrentPaperWordsEmbedding.append(CurrentWordVector)

        TV = EncodedLayer[11][0]

        SentenceEmbedding = torch.mean(TV, dim=0)

        SentenceEmbedding = np.array(SentenceEmbedding)
        SentenceEmbeddingList.append(SentenceEmbedding)
   
  
  AveragePaperEmbedding = np.mean(SentenceEmbeddingList, axis=0)
  PaperEmbedding.append(AveragePaperEmbedding)

  AllPaperWords.append(CurrentPaperWords)
  AllPaperWordsEmbedding.append(CurrentPaperWordsEmbedding)


PaperEmbedding = np.array(PaperEmbedding)

print("Shape Paper Embedding")
print(PaperEmbedding.shape)
print("Length Word List")
print(len(AllPaperWords))


# In[ ]:


from sklearn.cluster import KMeans

KMeansModel = KMeans(n_clusters=10)
KMeansModel.fit(PaperEmbedding)
Lables = KMeansModel.labels_
print(Lables)


# In[ ]:


from sklearn.decomposition import PCA

PCAModel = PCA(n_components=2)
Result = PCAModel.fit_transform(PaperEmbedding)


UniqueLables = np.unique(Lables)


# In[ ]:


print(Result.shape)
import matplotlib.pyplot as plt

print("Visualisation")
CplorList = ["red", "blue", "black", "orange", "green", "yellow", "cyan","magenta", "lime","violet"]
CurrentText = [i for i in range(1,11)]
Fig, Axis = plt.subplots(figsize=(25,10))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=30)
plt.rc('font', size=18)
for indexI, i in enumerate(UniqueLables):
    CurrentX = []
    CurrentY = []

    for indexJ, j in enumerate(Lables):
        if i == j:
            CurrentX.append(Result[indexJ,0])
            CurrentY.append(Result[indexJ,1])

    Axis.plot(CurrentX, CurrentY, marker="o", linewidth=0, color=CplorList[indexI], label ="Cluster " + str(CurrentText[indexI]))
Axis.grid()
Axis.set_xlabel("Component 1")
Axis.set_ylabel("Component 2")
plt.title("PCE Decomposition")
Axis.legend()
plt.show()


# # **The main idea is to build a vector representation of papers and clustering it. Assumption: tokens that are contextually specific to a domain will have a similar vector representation, which means that the cosine distance will be close. The T5 transformer neural network is used to extract the context of the cluster.**

# In[ ]:


from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = torch.device("cpu")

for idCluster, CurrentUniqueLable in enumerate(UniqueLables):
  CurrentText = ""
  Counter = 0
  for id, CurrentLable in enumerate(Lables):
    if CurrentLable == CurrentUniqueLable:
      if Counter > 2:
        break
      CurrentText += Papers[id]
      Counter += 1

  preprocess_text = CurrentText.strip().replace("\n","")
  t5_prepared_Text = "summarize: " + preprocess_text
  tokenized_text = tokenizer.encode(t5_prepared_Text,return_tensors="pt").to(device)

  summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  print("Description " + str(idCluster) + " Cluster")
  print(output)
  print("\n")


# # **As can be seen from the figure, the clusters are quite well separable. Apparently, the last cluster contains uninformative articles that can be attributed to "garbage"**
