#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Open Research Dataset (CORD-19) Analysis
# 
# <p align="center">
#     <img src="https://pages.semanticscholar.org/hs-fs/hubfs/covid-image.png?width=300&name=covid-image.png"/>
# </p>
# 
# ***NOTE: There is a [Report Builder Notebook](https://www.kaggle.com/davidmezzetti/cord-19-report-builder) that runs on a prebuilt model. If you just want to try this out without a full build, this is the best choice.***
# 
# COVID-19 Open Research Dataset (CORD-19) is a free resource of scholarly articles, aggregated by a coalition of leading research groups, covering COVID-19 and the coronavirus family of viruses. The dataset can be found on [Semantic Scholar](https://pages.semanticscholar.org/coronavirus-research) and [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).
# 
# This notebook builds an index over the CORD-19 dataset to assist with analysis and data discovery using the [paperai](https://github.com/neuml/paperai) project. A series of COVID-19 related research topics were explored to identify relevant articles and help find answers to key scientific questions.
# 
# ### Tasks
# A full list of Kaggle CORD-19 Challenge tasks are [referenced below](#Round-1-Tasks). This notebook and corresponding report notebooks won 🏆 [7 awards](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/161447) 🏆 in the Kaggle CORD-19 Challenge.
# 
# The latest tasks are also stored in the [cord19q repository](https://github.com/neuml/cord19q/tree/master/tasks).
# 

# # Install
# 
# [paperai](https://github.com/neuml/paperai) is installed via the [cord19reports](https://www.kaggle.com/davidmezzetti/cord19reports) package. This ensures version consistency across this and all related notebooks. 

# In[ ]:


from cord19reports import install

# Install paperai project
install()


# # CORD-19 ETL - Articles SQLite Database
# 
# This project depends on the [CORD-19 ETL notebook](https://www.kaggle.com/davidmezzetti/cord-19-etl). The CORD-19 ETL notebook has a full overview of the raw CORD-19 dataset, parsing rules and other important information on how the data stored in SQLite is derived. Previous versions of this notebook had the full build process for both the SQLite database and embeddings index here but modularization was necessary as the dataset grew.
# 
# This following step copies over the previously constructed Articles SQLite database and related files. 
# 

# In[ ]:


import os
import shutil

# Copy articles.sqlite and related files locally
os.mkdir("cord19q")
shutil.copy("../input/cord-19-etl/cord19q/articles.sqlite", "cord19q")
shutil.copy("../input/cord-19-etl/cord19q/attribute", "cord19q")
shutil.copy("../input/cord-19-etl/cord19q/design", "cord19q")


# # Build Embedding Index
# 
# An embeddings index is created with [FastText](https://fasttext.cc/) + [BM25](https://en.wikipedia.org/wiki/Okapi_BM25). Background on this method can be found in this [Medium article](https://towardsdatascience.com/building-a-sentence-embedding-index-with-fasttext-and-bm25-f07e7148d240) and an existing repository using this method [codequestion](https://github.com/neuml/codequestion).
# 
# The embeddings index takes each COVID-19 tagged, not labeled a question/fragment, having a detected study type, tokenizes the text, and builds a sentence embedding. A sentence embedding is a BM25 weighted combination of the FastText vectors for each token in the sentence. The embeddings index takes the full corpus of these embeddings and builds a [Faiss](https://github.com/facebookresearch/faiss) index to enable similarity searching. 
# 
# Important source files to highlight
# 
# * Indexing Process -> [index.py](https://github.com/neuml/paperai/blob/master/src/python/paperai/index.py)
# * Tokenizer -> [tokenizer.py](https://github.com/neuml/txtai/blob/master/src/python/txtai/tokenizer.py)
# * Embeddings Model -> [embeddings.py](https://github.com/neuml/txtai/blob/master/src/python/txtai/embeddings.py)
# * BM25 Scoring -> [scoring.py](https://github.com/neuml/txtai/blob/master/src/python/txtai/scoring.py)
# 
# FastText vectors trained on the full CORD-19 corpus are required. A [dataset with pre-trained vectors](https://www.kaggle.com/davidmezzetti/cord19-fasttext-vectors) is included and used in this notebook. Building the vectors takes a couple of hours when locally trained and would most likely take much longer within a notebook. 
# 
# Vectors can optionally be (re)built by running the following command with the project and articles.sqlite database installed locally:
# 
# ```
# python -m paperai.vectors
# ```
# 
# The following code builds the embeddings index using fastText vectors trained on the full CORD-19 dataset. Alternatively, any [pymagnitude vector file](https://github.com/plasticityai/magnitude#pre-converted-magnitude-formats-of-popular-embeddings-models) can be used the build the sentence embeddings.

# In[ ]:


import shutil

from paperai.index import Index

# Copy vectors locally for predictable performance
shutil.copy("../input/cord19-fasttext-vectors/cord19-300d.magnitude", "/tmp")

# Build the embeddings index, limit to most recent documents
Index.run("cord19q", "/tmp/cord19-300d.magnitude", 75000)


# In[ ]:


import os

# Workaround for mdv terminal width issue
os.environ["COLUMNS"] = "80"

from paperai.highlights import Highlights
from txtai.pipeline import Tokenizer

from nltk.corpus import stopwords
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pycountry

# Use paperai + NLTK stop words
STOPWORDS = Highlights.STOP_WORDS | set(stopwords.words("english"))

# Tokenizes text and removes stopwords
def tokenize(text, case_sensitive=False):
    # Get list of accepted tokens
    tokens = [token for token in Tokenizer.tokenize(text) if token not in STOPWORDS]
    
    if case_sensitive:
        # Filter original tokens to preserve token casing
        return [token for token in text.split() if token.lower() in tokens]

    return tokens
    
# Country data
countries = [c.name for c in pycountry.countries]
countries = countries + ["USA"]

# Lookup country name for alpha code. If already an alpha code, return value
def countryname(x):
    country = pycountry.countries.get(alpha_3=x)
    return country.name if country else x
    
# Resolve alpha code for country name
def countrycode(x):
    return pycountry.countries.get(name=x).alpha_3

# Tokenize and filter only country names
def countrynames(x):
    return [countryname(country) for country in countries if country.lower() in x.lower()]

# Word Cloud colors
def wcolors(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    colors = ["#7e57c2", "#03a9f4", "#011ffd", "#ff9800", "#ff2079"]
    return np.random.choice(colors)

# Word Cloud visualization
def wordcloud(df, title = None):
    # Set random seed to have reproducible results
    np.random.seed(64)
    
    wc = WordCloud(
        background_color="white",
        max_words=200,
        max_font_size=40,
        scale=5,
        random_state=0
    ).generate_from_frequencies(df)

    wc.recolor(color_func=wcolors)
    
    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')

    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wc),
    plt.show()

# Dataframe plot
def plot(df, title, kind="bar", color="bbddf5"):
    # Remove top and right border
    ax = plt.axes()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set axis color
    ax.spines['left'].set_color("#bdbdbd")
    ax.spines['bottom'].set_color("#bdbdbd")

    df.plot(ax=ax, title=title, kind=kind, color=color);

# Pie plot
def pie(labels, sizes, title):
    patches, texts = plt.pie(sizes, colors=["#4caf50", "#ff9800", "#03a9f4", "#011ffd", "#ff2079", "#7e57c2", "#fdd835"], startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.title(title)
    plt.show()
    
# Map visualization
def mapplot(df, title, bartitle):
    fig = go.Figure(data=go.Choropleth(
        locations = df["Code"],
        z = df["Count"],
        text = df["Country"],
        colorscale = [(0,"#fffde7"), (1,"#f57f17")],
        showscale = False,
        marker_line_color="darkgray",
        marker_line_width=0.5,
        colorbar_title = bartitle,
    ))

    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        )
    )
    
    fig.show(config={"displayModeBar": False, "scrollZoom": False})


# # Exploring the data
# The articles database has a copy of all articles that were found in metadata.csv. Pure duplicate articles (based on the sha hash) are filtered out. In addition to the metadata and text fields, a field named tags is added. Each article is tagged based on the topic. The only tag at this time is COVID-19 for articles that directly mention COVID-19 and related terms. This field is important as the embedding index and all model searches will go against the subset of data tagged as COVID-19.

# ## Articles Table
# A sample of the articles table is shown below.

# In[ ]:


import pandas as pd
import sqlite3

# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

# Articles
pd.set_option("max_colwidth", 125)
articles = pd.read_sql_query("select * from articles where tags is not null LIMIT 5", db)
articles


# ## Sections Table
# In addition to the articles table, another table named sections is also created. The full text content is stored here. Each row is a single sentence from an article. Sentences are parsed using [NTLK's](https://www.nltk.org/) sent_tokenize method. The article id and tags are also stored with each section. The sections schema and sample rows are shown below.

# In[ ]:


# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

# Sections
pd.set_option("max_colwidth", 125)
sections = pd.read_sql_query("select * from sections where tags is not null LIMIT 5", db)
sections


# ## Most Frequent Words in Tagged Articles
# The following wordcloud shows the most frequent words within the titles of tagged articles.

# In[ ]:


# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

# Select data
articles = pd.read_sql_query("select title from articles where tags is not null and title is not null order by published desc LIMIT 100000", db)

# Build word frequencies on filtered tokens
freqs = pd.Series(np.concatenate([tokenize(x) for x in articles.Title])).value_counts()
wordcloud(freqs, "Most frequent words in article titles tagged as COVID-19")


# ## Tagged Articles by Country Mentioned
# The following map shows the Articles by Country mentioned. China is mentioned significantly more and it's count is clipped in this graphic to allow showing distribution across the globe.

# In[ ]:


# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

sections = pd.read_sql_query("select text from sections where tags is not null order by id desc LIMIT 500000", db)

# Filter tokens to only country names. Build dataframe of Country, Count, Code
mentions = pd.Series(np.concatenate([countrynames(x) for x in sections.Text])).value_counts()
mentions = mentions.rename_axis("Country").reset_index(name="Count")
mentions["Code"] = [countrycode(x) for x in mentions["Country"]]

# Set max to 5000 to allow shading for multiple countries
mentions["Count"] = mentions["Count"].clip(upper=5000)

mapplot(mentions, "Tagged Articles by Country Mentioned", "Articles by Country")


# ## Tagged Articles by Source
# The following graph shows the articles grouped by the source field in the metadata. Only the Top 15 sources are shown.

# In[ ]:


# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

articles = pd.read_sql_query("select source from articles where tags is not null order by published desc LIMIT 100000", db)

freqs = articles.Source.value_counts().sort_values(ascending=True)[-15:]
plot(freqs, "Tagged Articles by Source", "barh", "#1976d2")


# ## Tagged Articles by Publication
# The graph below shows the articles grouped by publication. Only the Top 15 publications are shown and many articles have no publication.

# In[ ]:


# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

articles = pd.read_sql_query("select case when (Publication = '' OR Publication IS NULL) THEN '[None]' ELSE Publication END AS Publication from articles where tags is not null order by published desc LIMIT 100000", db)

freqs = articles.Publication.value_counts().sort_values(ascending=True)[-15:]

plot(freqs, "Tagged Articles by Publication", "barh", "#7e57c2")


# ## Tagged Articles by Publication Month
# The following graph shows articles by publication month. All of the articles have a publication date of 2020 or later (or the date is null). Many publication dates only include the year but there is a significant portion of articles this month, which shows the rapid pace things are moving. Also note that some publication dates are in the future. The articles have been released early to help find answers.

# In[ ]:


# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

articles = pd.read_sql_query("select strftime('%Y-%m', published) as Published from articles where tags is not null and published >= '2020-01-01' order by published desc LIMIT 100000", db)

freqs = articles.Published.value_counts().sort_index()
plot(freqs, "Tagged Articles by Publication Month", "bar", "#ff9800")


# ## Tagged Articles by Study Design
# The chart below shows articles grouped by study design type. The study design gives researchers insight into the overall structure and quality of a study. The more rigor and hard data that goes into a study, the more reliable. This is a distinction compared to many other search systems, where we look for the best matching text. Credibility of the information is very important 
# in helping judge whether the conclusions are reliable. 
# 
# The medical field is rightfully a skeptical field. Many technologists are accustomed to running a web search and quickly trying the top results until you get to something that works. Lets be glad our doctors don't do the same. 

# In[ ]:


# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

articles = pd.read_sql_query('select count(*) as count, case when design=1 then "systematic review" when design in (2, 3) then "control trial" ' + 
                             'when design in (4, 5) then "prospective studies" when design=6 then "retrospective studies" ' +
                             'when design in (7, 8) then "case series" else "modeling" end as design from articles ' +
                             'where tags is not null and design > 0 group by design order by published desc LIMIT 100000', db)

articles = articles.groupby(["design"]).sum().reset_index()

# Plot a pie chart of study types
pie(articles["design"], articles["count"], "Tagged Articles by Study Design")


# ## Exploration Takeaways
# Given the urgency to find any data to help, many of the tagged articles are recent. Publications by nature put hypothesises and theories through a rigorious scientific method/peer review to ensure accuracy and reliability. It's a balancing act of not holding on to data that can help against making sure decisions are based on accurate data. Given that all searches are against this subset of data, conclusions should be carefully drawn. 

# # Testing the model
# 
# Now that both the articles.sqlite database and embeddings index are both created, lets test that everything is working properly.

# ## Word Embeddings
# The foundation of sentence embeddings are word embeddings. As previously explained, sentence embeddings are just word embeddings joined together (each token weighted by a BM25 index). 

# In[ ]:


from txtai.embeddings import Embeddings

embeddings = Embeddings()
embeddings.load("cord19q")

# Get word vectors model
vectors = embeddings.model.model

pd.DataFrame(vectors.most_similar("covid-19", topn=10), columns=["key", "value"])


# The vector model is good at identifying near matches, which helps increase the accuracy of the overall model. Notice that the top hits are typos (covid-10 mistyped 0 instead of 9). 
# 
# Below shows similarity for a list of terms, numbers look overall as expected, model has learned an association between the various diseases and knows phone is not related.

# In[ ]:


vectors.similarity("coronavirus", ["sars", "influenza", "ebola", "phone"])


# ## Sentence Embeddings
# At the highest level, the model builds embeddings for each sentence in the corpus. For input queries, it compares each sentence against the input query. Faiss enables that similarity search to be fast. An example of how this works at a small level below.

# In[ ]:


sentence1 = "Range of incubation periods for the disease in humans"
sentence2 = "The incubation period of 2019-nCoV is generally 3-7 days but no longer than 14 days, and the virus is infective during the incubation period"

embeddings.similarity(Tokenizer.tokenize(sentence1), [Tokenizer.tokenize(sentence2)])


# In[ ]:


sentence1 = "Range of incubation periods for the disease in humans"
sentence2 = "The medical profession is short on facemasks during this period, more are needed"

embeddings.similarity(Tokenizer.tokenize(sentence1), [Tokenizer.tokenize(sentence2)])


# ## Run a query
# Run a full query to ensure model is working.

# In[ ]:


from paperai.query import Query

# Execute a test query
Query.run("antiviral covid-19 success treatment", 5, "cord19q")


# # Building task reports
# Task reports are an aggregation of each question within a task in the challenge. For each question, a query is run and the top articles are returned. For each article, text matches are shown as bulleted points and these are the best matching sentences within the article. The full list of result sentences are also analyzed and run through a [textrank algorithm](https://en.wikipedia.org/wiki/Automatic_summarization#TextRank_and_LexRank). Highlights or top sentences within the results are also shown within the report. 
# 
# Important source files to highlight
# * Report Process -> [execute.py](https://github.com/neuml/paperai/blob/master/src/python/paperai/report/execute.py)
# * Textrank algorithm to highlight best sentences -> [highlights.py](https://github.com/neuml/paperai/blob/master/src/python/paperai/highlights.py)
# 
# Queries use a YAML formatted syntax that allows customizing the query string and result columns. The following example shows how to build a task report.

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', '\nfrom paperai.report.execute import Execute as Report\nfrom IPython.display import display, Markdown\n\nquery = """\nname: query\n\nantiviral covid-19 success treatment:\n    query: antiviral covid-19 success treatment\n    columns:\n        - name: Date\n        - name: Study\n        - name: Study Type\n        - name: Sample Size\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n"""\n\n# Execute report query\nReport.run(query, 10, "md", "cord19q")\n\n# Render report\ndisplay(Markdown(filename="query.md"))')


# # Round 1 Tasks
# 
# The following is a list of submitted [Round 1](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/148807) task notebooks. These notebooks are no longer updated as of 2020-05-02. 
# 
# - [What is known about transmission, incubation, and environmental stability?](https://www.kaggle.com/davidmezzetti/cord-19-transmission-incubation-environment)
# - [What do we know about COVID-19 risk factors?](https://www.kaggle.com/davidmezzetti/cord-19-risk-factors?scriptVersionId=33173909)
# - [What do we know about virus genetics, origin, and evolution?](https://www.kaggle.com/davidmezzetti/cord-19-virus-genetics-origin-and-evolution)
# - [What do we know about vaccines and therapeutics?](https://www.kaggle.com/davidmezzetti/cord-19-vaccines-and-therapeutics)
# - [What do we know about non-pharmaceutical interventions?](https://www.kaggle.com/davidmezzetti/cord-19-non-pharmaceutical-interventions)
# - [What has been published about medical care?](https://www.kaggle.com/davidmezzetti/cord-19-medical-care)
# - [What do we know about diagnostics and surveillance?](https://www.kaggle.com/davidmezzetti/cord-19-diagnostics-and-surveillance)
# - [What has been published about information sharing and inter-sectoral collaboration?](https://www.kaggle.com/davidmezzetti/cord-19-sharing-and-collaboration)
# - [What has been published about ethical and social science considerations?](https://www.kaggle.com/davidmezzetti/cord-19-ethical-and-social-science-considerations)
# 
# # Round 2 Tasks
# 
# The following notebooks map to the tables in the [Kaggle COVID-19 Literature Review](https://www.kaggle.com/covid-19-contributions) and [Round 2](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/161447).
# 
# - [Task 1: Population](https://www.kaggle.com/davidmezzetti/cord-19-population)
# - [Task 2: Relevant Factors](https://www.kaggle.com/davidmezzetti/cord-19-relevant-factors)
# - [Task 3: Patient Descriptions](https://www.kaggle.com/davidmezzetti/cord-19-patient-descriptions)
# - [Task 4: Models and Open Questions](https://www.kaggle.com/davidmezzetti/cord-19-models-and-open-questions)
# - [Task 5: Materials](https://www.kaggle.com/davidmezzetti/cord-19-materials)
# - [Task 6: Diagnostics](https://www.kaggle.com/davidmezzetti/cord-19-diagnostics)
# - [Task 7: Therapeutics](https://www.kaggle.com/davidmezzetti/cord-19-therapeutics)
# - [Task 8: Risk Factors](https://www.kaggle.com/davidmezzetti/cord-19-risk-factors)
# - [Full Task CSV Export List](https://www.kaggle.com/davidmezzetti/cord-19-task-csv-exports)
# 
# 
