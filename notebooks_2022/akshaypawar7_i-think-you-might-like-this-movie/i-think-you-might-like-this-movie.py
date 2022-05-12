#!/usr/bin/env python
# coding: utf-8

# # Recommender system...🤔
# 
# How to design a recommendation system is a popular question appearing in the most recent system design interviews for big tech companies. The reason is simple. Recommendation systems have become very important over the past few years. Amazon, Netflix, YouTube, Hulu, Instagram and several other companies have their own recommendation systems to make different kinds of suggestions to their users. Different types of algorithms are used to track patterns in the kind of data utilized by the users on the application, to suggest the most relevant material to each user from a large store of content.
# 
# Let’s discuss how a recommendation system, such as the Movies Recommender, is designed.
# 
# 
# ![](https://miro.medium.com/max/875/1*6gDIv9mWyc8vkdGerBKiXw.jpeg)
# 
# 
# 
# # Overview
# Recommender systems usually make use of either or both collaborative filtering and content-based filtering, as well as other systems such as knowledge-based systems.
# * Collaborative filtering approaches build a model from a user's past behavior (items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that the user may have an interest in.
# * Content-based filtering approaches utilize a series of discrete, pre-tagged characteristics of an item in order to recommend additional items with similar properties.
# 
# # Approaches
# 
# ## Content-based filtering
# Content-based filtering, also called classification based or item-item collaborative filtering, is a machine learning technique that establishes a correlation between the products to make decisions.
# 
# 
# ![content_based](https://miro.medium.com/max/500/1*O278BoepckGzdtcQlnJM_g.png)
# 
# Consider a database of movies. Content-based filtering will process the features or characteristics of these movies. Some of the properties that the algorithm considers can be:
# * Whether the movie is animated or not?
# * Is the movie based on Marvel comics?
# * Whether the movie is a thriller or not?
# * Is the movie rated PG-13?
# * Is Emma Watson starring in it?
# 
# Suppose our recommendation system holds 6 movies. The table below matches the movies with 5 features that are important to our application:
# 
# 
# ![](https://miro.medium.com/max/875/0*2bPoe1jFswpwlTIC)
# 
# 
# Now if a user has watched Movie 2, which has features 2, 3, and 5 in it, the content-based filtering algorithm finds other movies that share the most features with Movie 2. 
# 
# Since Movie 3 and Movie 5 share the most features with Movie 2 among all the movies in the database, the user is suggested Movie 3 and Movie 5.
# 
# Content-based filtering does not take into account other user’s activities when recommending content to a user.
# 
# 
# 
# ## Collaborative filtering
# 
# Collaborative filtering or user-user collaborative model is based on the assumption that users will like products that are consumed by other users with similar taste. Instead of assigning features to content and then recommending the content to the users based on their feature preferences, as was the case with content-based filtering, this approach attempts at detecting patterns between the data of different users.
# 
# 
# ![](https://miro.medium.com/max/875/0*DYPV_AtSVsXAjuwg)
# 
# While collaborative filtering is a complex machine learning approach and can be employed in multiple forms, let’s discuss the most basic solution.
# So instead of storing a table of features against movies, let’s store a table of movies against users.
# 
# ![](https://miro.medium.com/max/875/0*oCb8XandSEcGSOeW)
# 
# The table above shows the available data of 5 users as to whether or not they have watched 4 different movies. Using this data, our collaborative filtering approach can identify which movie to suggest to User 5.
# 
# Consider ticks as a representation that the user has watched that particular movie and crosses to tell that the user has not watched that movie. Now we know that User 5 has watched Movie 1. Our algorithm considers User 1 and User 3 as similar users since they have also watched Movie 1 but Users 2 and 4 have not. Now looking at the data for User 1 and 3 (similar users), you can see from the table that both have watched Movie 3, while only User 1 has watched Movie 2 and Movie 4.
# 
# ![](https://miro.medium.com/max/500/1*E2MqFWv1nz2uhWMSk3L9Ag.png)
# 
# Our algorithm gives Movie 3 two votes, while it gives Movie 2 and Movie 4 a single vote each as shown in the diagram above. Since Movie 3 has the most votes, the recommendation system will suggest Movie 3 to User 5.
# 
# ## that's the basic things, now we can build one.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


df = pd.read_csv('../input/millions-of-movies/movies.csv')
df.drop('imdb_id',axis = 1, inplace =True)
df.head()


# # Feature extraction
# We will compute pairwise similarity scores for all movies based on their overview and recommend movies based on that similarity score.
# 
# For any of you who has done even a bit of text processing before knows we need to convert the word vector of each overview. Now we'll compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each overview.
# 
# Now if you are wondering what is term frequency , it is the relative frequency of a word in a document and is given as (term instances/total instances). Inverse Document Frequency is the relative count of documents containing the term is given as log(number of documents/documents with term) The overall importance of each word to the documents in which they appear is equal to TF * IDF
# 
# This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each row represents a movie, as before.This is done to reduce the importance of words that occur frequently in plot overviews and therefore, their significance in computing the final similarity score.
# 
# Fortunately, scikit-learn gives you a built-in TfIdfVectorizer class that produces the TF-IDF matrix in a couple of lines. That's great, isn't it?

# In[ ]:


def lammitization():
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    from nltk.corpus import wordnet, stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    df['overview']= df['overview'].fillna('')

    # Tokenizing the tweet base texts.

    df['overview']=df['overview'].str.lower().apply(word_tokenize).apply(nltk.tag.pos_tag) # Applying part of speech tags.


    # Converting part of speeches to wordnet format.

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


    df['overview']= df['overview'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

    # Applying word lemmatizer.

    wnl = WordNetLemmatizer()

    df['overview']= df['overview'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])


    df['overview']= df['overview'].apply(lambda x: ' '.join(x))


# In[ ]:


df.drop_duplicates(inplace=True, ignore_index=True)
#df = df.groupby('title').first().reset_index()
df.fillna(value={i: ' ' for i in ['overview', 'genres', 'keywords', 'credits']}, inplace=True)

# lambda func for str split join
strOp= lambda x: ' '.join(x.split('-'))

df.overview = df.overview + df.keywords.apply(strOp) + df.genres.apply(strOp) + df.credits.apply(lambda x: ' '.join(x.replace(' ', '').split('-')[:3]))
 
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['overview'])

display(pd.DataFrame(
    tfidf_matrix[:10, 7000:7070].toarray(),
    columns= tfidf.get_feature_names_out()[7000:7070],
    index = df.title[:10]).round())

print(tfidf_matrix.shape)


# We see that over 460000 different words were used to describe the 230160 movies in our dataset.
# 
# With this matrix in hand, we can now compute a similarity score.
# 
# We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. We use the cosine similarity score since it is independent of magnitude and is relatively easy and fast to calculate.
# 
# # Finally The Recommender Function

# In[ ]:



# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title):
    # Get the index of the movie that matches the title
    idx = df.index[df['title'] == title][0]
    # show given movie poster
    try:
        a = io.imread(f'https://image.tmdb.org/t/p/w500/{df.loc[idx, "poster_path"]}')
        plt.imshow(a)
        plt.axis('off')
        plt.title(title)
        plt.show()
    except:pass
    
    print('Recommendations\n')


    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(
        cosine_similarity(
            tfidf_matrix,
            tfidf_matrix[idx])))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:10]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    result = df.iloc[movie_indices]
    
    # show reco. movie posters
    fig, ax = plt.subplots(3, 3, figsize=(15,20))
    ax=ax.flatten()
    for i, j in enumerate(result.poster_path):
        try:
            ax[i].axis('off')
            ax[i].set_title(result.iloc[i].title,fontsize=22)
            a = io.imread(f'https://image.tmdb.org/t/p/w500/{j}')
            ax[i].imshow(a)
        except: pass
    fig.tight_layout()
    fig.show()


# In[ ]:


get_recommendations("Avatar")

