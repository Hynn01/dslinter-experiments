#!/usr/bin/env python
# coding: utf-8

# # H&M Consumer Analytics-RFM Segmentation and Collaborative Filtering

# ### We have applied consumer analytics on H&M Data set
# 
# https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendation
# 
# ***Overview:
# 
# 
# * Overview of Data
# * Handling Missing value treatment / Feature Engineering
#  
#  Find the EDA-
#  https://github.com/techanalyst84/customer-analytics-project/blob/main/h-m-sales-and-customers-deep-analysis_final.ipynb 
#  
# Solution Approach
# ##RFM
# ##Recommender Algorithm ##Singular Value Decomposition ##
# 
# The idea is to classify the customers based on sale transactions info over the last 1 year, into segments on basis of RFM(recency of purchase, frequency of purchase, and Montetory value they bring to the table). 
# Then to build a base recommender model based on Top RFM segments as training data and based on ML concepts like matrix factorization, Cosine similarity and Singular Value decomposition.
# Further to use this model to come up with predictions/recommendations of products/articles for the entire customer list. 
# To begin off this is good with Score of 0.0068, as the model is based on popular items in the recent past(amongst the popular customers); however, one can improvise upon this by building models specific to the customer's age group.
# 
# Thank you for your time !!
# 
# Please dont forget to upvote, if you find this informative for your work!!

# In[ ]:


## https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview -Score of Score: 0.0068-Please upvote!!
## Version Update- Made changes to train the model based on purchases in last year on Top 5 RFM segment and then come up with recommendation for entire customers
## Score of 0.0068, with much better run time. This model can be further improvised to include customer age into account..Wait for the next update!!
import numpy as np 
import pandas as pd
 

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-white')
sns.set_style("whitegrid")
sns.despine()
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

import matplotlib as mpl

mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# ### Data import

# In[ ]:



#articles = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv", 
 #                      encoding="ISO-8859-1", header=0)
#customers = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv",
 #                       encoding="ISO-8859-1", header=0)
transactions =  pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv",
                           encoding="ISO-8859-1",dtype={'article_id':str}, header=0).drop_duplicates()


# In[ ]:


transactions.head(7)


# # RFM Analysis
# 

# In[ ]:


# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[ ]:




transactions['InvoiceDate'] = pd.to_datetime(transactions['t_dat'],format='%Y-%m-%d')
transactions=transactions[["InvoiceDate", "customer_id", "article_id", "price","sales_channel_id"]].drop_duplicates()


# In[ ]:


transactions.shape


# In[ ]:


# checking df's missing value's attribution in %
df_null = round(100*(transactions.isnull().sum())/len(transactions), 2)
df_null


# In[ ]:


# checking df's missing value's attribution in %
df_null = round(100*(transactions.isna().sum())/len(transactions), 2)
df_null


# In[ ]:


##Generate Invoice ID as combination of Customer id and Transaction Date.
#transactions['_ID'] = transactions['customer_id']  + transactions['InvoiceDate'].astype(str) 

#transactions['Invoice_id'] = pd.factorize(transactions['_ID'])[0]


# In[ ]:


transactions.head()


# In[ ]:


import datetime as dt


# In[ ]:


start_date = dt.datetime(2020,3,1)

# Filter transactions by date
transactions["t_dat"] = pd.to_datetime(transactions["InvoiceDate"])
transactions = transactions.loc[transactions["t_dat"] >= start_date]


# In[ ]:


#analysis_date = max(transactions['InvoiceDate']) + dt.timedelta(days= 1)
analysis_date=dt.datetime(2020,9,23)
print((analysis_date).date())


# In[ ]:


transactions['date']=transactions['InvoiceDate']


# In[ ]:


rfm = transactions.groupby('customer_id').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,
    'date': 'count',
    'price': 'sum'})
#rfm.head()
rfm.columns=["Recency","Frequency","Monetary"]
rfm = rfm[rfm["Monetary"] > 0]
 
 #https://www.kaggle.com/code/kanberburak/rfm-analysis/notebook


# In[ ]:


transactions.head(1)


# In[ ]:


#Date from customer's last purchase.The nearest date gets 5 and the furthest date gets 1.
rfm["recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
# Total number of purchases.The least frequency gets 1 and the maximum frequency gets 5.
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
#Total spend by the customer.The least money gets 1, the most money gets 5.
rfm["monetary_score"]= pd.qcut(rfm["Monetary"],5,labels=[1,2,3,4,5])
rfm.head()


# In[ ]:


#RFM - The value of 2 different variables that were formed was recorded as a RFM_SCORE
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))


# In[ ]:


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm.head()


# In[ ]:


rfm[["segment", "Recency","Frequency","Monetary"]].groupby("segment").agg(["mean","count","max"]).round()


# In[ ]:


import plotly.express as px


# In[ ]:



x = rfm.segment.value_counts()
fig = px.treemap(x, path=[x.index], values=x)
fig.update_layout(title_text='Distribution of the RFM Segments', title_x=0.5,
                 title_font=dict(size=20))
fig.update_traces(textinfo="label+value+percent root")
fig.show()


# # Recommend Items Frequently Purchased Together
# 

# Item-Item Based Collaborative Filtering
# 
# * Objective-To produce recommendations of Items for Hibernating customer-User 5(from RFM) for their upcoming purchase. 
# 
# Step 1- Matrix Factorization.
# The Entries in table are based on time-adjusted count of # of purchase by user- item A bought 5 times on the first day of the train period is inferior to item B bought 4 times on the last day of the train period. This is done by weighted down exponentially by day of purchase
# 
# Step 2- See Recommendation as optimization problem, Rating Prediction-make good Recommendation or prediction.
# Quantify Goodness using RMSE:
# Lower RMSE =>better recommendation.
# Want to make good recommendation on items that user has not yet seen or purchase before(example – Hibernating customer-User 5 from RFM). Purely based on Popularity of the item. How we do?
# 
# Let’s build a system such that it works well on known (User, Product) rating/purchase counts. And hope the system will also predict well the unknown ratings.
# 
# Done by optimization method– Epoch. Then use this system to predict/recommend items unknown users
# 
# Use Latent Factor Model like SVD to Dimension Reduction, handling nulls.
# Now this is can be assumed as vector space in 2D
# And we can calculate the distant of two point using Cosine-get the nearest neighbor.
# 
# Step 6-Use this system/model to predict hibernating users-User 5 recommendation.
# 
# Concept based on -
# 
# https://www.youtube.com/watch?v=E8aMcwmqsTg 
# 
# 
# 
# https://www.analyticsvidhya.com/blog/2021/07/recommendation-system-understanding-the-basic-concepts/#:~:text=A%20recommendation%20system%20is%20a,suggests%20relevant%20items%20to%20users.

# ![](https://github.com/techanalyst84/customer-analytics-project/blob/main/Recommendator%201.jpg?raw=true)

# ![](https://github.com/techanalyst84/customer-analytics-project/blob/main/Recommendator%202.JPG?raw=true)

# ![](https://github.com/techanalyst84/customer-analytics-project/blob/main/Recommendator%203.JPG?raw=true)

# In[ ]:





# # Item-Based Collaborative Filtering -using Probabilistic Matrix Factorization
# 
# 

# **Preparing the data** 
# We need to restrict the data respect to a minimum transaction date. In that way, we reduce the dimensionality of the problem and we get rid of transactions that are not important in terms of the time decaying popularity.
# 
# Also, we are getting rid of articles that have not been bought enough. (Minimum 10 purchases are required)
# 
# 
# https://www.kaggle.com/code/luisrodri97/item-based-collaborative-filtering

# In[ ]:


import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm


# In[ ]:





# In[ ]:


rfm=rfm.reset_index()


# In[ ]:


transactions.head(1)


# In[ ]:


transactions=pd.merge(transactions,rfm[["customer_id","segment"]],how='inner',on='customer_id')
training_segment = ['champions', 'potential_loyalists', 'new_customers','promising','loyal_customers']
transactions = transactions[transactions['segment'].isin(training_segment)]
transactions=transactions.drop('segment', axis=1)


# In[ ]:


start_date = datetime.datetime(2020,9,1)
# Filter transactions by date
transactions["t_dat"] = pd.to_datetime(transactions["InvoiceDate"])
transactions = transactions.loc[transactions["InvoiceDate"] >= start_date]


# In[ ]:



# Filter transactions by number of an article has been bought
article_bought_count = transactions[['article_id', 'InvoiceDate']].groupby('article_id').count().reset_index().rename(columns={'InvoiceDate': 'count'})
most_bought_articles = article_bought_count[article_bought_count['count']>10]['article_id'].values
transactions = transactions[transactions['article_id'].isin(most_bought_articles)]
transactions["bought"]=1 


# In[ ]:




Due to the big amount of items, we can not consider the whole matrix in order to train. Therefore, we need to generate some negative samples: transactions that have never occured.


# In[ ]:


# Generate negative samples
np.random.seed(0)

negative_samples = pd.DataFrame({
    'article_id': np.random.choice(transactions.article_id.unique(), transactions.shape[0]),
    'customer_id': np.random.choice(transactions.customer_id.unique(), transactions.shape[0]),
    'bought': np.zeros(transactions.shape[0])
})

Model will be based on recommendations computed through the time decaying popularity and the most similar items to those items bought the most times by each user. Similarity among items is computed through cosine distance.


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


class ItemBased_RecSys:
    ''' Collaborative filtering using a custom sim(u,u'). '''

    def __init__(self, positive_transactions, negative_transactions, num_components=10):
        ''' Constructor '''
        self.positive_transactions = positive_transactions
        self.transactions = pd.concat([positive_transactions, negative_transactions])
        self.customers = self.transactions.customer_id.values
        self.articles = self.transactions.article_id.values
        self.bought = self.transactions.bought.values
        self.num_components = num_components

        self.customer_id2index = {c: i for i, c in enumerate(np.unique(self.customers))}
        self.article_id2index = {a: i for i, a in enumerate(np.unique(self.articles))}
        
    def __sdg__(self):
        for idx in tqdm(self.training_indices):
            # Get the current sample
            customer_id = self.customers[idx]
            article_id = self.articles[idx]
            bought = self.bought[idx]

            # Get the index of the user and the article
            customer_index = self.customer_id2index[customer_id]
            article_index = self.article_id2index[article_id]

            # Compute the prediction and the error
            prediction = self.predict_single(customer_index, article_index)
            error = (bought - prediction) # error
            
            # Update latent factors in terms of the learning rate and the observed error
            self.customers_latent_matrix[customer_index] += self.learning_rate *                                     (error * self.articles_latent_matrix[article_index] -                                      self.lmbda * self.customers_latent_matrix[customer_index])
            self.articles_latent_matrix[article_index] += self.learning_rate *                                     (error * self.customers_latent_matrix[customer_index] -                                      self.lmbda * self.articles_latent_matrix[article_index])
                
                
    def fit(self, n_epochs=10, learning_rate=0.001, lmbda=0.1):
        ''' Compute the matrix factorization R = P x Q '''
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        n_samples = self.transactions.shape[0]
        
        # Initialize latent matrices
        self.customers_latent_matrix = np.random.normal(scale=1., size=(len(np.unique(self.customers)), self.num_components))
        self.articles_latent_matrix = np.random.normal(scale=1., size=(len(np.unique(self.articles)), self.num_components))

        for epoch in range(n_epochs):
            print('Epoch: {}'.format(epoch))
            self.training_indices = np.arange(n_samples)
            
            # Shuffle training samples and follow stochastic gradient descent
            np.random.shuffle(self.training_indices)
            self.__sdg__()

    def predict_single(self, customer_index, article_index):
        ''' Make a prediction for an specific user and article '''
        prediction = np.dot(self.customers_latent_matrix[customer_index], self.articles_latent_matrix[article_index])
        prediction = np.clip(prediction, 0, 1)
        
        return prediction

    def default_recommendation(self):
        ''' Calculate time decaying popularity '''
        # Calculate time decaying popularity. This leads to items bought more recently having more weight in the popularity list.
        # In simple words, item A bought 5 times on the first day of the train period is inferior than item B bought 4 times on the last day of the train period.
        self.positive_transactions['pop_factor'] = self.positive_transactions['t_dat'].apply(lambda x: 1/(datetime.datetime(2020,9,23) - x).days)
        transactions_by_article = self.positive_transactions[['article_id', 'pop_factor']].groupby('article_id').sum().reset_index()
        return transactions_by_article.sort_values(by='pop_factor', ascending=False)['article_id'].values[:12]


    def predict(self, customers):
        ''' Make recommendations '''
        recommendations = []
        self.articles_latent_matrix[np.isnan(self.articles_latent_matrix)] = 0
        # Compute similarity matrix (cosine)
        similarity_matrix = cosine_similarity(self.articles_latent_matrix, self.articles_latent_matrix, dense_output=False)

        # Convert similarity matrix into a matrix containing the 12 most similar items' index for each item
        similarity_matrix = np.argsort(similarity_matrix, axis=1)
        similarity_matrix = similarity_matrix[:, -12:]

        # Get default recommendation (time decay popularity)
        default_recommendation = self.default_recommendation()

        # Group articles by user and articles to compute the number of times each article has been bought by each user
        transactions_by_customer = self.positive_transactions[['customer_id', 'article_id', 'bought']].groupby(['customer_id', 'article_id']).count().reset_index()
        most_bought_article = transactions_by_customer.loc[transactions_by_customer.groupby('customer_id').bought.idxmax()]['article_id'].values

        # Make predictions
        for customer in tqdm(customers):
            try:
                rec_aux1 = []
                rec_aux2 = []
                aux = []

                # Retrieve the most bought article by customer
                user_most_bought_article_id = most_bought_article[self.customer_id2index[customer]]

                # Using the similarity matrix, get the 6 most similar articles
                rec_aux1 = self.articles[similarity_matrix[self.article_id2index[user_most_bought_article_id]]]
                # Return the half of the default recommendation
                rec_aux2 = default_recommendation

                # Merge half of both recommendation lists
                for rec_idx in range(6):
                    aux.append(rec_aux2[rec_idx])
                    aux.append(rec_aux1[rec_idx])

                recommendations.append(' '.join(aux))
            except:
                # Return the default recommendation
                recommendations.append(' '.join(default_recommendation))
        
        return pd.DataFrame({
            'customer_id': customers,
            'prediction': recommendations,
        })

Define your hyperparameters and fit the model. Take into account that there are more customizable parameters in the data processing section.


# In[ ]:


rec = ItemBased_RecSys(transactions, negative_samples, num_components=1000)
rec.fit(n_epochs=1)


# In[ ]:


customers = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv'
                       ,encoding="ISO-8859-1", dtype={'article_id':str},header=0  ).customer_id.unique()


# In[ ]:


recommendations = rec.predict(customers)


# In[ ]:


recommendations.head()


#  
