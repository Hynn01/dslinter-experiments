#!/usr/bin/env python
# coding: utf-8

# # Getting Started | EDA➡️Preprocessing
# 
# In this notebook I will walk you through the beginning steps of taking on a Kaggle competition -- from understanding the dataset to geting data prepared to be sent to a machine learning model. We will go over the follow tasks:
# 
# * Reading in the dataset
# * Calculating statistics about the dataset
# * Visualizing single features of the data
# * Visualizing multiple features (multivariate analysis)
# * Preprocessing data for machine learning model
# * Further steps to create a winning model
# 
# 
# ### *If you enjoy, please upvote* 😊

# # Read in Dataset
# 
# To begin, we will read in the dataset with the `pandas` library. `pandas` is a great tool for manipulating and displaying tabular data. 
# 
# For more information on `pandas`, visit this link:  
# 
# > https://pandas.pydata.org/docs/ 

# In[ ]:


import pandas as pd #data manipulation

#removes limit on number of columns to display
pd.set_option('display.max_columns', None)

#This line saves the training data into a dataframe 
#for more on dataframes see below
#-> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
train_data = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv",
                        index_col=0)


#shows first 5 rows of dataframe
train_data.head()


# ## What this shows us: 
# 
# From the first 5 samples of the dataset we see there are **31 anonymous features** (their meaning is unknown) and a **target feature**. All of the **anonymous features are numeric except '*f_27*'** which seems to be categorical.

# # Dataset Overview
# 
# In the previous section we looked at the first 5 samples of the training data. From our observations we could come up with some conclusions and speculations about the dataset. In this section we will use the `describe` method to view statistics about the columns of the dataset. 
# 
# For more information on `describe`, visit this link: 
# 
# > https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html

# In[ ]:


#shows basic statistics of each numeric column of the dataframe
train_data.describe()


# In[ ]:


#shows basic statistics of each categoric column of the dataframe (f_27)
train_data.describe(include=['object'])


# ## What this shows us: 
# 
# An initial glance of the dataset's summary statistics show us that there are **900,000** data points and **no null values**. Also we see that the **features (columns) are not the same scale**. The only non-numeric feature **(*f_27*) has almost as many unique values as there are rows** in the dataset. This could mean a couple of things, the *f_27* feature is either useless to us, needs to be represented numerically, or needs to be grouped into a smaller set of categories. Finally, the **target feature is balanced** -- meaning there are roughly the same number of *1* values as there are *0* values.
# 
# > **Note on the target feature:** This is a binary classification problem (as it states on the competition page). This means that there are only 2 classes to predict. It is common to numerically encode classes in a binary classification problem as 1 or 0.
# 

# # Basic Exploratory Data Analysis (EDA)
# 
# It is often easier for us to understand patterns when we can see it visually. In this section we will use the `matplotlib.pyplot` library to display some information on our dataset. We will **create 2 types of graphs** to get a better understanding of our data. The first graph will be a histogram. **Histograms** are used to display the distribution of values of a feature. The second graph we make will be a bar chart to display the *f_27* categoric values. **Bar Charts** are an easy and clear way to display categoric values.
# 
# For more information on `matplotlib.pyplot`, visit this link: 
# 
# > https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.html

# In[ ]:


import matplotlib.pyplot as plt #data viz
# v Sets matplotlib figure size defaults to 25x20
plt.rcParams["figure.figsize"] = (25,20)

# Creates a figure with 30 subplots to plot each numeric feature
#on 'subplots'->https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.subplots.html
fig, ax = plt.subplots(#This functions lets us place many plots within a single figure
    5, #number of rows
    6  #number of columns
)

#adds title to figure            
fig.text(
    0.35, # text position along x axis
    1, # text position along y axis
    'EDA of Numeric Features', #title text
    {'size': 35} #Increase font size to 35
         )

#The below code will display all numeric feature distributions with a histogram

# subplots can be accessed with an index similar to python lists
i = 0 # subplot column index
j = 0 # subplot row index
for col in train_data.columns: #iterate thru all dataset columns
    if col not in ['f_27', 'target']: #dont plot f_27 or target feature-will error
        ax[j, i].hist(train_data[col], bins=100) #plots histogram on subplot [j, i]
        ax[j, i].set_title(col, #adds a title to the subplot
                           {'size': '14', 'weight': 'bold'}) 
        if i == 5: #if we reach the last column of the row, drop down a row and reset
            i = 0
            j += 1
        else: #if not at the end of the row, move over a column
            i += 1


plt.show() #displays figure


# In[ ]:


#The below code will display value counts of the categoric feature 'f_27'
train_data['f_27'].value_counts()[:50].plot(kind='bar') #shows the top 50 common values
plt.title('f_27 Top 50 Most Common Values', {'size': '35'}) #Adds title
plt.show() #displays figure


# ## What this shows us: 
# 
# From the graphs we created we can see that **not all the numeric features are continuous**. There are a handful of numeric features that are **discrete and may be a categoric representation**. Of the continuous numeric features, **all had an almost perfect normal distribution**. Visualization of the non-numeric categorical feature **(*f_27*) did not give any obvious insight** to what each category means or how they are related to one another. 

# # Target Feature EDA
# 
# Now that we have a better understanding of the dataset features, we will look at how the target feature (what we are predicting) relates to the other dataset features. **This is a very important step** in creating a great model. We need to have a really good understanding of which features are good indicators of the target feature. We can use this information to cherry-pick the most important features or to create our own custom features. A quick way to see correlation between the target feature and other dataset feature is the `corr` method in `pandas`. This method will show us the Pearson correlation coefficients of each feature in relation to one another. 
# 
# For more information on `corr`, visit this link: 
# 
# > https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
# 
# 
# *For demonstration purposes, I will only look at 2 features and how they interact with the target feature. You should look many combinations of features when you try this yourself.*
# 
# We will also be using the `seaborn` library to create KDE plots (kernel density estimate) and scatter plots. KDE plots are similar to histograms but they show a continous distribution of the data without binning. Scatter plots show how 2 numeric variables relate to one another. 
# 
# For more information on `seaborn`, visit this link:
# 
# > https://seaborn.pydata.org/

# In[ ]:


import seaborn as sns #data viz

#Displays Pearson correlation coefficients of each feature in relation to one another
#This value ranges from -1 to 1... 0 means no correlation
#seaborn heatmap link: https://seaborn.pydata.org/generated/seaborn.heatmap.html
plt.figure(figsize=(30, 2))         #changes figure size
sns.heatmap(train_data.corr()[-1:], #takes correlations of only target feature
            cmap="viridis",         #changes the color palete
            annot=True              #display coeficient value
           )
plt.title('Correlation with Target Feature', {'size': '35'}) #sets title
plt.show() #displays figure


# **We will look at the 2 most correlated features -- *f_19* (-0.088) and *f_21* (0.13) -- but you can show any features you would like by change the variable values in the next cell.**

# In[ ]:


# Change the variable values to other feature names to display that feature
FEAT_1 = 'f_19'
FEAT_2 = 'f_21'


# In[ ]:


#The below code will display KDE plots split by the features target value
#kdeplot link: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
# v Sets matplotlib figure size defaults to 12x6
plt.rcParams["figure.figsize"] = (12,6)

# Creates a figure with 2 subplots to plot 2 features x 2 target values
fig, ax = plt.subplots(#This functions lets us place many plots within a single figure
    1, #number of rows
    2  #number of columns
)

#Plots kde for FEAT_1 
sns.kdeplot(
    data=train_data[train_data.target==0], #data for target value == 0
    x=FEAT_1,                              
    shade=True,                            #shade under curve
    color='blue',
    label='0',
    ax=ax[0]
)
sns.kdeplot(
    data=train_data[train_data.target==1], #data for target value == 1
    x=FEAT_1,
    shade=True,
    color='orange',
    label='1',
    ax=ax[0]
)
#set title for subplot
ax[0].set_title(FEAT_1, {'size':'18', 'weight': 'bold'})
ax[0].legend()#displays legend 


#Plots kde for FEAT_2 
sns.kdeplot(
    data=train_data[train_data.target==0], #data for target value == 0
    x=FEAT_2,                              
    shade=True,                            #shade under curve
    color='blue',
    label='0',
    ax=ax[1]
)
sns.kdeplot(
    data=train_data[train_data.target==1], #data for target value == 1
    x=FEAT_2,
    shade=True,
    color='orange',
    label='1',
    ax=ax[1]
)
#set title for subplot
ax[1].set_title(FEAT_2, {'size':'18', 'weight': 'bold'})
ax[1].legend()#displays legend 

#sets the figure title
fig.text(
    0.05, #x position of text
    1,    #y position of text
    'Distribution for Different Target Values', #text
    {'size':'35'} #style
)
plt.show() #displays figure


# In[ ]:


#The below code will display a scatter plot with different colors for target values

plt.figure(figsize=(10,10)) #changes figure size to 10x10
#seaborn scarrplot link: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
#TIP: Adjust 'alpha' value to change transparency
sns.scatterplot(data=train_data, x=FEAT_1, y=FEAT_2, hue='target', alpha=0.1) 
#sets figure title
plt.title("{} v {}: Target Values".format(FEAT_1, FEAT_2),
         {'size': 35})
plt.show() #displays figure


# ## What this shows us: 
# 
# The **most correlated features with the target are *f_19* and *f_21***. Different target value distributions for *f_19* and *f_21* **both have a slight shift** from eachother. This shows us that these features have some predictive power on their own in predicting the target feature. The distribution shift was not large so they will **not be very useful on their own**. In our bivariate analysis of *f_19* and *f_21* (scatter plot) we observed different target values in different areas of the feature space. This means **combining features has greater predictive power**.
# 
# *Now try these visualizations on other features.*

# # Preprocess Data
# 
# Before we can properly train a machine learning model, we need to preprocess our data. The preprocessing steps can differ greatly depending on the type of model used and the problem at hand. We will just be doing basic preprocessing steps that will be helpful for most models. 
# 
# The preprocessing steps will be as follow:
# 
# 1. Split data into X and y (only on training data)
# 2. Split into training|validation set (only on training data)
# 3. Convert non-numeric features into a numerical representation (training + test)
# 4. Scale each feature to be between 0-1 (training + test)
# 
#  > **Note on Preprocessing**: When preprocessing data you will want to make sure to only use training data to scale or impute on the validation/test data. When we do scaling between 0-1 we will use the training set's minimum and maximum values to do this.
# 
# 
# We will be using `numpy` to manipulate the data. `numpy` is a high-performance mathematical library for Python. 
# 
# For more information on `numpy`, visit this link: 
# 
# > https://numpy.org/doc/stable/

# *Below cell reads in test data file.*

# In[ ]:


#Need to read in the test data to preprocess
test_data = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv",
                       index_col=0)
test_X = test_data.values #converts features into numpy array


# ### Step 1. Split data into X and y (only on training data)
# 
# In supervised learning tasks like this one (binary classification) there needs to be a target variable (the thing we want to predict). In this case it is the feature labeled as *'target'*. We will seperate this target feature from the other dataset features. Most ML algorithms require the target variable to be a seperate parameter.

# In[ ]:


#splits the training data into predictive feature and target feature
X = train_data.drop('target', axis=1).values #train features converted to numpy array
y = train_data.target.values #target converted to numpy array

#displays a sample
print("PREDICTIVE FEATURES:")
print(X[0, :])
print('---------------------')
print('TARGET:')
print(y[0])


# ### Step 2. Split into training|validation set (only on training data)
# 
# A validation set is a number of data samples that we reserve to use for observing our model's performance. This is a **very important step**. A validation set is similar to a test set in that we use it to test our model's performance on unseen data. However, if we keep tuning our model so that it reaches the best possible score on the validation set we are at a high risk of **overfitting on the validation set**. This happens very often in Kaggle competitions - the **top scoring teams on the public scoreboard are many times overfitting to the public scoreboard data** so when the competition ends, their score is a lot worse. This can be avoided by making sure you have your own validation set and a test set.

# In[ ]:


import numpy as np #math/data manipulation

split_size = 0.8 # 80% training -> 20% validation
num_rows = X.shape[0] #number of rows of the dataset
split_ix = int(num_rows * split_size) #gets the index of where to split the data

#splits into training data
train_X = X[:split_ix]
train_y = y[:split_ix]
#splits into validation data
val_X = X[split_ix:]
val_y = y[split_ix:]

#displays size of new training/validation set
print("TRAINING SET LENGTH:", train_X.shape[0])
print("VALIDATION SET LENGTH:", val_X.shape[0])
print("TESTING SET LENGTH:", test_X.shape[0])


# ### Step 3. Convert non-numeric features into a numerical representation (training + test)
# 
# There are 2 different ways we can go about converting *f_27* characters into a numerical representation. The first way is to one-hot encode each character. The other way is to count how many times each character appeared in the sequence (also called *bag of words*).
# Lets take a look at what each of this would look like and weigh the pros and cons.
# 
# * Suppose there are 5 unique characters for this example: A->E (*actual dataset has 20 unique characters A->T*)
# * Each data point has exactly 5 characters total (*actual dataset has 10 character total*)
# * Our sample data point is "ABDBD"
# 
# 
# **One-Hot Encoded Representation**
# ```
# "ABDBD" -> [
#             [1, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0],
#             [0, 0, 0, 1, 0],
#             [0, 1, 0, 0, 0],
#             [0, 0, 0, 1, 0]
#            ]
# ```
# * **Pros**: Captures order/placement of characters
# * **Cons**: Greatly increases number of features (this example would add 25 features), sparse (mostly 0 values)
# 
# **Bag-of-Words Representation**
# ```
# "ABDBD" -> [1, 2, 0, 2, 0]
# ```
# * **Pros**: Limited extra features, captures character count
# * **Cons**: Loses all sense of character order
# 
# > **Tf-idf (Advanced)**: Tf-idf is a simple twist on Bag-of-Words. It stands for *term frequency-inverse document frequency*. This approach will look at a normalized count where each character count is divided by the number of sequences that character appears in. This method may be better in capturing the meaning of characters that do not appear often. For more information on tf-idf, visit this link: https://monkeylearn.com/blog/what-is-tf-idf/
# 
# #### **Moving Forward**
# 
# To keep our data as simple and easy as possible, lets opt for the **Bag-of-Words** approach. Feel free to try the One-Hot Encoder approach on your own with `sklearn.preprocessing.OneHotEncoder` available here:
# 
# > https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

# In[ ]:


from string import ascii_uppercase #list of all upercase characters 
                                   #(easier than writing 'A'->'T' myself)


def bag_of_characters(dataset):
    """
    This function takes a dateset and transforms the 'f_27'
    into a bag-of-words representation as mentioned in the markdown above
    """
    new_features = [] #bag-of-words features
    for i in range(dataset.shape[0]): #iterates thru every row
        # the below 3 lines make a dictionary that maps a character to the 
        # total amount present -- initializes at 0 (ex. 'A': 0, 'B': 0 ...)
        bag = {} 
        for c in ascii_uppercase[:20]: #ix 0 = 'A', ix 19 = 'T'
            bag[c] = 0
        
        for char in dataset[i, 27]: #iterates thru the 10 total characters in the feature
            bag[char] += 1 #adds 1 to that characters value in the 'bag' dict
        new_features.append(list(bag.values())) #append this row 
    return np.array(new_features)

#adds new features by appending the old numpy array with the new features along
#axis 1 (axis 1 = columns, axis 0 = rows)
train_X = np.append(train_X, bag_of_characters(train_X), axis=1) 
val_X = np.append(val_X, bag_of_characters(val_X), axis=1)
test_X = np.append(test_X, bag_of_characters(test_X), axis=1)

#deletes the character feature 'f_27' from the numpy array
# np.delete(array, row/column#, axis#)
train_X = np.delete(train_X, 27, 1)
val_X = np.delete(val_X, 27, 1)
test_X = np.delete(test_X, 27, 1)


#display a sample of the data
print('SAMPLE OF BAG-OF-WORDS TRANSFORMED DATA')
print(train_X[0])


# ### Step 4. Scale each feature to be between 0-1 (training + test)
# 
# Not all features of this data set are on the same scale. Many ML models require that data be on the same scale to properly train. We will scale all of our features to be between 0 and 1. To do this we will execute the following formula on each feature:
# 
# ```
# x_new = ( x - min(X) ) / ( max(X) - min(X) )
# ```
# *where X is a feature and x is an element of that feature*
# 
# 
# 
# > **Note on scaling:** Scaling is not required for all ML algorithms. Algorithms based on decision trees (such as the popular **XGBoost**) do not require that features be scaled. Scaling will not negatively affect their performance however.

# In[ ]:


min_max = {} #dictionary to hold min max values for each feature


for i in range(train_X.shape[1]): #iterates thru training data columns
    min_max[i] = (train_X[:, i].min(), train_X[:, i].max()) #populates min_max dictionary 
    
    
def min_max_scale(dataset):
    """
    This function will take in a dataset and apply min-max scaling to it.
    The values of the min-max scaling will be taken from the 'min_max' dictionary
    """
    for i in range(dataset.shape[1]): #iterate thru dataset columns
        #new value = (value - min) / (max - min)
        dataset[:,i] = (dataset[:,i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
    return dataset

#apply our min_max_scale funtion to all datasets
train_X = min_max_scale(train_X)           #training
val_X = min_max_scale(val_X)   #validation
test_X = min_max_scale(test_X) #testing


#displays sample of scaled data:
print("SAMPLE OF SCALED DATA:")
print(train_X[0])


# # Where to go from here...
# 
# ### Models
# 
# From this point on you can **start training basic machine learning models** on the processed data we created. I recommend, for beginners, starting with some of the following models available in the `sklearn` library:
# 
# * **Logistic Regression** (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# * **Decision Tree** (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
# 
# Once you understand these basics, a great algorithm (based off of the decision tree) is **XGBoost** (Extreme Gradient Boost(ing)). 
# 
# Here is a link for `xgboost`: 
# 
# > https://xgboost.readthedocs.io/en/stable/python/python_api.html?highlight=XGBClassifier#xgboost.XGBClassifier
# 
# 
# ### Feature Engineering
# 
# There is still a lot left to explore within the features of the dataset. I recommend exploring further how different features relate to the target feature. Based on these relations, you may **create more features that are combinations of two or more features**. An example would be a feature *C* that is feature *A* x feature *B*. Also, the feature ***f_27* needs to be examined in more detail**. There may be important information hidden in all of those seemingly meaningless characters.
# 
# 
# ### Evaluating Performance
# 
# The performance metric for this competition is **Area Under the ROC Curve**. It is very important to understand this to create a winning model. 
# 
# Here is a link to learn more about AUC (ROC): 
# 
#  > https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
# 
# Here is a function to calculate area under the roc curve, `sklearn.metrics.roc_auc_score`:
# 
# > https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
# 
# 
# 
# ### I hope this information helped you. Good luck in the competition! 🍀
