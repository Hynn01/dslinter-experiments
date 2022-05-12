#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-primary" style="border-radius:25px" role="alert">
#   <h1 style="text-align: center; color:blue ;font-size:200%;">Titanic Spaceship- Let's Rescue Together</h1>
# </div>

# <div class="alert alert-success" style="border: 1px solid white; border-radius:25px" role="alert">
#   <h4 class="alert-heading">Welcome!</h4>
#     <p>Welcome to Spaceship-Titanic Kernel! </p>
#   <hr>
#   
#   <p class="mb-0">Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.
# 
# The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.</p>
#       
#   <p class="mb-0">While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!</p>
# </div>
# 
# <div class="alert alert-success" style="border-radius:25px" role="alert">
#   
#   <h4 class="alert-heading">Challenge!</h4>
#     <p class="text-dark">To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.</p>
# 
# <p class="text-dark">Help save them and change history!</p>
#     
# </div>
# 
# <div class="alert alert-success" style="border-radius:25px" role="alert">
#   
#   <h4 class="alert-heading">Task!</h4>
#     <p class="text-dark"> Create a machine learning classification model to acuurately predict the status of remaining passengers using the information recovered from the spaceship's damaged computer system</p>
# </div>
# 

# ## Contents
# <a id='Contents'></a>
#  * 1. [Library and Data Import](#phase1)<hr>
#  
#  * 2. [Exploratory Data Analysis](#phase2)
#      * 2.1 [Dataset Review](#phase2.1)
#      * 2.2 [Correlations](#phase2.2)
#      * 2.3 [Tabular Results](#phase2.3)
#      * 2.4 [Graphical Presentation](#phase2.4)
#   <hr>    
#  
#  * 3. [Missing Data Analysis](#phase3)<hr>
#  
#  * 4. [Detection of the Outliers](#phase4)<hr>
#  
#  * 5. [Data Pre-processing](#phase5)
#      * 5.1 [Normalaty Check](#phase5.1)
#      * 5.2 [Train and Test Dataset](#phase5.2)
#      * 5.3 [Feature Scaling](#phase5.3)
#  <hr>
#  * 6. [Classification ML Models](#phase6)
#      * 6.1 [Base Classification Models](#phase6.1)
#      * 6.2 [Grid Search CV](#phase6.2)
#      * 6.3 [Hypertuned Classification Models](#phase6.3)
# <hr>
#  * 7. [Test Dataset](#phase7) (**under construction**)
# 

# <div>
#     
# <h1>1. Library and Data Import <a id='phase1'></a></h1>
#     
# </div>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Machine Learning Libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgbm


# In[ ]:


train=pd.read_csv('../input/spaceship-titanic/train.csv')


# <div>
#     
# <h1>2. Exploratory Data Analysis <a id='phase2'></a></h1>
#     
# </div>

# In[ ]:


train.head()


# <div>
#     
# <h2>2.1 Dataset Review <a id='phase2.2'></a></h2>
#     
# </div>

# In[ ]:


train.info()


# In[ ]:


train.describe().T


# In[ ]:


pd.DataFrame((train.isna().sum()/ train.shape[0])*100)


# <p class="text-primary"> Missing values table displays the percentage of missing values in this dataset. Except the PassengerId and the target variable (Transported), all the other features have nearly missing values in between two to three percent. Such an extent of percentage missing values can be replaced using different characteristics.</p>

# In[ ]:


pd.DataFrame(train.nunique())


# <p class="text-primary"> It is observed that features HomePlanet, CryoSleep, Destination, VIP are categorical variables.</p>
# <hr> 
# <p class="text-primary"> Remaining features have more than 10 unique values. These features are likely numerical features.</p>

# <div>
#     
# <h2>2.2 Correlations <a id='phase2.2'></a></h2>
#     
# </div>

# In[ ]:


plt.figure(figsize=(14,6))
sns.heatmap(train.corr(), annot=True, fmt='.3f', cmap='CMRmap_r')


# <p class="text-primary"> Most associated feature with the target variable is 'RoomService' while ShoppingMall is the least associated feature.</p>

# In[ ]:


sns.set_theme()
train.corr()['Transported'].drop('Transported').sort_values().plot.barh()


# <div>
#     
# <h2>2.3 Tabular Results <a id='phase2.3'></a></h2>
#     
# </div>

# In[ ]:


train['Transported'].value_counts()


# <p class='text-primary'> Nearly 50% passengers were lost during this maiden voyage.</p>

# In[ ]:


pd.DataFrame(train.groupby(['HomePlanet'])['Transported'].value_counts())


# In[ ]:


pd.DataFrame(train.groupby(['CryoSleep'])['Transported'].value_counts())


# In[ ]:


pd.DataFrame(train.groupby(['HomePlanet', 'Destination'])['Transported'].value_counts())


# In[ ]:


pd.DataFrame(train.groupby(['HomePlanet', 'Destination', 'CryoSleep'])['Transported'].value_counts())


# In[ ]:


pd.DataFrame(train.groupby(['HomePlanet', 'Destination', 'VIP', 'CryoSleep'])['Transported'].value_counts())


# <div>
#     
# <h2>2.4 Graphical Presentation <a id='phase2.4'></a></h2>
#     
# </div>

# <p class='text-primary'> Let's visualize same tables using the graphs. </p>

# In[ ]:


fig=plt.figure(figsize=(20,6))
sns.histplot(x=train['HomePlanet'], shrink=0.8, bins=12, hue=train['Transported'], multiple="dodge",palette='brg' )


# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(18,6))

fig.add_subplot(1,3,1)
sns.histplot(x=train[train['HomePlanet']=='Europa']['Destination'], shrink=0.8, hue=train['Transported'], multiple="dodge",palette='Reds' )

fig.add_subplot(1,3,2)
sns.histplot(x=train[train['HomePlanet']=='Earth']['Destination'], shrink=0.8, hue=train['Transported'], multiple="dodge",palette='Blues' )

fig.add_subplot(1,3,3)
sns.histplot(x=train[train['HomePlanet']=='Mars']['Destination'], shrink=0.8, hue=train['Transported'], multiple="dodge",palette='Greens' )


# In[ ]:


train['VIP']=train['VIP'].astype('float32')


# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(18,6))

fig.add_subplot(1,3,1)
sns.countplot(x=train[train['HomePlanet']=='Europa']['VIP'], hue=train['Transported'],palette='viridis_r' )

fig.add_subplot(1,3,2)
sns.countplot(x=train[train['HomePlanet']=='Earth']['VIP'], hue=train['Transported'], palette='viridis_r' )

fig.add_subplot(1,3,3)
sns.countplot(x=train[train['HomePlanet']=='Mars']['VIP'], hue=train['Transported'], palette='viridis_r' )


# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(18,6))

fig.add_subplot(1,3,1)
sns.countplot(x=train[train['HomePlanet']=='Europa']['VIP'], hue=train['CryoSleep'],palette='magma' )

fig.add_subplot(1,3,2)
sns.countplot(x=train[train['HomePlanet']=='Earth']['VIP'], hue=train['CryoSleep'], palette='magma' )

fig.add_subplot(1,3,3)
sns.countplot(x=train[train['HomePlanet']=='Mars']['VIP'], hue=train['CryoSleep'], palette='magma' )
plt.show()


# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(18,6))

fig.add_subplot(1,3,1)
sns.countplot(x=train[train['HomePlanet']=='Europa']['Destination'], hue=train['CryoSleep'],palette='summer' )

fig.add_subplot(1,3,2)
sns.countplot(x=train[train['HomePlanet']=='Earth']['Destination'], hue=train['CryoSleep'], palette='summer' )

fig.add_subplot(1,3,3)
sns.countplot(x=train[train['HomePlanet']=='Mars']['Destination'], hue=train['CryoSleep'], palette='summer' )


# <p class='text-danger'> Let's analyse numerical characteristics to related target variables as well as other features in this dataset</p>

# In[ ]:


train['Id']=train.index + 1


# In[ ]:


columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' ]
sns.set_theme()
fig=plt.figure(figsize=(18,10))
j=1

for i in columns:
    fig.add_subplot(2,3,j)
    sns.scatterplot(data=train, x='Id', y=i, hue='Transported', palette='dark')
    j+=1


# In[ ]:


columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' ]
sns.set_theme()
fig=plt.figure(figsize=(18,10))
j=1

for i in columns:
    fig.add_subplot(2,3,j)
    sns.scatterplot(data=train, x='Id', y=i, hue='CryoSleep', palette='viridis')
    j+=1


# In[ ]:


columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' ]
sns.set_theme()
fig=plt.figure(figsize=(18,10))
j=1
m=0
for i in columns:
    fig.add_subplot(2,3,j)
    sns.scatterplot(data=train,x='Id' , y=i, hue='HomePlanet', palette='dark')
    j+=1


# In[ ]:


columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' ]
sns.set_theme()
fig=plt.figure(figsize=(18,10))
j=1

for i in columns:
    fig.add_subplot(2,3,j)
    sns.scatterplot(data=train, x=range(len(train)), y=i, hue='Destination', palette='dark')
    j+=1


# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(18,10))

fig.add_subplot(2,2,1)
sns.scatterplot(data=train, x='FoodCourt', y='RoomService', hue='Transported', palette='Reds')

fig.add_subplot(2,2,2)
sns.scatterplot(data=train, x='ShoppingMall', y='RoomService', hue='Transported', palette='Greens')

fig.add_subplot(2,2,3)
sns.scatterplot(data=train, x='Spa', y='RoomService', hue='Transported', palette='Blues')


fig.add_subplot(2,2,4)
sns.scatterplot(data=train, x='VRDeck', y='RoomService', hue='Transported', palette='Blues')


# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(18,10))

fig.add_subplot(2,2,1)
sns.scatterplot(data=train, x='RoomService', y='FoodCourt', hue='Transported', palette='Reds')

fig.add_subplot(2,2,2)
sns.scatterplot(data=train, x='ShoppingMall', y='FoodCourt', hue='Transported', palette='Greens')

fig.add_subplot(2,2,3)
sns.scatterplot(data=train, x='Spa', y='FoodCourt', hue='Transported', palette='Blues')


fig.add_subplot(2,2,4)
sns.scatterplot(data=train, x='VRDeck', y='FoodCourt', hue='Transported', palette='Blues')


# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(18,10))

fig.add_subplot(2,2,1)
sns.scatterplot(data=train, x='RoomService', y='Spa', hue='Transported', palette='Reds')

fig.add_subplot(2,2,2)
sns.scatterplot(data=train, x='ShoppingMall', y='Spa', hue='Transported', palette='Greens')

fig.add_subplot(2,2,3)
sns.scatterplot(data=train, x='FoodCourt', y='Spa', hue='Transported', palette='Blues')


fig.add_subplot(2,2,4)
sns.scatterplot(data=train, x='VRDeck', y='Spa', hue='Transported', palette='Blues')


# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(18,10))

fig.add_subplot(2,2,1)
sns.scatterplot(data=train, x='RoomService', y='VRDeck', hue='Transported', palette='Reds')

fig.add_subplot(2,2,2)
sns.scatterplot(data=train, x='ShoppingMall', y='VRDeck', hue='Transported', palette='Greens')

fig.add_subplot(2,2,3)
sns.scatterplot(data=train, x='FoodCourt', y='VRDeck', hue='Transported', palette='Blues')


fig.add_subplot(2,2,4)
sns.scatterplot(data=train, x='Spa', y='VRDeck', hue='Transported', palette='Blues')


# <div>
#     
# <h1>3. Missing Data Analysis <a id='phase3'></a></h1>
#     
# </div>

# In[ ]:


pd.DataFrame(train.isna().sum())


# <h2>Approach:</h2>
# <p class='text-primary'>1. First, missing values of categorical features are addressed.</p>
#    
# <p class='text-primary'> 2. Numerical features are addressed using mean/mode values obtained by group_by option in python using different categorical characteristics.</p>

# <p class='text-danger'> For the features mentioned in the columns list, mode of the respective column is used to fill the missing columns.</p>

# In[ ]:


columns=['HomePlanet', 'CryoSleep', 'VIP']
for i in columns:
    train[i]=train[i].fillna(train[i].mode()[0])


# <p class='text-primary'> As provided in the documentation, The cabin number where the passenger is staying, takes the form deck/num/side, where side can be either P for Port or S for Starboard. </p>
# 
# <p class='text-primary'> Since each cabin has a limited capacity, missing values can not be replaced using mean/mode/median strategy.</p> 
# 
# <p class= 'text-danger'> Two different features are created i.e. 'Deck' and 'Side'. The mode values of deck and the side are used to fill the missing values in this feature. 
#     
# **Note: Cabin feature is dropped in the pre-processing phase.**

# In[ ]:


train['Cabin']=train['Cabin'].astype(str)


# In[ ]:


train['Deck']=train['Cabin'].apply(lambda x : x.split('/',1)[0])


# In[ ]:


train['Side']=train['Cabin'].apply(lambda x :  (x.split('/',1)[-1]).split('/')[-1])


# In[ ]:


train['Side']=train['Side'].replace('nan', np.nan)

train['Deck']=train['Deck'].replace('nan', np.nan)
train['Cabin']=train['Cabin'].replace('nan', np.nan)


# <p class='text-primary'>I found the mode values for different groups obtained using group_by function in python. Missing values are replaced using these mode values.</p>

# In[ ]:


train['Deck'] = train['Deck'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP'])['Deck'].transform(lambda x: x.value_counts().idxmax()))
train['Side'] = train['Side'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP'])['Side'].transform(lambda x: x.value_counts().idxmax()))
train['Destination'] = train['Destination'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP'])['Destination'].transform(lambda x: x.value_counts().idxmax()))


# <p class='text-primary'>To maintain constistency in the assumptions above, I used same features for the grouping.</p>

# <p class='text-primary'>Same strategy is applied for replacing missing values in numerical characteristics as well. Instead of 
# the mode value in case of categorical features, the missing values for numerical characteristics are replaced using mean values. </p>

# In[ ]:


train['Age'] = train['Age'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP', 'Deck', 'Side'])['Age'].transform('mean'))


# In[ ]:


train['RoomService'] = train['RoomService'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP', 'Deck', 'Side'])['RoomService'].transform('mean'))


# In[ ]:


train['FoodCourt'] = train['FoodCourt'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP', 'Deck', 'Side'])['FoodCourt'].transform('mean'))


# In[ ]:


train['Spa'] = train['Spa'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP', 'Deck', 'Side'])['Spa'].transform('mean'))


# In[ ]:


train['ShoppingMall'] = train['ShoppingMall'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP', 'Deck', 'Side'])['ShoppingMall'].transform('mean'))
train['ShoppingMall']=train['ShoppingMall'].fillna(train['ShoppingMall'].mean())


# In[ ]:


train['VRDeck'] = train['VRDeck'].fillna(train.groupby(['CryoSleep', 'HomePlanet', 'VIP', 'Deck', 'Side'])['VRDeck'].transform('mean'))
train['VRDeck']=train['VRDeck'].fillna(train['VRDeck'].mean())


# In[ ]:


pd.DataFrame(train.isna().sum())


# <p class='text-primary'>No values missing!!!! </p>
# 
# **PS: 'Name' column is dropped. 'Cabin' column is already divided into 'Deck' and 'Side' columns. Columns are re-arranged and mentioned columns will be dropped in data-preprocessing phase.** 

# <div>
#     
# <h1>4. Outliers detection <a id='phase4'></a></h1>
#     
# </div>

# In[ ]:


columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' ]
sns.set_theme()
fig=plt.figure(figsize=(18,10))
j=1

for i in columns:
    fig.add_subplot(2,3,j)
    sns.scatterplot(data=train, x=range(len(train)), y=i, hue='Destination', palette='dark')
    j+=1


# In[ ]:


train=train[~ ((train['Spa']>7500) | (train['VRDeck']>10000) | (train['FoodCourt']>11000) | (train['RoomService']>=6000) | (train['ShoppingMall']>4000) ) ] 


# <div>
#     
# <h1>5. Data Pre-processing <a id='phase5'></a></h1>
#     
# </div>

# In[ ]:


Y=train['Transported']


# In[ ]:


train['Total_expenses']= train['VRDeck']+train['Spa']+train['ShoppingMall']+train['FoodCourt']+train['RoomService']


# In[ ]:


train=train.drop(['Cabin', 'PassengerId', 'Name', 'Id', 'Transported'], axis=1)


# In[ ]:


dictionary= {'A':6 , 'B':5, 'C':5 , 'D':4 , 'E':3 , 'F':2 , 'G':1, 'T':0 }


# In[ ]:


train=train.replace(({"Deck": dictionary}))


# In[ ]:


train=train[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_expenses','Deck', 'CryoSleep', 'VIP', 'Side', 'HomePlanet', 'Destination' ]]


# <p class='text-primary'> Re-arranging the columns helps to carry out standard scaling </p>

# In[ ]:


train['CryoSleep']=train['CryoSleep'].astype('float32')


# In[ ]:


train['Side']=train['Side'].apply(lambda x: 1 if x=='S' else 0)


# In[ ]:


train=pd.get_dummies(train, drop_first=True)


# <div>
#     
# <h2>5.1 Normality check <a id='phase5.1'></a></h2>
#     
# </div>

# In[ ]:


train.skew()


# In[ ]:


skewed_columns=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_expenses']
from scipy.special import boxcox1p
lam=0.2
for i in skewed_columns:
    train[i]= boxcox1p(train[i],lam)


# In[ ]:


train.skew()


# In[ ]:


X=train.values
Y=Y.values


# <div>
#     
# <h2>5.2 Splitting the dataset to train and test datasets <a id='phase5.2'></a></h2>
#     
# </div>

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=1)


# <div>
#     
# <h2>5.3 Feature Scaling <a id='phase5.3'></a></h2>
#     
# </div>

# In[ ]:


from sklearn.preprocessing import RobustScaler
sc= RobustScaler()
X_train[:,0:7]= sc.fit_transform(X_train[:, 0:7])
X_test[:,0:7]= sc.transform(X_test[:, 0:7])


# <div>
#     
# <h1>6. Machine Learning Models <a id='phase6'></a></h1>
#     
# </div>

# <div>
#     
# <h2>6.1 Base ML Models <a id='phase6.1'></a></h2>
#     
# </div>

# In[ ]:


model_pipeline=[]
model_pipeline.append(DecisionTreeClassifier(random_state=41))
model_pipeline.append(LogisticRegression(solver='saga', penalty='l1',random_state=42))
model_pipeline.append(KNeighborsClassifier())
model_pipeline.append(RandomForestClassifier(random_state=44))
model_pipeline.append(SVC(random_state=45, probability=True))
model_pipeline.append(xgb.XGBClassifier(random_state=46))
model_pipeline.append(lgbm.LGBMClassifier(random_state=47))


# In[ ]:


model_list=['Decision Tree', 'Logistic Regression', 'K-Nearest Neighbors', 'Random_Forest_Classification', 'SVM', 'XG', 'LGBM']
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn import metrics
i=0
acc=[]
cm=[]

plt.figure(figsize=(16,10))

for classifier in model_pipeline:
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    acc.append(round(accuracy_score(Y_test, Y_pred),2))
    cm.append(confusion_matrix(Y_test, Y_pred))
    Pred_prob=classifier.predict_proba(X_test)
    fpr, tpr, thresh = metrics.roc_curve(Y_test, Pred_prob[:,1])
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve {}'.format(model_list[i]))
    i+=1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random_Guess')
plt.title('ROC curve for Several ML models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend()
plt.show()


# In[ ]:


# Let us plot confusion matrix for all the model and compare.

fig=plt.figure(figsize=(20,10))

for i in range(0,len(cm)):
    cm_con=cm[i]
    model=model_list[i]
    sub_fig_title=fig.add_subplot(2,4,i+1).set_title(model)
    plot_map=sns.heatmap(cm_con,annot=True,cmap='Greens_r',fmt='g')
    plot_map.set_xlabel('Predicted_Values')
    plot_map.set_ylabel('Actual_Values')


# In[ ]:


result=pd.DataFrame({'Model': model_list, 'Accuracy': acc})
result


# <div>
#     
# <h2>6.2 Hypertuning parameters <a id='phase6.2'></a></h2>
#     
# </div>

# <div>
#     
# <h4>6.2.1 Logistic Regression <a id='phase6.2.1'></a></h4>
#     
# </div>

# from sklearn.model_selection import GridSearchCV
# parameters = [{'solver': ['sag','saga','lbfgs', 'newton-cg', 'liblinear'],
#               'penalty': ['l1', 'l2', 'elasticnet' 'none'],
#                
#                 }]
# grid_search = GridSearchCV(estimator = LogisticRegression(),
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search.fit(X_train, Y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LogisticRegression(random_state = 0, solver='sag', penalty='l2'), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# <div>
#     
# <h4>6.2.2 Decision Tree Classifier <a id='phase6.2.2'></a></h4>
#     
# </div>

# from sklearn.model_selection import GridSearchCV
# parameters = [{'criterion': ['gini', 'entropy'],
#                'splitter': ['best', 'random'],
#                'max_depth': [*range(1,10,1)],
#                'min_samples_split': [2,5,6,7,8,9,10],
#                'max_features': [1,2,3,4,5,7,9,11,13,15,'auto', 'sqrt', 'log2']
#                
#                 }]
# grid_search = GridSearchCV(estimator = DecisionTreeClassifier(random_state=1),
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search.fit(X_train, Y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = DecisionTreeClassifier(criterion='entropy', max_depth=9, max_features=13, min_samples_split=10, splitter='best'), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# <div>
#     
# <h4>6.2.3 XG-Boost Classifier <a id='phase6.2.3'></a></h4>
#     
# </div>

# from sklearn.model_selection import GridSearchCV
# parameters = [{'learning_rate': [0.01,0.02,0.03,0.04,0.05, 0.06,0.07],
#                'gamma': [0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8]
#               }]
# grid_search = GridSearchCV(estimator = xgb.XGBClassifier(random_state=10, booster='gbtree'),
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search.fit(X_train, Y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb.XGBClassifier(random_state=10, booster='gbtree', gamma=0.2, learning_rate=0.07), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# <div>
#     
# <h4>6.2.4 LGBM Classifier <a id='phase6.2.4'></a></h4>
#     
# </div>

# from sklearn.model_selection import GridSearchCV
# parameters = [{'learning_rate': [0.1,0.2,0.3,0.25,0.15],
#                'boosting_type': ['gbdt', 'rf', 'goss', 'dart'],
#                'max_depth': [*range(1,10,1)]
#               }]
# grid_search = GridSearchCV(estimator = lgbm.LGBMClassifier(random_state=10),
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search.fit(X_train, Y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lgbm.LGBMClassifier(random_state=10, boosting_type='gbdt', learning_rate=0.1, max_depth=6), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# <div>
#     
# <h4>6.2.5 SVM Classifier <a id='phase6.2.5'></a></h4>
#     
# </div>

# from sklearn.model_selection import GridSearchCV
# parameters = [{'gamma': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7],
#                'C': [1,2,3,4,5,6,7]
#               }]
# grid_search = GridSearchCV(estimator = SVC(random_state=4, kernel='rbf'),
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search.fit(X_train, Y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = SVC(kernel='rbf', gamma=0.1, C=3, random_state=4), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# <div>
#     
# <h2>6.3 Hypertuned ML Model Results <a id='phase6.2.5'></a></h2>
#     
# </div>

# In[ ]:


model_pipeline=[]
model_pipeline.append(DecisionTreeClassifier(criterion='entropy', max_depth=9, max_features=7, min_samples_split=10, splitter='best'))
model_pipeline.append(LogisticRegression(solver='saga', penalty='l1',random_state=42))
model_pipeline.append(SVC(kernel='rbf', gamma=0.1, C=3, random_state=4,probability=True))
model_pipeline.append(xgb.XGBClassifier(random_state=10, booster='gbtree', gamma=0.2, learning_rate=0.07))
model_pipeline.append(lgbm.LGBMClassifier(random_state=10, boosting_type='dart', learning_rate=0.25, max_depth=8))


# In[ ]:


model_list=['Decision Tree', 'Logistic Regression', 'SVM', 'XG', 'LGBM']
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn import metrics
i=0
acc=[]
cm=[]

plt.figure(figsize=(16,10))

for classifier in model_pipeline:
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    acc.append(round(accuracy_score(Y_test, Y_pred),2))
    cm.append(confusion_matrix(Y_test, Y_pred))
    Pred_prob=classifier.predict_proba(X_test)
    fpr, tpr, thresh = metrics.roc_curve(Y_test, Pred_prob[:,1])
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve {}'.format(model_list[i]))
    i+=1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random')
plt.title('ROC curve for Several ML models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend()
plt.show()


# <div>
#     
# <h1>7. Test Dataset (Under Construction) <a id='phase7'></a></h1>
#     
# </div>

# <p class='text-danger'>In this stage, it is expected to repeat same procedure as one did for the train.csv file for model development.</p> 

# In[ ]:





# In[ ]:





# <p class="text-secondary" style="text-align: center; font-size:100%"> 🚀🚀 If you like this kernel, do upvote 🚀 🚀</p>
