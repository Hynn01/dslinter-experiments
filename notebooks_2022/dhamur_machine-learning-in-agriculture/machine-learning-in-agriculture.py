#!/usr/bin/env python
# coding: utf-8

# <div style="display:fill;
#             border-radius:15px;
#             background-color:#00bd35;
#             font-size:210%;
#             font-family:cursive;
#             letter-spacing:0.5px;
#             padding:10px;
#             color:white;
#             border-style: solid;
#             border-color: black;
#             text-align:center;">
# <b>
# 🌱🍎Suitable crop for suitable soil 🌾🌿</b>
# </div>

# <h1><b>1 <span style='color:#00bd35;'>|</span> Importing Libraries and Loading dataset</b></h1>
# 
# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.1 | Names and uses</b></p>
# </div>
# 
# <ul>
#     <li style="font-size:15px"><h4 style="line-height:25px"><mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Pandas</b></mark> for handling the data. It contains n number of function for data handling.<a href="https://github.com/Dhamu785/py/tree/master/pandas"> Refer</a></h4></li>
#     <li style="font-size:15px"><h4 style="line-height:25px"><mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Numpy</b></mark> for creating and handling the array.</h4></li>
#     <li style="font-size:15px"><h4 style="line-height:25px">
#         <mark style="background-color:#00bd35;color:white;border-radius:4px;">
#             <b>Matplotlib</b>
#         </mark>
#         for visualizing the data. It also hulpfull in finding patterns in the dataset.<a href="https://www.kaggle.com/dhamur/matplotlib-different-charts"> Refer</a></h4></li>
#     <li style="font-size:15px"><h4 style="line-height:25px"><mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Seaborn</b></mark> is a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures. Seaborn helps you explore and understand your data.</h4></li>
#     <li style="font-size:15px"><h4 style="line-height:25px"><mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Scikit-learn (Sklearn)</b></mark> is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>machine learning and statistical modeling</b></mark> including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.</h4></li>
# 
# </ul>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.2 | Loading and reading the data</b></p>
# </div>
# <h4 style="line-height:30px">This is dataset which is used to recommend the crop for the suitable soil. This will be very useful in crop production (Agriculture) without looses based on soli ph, rainfall, humadity and other chemical components present in the soil. </h4>

# In[ ]:


df = pd.read_csv('../input/smart-agricultural-production-optimizing-engine/Crop_recommendation.csv')


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.3 | About the data</b></p>
# </div>
# <ul>
#     <li><h4 style="line-height:25px"><mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Nitrogen</b></mark> is so vital because it is a <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>major component of chlorophyll</b></mark>, the compound by which plants use sunlight energy to produce sugars from water and carbon dioxide (i.e., photosynthesis). It is also a major component of amino acids, the building blocks of proteins. Without proteins, plants wither and die.</h4></li>
#     <hr>
#     <li><h4 style="line-height:25px;"><mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Phosphorus</b></mark> is, therefore, important in <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>cell division and development of new tissue.</b></mark> Phosphorus is also associated with complex energy transformations in the plant. Adding phosphorus to soil low in available phosphorus promotes root growth and winter hardiness, stimulates tillering, and often hastens maturity.</h4></li>
#     <hr>
#     <li><h4 style="line-height:25px"><mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Potassium</b></mark> is a critical nutrient that plants absorb from the soil, and from fertilizer. It <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>increases disease resistance</b></mark>, helps stalks to grow upright and sturdy, improves drought tolerance, and helps plants get through the winter.</h4></li>
#     <hr>
#     <li><h4 style="line-height:25px">The average <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>soil temperatures</b></mark> for bioactivity range from 50 to 75F. These values are favorable for normal life functions of earth biota that ensure proper organic matter decomposition, <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>increased nitrogen mineralization</b></mark>, uptake of soluble substances, and metabolism.</h4></li>
#     <hr>
#     <li><h4 style="line-height:25px">The pH range <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>5.5–6.5</b></mark> is optimal for plant growth as the availability of nutrients is optimal.</h4></li>
#     <hr>
#     <li><h4 style="line-height:25px">Besides disease, <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>rainfall</b></mark> can also determine how fast a crop will grow from seed, including when it will be ready for harvesting. A good balance of rain and proper irrigation can lead to <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>faster-growing plants</b></mark>, which can cut down on germination time and the length between seeding and harvest.</h4></li>
#     <hr>
# </ul>

# In[ ]:


df.head()


# In[ ]:


print("Shape of the dataframe: ",df.shape)
df.isna().sum()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# 
# <h1><b>2 <span style='color:#00bd35;'>|</span> 🎯📊EDA-Exploratory data analysis📈📉</b></h1>
# 
# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1 | Data distribution</b></p>
# </div>
# 

# In[ ]:


sns.displot(x=df['N'], bins=20,kde=True,edgecolor="black",color='black',facecolor='#ffb03b')
plt.title("Nitrogen",size=20)
plt.show()


# In[ ]:


sns.displot(x=df['P'],bins=20,color='black',edgecolor='black',kde=True,facecolor='#ffb03b')
plt.title("Phosphorus", size=20)
plt.xticks(range(0,150,20))
plt.show()


# In[ ]:


sns.displot(x=df['K'],kde=True, bins=20, facecolor='#ffb03b',edgecolor='black', color='black')
plt.title("Potassium",size=20)
plt.show()


# In[ ]:


sns.displot(x=df['temperature'], bins=20,kde=True,edgecolor="black",color='black',facecolor='#ffb03b')
plt.title("Temperature",size=20)
plt.show()


# In[ ]:


sns.displot(x=df['humidity'], color='black',facecolor='#ffb03b',kde=True,edgecolor='black')
plt.title("Humidity",size=20)
plt.show()


# In[ ]:


sns.displot(x=df['rainfall'], color='black',facecolor='#ffb03b',kde=True,edgecolor='black')
plt.title("Rainfall",size=20)
plt.show()


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2 | Categorical plot </b></p>
# </div>
# 

# In[ ]:


sns.relplot(x='rainfall',y='temperature',data=df,kind='scatter',hue='label',height=5)
plt.show()


# In[ ]:


sns.pairplot(data=df,hue='label')
plt.show()


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.3 | Outerlier detection using graphs</b></p>
# </div>
# 

# In[ ]:


# Unique values in the label column

crops = df['label'].unique()
print(len(crops))
print(crops)
print(pd.value_counts(df['label']))


# In[ ]:


# Filtering each unique label and store it in a list df2 for to plot the box plot

df2=[]
for i in crops:
    df2.append(df[df['label'] == i])
df2[1].head()


# In[ ]:


sns.catplot(data=df, x='label', y='temperature', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Temperature", size=20)
plt.show()


# In[ ]:


sns.catplot(data=df, x='label', y='humidity', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Humidity", size=20)
plt.show()


# In[ ]:


sns.catplot(data=df, x='label', y='temperature', kind='box', height=10, aspect=20/8.27)
plt.show()


# In[ ]:


sns.catplot(data=df, x='label', y='N', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


sns.catplot(data=df, x='label', y='ph', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Nitrogen",size=20)
plt.show()


# In[ ]:


sns.catplot(data=df, x='label', y='P', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Phosphorus",size=20)
plt.show()


# In[ ]:


sns.catplot(data=df, x='label', y='K', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Potassium",size=20)
plt.show()


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:white;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;
#               color:black;
#               text-align:center;"><b>These graphs shows that thers is no outliers present in this dataset</b></p>
# </div>
# 
# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.4 | Lets check through Mathematics (Statistics)</b></p>
# </div>
# 

# In[ ]:


def detect_outlier(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (1.5*IQR)
    upper_limit = q3 + (1.5*IQR)
    print(f"Lower limit: {lower_limit} Upper limit: {upper_limit}")
    print(f"Minimum value: {x.min()}   MAximum Value: {x.max()}")
    for i in [x.min(),x.max()]:
        if i == x.min():
            if lower_limit > x.min():
                print("Lower limit failed - Need to remove minimum value")
            elif lower_limit < x.min():
                print("Lower limit passed - No need to remove outlier")
        elif i == x.max():
            if upper_limit > x.max():
                print("Upper limit passed - No need to remove outlier")
            elif upper_limit < x.max():
                print("Upper limit failed - Need to remove maximum value")
detect_outlier(df['K'][df['label']=='grapes'])


# In[ ]:


for i in df['label'].unique():
    detect_outlier(df['K'][df['label']==i])
    print('---------------------------------------------')


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:white;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;
#               color:black;
#               text-align:center;"><b>These graphs shows that thers is no outliers present in this dataset and it is confirmed with the help of Statistics(IQR)</b></p>
# </div>
# 
# <h1><b>3 <span style='color:#00bd35;'>|</span> 🎯📊Prediction🍅🥭🍎🍐</b></h1>
# 
# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.1 | Spliting the train and test data</b></p>
# </div>
# 

# In[ ]:


x = df.drop(['label'], axis=1)
x.head()


# In[ ]:


Y = df['label']
encode = preprocessing.LabelEncoder()
y = encode.fit_transform(Y)
print("Label length: ",len(y))


# In[ ]:


x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y)
print(len(x_train),len(y_train),len(x_test),len(y_test))


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.2 | Best model choosing</b></p>
# </div>
# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#636363;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><i>3.2.1 <b>|</b> Decision Tree, Support vector mechanism, Random forest</i></p>
# </div>

# In[ ]:


a={'decision tree' : {
        'model' : DecisionTreeClassifier(criterion='gini'),
        'params':{'decisiontreeclassifier__splitter':['best','random']}
    },
    'svm': {
        'model': SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
   'k classifier':{
       'model':KNeighborsClassifier(),
       'params':{'kneighborsclassifier__n_neighbors':[5,10,20,25],'kneighborsclassifier__weights':['uniform','distance']}
   }
}


# In[ ]:


score=[]
details = []
best_param = {}
for mdl,par in a.items():
    pipe = make_pipeline(preprocessing.StandardScaler(),par['model'])
    res = model_selection.GridSearchCV(pipe,par['params'],cv=5)
    res.fit(x_train,y_train)
    score.append({
        'Model name':mdl,
        'Best score':res.best_score_,
        'Best param':res.best_params_
    })
    details.append(pd.DataFrame(res.cv_results_))
    best_param[mdl]=res.best_estimator_
pd.DataFrame(score)


# In[ ]:


details[0]


# In[ ]:


details[1]


# In[ ]:


details[2]


# In[ ]:


details[3]


# In[ ]:


score


# In[ ]:


pd.DataFrame(score)


# In[ ]:


for i in best_param.keys():
    print(f'{i} : {best_param[i].score(x_test,y_test)}')


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#636363;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><i>3.2.2 <b>|</b> Best model - Random forest</i></p>
# </div>

# In[ ]:


predicted = best_param['random_forest'].predict(x_test)
predicted


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(confusion_matrix(y_test,predicted),annot=True)
plt.xlabel("Original")
plt.ylabel("Predicted")
plt.show()


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#636363;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><i>3.2.3 <b>|</b> Bagging classifier for more accuracy</i></p>
# </div>

# In[ ]:


pipe1 = make_pipeline(preprocessing.StandardScaler(),RandomForestClassifier(n_estimators = 10))
bag_model = BaggingClassifier(base_estimator=pipe1,n_estimators=100,
                              oob_score=True,random_state=0,max_samples=0.8)


# In[ ]:


bag_model.fit(x_train,y_train)


# In[ ]:


bag_model.score(x_test,y_test)


# In[ ]:


predict = bag_model.predict(x_test)


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(confusion_matrix(y_test,predict),annot=True)
plt.show()


# <h1><b>4 <span style='color:#00bd35;'>|</span> 🎊🎊🎊 Conclusion 🎊🎊🎊</b></h1>
# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.1 | Value mapping</b></p>
# </div>
# <h4 style= "line-height:20px;"> <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Value mapping</b></mark> shows that which value is belongs to which crop. It helps in easy reading the predicted value. Eg: If predicted value id <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>20</b></mark> then its belongs to <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Crop rice.</b></mark> So on...</h4>

# In[ ]:


dha2 =pd.DataFrame(Y)
code = pd.DataFrame(dha2['label'].unique())


# In[ ]:


dha = pd.DataFrame(y)
encode = pd.DataFrame(dha[0].unique())
refer = pd.DataFrame()
refer['code']=code
refer['encode']=encode
refer


# <div style="color:white;
#             display:fill;
#             border-radius:8px;
#             background-color:#2b2b2b;
#             font-size:120%;
#             font-family:fantasy;
#             letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.2 | Report</b></p>
# </div>
# <h4 style="line-height:25px;"> Score for each crops. Want to know more more about <mark style="background-color:#00bd35;color:white;border-radius:4px;"><b>Precision and recall</b></mark> <br>- <a href="https://en.wikipedia.org/wiki/Precision_and_recall">Wikipedia</a><br> -
#     <a href="https://www.youtube.com/watch?v=2osIZ-dSPGE&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=19">Youtube</a></h4>

# In[ ]:


print(classification_report(y_test,predict))


# <div style="display:fill;
#             border-radius:15px;
#             background-color:#00bd35;
#             font-size:210%;
#             font-family:cursive;
#             letter-spacing:0.5px;
#             padding:10px;
#             color:white;
#             border-style: solid;
#             border-color: black;
#             text-align:center;">
#     <b>The End <br><p style = "text-align:center;font-size:20px; color:white"><i>Thank you for visiting</i></p></b>
# </div>
