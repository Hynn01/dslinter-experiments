#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from sklearn import preprocessing
import itertools
from sklearn.metrics import precision_recall_curve,average_precision_score, accuracy_score,roc_curve,roc_auc_score,confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# ## Importing dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/personal-key-indicators-of-heart-disease/heart_2020_cleaned.csv')


# ## Exploratory Analysis

# In[ ]:


df.head(10)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include='object')


# In[ ]:


df.corr()


# In[ ]:


l=df.shape[0] ## no of observations


# ### Target (HeartDisease)

# In[ ]:


df['HeartDisease'].value_counts()/l*100  #in percentage


# In[ ]:


fig,ax = plt.subplots(figsize=(10,6))
labels = df['HeartDisease'].unique()

ax.pie(df['HeartDisease'].value_counts(), labels=labels,autopct='%1.2f%%',textprops = {"fontsize":14})
plt.show()


# In[ ]:


## 8.5% are at a risk of heart disease out of the total(319795). So we have to keep this in mind that it is a imbalanced dataset


# ### Features

# In[ ]:


dummy_df=df.copy()


# In[ ]:


dummy_df.loc[dummy_df['HeartDisease'] =='Yes','HeartDisease']=1
dummy_df.loc[dummy_df['HeartDisease'] == 'No','HeartDisease']=0


# #### BMI

# In[ ]:


df['BMI'].describe()


# In[ ]:


sns.boxplot(data=df, x='BMI')


# In[ ]:


## it has a lot of spread in the data, let us make it categorical based on the health industry specifications


# In[ ]:


t=df['BMI']
cond = [(t.between(0,18.5)),(t.between(18.5,24.9)),(t.between(24.9,29.9)),(t.between(29.9,34.9)), (t.between(34.9,100))]

labels = ['Uwt','N','Owt','O','EO']    # Uwt-underweight, N-normal, Owt-overweight, O-Obese, EO-extremely obese 

df['BMI_cat'] = np.select(cond,labels)


# In[ ]:


df.drop(columns=['BMI'],inplace=True)


# In[ ]:


df['BMI_cat'].value_counts()


# In[ ]:


## we have more people that are overweight which matches with the avg BMI of the US, where the data is from.


# In[ ]:


dummy_df['BMI_cat']=df['BMI_cat']
t1=dummy_df[["HeartDisease",'BMI_cat']].groupby(['BMI_cat'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t1.sort_values(by='HeartDisease'),x='BMI_cat',y='HeartDisease')


# In[ ]:


## here we see that Extremely obese people have higher risk of heart disease, also underweight people also have it higher than
## normal ones, which some recent studies also show


# #### PhysicalHealth

# In[ ]:


df['PhysicalHealth'].describe()


# In[ ]:


sns.boxplot(data=df, x='PhysicalHealth')


# In[ ]:


fig, ax = plt.subplots(figsize = (15,8))
sns.kdeplot(data=df, x='PhysicalHealth', hue='HeartDisease', fill=True, palette="crest", alpha=.5, linewidth=0)


# In[ ]:


## here the plots [P(X|Y=No)P(Y=No) and P(X|Y=Yes)P(Y=Yes), where X is the PhysicalHealth] take into account the prior prob of
## each class(P(Y=No) and P(Y=Yes) respectively),thus due to the imbalance, the prob plot for the majority class would always 
## be higher than the minority.
## { 
##   Due to this reason only, a classifier predicting based on the baysian prob [P(Y=Yes|X) = P(X|Y=Yes)P(Y=Yes)/P(X) and 
##   P(Y=No|X) = P(X|Y=No)P(Y=No)/P(X)] would choose the majority class only and get high accuracy 
##    [since P(Y=No|X) > P(Y=Yes|X), i.e., 'No' class is always more likely than the 'Yes' class for a given X]
## }


# In[ ]:


## Thus let us look the the probability independently for the classes (P(X|Y=No) and P(X|Y=Yes)), i.e., not taking into 
## account the prior prob, to better see the effect of X on the classes


# In[ ]:


fig, ax = plt.subplots(figsize = (15,8))
sns.kdeplot(data=df, x='PhysicalHealth', hue='HeartDisease', fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)


# In[ ]:


## From this plot we can see that the prob for people with heart disease being healthy for the pass 30 days (X=0) is lower than
## the people with no heart disease, whereas it increases above that and is much more near the 30 mark (ill past 30 days)


# In[ ]:


sns.barplot(y='PhysicalHealth',data=df,x='HeartDisease')


# In[ ]:


## the mean sick days(in 30 days) are significantly higher for people with heart disease


# #### MentalHealth

# In[ ]:


df['MentalHealth'].describe()


# In[ ]:


sns.boxplot(data=df, x='MentalHealth')


# In[ ]:


fig, ax = plt.subplots(figsize = (15,8))
sns.kdeplot(data=df, x='MentalHealth', hue='HeartDisease', fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)


# In[ ]:


## even for more days of bad mental health the prob is around the same or even lower for heart disease people than the ones
## with no heart problem, a reason for that could be that many people nowadays face bad mental health problems, but all wont
## have the heart problem.


# In[ ]:


sns.barplot(y='MentalHealth',data=df,x='HeartDisease')


# In[ ]:


## The difference in mean days for bad mental health is also around the same for both the cases, which could mean that mental 
## health might not be a big factor for the heart disease, and thus we can drop it for the model


# In[ ]:


df.drop(columns='MentalHealth', inplace=True)


# #### SleepTime

# In[ ]:


df['SleepTime'].describe()


# In[ ]:


## the max is 24! The central value is around 7 hours


# In[ ]:


df['SleepTime'].value_counts()


# In[ ]:


sns.boxplot(data=df, x='SleepTime')


# In[ ]:


fig, ax = plt.subplots(figsize = (15,8))
sns.kdeplot(data=df, x='SleepTime', hue='HeartDisease', fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)


# In[ ]:


t2=dummy_df[["HeartDisease",'SleepTime']].groupby(['SleepTime'],as_index=False).mean()


# In[ ]:


fig, ax = plt.subplots(figsize = (15,8))
sns.barplot(data=t2.sort_values(by='HeartDisease'),x='SleepTime',y='HeartDisease')


# In[ ]:


## we can see from the above plots that the people with less sleep time or more than the avg sleep time (7hrs) have higher
## risk of heart disease


# In[ ]:


## Considering the spread and the outliers & also its trend on the target, it would be best to bin this feature


# In[ ]:


st=df['SleepTime']
cond = [(st.between(0,6)),(st.between(6,9)),(st.between(9,24))]

labels = ['Low','Normal','High']

df['SleepTime_cat'] = np.select(cond,labels)


# In[ ]:


df.drop(columns=['SleepTime'],inplace=True)


# In[ ]:


df['SleepTime_cat'].value_counts()


# In[ ]:


dummy_df['SleepTime_cat']=df['SleepTime_cat']
t3=dummy_df[["HeartDisease",'SleepTime_cat']].groupby(['SleepTime_cat'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t3.sort_values(by='HeartDisease'),x='SleepTime_cat',y='HeartDisease')


# #### Smoking

# In[ ]:


df['Smoking'].value_counts()


# In[ ]:


t4=dummy_df[["HeartDisease",'Smoking']].groupby(['Smoking'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t4.sort_values(by='HeartDisease'),x='Smoking',y='HeartDisease')


# In[ ]:


## Smoking leads to greater heart risk. Another reason to quit it!


# #### AlcoholDrinking

# In[ ]:


df['AlcoholDrinking'].value_counts()/l*100


# In[ ]:


t5=dummy_df[["HeartDisease",'AlcoholDrinking']].groupby(['AlcoholDrinking'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t5.sort_values(by='HeartDisease'),x='AlcoholDrinking',y='HeartDisease')


# In[ ]:


## Since there are very few heavy drinkers (6.8%), so people with high heart problem risk dont look to be influenced by heavy drinking


# In[ ]:


df[df['AlcoholDrinking']=='Yes']['HeartDisease'].value_counts()


# In[ ]:


df['HeartDisease'].value_counts()


# In[ ]:


## Only 4% (1141/27373*100) of people with heart disease are heavy drinkers, so this might not be that good indicator for heart disease, and
## it's better to drop it


# In[ ]:


df.drop(columns='AlcoholDrinking', inplace=True)


# #### Stroke

# In[ ]:


df['Stroke'].value_counts()/l*100


# In[ ]:


t6=dummy_df[["HeartDisease",'Stroke']].groupby(['Stroke'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t6.sort_values(by='HeartDisease'),x='Stroke',y='HeartDisease')


# In[ ]:


## So a past stroke could be a big indicator for heart disease


# #### DiffWalking

# In[ ]:


df['DiffWalking'].value_counts()/l*100


# In[ ]:


t7=dummy_df[["HeartDisease",'DiffWalking']].groupby(['DiffWalking'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t7.sort_values(by='HeartDisease'),x='DiffWalking',y='HeartDisease')


# In[ ]:


## Difficulty in walking is also a good indicator for heart problem


# #### Sex

# In[ ]:


df['Sex'].value_counts()/l*100


# In[ ]:


t8=dummy_df[["HeartDisease",'Sex']].groupby(['Sex'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t8.sort_values(by='HeartDisease'),x='Sex',y='HeartDisease')


# #### AgeCategory	

# In[ ]:


df['AgeCategory'].value_counts()/l*100


# In[ ]:


t9=dummy_df[["HeartDisease",'AgeCategory']].groupby(['AgeCategory'],as_index=False).mean()


# In[ ]:


fig, ax = plt.subplots(figsize = (15,8))
sns.barplot(data=t9.sort_values(by='HeartDisease'),x='AgeCategory',y='HeartDisease')


# In[ ]:


## we could see here that the risk of heart disease increases with age


# In[ ]:


## Let us replace the category of Age with the mean value in that range (or we can do encoding aswell)


# In[ ]:


AgeCategory_mean = {'18-24':21,'25-29':27,'30-34':32,'35-39':37,'40-44':42,'45-49':47,'50-54':52,'55-59':57, 
                    '60-64':62,'65-69':67,'70-74':72,'75-79':77,'80 or older':80}

df['Mean_Age'] = df['AgeCategory'].apply(lambda x: AgeCategory_mean[x])


# In[ ]:


df.drop(columns=['AgeCategory'],inplace=True)


# #### Race	

# In[ ]:


df['Race'].value_counts()/l*100


# In[ ]:


t10=dummy_df[["HeartDisease",'Race']].groupby(['Race'],as_index=False).mean()


# In[ ]:


fig, ax = plt.subplots(figsize = (15,8))
sns.barplot(data=t10.sort_values(by='HeartDisease'),x='Race',y='HeartDisease')


# In[ ]:


## The Native Americans look to be at higher risk of the heart disease, which according to recent reports by CDC also is shown
## that heart disease is a leading cause of death amongst them and they are more likely to develop a heart issue than the white
## people. The asians seem to be having the lowest risk for a heart disease.


# In[ ]:





# #### Diabetic	

# In[ ]:


df['Diabetic'].value_counts()/l*100


# In[ ]:


t11=dummy_df[["HeartDisease",'Diabetic']].groupby(['Diabetic'],as_index=False).mean()


# In[ ]:


fig, ax = plt.subplots(figsize = (15,8))
sns.barplot(data=t11.sort_values(by='HeartDisease'),x='Diabetic',y='HeartDisease')


# In[ ]:


## People with Diabetes are at higher risk of a heart disease


# #### KidneyDisease	

# In[ ]:


df['KidneyDisease'].value_counts()/l*100


# In[ ]:


t12=dummy_df[["HeartDisease",'KidneyDisease']].groupby(['KidneyDisease'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t12.sort_values(by='HeartDisease'),x='KidneyDisease',y='HeartDisease')


# #### SkinCancer	

# In[ ]:


df['SkinCancer'].value_counts()/l*100


# In[ ]:


t13=dummy_df[["HeartDisease",'SkinCancer']].groupby(['SkinCancer'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t13.sort_values(by='HeartDisease'),x='SkinCancer',y='HeartDisease')


# #### Asthma	

# In[ ]:


df['Asthma'].value_counts()/l*100


# In[ ]:


t14=dummy_df[["HeartDisease",'Asthma']].groupby(['Asthma'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t14.sort_values(by='HeartDisease'),x='Asthma',y='HeartDisease')


# In[ ]:


## So asthama is also a good indicator but not as strong as skin cancer and kidney disease


# #### PhysicalActivity	

# In[ ]:


df['PhysicalActivity'].value_counts()/l*100


# In[ ]:


t15=dummy_df[["HeartDisease",'PhysicalActivity']].groupby(['PhysicalActivity'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t15.sort_values(by='HeartDisease'),x='PhysicalActivity',y='HeartDisease')


# In[ ]:


## Doing some physical activity can reduce the risk for heart disease


# #### GenHealth	

# In[ ]:


df['GenHealth'].value_counts()/l*100


# In[ ]:


t16=dummy_df[["HeartDisease",'GenHealth']].groupby(['GenHealth'],as_index=False).mean()


# In[ ]:


sns.barplot(data=t16.sort_values(by='HeartDisease'),x='GenHealth',y='HeartDisease')


# In[ ]:


## The general health is affected with heart disease, which comes as so surprise. Although we can see that even with the general
## health in the good region, the heart disease cases are not zero, i.e., your general health being good doesnt exempt you from
## the risk of having a heart issue.


# In[ ]:





# ## Model building

# ### Encoding

# In[ ]:


df


# In[ ]:


encoded_df = df.copy()

##Label Encoding
le = preprocessing.LabelEncoder()
cols=['HeartDisease', 'Smoking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']

for i in cols:
    encoded_df[i] = le.fit_transform(encoded_df[i])
    
le_1=le.fit(encoded_df['GenHealth'])
le_1.classes_ = np.array(['Poor', 'Fair','Good','Very good','Excellent'])   ## to assign 0 to Poor and 4 to Excellent
encoded_df['GenHealth'] = le_1.transform(encoded_df['GenHealth'])

## One hot encoding
encoded_df=pd.concat([encoded_df,pd.get_dummies(encoded_df['Race'],prefix='Race',drop_first=True)], axis=1)
encoded_df=pd.concat([encoded_df,pd.get_dummies(encoded_df['Diabetic'],prefix='Diabetic',drop_first=True)], axis=1)
encoded_df=pd.concat([encoded_df,pd.get_dummies(encoded_df['BMI_cat'],prefix='BMI',drop_first=True)], axis=1)
encoded_df=pd.concat([encoded_df,pd.get_dummies(encoded_df['SleepTime_cat'],prefix='SleepTime',drop_first=True)], axis=1)

encoded_df.drop(columns=['Race','Diabetic','BMI_cat','SleepTime_cat'],axis=1,inplace=True)


# In[ ]:


from IPython.display import display
with pd.option_context('display.max_columns', 30):
    display(encoded_df)


# ### Standardization

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


X=encoded_df.drop(columns=['HeartDisease'],axis=1)
y=encoded_df['HeartDisease']

col=X.columns


# In[ ]:


encod_stand_df = sc.fit_transform(X)
es_df = pd.DataFrame(encod_stand_df, columns = col)


# ### Train test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(es_df,y,test_size =0.2,random_state = 42)


# ### Creating evaluation metrics

# In[ ]:


def plot_confusion_matrix(model,X_test,y_test,normalize=False):
    """
        Computes and plots the confusion matrix.         
        
        
        Parameters
        ----------
        
        model : ML model
            A trained model for which ROC is to be computed
        X_test: pandas.DataFrame
            Testing data for independent variables(features)
        y_test: pandas.Series
            Testing data for target variables
        normalize : boolean, default=False
            To normalize the matrix
        
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    classes=["No", "Yes"]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


def plot_roc(model,X_test,y_test,plot_threshold=False,t=0.5):
    """
        Computes and plots the ROC curve and AUC. It also plots a given threshold value        
        
        
        Parameters
        ----------
        
        model : ML model
            A trained model for which ROC is to be computed
        X_test: pandas.DataFrame
            Testing data for independent variables(features)
        y_test: pandas.Series
            Testing data for target variables
        plot_threshold : boolean, default=False
            Whether to plot the given threshold or not.
        t : float, default=0.5
            Threshold value to plot on the graph
    """
    
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, thresh = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    

    # margins for text threshold
    y_text_margin=0.05
    x_text_margin=0.2
    
    
    x_thresh, y_thresh = fpr[np.argmin(abs(thresh - t))], tpr[np.argmin(abs(thresh - t))]
    
    ## np.argmin(abs(thresh - t)) will give you the index of the min element in the array (thresh-t)
    ## since 'thresh' array wont have the exact threshold 't' in it, so we need the
    ## one which is the closest (if t is in thresh then (thresh - t) would be zero and the min in abs(thresh - t) )

    # Roc curve
    plt.plot(fpr, tpr,lw=2,label='ROC curve (AUC = %0.2f)' % auc,linestyle='-')
    
    # Reference line
    plt.plot([0, 1], [0, 1],label="Random guess",color='red',lw=2, linestyle='--')
    
    # Plotting threshold
    if plot_threshold:
    
        plt.axhline(y=y_thresh,lw=2,color='black',linestyle=':')
        plt.axvline(x=x_thresh,lw=2,color='black',linestyle=':')
        
        # text for threshold
        if x_thresh > 0.5 and y_thresh > 0.5:
            plt.text(x=x_thresh - x_text_margin, y=y_thresh - y_text_margin,s='Threshold : {:.2f}'.format(t))
        elif x_thresh <= 0.5 and y_thresh <= 0.5:
            plt.text(x=x_thresh + x_text_margin, y=y_thresh + y_text_margin,s='Threshold : {:.2f}'.format(t))
        elif x_thresh <= 0.5 < y_thresh:
            plt.text(x=x_thresh + x_text_margin, y=y_thresh - y_text_margin,s='Threshold : {:.2f}'.format(t))
        elif x_thresh > 0.5 >= y_thresh:
            plt.text(x=x_thresh - x_text_margin, y=y_thresh + y_text_margin,s='Threshold : {:.2f}'.format(t))
        
        plt.plot(x_thresh, y_thresh, 'ro')  # redpoint of threshold on the ROC curve
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)


# In[ ]:


def plot_prc(model,X_test,y_test):
    """
        Computes and plots the precision-recall curve and AUC.        
        
        
        Parameters
        ----------
        
        model : ML model
            A trained model for which ROC is to be computed
        X_test: pandas.DataFrame
            Testing data for independent variables(features)
        y_test: pandas.Series
            Testing data for target variables
        
    """
    
    y_pred_proba = model.predict_proba(X_test)[::,1]
    prec, rec, thresh = precision_recall_curve(y_test,  y_pred_proba)
    auc_val = average_precision_score(y_test, y_pred_proba)

    
    rand = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [rand, rand],color='red', linestyle='--', label='Random')
    plt.plot(rec, prec, marker='.', label='PR curve (AUC = %0.2f)' % auc_val,linestyle='-',color='orange')
    

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc=0)
    


# ### Model

# #### Random Forest

# In[ ]:


model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train,y_train)


y_pred = model_rf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("\n\t\t\tClassification report\n",classification_report(y_test, y_pred))


# In[ ]:


## Since this is an imbalanced dataset, so we focus on better precision or recall than accuracy of the minority class. 
## Recall is much preferred in this case since its better to detect a heart patient at the cost of more False positives rather
## than high precision where we might ignore a lot of True cases (heart risk patients) to improve our correct predictions.

## (Its better to tell a false result to a no heart risk patient than telling a heart patient, as the former might cause 
## unnecessary doctor visit but the later might lead to the ignore of an underlying disease, which is obviously much worse.)


# In[ ]:





# In[ ]:


model_rf = RandomForestClassifier(random_state=42)
model_rf.set_params(n_estimators=500, class_weight="balanced",max_depth=30,min_samples_split= 30,min_samples_leaf=24)
model_rf.fit(X_train,y_train)

y_pred = model_rf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("\n\t\t\tClassification report\n",classification_report(y_test, y_pred))


# In[ ]:


## here we have good recall for the minority class


# In[ ]:


print('Cross validation score: ',cross_val_score(model_rf, X_train, y_train, cv=10).mean())


# In[ ]:


plot_confusion_matrix(model_rf,X_test,y_test,normalize=True)


# In[ ]:


plot_prc(model_rf,X_test,y_test)


# In[ ]:


plot_roc(model_rf,X_test,y_test)


# #### Logistic Regression

# In[ ]:


model_lr = LogisticRegression(random_state=0,class_weight='balanced', max_iter=800)

model_lr.fit(X_train, y_train)    

y_pred = model_lr.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("\n\t\t\tClassification report\n",classification_report(y_test, y_pred))


# In[ ]:


plot_confusion_matrix(model_lr,X_test,y_test,normalize=True)


# In[ ]:


plot_prc(model_lr,X_test,y_test)


# #### Naive Bayes

# In[ ]:


model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

y_pred=model_nb.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
print("\n\t\t\tClassification report\n",classification_report(y_test, y_pred))


# In[ ]:


plot_confusion_matrix(model_nb,X_test,y_test,normalize=True)


# In[ ]:


plot_prc(model_nb,X_test,y_test)


# ## Conclusion

# Random Forest is performing the best out of the other models on this dataset. It is giving us a good recall for the minority class and AUC for the PR curve is also the best amongst others.Based on this modelling I have made a web app using Flask and deployed it on [AWS](http://ec2-54-91-30-108.compute-1.amazonaws.com:8080/) (and [Heroku](https://risk-heart-disease.herokuapp.com/)) where you could answer some questions about yourself and have the model predict whether you have a risk of heart disease or not. Check out the project [here](https://github.com/prateek-py/Heart-disease-risk-Prediction)
