#!/usr/bin/env python
# coding: utf-8

# ### Module 1: Data Exploration
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
#------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv("../input/creditcard-test-train/creditcard_train.csv")

#Check DateFrame
print('\33[91m'+"DateFrame Preview")
display(df.head())
print("")

#Check General information of df
print('\33[91m'+"Information of DateFrame")
print('\33[0m'+"")
print("Dimension = ", df.ndim)
print("Size = ", df.size)
print("Shape = ", df.shape)
print("Positive Class: ", df[df['Class']>0].shape)
print("Negative Class: ", df[df["Class"]<1].shape)
orlen = len(df.index)

#Check data type
df_dt = pd.DataFrame(df.dtypes, columns = ['data type'])
display(df_dt)
display(df.describe())
print("")
#df.info() #may lead to error due to diff. numpy version 

#Check for NaN value
print('\33[91m'+"Missing data in DateFrame")
print('\33[0m'+"")
missing_data = pd.concat([df.isnull().sum(),(((df.isnull().sum()/len(df.index))*100))], axis=1, keys=['Total', 'Percent %'])
display(missing_data[missing_data["Total"]>0])
print("Number of variables contain Missing values = ", len(missing_data[missing_data["Total"]>0].index))
print("Number of Rows contain Missing values = ", missing_data["Total"].sum())
print("Percentage of Missing value = ", math.ceil(missing_data["Percent %"].sum()*1000)/1000,"%")
print("")

#Handle Missing value -> Remove rows with missing value 
#                        Because it still contains 99.7% of data
print('\33[91m'+"Handling Missing values -> Remove rows that contains missing value ")
print('\33[0m'+"")
df = df.dropna()
print("After handling the missing values, Dimension = ", df.ndim)
print("After handling the missing values, Size = ", df.size)
mis = df.size
print("After handling the missing values, shape = ", df.shape)
print(f"Data removed = {missing_data['Total'].sum()} during handling the missing values")
print(f"Data remaining = {orlen-missing_data['Total'].sum()} after handling the missing values")
print("Positive Class: ", df[df['Class']>0].shape)
print("Negative Class: ", df[df["Class"]<1].shape)

#Check for Duplicated value
print("")
print('\33[91m'+'Duplicated value in DataFrame')
print('\33[0m'+"")
df123 = df.copy()
df123.drop_duplicates(subset=None, inplace=True)
Duplicated = len(df)-len(df123)
print(f'Duplicated values found: {Duplicated} with Percentage: {round((Duplicated/len(df)*100),4)}% in total')
print("")
print('\33[91m'+"Handling Duplicated value -> Removal")
print('\33[0m'+"")
df = df.drop_duplicates()
print("After handling the duplicated value, Dimension = ", df.ndim)
print("After handling the duplicated value, Size = ", df.size)
print("After handling the duplicated value, shape = ", df.shape)
print(f"Data removed = {(mis-df.size)//31} during handling the duplicated value")
print(f"Data remaining = {df.size//len(df.columns)} after handling the duplicated value")
print("Positive Class: ", df[df['Class']>0].shape)
print("Negative Class: ", df[df["Class"]<1].shape)

print("")
print('\33[91m'+"Handling Outliers -> Keeping All Outliers")
print('\33[0m'+"")
print("Reason on keeping all outliers will be deliberate in the Summary part.")
#Q1 = df.quantile(0.25)
#Q3 = df.quantile(0.75)
#IQR = Q3 - Q1
#cols = df.columns.values[:-1]
#df = df[~((df[cols] < (Q1 - 2.5 * IQR)) |(df[cols] > (Q3 + 2.5 * IQR))).any(axis=1)]
#print("After handling the outliers, Dimension = ", df.ndim)
#print("After handling the outliers, Size = ", df.size)
#print("After handling the outliers, shape = ", df.shape)
#print(f"Data removed = {(mis-df.size)//31} during handling the outliers")
#print(f"Data remaining = {df.size//len(df.columns)} after handling the outliers")
#print("Positive Class: ", df[df['Class']>0].shape)
#print("Negative Class: ", df[df["Class"]<1].shape)


# ### Module 2: Data Visualization
# 

# In[ ]:


#1. Ratio of two Class
fig=plt.figure(figsize=(10,10))
print('\33[91m'+"bar_label may cause error")
print("If so, please update matplotlib to 3.4 or higher version")
ax = sns.countplot(x = 'Class',data = df)
ax.bar_label(ax.containers[0])
ax.set_title("1. Number of Fraudulent Transactions vs. Non-Fraudulent Transactions",color="r", y=1.03)
plt.xlabel('Class: "0" for Non-Fraudulent, "1" for Fraudulent')
plt.show()
#------------------------------------------------------------------------------------------------------------------------------
df_fraud_C = df[df['Class']>0].copy().drop(["Class"],axis=1)
df_normal_C = df[df["Class"]<1].copy().drop(["Class"],axis=1)
#------------------------------------------------------------------------------------------------------------------------------
#2. Distribution of Non-Fraudulent Transactions
fig=plt.figure(figsize=(20,20))
fig.suptitle("2. Distribution of variables for Non-Fraudulent Transactions", fontsize=16,color = "r")
for i,x in enumerate(df_normal_C.columns):
    ax = plt.subplot(8,4,i+1)
    sns.histplot(df_normal_C[x],bins=20,kde=True).set_title("Distribution of "+x, fontweight ="bold")
fig.tight_layout()  
fig.subplots_adjust(top=0.93)
plt.show()
#------------------------------------------------------------------------------------------------------------------------------
#3. Distribution of Fraudulent Transactions
fig=plt.figure(figsize=(20,20))
fig.suptitle("3. Distribution of variables for Fraudulent Transactions", fontsize=16,color = "r")
for i,x in enumerate(df_fraud_C.columns):
    ax = plt.subplot(8,4,i+1)
    sns.histplot(df_fraud_C[x],bins=20,kde=True,color = "darkorange").set_title("Distribution of "+x, fontweight ="bold")
fig.tight_layout()  
fig.subplots_adjust(top=0.93)
plt.show()
#------------------------------------------------------------------------------------------------------------------------------
#4. Distribution of Financial Amounts Transactions in Different Class
plt.figure(figsize=(10,8))
sns.boxplot(data = df, x= 'Class', y='Amount')
plt.ylim(-10,500)
plt.title("Distribution of Financial Amounts Transacted in Different Class",color = "r", y=1.03)
plt.show()
#print('Mean:')
#mean = df.groupby('Class')['Amount'].mean()
#mean_df = pd.DataFrame(data = mean)
#display(mean_df)
#print('Median:')
#median = df.groupby('Class')['Amount'].median()
#median_df = pd.DataFrame(data = median)
#display(median_df)
#------------------------------------------------------------------------------------------------------------------------------
#5. Correlation Matrix of Fraudulent Transactions between Lable & Features
corr = df.corr()
corr = corr[['Class']]
fig = plt.figure(figsize=(10,8))
fig = sns.heatmap(corr, annot=True, linewidths=0.3)
plt.title("Correlation between Class & Features",color = "r", y=1.03)
plt.show()


# ### Module 3: Dimension Reduction
# 

# In[ ]:


#define function
def decide(df,df_inverse,y_data):
    error_rate = np.sum((np.array(df)-np.array(df_inverse))**2,axis=1)
    error_df = pd.DataFrame(error_rate,index=df.index)
    decide_df = pd.concat([error_df,y_data],axis=1)
    decide_df.columns = ["error_rate","correct_label"]
    #choose top 500 error rate
    decide_df = decide_df.sort_values(by="error_rate")
    decide_df_500 = decide_df.tail(500)
    precision = np.round(decide_df_500.error_rate[decide_df_500.correct_label==1].count()/500,4)
    recall = np.round(decide_df_500.error_rate[decide_df_500.correct_label==1].count()/y_data.sum(),4)
    F1 = np.round((2 * (precision * recall) / (precision + recall)),4)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("F1-score: ",F1)
#------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#------------------------------------------------------------------------------------------------------------------------------
#Drop Label -> "Class"
df_C = df.copy().drop(['Class'],axis = 1)

#Re-scaling Before PCA -> Use StandardScaler but not MinMaxScaler
scaler = StandardScaler(copy=True)
df_C.loc[:,df_C.columns] = scaler.fit_transform(df_C[df_C.columns])

#Use PCA to reduce dimension
#1. Check For Number of Principal Components To Take
print('\33[91m'+"Cumulative Explained Variance with All 30 Components")
print('\33[0m'+"")
pca = PCA(n_components=30)
pca.fit(df_C)
cum_var = []
for i in range(0,len(pca.explained_variance_ratio_)):
    if i==0:
        cum_var.append(round(pca.explained_variance_ratio_[i],4))
    else:
        cum_var.append(round(pca.explained_variance_ratio_[i]+cum_var[i-1],4))
print(cum_var)
print("")
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title("Cumulative Explained Variance with All 30 Components",color="r")
plt.show()
print("")

#2. Number of Principal Components: "5% error": 27
components = (next(x for x,ele in enumerate(cum_var) if ele > 0.95))+1
print('\33[91m'+ f"Number of Components Chosen: {components} for 95% of Explained Variance")
pca = PCA(n_components=components)
X_trained = pca.fit_transform(df_C)
dumdf = pd.DataFrame(data = X_trained[:, :2])
dumdf = pd.concat((dumdf,df["Class"].copy()),join="inner",axis=1)
dumdf.columns = ['PCA1', 'PCA2','Class']
sns.lmplot(x='PCA1', y='PCA2', hue='Class', data=dumdf, fit_reg=False)
plt.title("PCA: PCA1 vs.PCA2",color="r")
plt.show()
#------------------------------------------------------------------------------------------------------------------------------
#3. Interpretation
df_DimRed = pd.DataFrame(data = pca.inverse_transform(X_trained), index = df_C.index)
print("")
print('\33[91m'+ "Interpretation")
print('\33[0m'+"")
print("PCA Type: Normal PCA")
print("PCA Plot Used: 2D PCA")
print(f"Number of Components Chosen: {components} for 95% of Explained Variance")
print("Result with PCA unsupervised learning: ")
decide(df_C,df_DimRed,df["Class"].copy())


# ### Module 4: Classification
# 

# In[ ]:


#define function
def info_model(y_test, y_pred):
    #Info. of the model
    acc_knn = round(accuracy_score(y_test, y_pred)*100, 4)
    print('\33[0m'+'Accuracy of the Model Classifier: ', acc_knn)
    print("")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cm_df=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
    sns.heatmap(cm_df, annot = True)
    plt.title('Confusion Matrix',color='r')
    plt.show()
    print('True Positives: ',cm_df.iloc[1][1])
    print('True Negatives: ',cm_df.iloc[0][0])
    print('False Positives (Type I error): ',cm_df.iloc[0][1])
    print('False Negatives ( Type II error): ',cm_df.iloc[1][0])
    #PR Curve
    average_precision = average_precision_score(y_test, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    plt.plot(recall, precision)
    plt.title(f'Precision-Recall Curve: Average Precision = {round(average_precision,2)}',color='r')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    #Roc Curve
    fpr, tpr, thresholds = roc_curve(y_test,y_pred)
    areaUnderROC = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='r', lw=2, label= f'ROC curve (area = {round(areaUnderROC,4)})')
    plt.plot([0, 1], [0, 1], color='k', lw=3, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve',color='r')
    plt.legend(loc="lower right")
    plt.show()
#------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#------------------------------------------------------------------------------------------------------------------------------
#Balanced the "Class" Data
dum_df_C = df_C.copy()
dum_df_C["Class"] = df['Class'].copy()
df_fraud = dum_df_C[dum_df_C['Class']==1]
df_normal = dum_df_C[dum_df_C['Class']==0]
fraud_len = len(df_fraud)
df_normal_bal = df_normal.sample(n=fraud_len,random_state=40)
sample_df = pd.concat([df_fraud,df_normal_bal.set_index(df_fraud.index)])
sample_df = sample_df.sample(frac=1, random_state=20)
sample_df = sample_df.reset_index(drop=True)

#Re-scale again to prevent any changes occur
scaler_bal = StandardScaler()
X = sample_df.copy().drop(['Class'],axis=1)
y_train = sample_df['Class'].copy()
X_train = scaler_bal.fit_transform(X)

#load test case -> X_test & y_test
test_df = pd.read_csv("../input/creditcard-test-train/creditcard_test.csv")
X_test = test_df.copy().drop(['Class'],axis=1)
y_test = test_df['Class'].copy()
X_test = StandardScaler().fit_transform(X_test)
#------------------------------------------------------------------------------------------------------------------------------
#1. K Nearest Neighbor 
print('\33[91m'+"1. K Nearest Neighbor Model")
print('\33[0m'+"")
from sklearn.neighbors import KNeighborsClassifier

# Hyperparameter Tuning w/ 5-fold cross-validation
param_grid = {'n_neighbors' : list(range(1,51))}
gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
g_res = gs.fit(X_train, y_train)
print(f'Best score (KNN Model): {g_res.best_score_} for {g_res.best_params_}')
plt.plot(range(1,51),g_res.cv_results_.get('mean_test_score'))
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Accuracy for different value of K for KNN model',color='r')
plt.show()
print("")

#Create model
print('\33[91m'+"Model Evaluation")
print("")
k_value = g_res.best_params_.get('n_neighbors')
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train, y_train)

#Test model
y_pred = model.predict(X_test)

info_model(y_test, y_pred)
#------------------------------------------------------------------------------------------------------------------------------
print('------------------------------------------------------------------------------------------------------------------------------')
#2. Random Forest 
print('\33[91m'+"2. Random Forest Model")
print('\33[0m'+"")
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter Tuning w/ 5-fold cross-validation
param_grid = {'n_estimators' : list(range(1,101))}
gs = GridSearchCV(RandomForestClassifier(random_state=123), param_grid, cv=5, scoring='accuracy')
g_res = gs.fit(X_train, y_train)
print(f'Best score (RF Model): {g_res.best_score_} for {g_res.best_params_}')
plt.plot(range(1,101),g_res.cv_results_.get('mean_test_score'))
plt.xlabel('Value of E for RF')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Accuracy for different value of E for RF model',color='r')
plt.show()
print("")

#Create model
print('\33[91m'+"Model Evaluation")
print("")
k_value = g_res.best_params_.get('n_estimators')
model = RandomForestClassifier(n_estimators=k_value)
model.fit(X_train, y_train)

#Test model
y_pred = model.predict(X_test)

info_model(y_test, y_pred)
#------------------------------------------------------------------------------------------------------------------------------
print('------------------------------------------------------------------------------------------------------------------------------')
#3. XGBoost
print('\33[91m'+"3. XGBoost Model")
print('\33[0m'+'Remark: XGBoost required to install manually. Installation guide has been provided in the code quoted by "###"')
print('\33[0m'+"")
### Installation -> XGBoost
###import sys###
###!{sys.executable} -m pip install xgboost###
from xgboost import XGBClassifier

# Hyperparameter Tuning w/ 5-fold cross-validation
param_grid = {'n_estimators' : list(range(1,101))}
gs = GridSearchCV(XGBClassifier(eval_metric='mlogloss',use_label_encoder =False), param_grid, cv=5, scoring='accuracy')
g_res = gs.fit(X_train, y_train)
print(f'Best score (xgb Model): {g_res.best_score_} for {g_res.best_params_}')
plt.plot(range(1,101),g_res.cv_results_.get('mean_test_score'))
plt.xlabel('Value of E for xgb')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Accuracy for different value of E for xgb model',color='r')
plt.show()
print("")

#Create model
print('\33[91m'+"Model Evaluation")
print("")
k_value = g_res.best_params_.get('n_estimators')
model = XGBClassifier(n_estimators=k_value, eval_metric='mlogloss',use_label_encoder =False)
model.fit(X_train, y_train)

#Test model
y_pred = model.predict(X_test)

info_model(y_test, y_pred)


# ### Module 5: Summary
# 

# **Data Exploration** <br>
# The dataset used in this project contains 31 columns: ' Time,' 'Amount,' 'Class,' and 'V1' to 'V28' with around 290,000 data. We firstly had checked for null values and duplicated values in the dataset. The result indicated there are 798 missing values distributed under the 'V22' and 'V23' columns and 1076 duplicated values. We had filtered out the whole row instead of adopting the imputation method. It is believed that the dropped data would not significantly impact the result since there are around 99.3% remaining data. We also had reviewed the outliers in the dataset. However, we had chosen to keep all outliers rather than remove them. The reason for not releasing the outliers is that it may affect the accuracy of the model 
# as the model may mislabel for the extreme cases in further analysis, which could be vital in discovering the fraudulent transactions. <br> 
# After the data preprocessing, 282,783 observations remained with 427 data for the positive class and 282,356 data for the negative class. <br> <br>
# We have visualized five figures on the dataset. The first one is a count plot on the number of fraudulent transactions and non-fraudulent transactions. The result had indicated that the dataset is highly imbalanced, with 282,356 data on non-fraudulent transactions but only 427 data on fraudulent transactions. We also had plotted the distribution of the variables in the two classes. It is discovered that the distribution of the following columns holds a similar distribution curve: 'V3,' 'V8,' 'V13,' 'V15,' 'V19,' 'V21' and 'V24.' However, due to confidentiality issues, we cannot interpret any outcome. There is an intersecting finding on the distribution of monetary amounts transacted in a different class. The average quantity transacted in the fraud group (\\$123.12) is higher than in the non-fraud group ($88.50). However, we obtained the opposite result when calculating the median, which was \\$11.4 for the fraud group and \\$22.0 for the non-fraud group. It may be because the criminals thought a small number of transactions could be hard to discover by the cardholder. Furthermore, we had examined the correlation between the  'Class' and Features. It is possible to see those Features only having a weak correlation to the 'Class'.

# **Model Evaluation** <br>
# ###### Dimension Reduction 
# To achieve dimension reduction, we have applied Principal component analysis. We had calculated the cumulative explained variance with all 30 components and had chosen 27 components for 95% explained variance. <br><br>
# Cumulative Explained Variance : <br>
# [0.0657, 0.1222, 0.1567, 0.191, 0.2251, 0.2591, 0.293, 0.3269, 0.3607, 0.3945, 0.4282, 0.4618, 0.4954, 0.5289, 0.5624, 0.5959, 0.6293, 0.6627, 0.6961, 0.7294, 0.7627, 0.796, 0.8292, 0.8624, 0.895, 0.9274, 0.9594, 0.988, 0.9986, 1.0] <br><br>
# Result: <br>
# Precision:  0.676<br>
# Recall:  0.7916<br>
# F1-score:  0.7292<br>
# ###### Classification
# We had chosen three classification models: K-Nearest Neighbor model, Random Forest classification model, and XGBoost classification Model. To maximize the accuracy, we applied the Hyperparameter Tuning method with 5-fold cross-validation to choose the best estimators for the model. <br> <br>
# For the K-Nearest Neighbor model, the estimators (n_neighbors) utilized to construct the model is 9, with the best score of 0.915 under the training process. After the testing process, the model accuracy is 96.7%, with the following result: <br> <br>
# True Positives:  47<br>
# True Negatives:  98<br>
# False Positives (Type I error):  2<br>
# False Negatives ( Type II error):  3<br>
# Area under the ROC curve: 0.96 <br> <br>
# For the Random Forest classification model, the estimators (n_estimators) utilized to construct the model is 35, with the best score of 0.940 under the training process. After the testing process, the model accuracy is 94.0%, with the following result: <br><br>
# True Positives:  49<br>
# True Negatives:  92<br>
# False Positives (Type I error):  8<br>
# False Negatives ( Type II error):  1<br>
# Area under the ROC curve: 0.94 <br> <br>
# For the XGBoost classification model, the estimators (n_estimators) utilized to construct the model is 61, with the best score of 0.939 under the training process. After the testing process, the model accuracy is 92.0%, with the following result: <br><br>
# True Positives:  49<br>
# True Negatives:  89<br>
# False Positives (Type I error):  11<br>
# False Negatives ( Type II error):  1<br>
# Area under the ROC curve: 0.935 <br> <br>

# **Conclusions** <br>
# Based on the model evaluation, it is discovered that the K-Nearest Neighbor model had the best accuracy with 96.7%. However, it should be noticed that the model accuracy under our evaluation existing error as we had initiated the "seed" in our model for repeatability on the result, which leads the model could achieve a better result by using a better seed. Another point that should be noticed is the weighting of Type  I error and Type II error. For example, if the business cares more about detecting the fraud group, the XGBoost classification model should be adopted as this model perform better in catching the fraud transactions in our evaluation.
