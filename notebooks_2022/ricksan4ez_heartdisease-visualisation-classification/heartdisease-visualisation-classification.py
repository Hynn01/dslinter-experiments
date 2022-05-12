#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.ibb.co/hDphhrF/ezgif-com-gif-maker.jpg" align="center" width=100%>

# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Import Libraries</font>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Upload dataset</font>

# In[ ]:


dataset = pd.read_csv('/kaggle/input/personal-key-indicators-of-heart-disease/heart_2020_cleaned.csv')
dataset.head(3)


# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Some Prior Data Analysis (aka EDA)</font>

# In[ ]:


# iclude='all' - just means that we need all columns.
dataset.describe(include='all')


# In[ ]:


# check the data types of columns
dataset.dtypes


# In[ ]:


# check null values and get a little bit more information about dataset
dataset.info()


# In[ ]:


# look into the unique values to check if there are any mismatches
dataset.nunique()


# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Now, lets dive into exploring the features</font>

# As we have almost 320000 of rows in data I want to use sample of it that will represant the whole popilation.

# In[ ]:


with_hd = dataset[dataset['HeartDisease']=='Yes'].sample(n=100000, random_state=42)
without_hd = dataset[dataset['HeartDisease']=='No'].sample(n=100000, random_state=42)
sample = pd.concat([with_hd, without_hd])


# In[ ]:


sample.shape


# In[ ]:


sample.head(5)


# <font face="Helvetica, sans-serif" size="4px" color="#000">The BMI 🍩 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('rocket_r')
plt.figure(figsize=(15, 4))
sns.stripplot(data=sample, x='HeartDisease', y='BMI')
plt.title('BMI vs Heart Disease')
plt.xlabel('HeartDisease')
plt.ylabel('BMI')
plt.show()


# In[ ]:


sns.catplot(data=sample, x='HeartDisease', y='BMI', kind='box')
plt.title('BMI vs Heart Disease')
plt.ylabel('BMI')
plt.xlabel('Heart Disease')
plt.show()


# Personally I thought that Body Mass Index (BMI) will have a huge impact on the health and namely on the heart but it doesn't look like there is any strong correlation here. Okay, lets proceed our research!

# <font face="Helvetica, sans-serif" size="4px" color="#000">The Smoking 🚬 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette("Set2")
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='Smoking', hue='HeartDisease',)
plt.title('Smoking vs Heart Disease')
plt.xlabel('Smoking')
plt.ylabel('Number of Cases')
plt.show()


# Here we clearly see that the smoking can lead straight forward to the heart disease as Left two bars shows that if person smokes than there are almost 2x chance to get heart disease and two bars from the right shows that if the person doesn’t smoke then there is almost 2x chance to have a healthy heart.

# <font face="Helvetica, sans-serif" size="4px" color="#000">The Alcohol 🥃 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('Accent')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='AlcoholDrinking', hue='HeartDisease',)
plt.title('Alcohol vs Heart Disease')
plt.xlabel('Alcohol Drinking')
plt.ylabel('Number of Cases')
plt.show()


# It seems that people who drink alcohol have heart diseases rarelier than people who doesn’t. Lets take closer look.

# In[ ]:


drink_alcohol = dataset[dataset['AlcoholDrinking'] == 'Yes'].sample(10000)
plt.figure(figsize=(15,7))
sns.countplot(data=drink_alcohol, x='AlcoholDrinking', hue='HeartDisease')
plt.title('AlcoholDrinking vs Heart Disease')
plt.xlabel('Alcohol Drinking')
plt.ylabel('Number of Cases')
plt.show()


# Yes! Here we clearly can notice that there much people with no heart disease while they drinking then people who doesn’t. So it seems that alcohol makes a good influence on our heart.

# <font face="Helvetica, sans-serif" size="4px" color="#000">The Physical Health 💪 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('viridis')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='PhysicalHealth', hue='HeartDisease')
plt.title('PhysicalHealth vs Heart Disease')
plt.xlabel('PhysicalHealth')
plt.ylabel('Number of Cases')
plt.show()


# Here we can see that as we increase the number of days the man was ill the more chances to have any heart disease.

# <font face="Helvetica, sans-serif" size="4px" color="#000">The Mental Health 🧠 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('crest')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='MentalHealth', hue='HeartDisease')
plt.title('MentalHealth vs Heart Disease')
plt.xlabel('MentalHealth')
plt.ylabel('Number of Cases')
plt.show()


# It doesn't look like it has a huge impact on Heart Diseases.

# <font face="Helvetica, sans-serif" size="4px" color="#000">The Diff Walking 🚶 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('rocket')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='DiffWalking', hue='HeartDisease')
plt.title('Diff Walking vs Heart Disease')
plt.xlabel('Diff Walking')
plt.ylabel('Number of Cases')
plt.show()


# Here we can see that if the man hardly walk that can lead to heart disaese.

# <font face="Helvetica, sans-serif" size="4px" color="#000">Sex ⚧ vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('Set1')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='Sex', hue='HeartDisease')
plt.title('Sex vs Heart Disease')
plt.xlabel('Sex')
plt.ylabel('Number of Cases')
plt.show()


# Hmm, here is one more intresting fact, it seems that the men has more chances to have Heart Disease then women... May be because of women...:)

# <font face="Helvetica, sans-serif" size="4px" color="#000">Age Category 🚼 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('Pastel2')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='AgeCategory', hue='HeartDisease', order=['18-24', '25-29', '30-34', '35-39', '40-44',
                                                                      '45-49', '50-54', '55-59', '60-64', '65-69',
                                                                      '70-74', '75-79', '80 or older'])
plt.title('Age Category vs Heart Disease')
plt.xlabel('Age Category')
plt.ylabel('Number of Cases')
plt.show()


# On the plot we clearly see that when we becoming older - the more chances to get the Heart Disease...

# <font face="Helvetica, sans-serif" size="4px" color="#000">Race 🌈 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('Pastel1')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='Race', hue='HeartDisease')
plt.title('Race vs Heart Disease')
plt.xlabel('Race')
plt.ylabel('Number of Cases')
plt.show()


# Intresting.. According to this plot White and American Indian/Alaskan Native people have heart diseases more often than other races

# <font face="Helvetica, sans-serif" size="4px" color="#000">Diabetic 🍬 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('PRGn_r')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='Diabetic', hue='HeartDisease')
plt.title('Diabetic vs Heart Disease')
plt.xlabel('Diabetic')
plt.ylabel('Number of Cases')
plt.show()


# I suppose that from this plot we can conclude that the people with Diabet have more chances to get heart disease.

# <font face="Helvetica, sans-serif" size="4px" color="#000">Physical Activity 🏒 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('Set3')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='PhysicalActivity', hue='HeartDisease')
plt.title('Physical Activity vs Heart Disease')
plt.xlabel('Physical Activity')
plt.ylabel('Number of Cases')
plt.show()


# Sports - Helth!

# <font face="Helvetica, sans-serif" size="4px" color="#000">GenHealth 🧬 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('Set3_r')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='GenHealth', hue='HeartDisease')
plt.title('Gen Health vs Heart Disease')
plt.xlabel('Gen Health')
plt.ylabel('Number of Cases')
plt.show()


# Gens play an important role in our life, so our helth also depends on our parents in some way.

# <font face="Helvetica, sans-serif" size="4px" color="#000">SleepTime 😴 vs Heart Disease 🫀</font>
# 

# In[ ]:


sns.set_palette('Set3_r')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='SleepTime', hue='HeartDisease')
plt.title('Sleep Time vs Heart Disease')
plt.xlabel('Sleep Time')
plt.ylabel('Number of Cases')
plt.show()


# Okay. Now i am starting to wake up even more easily:) 7 hours seems to be the best choice

# <font face="Helvetica, sans-serif" size="4px" color="#000">Asthma 🫁 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('Spectral')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='Asthma', hue='HeartDisease')
plt.title('Asthma vs Heart Disease')
plt.xlabel('Asthma')
plt.ylabel('Number of Cases')
plt.show()


# It has some effect, but not as much as I expected:)

# <font face="Helvetica, sans-serif" size="4px" color="#000">KidneyDisease 🤧 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('Spectral_r')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='KidneyDisease', hue='HeartDisease')
plt.title('Kidney Disease vs Heart Disease')
plt.xlabel('Kidney Disease')
plt.ylabel('Number of Cases')
plt.show()


# It seems that it looks similar to asthma, has some impact...

# <font face="Helvetica, sans-serif" size="4px" color="#000">SkinCancer 🦀 vs Heart Disease 🫀</font>

# In[ ]:


sns.set_palette('RdPu')
plt.figure(figsize=(15,7))
sns.countplot(data=sample, x='SkinCancer', hue='HeartDisease')
plt.title('Skin Cancer vs Heart Disease')
plt.xlabel('Skin Cancer')
plt.ylabel('Number of Cases')
plt.show()


# The same as asthma or Kidney Disease.

# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Splitting dataset to features and target variable</font>

# Getting X and y

# In[ ]:


X = sample.iloc[:, 1:]
y = sample.iloc[:, 0]


# In[ ]:


X.head(3)


# In[ ]:


y.head(3)


# Getting dfs with only categorical and only numeric data

# In[ ]:


X_cat = X[['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
          'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']]
X_cat.head(2)


# In[ ]:


X_num = X[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']]
X_num.head(2)


# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Encoding Categorical Values</font>

# In[ ]:


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])],remainder='passthrough')
X_cat_encoded = ct.fit_transform(X_cat)


# In[ ]:


X_cat_encoded.shape


# In[ ]:


type(X_cat_encoded)


# In[ ]:


X_cat_encoded = X_cat_encoded.toarray()


# In[ ]:


X_cat_encoded.shape


# In[ ]:


X_cat_encoded[1]


# In[ ]:


X_cat_df = pd.DataFrame(X_cat_encoded)
X_cat_df.head(3)


# In[ ]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[ ]:


y.shape


# In[ ]:


y


# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Scaling Numeric Values</font>

# In[ ]:


sc = StandardScaler()
X_num_scaled = sc.fit_transform(X_num)


# In[ ]:


X_num_scaled.shape


# In[ ]:


type(X_num_scaled)


# In[ ]:


X_num_scaled_df = pd.DataFrame(X_num_scaled, columns=['46', '47', '48', '49'])
X_num_scaled_df.head(3)


# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Concatenating numeric and categorical values</font>

# In[ ]:


X_cat_df.reset_index(drop=True, inplace=True)
X_num_scaled_df.reset_index(drop=True, inplace=True)
X = pd.concat([X_cat_df, X_num_scaled_df], axis=1)
X.head(3)


# In[ ]:


X.shape


# In[ ]:


X = X.values


# In[ ]:


X


# In[ ]:


y


# In[ ]:


y.shape


# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">Creating and Evaluating Models

# <font face="Helvetica, sans-serif" size="4px" color="#000">Logistic Regression

# In[ ]:


log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
scores_log_reg = cross_val_score(log_reg, X, y, cv=10)
print(f'Mean: {np.mean(scores_log_reg)}')
print(f'Standard Deviation: {np.std(scores_log_reg)}')


# <font face="Helvetica, sans-serif" size="4px" color="#000">KNeighborsClassifier

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=23)
scores_knn = cross_val_score(knn, X, y, cv=10)
print(f'Mean: {np.mean(scores_knn)}')
print(f'Standard Deviation: {np.std(scores_knn)}')


# <font face="Helvetica, sans-serif" size="4px" color="#000">SVC

# In[ ]:


svc = SVC(kernel='linear')
scores_svc = cross_val_score(svc, X, y, cv=10)
print(f'Mean: {np.mean(scores_svc)}')
print(f'Standard Deviation: {np.std(scores_svc)}')


# <font face="Helvetica, sans-serif" size="4px" color="#000">Kernel SVC

# In[ ]:


kernel_svc = SVC(kernel='rbf')
scores_kernel_svc = cross_val_score(kernel_svc, X, y, cv=10)
print(f'Mean: {np.mean(scores_kernel_svc)}')
print(f'Standard Deviation: {np.std(scores_kernel_svc)}')


# <font face="Helvetica, sans-serif" size="4px" color="#000">Naive Bayes

# In[ ]:


naive_bayes = GaussianNB()
scores_naive_bayes = cross_val_score(naive_bayes, X, y, cv=10)
print(f'Mean: {np.mean(scores_naive_bayes)}')
print(f'Standard Deviation: {np.std(scores_naive_bayes)}')


# <font face="Helvetica, sans-serif" size="4px" color="#000">DecisionTreeClassifier

# In[ ]:


dtc = DecisionTreeClassifier()
scores_dtc = cross_val_score(dtc, X, y, cv=10)
print(f'Mean: {np.mean(scores_dtc)}')
print(f'Standard Deviation: {np.std(scores_dtc)}')


# <font face="Helvetica, sans-serif" size="4px" color="#000">RandomForestClassifier

# In[ ]:


rfc = RandomForestClassifier(n_estimators=70)
scores_rfc = cross_val_score(rfc, X, y, cv=10)
print(f'Mean: {np.mean(scores_rfc)}')
print(f'Standard Deviation: {np.std(scores_rfc)}')


# <font face="Helvetica, sans-serif" size="5px" color="#34baeb">The Bottom Line

# It seems that the SVC model suits best for this problem and provide us with almost 77% accuracy. 

# Thanks, everyone for viewing my notebook. I hope it was helpful and intresting journey, now I want to ask you to upvote and leave some feedback - it will help me to improve in professional way and create more exciting and useful notebooks!
