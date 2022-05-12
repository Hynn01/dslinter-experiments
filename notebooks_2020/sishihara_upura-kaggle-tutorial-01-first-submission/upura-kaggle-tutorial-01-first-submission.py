#!/usr/bin/env python
# coding: utf-8

# This Notebook is a kaggle tutorial for Japanese kaggle beginners writen in Japanese.
# 
# # 1. まずはsubmit！ 順位表に載ってみよう

# この[Notebook](https://www.kaggle.com/sishihara/upura-kaggle-tutorial-01-first-submission)では、Kaggleでのsubmitの方法を学びます。
# 
# Kaggleでは、いくつかの方法で自分が作成した機械学習モデルの予測結果を提出可能です。（Notebook経由でしか提出できないコンペティションも存在します）
# 
# - Notebook経由
# - csvファイルを直接アップロード
# - [Kaggle API](https://github.com/Kaggle/kaggle-api)を利用
# 
# 今回は、Notebook経由で提出してみましょう。
# 
# このNotebookにはいろいろなセルが含まれていますが、一旦は何も考えずに右上の「COMMIT」をクリックしてみてください。
# 
# 今回は一旦無視したNotebookの処理の流れは、次のNotebookで具体的に見ていきます。

# In[ ]:


import numpy as np
import pandas as pd


# ## データの読み込み

# In[ ]:


get_ipython().system('ls ../input/titanic')


# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


gender_submission.head()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


data = pd.concat([train, test], sort=False)


# In[ ]:


data.head()


# In[ ]:


print(len(train), len(test), len(data))


# In[ ]:


data.isnull().sum()


# ## 特徴量エンジニアリング

# ### 1. Pclass

# ### 2. Sex

# In[ ]:


data['Sex'].replace(['male','female'], [0, 1], inplace=True)


# ### 3. Embarked

# In[ ]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# ### 4. Fare

# In[ ]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)


# ### 5. Age

# In[ ]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)


# In[ ]:


delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)


# In[ ]:


train = data[:len(train)]
test = data[len(train):]


# In[ ]:


y_train = train['Survived']
X_train = train.drop('Survived', axis = 1)
X_test = test.drop('Survived', axis = 1)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# ## 機械学習アルゴリズム

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


y_pred[:20]


# ## 提出

# In[ ]:


sub = gender_submission
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)


# In[ ]:




