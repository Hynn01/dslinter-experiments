#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Question and problem definition
# 
#     อยากรู้การจ่ายเงินว่าลูกค้าแต่ละคนเป็นประเภทไหนและจ่ายเงินด้วยอะไรซึ่งอาจจะทำให้เราสับสนเพราะต้องลองดูว่าลูกค้านั้นจ่ายเงินด้วย เงินสดหรือจ่ายผ่านบัตร
#  

# # Acquire training and testing data

# In[ ]:


# ดึงข้อมูลจากไฟล์มาเก็บไว้ในตัวแปร data
data = pd.read_csv("../input/passakorn-supermarket/supermarket_sales - Sheet1.csv")

# ก็อปปี้ข้อมูลไว้
original_data = data.copy()
data.head()


# # Wrangle, prepare, cleanse the data

# In[ ]:


# ตรวจสอบรูปแบบของข้อมูล
data.info()


# In[ ]:


# ดูข้อมูล column 'Date'
data['Date'].head()


# In[ ]:


# แปลงค่า column 'Date' เป็นเดือน
dt_series = pd.to_datetime(data['Date'])
data['month'] = dt_series.dt.month
data['month'].head()


# In[ ]:


# ตรวจสอบว่ามีการชำระเงินแบบใดบ้าง
data['Payment'].value_counts()


# In[ ]:


# กำหนดสถานะผู้ที่ชำระด้วย Ewallet และ Credit card ให้เป็นกลุ่ม digitalpay
digitalpay = ["Ewallet","Credit card"]
# ส่วนสถานะผู้ที่ชำระด้วย Cash ให้เป็นกลุ่ม normalpay

# ฟังก์ชั่นแบ่งกลุ่มของลูกค้าเป็น catagory
def data_split(status):
    if status in digitalpay:
        return 'digitalpay'
    else:
        return 'normalpay'

# ฟังก์ชั่นแบ่งกลุ่มของลูกค้าเป็นตัวเลข  
def data_split_num(status):
    if status in digitalpay:
        return 1
    else:
        return 0
# สร้าง column และเรียกใช้ฟังก์ชั่น
data['data_split'] = data['Payment'].apply(data_split)
data['data_split_num'] = data['Payment'].apply(data_split_num)
data.head()


# In[ ]:


# plot ดูจำนวนข้อมูลให้แต่ละเดือน
sns.countplot(x="month", data=data);


# In[ ]:


# plot ดูค่าเฉลี่ยของการจ่ายเงินของลูกค้าในแต่ละเดือน
plt.figure(figsize=(12,8))
sns.barplot('Payment', 'month', data=data, palette='tab10', ci=None)
plt.title('Payment', fontsize=16)
plt.xlabel('Average payment amount', fontsize=14)
plt.ylabel('month', fontsize=14)


# จากรูปจะเห็นได้ว่าการจ่ายเงินแทบจะไม่แตกต่างกันมาก

# In[ ]:



colors = ["#FF6600", "#FFCCCC"]
labels ="digitalpay", "normalpay"

plt.suptitle('Information on data_split', fontsize=20)

data["data_split"].value_counts().plot.pie(autopct='%1.2f%%',  shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=70)


# จากกราฟจะเห็นได้ว้่ มีการจ่ายแบบ digitalpay มากกว่า normalpay 31.20%

# In[ ]:


# plot เพื่อดูจำนวนผู้ที่จ่ายด้วยเงินสดในแต่ละเดือนคิดเป็น%
graph_dims = (24,10)
fig, ax = plt.subplots(figsize=graph_dims)
palette = ["#FADEE1"]
sns.barplot(x="month", y="data_split_num", data=data, palette=palette, estimator=lambda x: sum(x)/len(x), ci = None)
ax.set(ylabel="(% of normalpay)")


# In[ ]:


# ตรวจสอบว่ามี columns มีค่าว่างหรือไม่ ถ้ามีจะเป็น True ถ้าไม่จะเป็น False
data.replace([np.inf, -np.inf], np.nan)
data.isnull().any()


# เมื่อเราเจอ columns ที่เสียหายแล้วนั้นเราต้องทำการใส่ค่าเข้าไปไม่เช่นนั้นจะเกิด Error ไม่สามารถ Train Model ได้

# In[ ]:


# หาจำนวนของ catagory (Product line = ประเภทสินค้า)
data['Product line'].value_counts()


# In[ ]:


# plot เพื่อดู distribution รายได้ในแต่ละเดือนของ supermarket
plt.figure(figsize=(24,10))
sns.distplot(data['Total'])


# In[ ]:


# หาค่ามัธยฐาน
data['Total'].median()


# # Analyze, identify patterns, and explore the data
# 
# เมื่อจัดการแยกกลุ่มได้แล้วต่อมาคือการลบข้อมูลที่ไม่จำเป็นออก

# In[ ]:


data_selected = ['Product line','Total','data_split_num','data_split','Payment','Date']
data = data[data_selected]
data.head()


# In[ ]:


# Check correlations
sns.heatmap(data.corr(), annot=True)


# In[ ]:


# check missing value อีกรอบเพื่อความชัว
data.isnull().any()


# In[ ]:


# แปลง categories ในเป็นตัวเลขโดยการทำ one hot encoding 
categories = ['Product line','Total','Payment','Date']
raw_model_data = pd.get_dummies(data.copy(), columns=categories,drop_first=True)


# # Model, predict and solve
# 
# หลังจากที่เราเตรียมข้อมูลและวิเคราะห์มาเรียบร้อยแล้ว ต่อไปเราต้องทำการ Train Model

# In[ ]:


# กำหนด feature ของ X และ y
X = raw_model_data.drop(columns=['data_split', 'data_split_num'],axis=1)
y = raw_model_data['data_split']

# แบ่ง X_train, X_test, y_train, y_test และกำหนดค่า randon_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)


# ทำการสร้าง Model โดยใช้วิธีแบบ Random Forest
# 
# 

# In[ ]:


# กำหนดค่าของ Forest
random_forest = RandomForestClassifier(n_estimators=600)
# ทำการ Train Model
random_forest.fit(X_train, y_train)
# ทำการ predict 
y_pred = random_forest.predict(X_test)
# ดูค่า accuracy ของ model
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[ ]:


# แสดงค่า accuracy ของ prediction
acc_random_forest = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_random_forest


# ทำการสร้าง Model โดยใช้วิธีแบบ DecisionTree

# In[ ]:


# กำหนดค่าของ DecisionTree 
decision_tree = DecisionTreeClassifier()
# ทำการ Train Model 
decision_tree.fit(X_train, y_train)
# ทำการ predict 
y_pred = decision_tree.predict(X_test)
# ดูค่า accuracy ของ model
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree


# In[ ]:


acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_decision_tree


# ทำการสร้าง Model โดยใช้วิธีแบบ NearestNeighbors

# In[ ]:


# กำหนดค่าของ KNN 
knn = KNeighborsClassifier(n_neighbors = 3)
# ทำการ train model
knn.fit(X_train, y_train)
# ทำการ predict 
y_pred = knn.predict(X_test)
# ดูค่า accuracy
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn


# In[ ]:


acc_knn = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_knn


# ทำการสร้าง Model โดยใช้วิธีแบบ Gaussian Naive Bayes

# In[ ]:


# กำหนดค่าของ Gaussian Naive Bayes
gaussian = GaussianNB()
# ทำการ train model
gaussian.fit(X_train, y_train)
# ทำการ predict 
y_pred = gaussian.predict(X_test)
# ดูค่า accuracy
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian


# In[ ]:


acc_gaussian = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_gaussian


# ทำการสร้าง Model โดยใช้วิธีแบบ Perceptron

# In[ ]:


perceptron = Perceptron()
# ทำการ train model
perceptron.fit(X_train, y_train)
# ทำการ predict
Y_pred = perceptron.predict(X_test)
# ดูค่า accuracy
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
acc_perceptron


# In[ ]:


acc_perceptron = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_perceptron


# ทำการสร้าง Model โดยใช้วิธีแบบ Stochastic Gradient Descent

# In[ ]:


# กำหนดค่าของ Stochastic Gradient Descent
sgd = SGDClassifier()
# ทำการ train model
sgd.fit(X_train, y_train)
# ทำการ predict
y_pred = sgd.predict(X_test)
# ดูค่า accuracy
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
acc_sgd


# In[ ]:


acc_sgd = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_sgd


# # Visualize, report, and present the problem solving steps and final solution
# 

# In[ ]:


# สร้าง dataframe เพื่อเก็บข้อมูล Score
models = pd.DataFrame({
    'Model': ['KNN', 'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Decision Tree'],
    'Score': [acc_knn, acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd,  acc_decision_tree]})

# แสดงผลออกมาโดยเรียงจากมากไปน้อย
models.sort_values(by='Score', ascending=False)


# **ในส่วนนี้ต้องมี library pydotplus ใน notebook ไม่สามารถรันได้

# In[ ]:


# Export Decision tree 
#dot_data = StringIO()
#export_graphviz(decision_tree, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True ,class_names=['0','1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Export เป็นไฟล์ png
#graph.write_png('dtree_pipe.png')
# Export เป็นไฟล์ pdf
#graph.write_pdf('dtree_pipe.pdf')
# แสดง Decision tree pipe 
#Image(graph.create_png())


# # Supply or submit the results

# In[ ]:


print("X_value = %s, Predicted=%s" % (X_test, y_pred[1]))

