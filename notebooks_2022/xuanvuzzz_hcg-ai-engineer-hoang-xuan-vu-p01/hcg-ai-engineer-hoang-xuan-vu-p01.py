#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[ ]:


df_data = pd.read_csv('/kaggle/input/hcg-test-data-services-dept/bookings.csv')
df_label = pd.read_csv('/kaggle/input/hcg-test-data-services-dept/LTV_class.csv')


# ### Data Exploration

# In[ ]:


df_data


# In[ ]:


df_label


# In[ ]:



data = pd.merge(df_data, df_label, how="inner", on="GuestID")


# In[ ]:


data


# In[ ]:


sns.histplot(df_label['LTV_Cluster'])


# 
# - Data bị mất cân bằng dữ liệu phần do đó có thể ảnh hưởng đến hiệu suất mô hình </br>
# => sử dụng biện pháp cân bằng dữ liệu </br>
# => sử dụng độ đo accuracy và f1-score để đánh giá hiệu suất mô hình

# In[ ]:


data.describe()


# In[ ]:


### Kiểm tra dữ liệu có bị null
data.isna().sum()


# - có 472 giá trị null ở cột Channel 
# - có 11 giá trị null cột RoomNo
# - có 34 giá trị null cột Country

# In[ ]:


median_RoomPrice = sorted(data["RoomPrice"].tolist())
print("giá trị trung vị của RoomPrice: " + str((median_RoomPrice[int(len(median_RoomPrice)/2)] + median_RoomPrice[int(len(median_RoomPrice)/2)+1])/2))
median_TotalPayment = sorted(data["TotalPayment"].tolist())
print("giá trị trung vị của TotalPayment: " + str((median_TotalPayment[int(len(median_TotalPayment)/2)] + median_TotalPayment[int(len(median_TotalPayment)/2)+1])/2))


# In[ ]:


data.plot.scatter(x='LTV_Cluster', y="RoomPrice")


# In[ ]:


data.plot.scatter(x='LTV_Cluster', y="TotalPayment")


# - Giá trị min của RoomPrice: 0
# - Giá trị max của RoomPrice: 399.540000
# - Giá trị Mean của RoomPrice: 205.361223
# - Giá trị median của RoomPrice: 209.67000000000002
# </br>
# "============================================================="
# - Giá trị min của TotalPayment: 8.520000
# - Giá trị max của TotalPayment: 8424.510000
# - Giá trị mean của TotalPayment: 725.491934
# - Giá trị median của TotalPayment: 448.57
# </br>
# "============================================================="
# </br>
# Median có khả năng đo lường xu hướng tập trung của dữ liệu mạnh nhất qua các số liệu cho thấy xu hướng giá phòng nằm trong khoảng 205.361223 Còn Mean do ảnh hưởng của các outlier (giá trị bất thường của dữ liệu) nên thường bị chênh lệch với median nhưng trong bộ dữ liệu này cho thấy mean và median chênh lệch không nhiều nên ít bị ảnh hường bởi outlier. Còn đối với TotalPayment giá trị mean và median khá chênh lệch nhiều giá trị mean cao hơn so với median hơn 1,5 lần

# In[ ]:


## Vẽ biểu đồ ma trận tương quan
# tinh su phu thuoc cua tung thuoc tinh
correlation = data.corr(method='pearson')
fig = plt.subplots(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Blues')


# In[ ]:


print(data.columns)


# In[ ]:


print(data["Channel"].unique())
print("="*20)
print(data["Country"].unique())
print("="*20)
print(data["Status"].unique())


# In[ ]:


data["Channel"].fillna("other", inplace = True) 

my_imputer = SimpleImputer(strategy = 'most_frequent')
data['Country'] = pd.DataFrame(my_imputer.fit_transform(data[['Country']]))
ordinal_encoder = OrdinalEncoder()
data[["Channel", 'Status', 'Country']] = ordinal_encoder.fit_transform(data[["Channel", 'Status', 'Country']])

data["RoomNo"].fillna(data["RoomNo"].median(), inplace = True) # fill by median value


# In[ ]:


data.isna().sum()


# In[ ]:


### create feature 
### distance1 = Distance between create date and arrival date
### distance1 = Distance between arrival date and DepartureDate
z0 = pd.to_datetime(data['CreatedDate'], format='%Y-%m-%d')
z1 = pd.to_datetime(data['ArrivalDate'], format='%Y-%m-%d')
z2 = pd.to_datetime(data['DepartureDate'], format='%Y-%m-%d')
data["distance1"] = [int(x.days) for x in (z1-z0).tolist()]
data['distance2'] = [int(x.days) for x in (z2-z1).tolist()]
data =  data[['Status', 'RoomGroupID', 'RoomPrice', 'Channel', 'RoomNo', 'Country', 'Adults',
       'Children', 'TotalPayment', 'distance1', 'distance2', 'LTV_Cluster']]
data


# In[ ]:


## Vẽ biểu đồ ma trận tương quan
# tinh su phu thuoc cua tung thuoc tinh
correlation = data.corr(method='pearson')
fig = plt.subplots(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Blues')


# - distance1 and distance2 có ảnh hưởng rất lớn đến TotalPayments và đến khả năng phân loại đây là những đặc trưng quan trọng có thể ảnh hưởng đến khả năng phân loại của mô hình dựa trên biểu đồ tương quan của ma trận

# In[ ]:


sns.set()
corrmat = data.corr()
cols = corrmat.nlargest(10, 'TotalPayment')['TotalPayment'].index
cols = [cols]
sns.pairplot(data[['TotalPayment', 'RoomPrice', 'distance1', 'distance2', 'RoomNo']], height = 2.5)
plt.show()


# ## Feature Engineer

# In[ ]:


data


# In[ ]:


data.columns


# In[ ]:


X = data[['Status', 'RoomGroupID', 'RoomPrice', 'Channel', 'RoomNo', 'Country',
       'Adults', 'Children', 'TotalPayment', 'distance1', 'distance2']]
y = data["LTV_Cluster"]


# In[ ]:


X.info()


# ### Split train test

# In[ ]:


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.1, random_state = 0)
print(x_train.shape)
print(x_test.shape)
y_train = y_train.tolist()
y_test = y_test.tolist()


# In[ ]:


print(x_train.shape)
print(len(y_train))


# ### Data normalize

# In[ ]:


scaler = StandardScaler()
X_train_transform = scaler.fit(x_train).transform(x_train)
X_test_transform = scaler.transform(x_test)
print(X_train_transform.shape)
X_test_transform.shape


# ### Built model

# In[ ]:


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train_transform, y_train)
preds = clf.predict(X_test_transform)


acc = accuracy_score(preds, y_test)
f1 = f1_score(preds, y_test, average="macro")
print("accuracy_score = "+ str(acc))
print("f1-score : "+ str(f1))


# In[ ]:


from sklearn import svm
clf = svm.SVC()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)


acc = accuracy_score(preds, y_test)
f1 = f1_score(preds, y_test, average="macro")
print("accuracy_score = "+ str(acc))
print("f1-score : "+ str(f1))


# - Khi sử dụng data normalize thì cho kết quả tốt hơn độ đo accuracy cao hơn 12% và f1-score cao hơn 9.8%

# - Sử dụng kỹ thuật smote để cân bằng nhãn để cải thiện performance

# In[ ]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
x_train_smote, y_train_smote = oversample.fit_resample(X_train_transform, y_train)


from sklearn import svm
clf = svm.SVC()
clf.fit(x_train_smote, y_train_smote)
preds = clf.predict(X_test_transform)


acc = accuracy_score(preds, y_test)
f1 = f1_score(preds, y_test, average="macro")
print("accuracy_score = "+ str(acc))
print("f1-score : "+ str(f1))


# -> Sau khi sử dụng kỹ thuật smote thì hiệu suất mô hình đã tăng thêm 1% accuracy và f1-score đã tăng 1.1%

# - Sử dụng kỹ thuật gán nhãn giả - pseudo labeling để cải thiện hiệu suất mô hình 
# https://towardsdatascience.com/pseudo-labeling-to-deal-with-small-datasets-what-why-how-fd6f903213af
# 

# In[ ]:


from sklearn import svm
clf = svm.SVC()
clf.fit(x_train_smote, y_train_smote)
preds = clf.predict(X_test_transform)

clf = svm.SVC()
# sử dụng nhãn vừa dự đoán để train lại mô hình 

x_train_pseudo = np.concatenate([x_train_smote, X_test_transform], axis=0)
y_train_pseudo = np.concatenate([y_train_smote, preds], axis= 0)
clf.fit(x_train_pseudo,y_train_pseudo  )

preds = clf.predict(X_test_transform)
acc = accuracy_score(preds, y_test)

f1 = f1_score(preds, y_test, average="macro")
print("accuracy_score = "+ str(acc))
print("f1-score : "+ str(f1))


# -- Sau khi áp dụng kỹ thuật pseudo labeling thì mô hình đã tăng lên 0.5% độ đo accuracy và 0.6% với độ đo f1-score </br>
# -- Bên cạnh đó còn thể sử dụng thêm một số thuật toán như gridsearch để tìm siêu tham số phù hợp cho mô hình hoặc sử dụng thêm một số thuật toán khác để xây dựng mô hình tốt hơn
