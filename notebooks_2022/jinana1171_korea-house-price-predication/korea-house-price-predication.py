#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **아파트 데이터 파일 불러오기**

# In[ ]:


house_df=pd.read_csv("../input/korean-real-estate-transaction-data/Apart Deal.csv")


# In[ ]:


house_size = house_df['전용면적']
house_prize = house_df['거래금액']
print(len(house_prize))
for i in range(4250000):
    if type(house_prize[i]) != int:
        house_prize[i] = int(house_prize[i].replace(',',''))


# In[ ]:


house_grade = house_df['층']
for i in range(10000):
    house_grade[i] = int(house_grade[i])


# **세로축 집 평수, 가로축 집 가격으로 분류**

# In[ ]:


plt.scatter(house_prize[:4250000], house_size[:4250000]) # price - size


# In[ ]:


house_size_mean = [0 for i in range(50)]
house_size_num = [0 for i in range(50)]
for i in range(4250000):
    if i%100000 == 0:
       print(i);
    house_size_mean[int(house_size[i]/10)] += house_prize[i];
    house_size_num[int(house_size[i]/10)] += 1;
for i in range(len(house_size_mean)):
    if house_size_num[i] != 0:
        house_size_mean[i]/=house_size_num[i];
plt.plot(house_size_mean)


# In[ ]:


plt.scatter(house_prize[:10000],house_grade[:10000]) # price - floor


# In[ ]:


house_grade_mean = [0 for i in range(60)]
house_grade_num = [0 for i in range(60)]
for i in range(10000):
    house_grade_mean[house_grade[i]] += house_prize[i];
    house_grade_num[house_grade[i]] += 1;
for i in range(len(house_grade_mean)):
    if house_grade_num[i] != 0:
        house_grade_mean[i]/=house_grade_num[i];
plt.plot(house_grade_mean)


# **우리 집의 층수와 평수 입력**

# In[ ]:


now_size = 34;
now_floor = 8;
now_size *= 3.31
plus = 0;
num = 0;
for i in range(42500):
    if (((int(house_size[i]) - int(now_size)) <= 10) and ((int(house_size[i]) - int(now_size)) >= -10)):
        plus += house_prize[i];
        num+=1;
plus /= num;
plusg = 0;
numg = 0;
for i in range(10000):
    if house_grade[i] == now_floor:
        plusg += house_prize[i];
        numg+=1;
plusg /= numg;
print("예상 가격 = ", (plusg+plus)/20000, "억");


# 실제 집값 = 3.1억

# 

# In[ ]:





# 
