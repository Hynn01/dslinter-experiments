#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# # Load Data

# In[ ]:


data = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
data


# # Info

# In[ ]:


data.info()


# ### column별 의미
# 1. age                      - 나이
# 2. anaemia                  - 빈혈 유무
# 3. creatinine_phosphokinase - 혈중 CPK 농도
# 4. diabetes                 - 당뇨 유무
# 5. ejection_fraction        - 박출계수 : 심장 수축 별 방출하는 혈액의 비율
# 6. high_blood_pressure      - 고혈압 유무
# 7. platelets - 혈소판의 양
# 8. serum_creatinine      -   혈중 크레아티닌 농도
# 9. serum_sodium  - 혈중 나트륨 농도
# 10. sex - 성별 (여자 0, 남자 1)
# 11. smoking - 흡연 여부
# 12. time - 관찰 기간(날짜 수)
# 13. DEATH_EVENT - 관찰 기관 내 사망 여부

# ### column 세부 사항
# #### CPK 혹은 CK
# * 심장, 뇌, 골격근에서 발견되는 효소
# * 근육이 수축할때 사용
# * 심장 세포가 손상된 경우 혈중 CK농도 상승
# 
# #### 박출계수
# * 좋은 기능을 하는 심장은 각 맥박당 혈액의 최소 50% 분출해야함
# 
# #### 크레아티닌
# * 근육에서 생성되는 노폐물, 근육량에 비례 ( 남자가 보통 수치가 높음)
# * 심부전으로 인해 증가
# * 정상 범위 0.5~1.4 mg/dL
# 
# #### 혈중 나트륨 농도
# * 심부전으로 인해 감소
# * 135 mmol/L (135 mEq/L)미만이면 저나트륨혈증

# # Data analysis & EDA

# In[ ]:


data.describe()


# ### 평균값 분석
# * 연령대 : 약 60세
# * 박출계수 : 약 38% ( 정상수치 50%이상)
# * 크레아티닌 : 약 1.4mg/dL ( 정상수치 0.5 ~ 1.4)
# * 혈중 나트륨 농도 : 약 137mmol/L ( 정상수치 135 이상)
# 
# > **정상 수치범위를 벗어나 있거나, 이탈에 근접하였음.**

# # 1. 결측치 유무 확인

# In[ ]:


data.isnull().sum()


# # 2. Heatmap

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(20,20)
sns.heatmap(data.corr(), square = True, annot = True)


# # 3. 부울 Column

# In[ ]:


sns.countplot(x = 'anaemia', data = data)
plt.show()

sns.countplot(x = 'diabetes', data = data)
plt.show()

sns.countplot(x = 'high_blood_pressure', data = data)
plt.show()

sns.countplot(x = 'sex', data = data)
plt.show()

sns.countplot(x = 'smoking', data = data)
plt.show()


# # 4. 실수형 Colum 

# In[ ]:


cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
for col in cols:
    sns.boxplot(x = data[col], color = 'teal')
    plt.title(col)
    plt.show()


# > 특정 환자들에게서 이상치가 상당히 많이 보임

# # 5. 연령대 분석

# 연령 분포를 관찰하기 위해 일의 자릿수를 버림

# In[ ]:


import collections

AGE = data['age']//10 * 10
AGE
collections.Counter(AGE)


# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

age_value = list(collections.Counter(AGE).values())
age_label = list(collections.Counter(AGE).keys())

'''fig = go.Pie(values = age_value, labels = age_label, name = "연령대")
layout = go.Layout(title = '심부전 환자 관찰 연령대 분포')
fig = go.Figure(data = [fig], layout = layout)
fig.update_traces(hole=.4, hoverinfo="label+percent")
pyo.iplot(fig)'''

fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Pie(values = age_value, labels = age_label,  title = '연령대'))
fig.update_traces(textposition='inside', hole=.4, textinfo="label+percent", hoverinfo="label+percent")
fig.show()


# > 60대와 50대가 가장 많이 분포함

# # 6. Death Event와 Column별 분석

# In[ ]:


fig = px.histogram(data, x='age', color = 'DEATH_EVENT', marginal = "box", hover_data = data.columns, title = 'Age 와 Death Event 분포',
                   labels = {'age':'AGE'},
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# > 사망한 환자들의 연령대가 높게 분포되어있음

# In[ ]:


fig = px.histogram(data, x='serum_creatinine', color = 'DEATH_EVENT', marginal = "box", hover_data = data.columns, title = '크레아티닌 과 Death Event 분포',
                   labels = {'serum_creatinine':'Creatinine'},
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# > 사망한 환자들의 크레아티닌 수치가 높게 분포되어있음

# In[ ]:


fig = px.histogram(data, x='ejection_fraction', color = 'DEATH_EVENT', marginal = "box", hover_data = data.columns, title = '박출계수 과 Death Event 분포',
                   labels = {'ejection_fraction':'박출계수'},
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# > 사망한 환자들의 심장에서 뿜어내는 혈액의 비율이 낮게 분포되어있음

# In[ ]:


fig = px.histogram(data, x='platelets', color = 'DEATH_EVENT', marginal = "box", hover_data = data.columns, title = '혈소판 과 Death Event 분포',
                   labels = {'platelets':'혈소판'},
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[ ]:


fig = px.histogram(data, x='serum_sodium', color = 'DEATH_EVENT', marginal = "box", hover_data = data.columns, title = '혈중 나트륨 과 Death Event 분포',
                   labels = {'creatinine_phosphokinase':'CPK'},
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# > 사망한 환자들의 혈당 나트륨 수치가 비교적 낮게 분포함

# In[ ]:




