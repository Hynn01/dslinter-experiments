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
# #### creatinine_phosphokinase : CPK 혹은 CK
# * 심장, 뇌, 골격근에서 발견되는 효소
# * 근육이 수축할때 사용
# * 심장 세포가 손상된 경우 혈중 CK농도 상승
# 
# #### ejection_fraction : 박출계수
# * 좋은 기능을 하는 심장은 각 맥박당 혈액의 최소 50% 분출해야함
# 
# #### platelets : 혈소판 
# * 혈소판이 과하게 활성화 될 경우 심부전 발생의 가장 중요한 지표인 심실 확장의 위험 증가
# 
# #### serum_creatinine : 혈청 크레아티닌
# * 근육에서 생성되는 노폐물, 근육량에 비례 ( 남자가 보통 수치가 높음)
# * 심부전으로 인해 증가
# * 정상 범위 0.5~1.4 mg/dL
# 
# #### serum_sodium : 혈중 나트륨 농도****
# * 심부전으로 인해 감소
# * 135 mmol/L (135 mEq/L)미만이면 저나트륨혈증

# # Data analysis & EDA

# In[ ]:


data.info()


# 나이를 나타내는 column인 **age**의 type이 **int64** 가 아닌 **float64**이다.

# In[ ]:


#age col이 float64로 나타나는 원인 찾기
array = np.array(data['age'])
print(array)


# 60.667 반올림 후 삽입

# In[ ]:


#age col - int64 type으로 변환
data['age'] = np.around(array).astype('int64')
data.info()


# In[ ]:


data.describe().T


# ### 평균값 분석
# * 연령대 : 약 60세
# * 박출계수(ejection_fraction) : 약 38% ( 정상수치 50%이상)
# * 크레아티닌(serum_creatinine) : 약 1.4mg/dL ( 정상수치 0.5 ~ 1.4)
# * 혈중 나트륨 농도(serum_sodium) : 약 137mmol/L ( 정상수치 135 이상)
# 
# > **심부전 환자들은 각 column에 대해 정상 수치범위를 벗어나 있거나, 이탈에 근접하였음.**

# # 1. 결측치

# In[ ]:


data.isnull().sum()


# > 결측치 없음 

# # 2. Heatmap

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(20,20)
sns.heatmap(data.corr(), square = True, annot = True)


# >**time**-**Death_event** : 심혈관 문제를 조기에 진단하여 적절한 시기에 치료를 받아 사망 가능성을 줄이는 것이 중요 (음의 관계)
# 
# >**serum_creatinine**-**Death_event** : 심부전이 발생하면 혈액 내의 혈청 크레아티닌이 증가함, Death Event와 양의 관계
# 
# >**Ejection_fraction** : 심장 수축시 혈액을 내뿜는 비율을 의미, 기본적으로 심장의 효율성이기 때문에 Death_event에 영향을 미침 (음의 관계)
# 
# >나이가 들수록 심장의 기능이 저하되는 역관계 패턴에서 알 수 있다, **하지만 age-Death_event의 수가 양수로 나옴**

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
# 
# > **creatinine_phosphokinase, platelets, serum_creatinine, serum_sodium** 에서 비교적 많은 이상치를 관찰할 수 있음

# # 5. Coluum 별 사망 환자 비교 분석

# ## 실수형 Column 이상치 분석/정상치 비교 : 사망 비율
# 
# <img width="491" alt="스크린샷 2022-05-02 오후 11 49 25" src="https://user-images.githubusercontent.com/81242086/166257634-d259fa68-033a-48d5-96a0-b579f099ad03.png">
# 
# * 이상점 기준 구하는 공식
# > * 낮은 이상치: Q1 - 1.5 * IQR 보다 작은 값 
# > * 높은 이상치: Q3 + 1.5 * IQR 보다 큰 값 
# > * Q1, Q3는 각각 25%, 75%에 해당하는 값
# > * IQR(사분위편차) = Q3 - Q1

# In[ ]:


#이상치와 정상치 분류한 후, 사망율 분석
outlier_list = ['creatinine_phosphokinase','platelets', 'serum_creatinine', 'serum_sodium']

import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

for col in outlier_list:
    q1=data[col].quantile(0.25)
    q2=data[col].quantile(0.5)
    q3=data[col].quantile(0.75)
    iqr=q3-q1
    condition_min=data[col] < q1 - 1.5*iqr
    condition_max=data[col] > q3 + 1.5*iqr
    
    condition = (data[col] <= q3 + 1.5*iqr) & (data[col] >= q1 - 1.5*iqr)
    if col=='platelets' :
        outlier_h = data[condition_max]
        outlier_l = data[condition_min]
        inlier = data[condition]
        death_out_h = len(outlier_h[outlier_h['DEATH_EVENT'] == 1]) / len(outlier_h) * 100
        death_out_l = len(outlier_l[outlier_l['DEATH_EVENT'] == 1]) / len(outlier_l) * 100
        death_in = len(inlier[inlier['DEATH_EVENT'] == 1]) / len(inlier) * 100
        
        value = [death_out_h,death_out_l,death_in]
        label = ['높은 이상치','낮은 이상치','정상치']
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Pie(values = value, labels = label,  title = col))
        fig.update_traces(textposition='inside', hole=.4, textinfo="label+percent", hoverinfo="label+percent")
        fig.show()
        
    else :
        outlier = data[condition_min + condition_max]
        inlier = data[condition]

        death_out = len(outlier[outlier['DEATH_EVENT'] == 1]) / len(outlier) * 100
        death_in = len(inlier[inlier['DEATH_EVENT'] == 1]) / len(inlier) * 100
        
        value = [death_out,death_in]
        label = ['이상치','정상치']
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Pie(values = value, labels = label,  title = col))
        fig.update_traces(textposition='inside', hole=.4, textinfo="label+percent", hoverinfo="label+percent")
        fig.show()


# Column 별 이상치 환자 분석을 한 결과  
# >* **혈중 나트륨 농도(serum_sodium)**가 낮을수록 사망 비율이 높아지고 
# 
# >* **혈소판의 양(platelets)**이 낮을수록 사망 비율이 높아진다.
# 
# >* **혈중 크레아티닌 농도(serum_creatinine)**가 높을 수록 사망 비율이 높아지는 것을 확인할 수 있다.  
# 
# >* 하지만 **혈중 CPK농도(creatinine_phosphokinase)**는 이상치 분석으로 사망 비율과 상관관계를 찾기 어렵다.

# ## 부울형 Column 비교 분석

# In[ ]:


outlier_list = ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking']

for col in outlier_list:
    
    T = data[data[col]==1]
    F = data[data[col]==0]
    
    death_T = len(T[T['DEATH_EVENT'] == 1]) / len(T) * 100
    death_F = len(F[F['DEATH_EVENT'] == 1]) / len(F) * 100
    
    value = [death_T,death_F]
    label = ['True','False']
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Pie(values = value, labels = label,  title = col))
    fig.update_traces(textposition='inside', hole=.4, textinfo="label+percent", hoverinfo="label+percent")
    fig.show()


# > 대부분의 column이 True와 False의 환자 사망 비율이 유사함.
# 
# > 빈혈(anaemia), 고혈압(high_blood_pressure) 증상이 있는 환자의 사망율이 다소 높긴 한다.
# 
# > 하지만 현재 데이터셋으로는 4개의 column들과 Death Event와의 뚜렷한 상관관계를 파악할 수 없음.

# # 6. 연령대 분석

# 연령 분포를 관찰하기 위해 일의 자릿수를 버리고, 10살 단위로 연령대를 구분함

# In[ ]:


#10살 단위로 연령대를 구분함
import collections

AGE = data['age']//10 * 10
AGE
collections.Counter(AGE)


# In[ ]:


#심부전 환자의 연령대 시각화
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

age_value = list(collections.Counter(AGE).values())
age_label = list(collections.Counter(AGE).keys())

fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Pie(values = age_value, labels = age_label,  title = '연령대'))
fig.update_traces(textposition='inside', hole=.4, textinfo="label+percent", hoverinfo="label+percent")
fig.show()


# > 60대와 50대가 가장 많이 분포함

# # 7. Death Event와 Column별 분석

# In[ ]:


fig = px.histogram(data, x='age', color = 'DEATH_EVENT', marginal = "box", hover_data = data.columns, title = 'Age 와 Death Event 분포',
                   labels = {'age':'AGE'},
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# > 사망한 환자들의 연령대가 비교적 높게 분포되어있음

# In[ ]:


fig = px.histogram(data, x='creatinine_phosphokinase', color = 'DEATH_EVENT', marginal = "box", hover_data = data.columns, title = '혈중 CPK 농도 와 Death Event 분포',
                   labels = {'creatinine_phosphokinase':'혈중 CPK 농도'},
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


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

# # Final. 결론

# ## Sunburst Charts in Plotly
# 
# > 계층 구조 데이터를 표시하는데 적합한 도표
# 
# > 하나의 고리가 어떤 구성요소로 이루어져있는지 효과적으로 보여줌

# In[ ]:


sun = data.groupby(['sex', 'diabetes', 'smoking', 'DEATH_EVENT'])['age'].count().reset_index()

sun.columns = [
    'sex', 
    'diabetes', 
    'smoking', 
    'DEATH_EVENT', 
    'count'
]

sun.loc[sun['sex'] == 0, 'sex'] = 'female'
sun.loc[sun['sex'] == 1, 'sex'] = 'male'
sun.loc[sun['smoking'] == 0, 'smoking'] = "doesn't smoke"
sun.loc[sun['smoking'] == 1, 'smoking'] = 'smoke'
sun.loc[sun['diabetes'] == 0, 'diabetes'] = "no diabetes"
sun.loc[sun['diabetes'] == 1, 'diabetes'] = 'diabetes'
sun.loc[sun['DEATH_EVENT'] == 0,'DEATH_EVENT'] = "ALIVE_EVENT"
sun.loc[sun['DEATH_EVENT'] == 1, 'DEATH_EVENT'] = 'DEATH_EVENT'

fig = px.sunburst(
    sun, 
    path=[
        'sex',
        'diabetes',
        'smoking', 
        'DEATH_EVENT'
    ], 
    values='count', 
    title='Sunburst chart for all patients',
    width=700,
    height=500
)

fig.show()


# ## 정리
# 
# - 사망한 환자들의 **평균 나이**는 그렇지 않은 환자보다 연령대가 높게 분포되어있음  
# 
# - 사망한 환자들의 **크레아티닌** 수치는 그렇지 않은 환자보다 수치가 높게 분포되어있음  
# 
# - 사망한 환자들의 **박출 계수**는 그렇지 않은 환자보다 계수가 낮게 분포되어있음  
# (사망한 환자들의 심장에서 뿜어내는 혈액의 비율이 낮게 분포되어있음)  
# 
# - 사망한 환자들의 **혈소판의 양**은 그렇지 않은 환자보다 계수가 낮게 분포되어있음
# 
# - 사망한 환자들의 **혈당 나트륨** 수치는 그렇지 않은 환자보다 수치가 낮게 분포되어있음 
# 
# - **혈중 CPK 농도**, **빈혈**, **당뇨**, **흡연**, **고혈압**은 상관관계를 찾을 수 없음

# ## 결론
# 
# 주어진 데이터에서 유추할수 있는 것은 다음과 같다.
# * 환자의 **나이**, **크레아티닌 수치**가 높거나 **혈소판의 양**, **박출 계수**, **혈당 나트륨 수치**가 낮다면 사망확률이 높다.
# * 조사한 바로는 **혈중 CPK 농도**, **빈혈**, **당뇨**, **흡연**, **고혈압**은 사망과 상관관계가 있다고 하였으나, 현 데이터셋에서는 데이터의 수가 많지 않아 단정하기 어렵다.
# 
