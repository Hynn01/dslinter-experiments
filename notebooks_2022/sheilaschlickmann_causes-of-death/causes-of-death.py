#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Importando o arquivo
data = pd.read_csv('../input/causes-of-death-our-world-in-data/20222703 Causes Of Death Clean Output V2.0.csv')


# In[ ]:


#Dropando algumas colunas desnecessárias, renomeando as colunas, retirando valores nulos e ordenando pelo maior numero de mortes em 2019
data = data.drop('Causes Full Description', axis=1)
data = data.drop('Code', axis=1)
data.columns = ['Causas','Mortes','Pais','Ano']
data.columns
data.head()
data = data.dropna()
filtro1 = data[(data.Ano==2019)].sort_values("Mortes", ascending = False)
filtro1


# In[ ]:


data.dtypes


# In[ ]:


#Filtrando somente o Brasil e o ano de 2019; Ordenando por numero de mortes.
filtro2 = data[(data.Pais=='Brazil') & (data.Ano==2019)].sort_values("Mortes", ascending = False)
filtro2


# In[ ]:


fig = px.bar(filtro2, x='Causas', y='Mortes',
             hover_data=['Causas', 'Mortes'], color='Causas',
             width=1200, height=1200, title="Principais causas de mortes no Brasil em 2019")
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


# In[ ]:


#Estatistica de mortes por doença cardiovascular
cardio= data[(data.Pais=='Brazil') & (data.Causas== 'Cardiovascular diseases')]
cardio.plot(x='Ano', y=['Mortes']);

#Estatistica de mortes por tumor
neoplasms= data[(data.Pais=='Brazil') & (data.Causas== 'Neoplasms')]
neoplasms.plot(x='Ano', y=['Mortes']);

#Infecções respiratórias inferiores
respiratory= data[(data.Pais=='Brazil') & (data.Causas== 'Lower respiratory infections')]
respiratory.plot(x='Ano', y=['Mortes']);

