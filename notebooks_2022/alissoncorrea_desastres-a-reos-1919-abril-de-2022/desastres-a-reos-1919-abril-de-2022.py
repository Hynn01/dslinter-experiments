#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# # Desastres aéreos (1919 - Abril de 2022)
# Esta análise de dados tem como objetivos obter insights sobre os acidentes e incidentes aeronáuticos ocorridos nesse período.

# In[ ]:


import os
print(os.listdir("../input"))


# # Importanto dados

# In[ ]:


accidents = pd.read_csv('../input/aviation-accidents-history1919-april-2022/aviation_accidents in countries - aviation_accidents.csv')


# In[ ]:


# Importando bibliotecas necessárias
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()
# Para Notebooks
init_notebook_mode(connected=True)

from seaborn import countplot
from matplotlib.pyplot import figure, show

def transformar_category(value):
    if value == 'A1':
        return 'Acidente perda de casco'
    if value == 'A2':
        return 'Acidente dano reparável'
    if value == 'C1':
        return 'Sabotagem/Abate perda de casco'
    if value == 'H2':
        return 'Sequestro dano reparável'
    if value == 'O1':
        return 'Fogo de solo/Sabotagem perda de casco'
    if value == 'U1':
        return 'Desconhecido perda de casco'
    if value == 'C2':
        return 'Sabotagem/Abate dano reparável'
    if value == 'O2':
        return 'Fogo de solo/Sabotagem dano reparável'
    if value == 'H1':
        return 'Sequestro perda de casco'
    if value == 'I2':
        return 'Incidente dano reparável'
    if value == 'I1':
        return 'Incidente perda de casco'

accidents['category_description'] = accidents['category'].map(transformar_category)
accidents = accidents.dropna()


# # Visualisando dados importados

# In[ ]:


accidents.head()


# # Alterando o nome das colunas

# In[ ]:


acidentes = accidents.copy();
acidentes.columns = ['país','data','aeronave_tipo','nome_marca','operador','fatilites','localização', 'categoria', 'descrição_categoria']
acidentes.columns
acidentes.head()


# # Quantidade de registros distintos em cada coluna

# In[ ]:


# Quantidade de registros distintos de cada coluna 
acidentes.nunique()


# # Classificação de ocorrências por categoria

# In[ ]:


plt.style.use("ggplot")
print(acidentes['categoria'].unique())

print(acidentes['categoria'].value_counts())

acidentes['categoria'].value_counts().plot(kind='pie', subplots=True, label="Classificação de Acidentes" ,figsize=(6, 6))


# # Classificação de ocorrências por categoria por país

# In[ ]:


# Quantidade de ocorrências por país 
pais = acidentes.groupby('país')['categoria'].count().sort_values(ascending=[False])
pais[:10]


# In[ ]:


# Quantidade de ocorrências por país em gráfico
plt.style.use("ggplot")
pais[:10].plot(kind='pie', subplots=True, label="Ocorrências por país" ,figsize=(7, 7))


# In[ ]:


#Visualizando melhor em um gráfico os 10 principais tipos de ocorrência
plt.style.use("ggplot")
a = acidentes['descrição_categoria'].value_counts()
a.head(10).plot(kind='barh', subplots=True, label="Descrição Categoria" ,figsize=(7, 7))
plt.xticks(rotation=80)
# Como podemos ver, Acidente com perda de casco é a principal ocorrência notificada.


# # Ocorrência por tipo de Aeronave

# In[ ]:


# Modelos de aviões 
modelo_aviao = acidentes['aeronave_tipo'].value_counts()
modelo_aviao[:10]


# In[ ]:


# Modelo de aviões em grafico
plt.style.use("ggplot")
sns.countplot(x='aeronave_tipo', data=acidentes, order = acidentes['aeronave_tipo'].value_counts()[:10].index)
plt.xticks(rotation=80)


# # Ocorrência por tipo de Aeronave no Brasil

# In[ ]:


# Modelos de aviões Brasil
acidentesBrasil = acidentes[acidentes['país']=='Brazil']
print(acidentesBrasil['aeronave_tipo'].value_counts()[:10])


# In[ ]:


# Modelo de aviões em grafico no Brasil
plt.style.use("ggplot")
sns.countplot(x='aeronave_tipo', data=acidentesBrasil, order = acidentesBrasil['aeronave_tipo'].value_counts()[:10].index)
plt.xticks(rotation=80)


# # Classificação de ocorrências por categoria no Brasil

# In[ ]:


# Categoria que mais aconteceram no brasil
plt.style.use("ggplot")
print(acidentesBrasil['categoria'].unique())

print(acidentesBrasil['categoria'].value_counts())

acidentesBrasil['categoria'].value_counts().plot(kind='pie', subplots=True, label="Classificação de Acidentes" ,figsize=(6, 6))


# In[ ]:


#Visualizando melhor em um gráfico os 10 principais tipos de ocorrência
plt.style.use("ggplot")
a = acidentesBrasil['descrição_categoria'].value_counts()
a.head(10).plot(kind='barh', subplots=True, label="Descrição Categoria" ,figsize=(7, 7))
plt.xticks(rotation=80)
# Como podemos ver, Acidente com perda de casco é a principal ocorrência notificada.


# # Listando as ocorrências no Brasil por Ano

# In[ ]:


acidentesBrasil['data']


# In[ ]:


import datetime as dt

acidentesBrasil['data'] = acidentesBrasil['data'].apply(lambda x: x.strip())

def trim_date(x):
    if len(x) >8 : return x[3:]
    else: return x

acidentesBrasil['data'] = acidentesBrasil['data'].apply(lambda x: trim_date(x))

def datefiller(x):
    
    if x == '??-???-????': return '01-Jan-1900'
    elif x[:6] == '??-???': return '01-Jan'+x[6:]
    elif x[:2] == '??': return '01'+x[2:]
    else: return x


acidentesBrasil['data'] = acidentesBrasil['data'].apply(lambda x:datefiller(x))


# In[ ]:


acidentesBrasil['Ano'] = acidentesBrasil['data'].str[4:].str.lower()


# In[ ]:


acidentesBrasil['Ano'].value_counts()[:20]


# In[ ]:


plt.style.use("ggplot")
a = acidentesBrasil['Ano'].value_counts()
a.head(20).plot(kind='barh', subplots=True, label="Ano" ,figsize=(7, 7))
plt.xticks(rotation=80)

