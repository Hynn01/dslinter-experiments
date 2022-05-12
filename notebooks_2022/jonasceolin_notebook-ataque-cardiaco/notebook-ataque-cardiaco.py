#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,classification_report,plot_confusion_matrix,confusion_matrix
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,PowerTransformer,FunctionTransformer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import plotly 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly import tools


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# ------------------ Meu código daqui para baixo -------------------------------

# Importa o dataset
data = pd.read_csv("../input/health-care-data-set-on-heart-attack-possibility/heart.csv")


# In[ ]:


# informações
data.head()


# In[ ]:


# verifica tipo de dados
data.dtypes


# In[ ]:


# Verificar nulos
data.info()


# In[ ]:


# checa linhas duplicadas
print("Linhas duplicadas qtd:", data.duplicated().sum())

# Tem apenas 1 linha duplicada no dataset


# In[ ]:


# elimina linha duplicada 
data=data.drop_duplicates(keep="first")


# In[ ]:


# Cria o Gráfico
plt.figure(figsize=(14,8))

sns.kdeplot(data = data , x = 'age' ,hue='sex')
plt.show()

# Sex: 0= feminino; 1= masculino 
# Objetivo: Vizualiar onde se encontra o pico de mortes por ataque cardiaco em cada genero

