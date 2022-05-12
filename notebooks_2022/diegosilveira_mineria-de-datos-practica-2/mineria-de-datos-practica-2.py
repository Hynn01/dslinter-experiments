#!/usr/bin/env python
# coding: utf-8

# # Máster en Ciencia de Datos. Curso 2021-2022
# 
# # Minería de Datos
# 
# ## Prática 2. Técnicas de Análisis
# 
# Autores:
# - Laura García
# - Diego Silveira
# 
# ## Descripción
# 
# > **ARREGLAR**
# A partir de los datos obtenidos de una pulsera de actividad Garmin, esta práctica pretende, en primer lugar, realizar un preprocesamiento de los datos que permita: seleccionar qué características de los datos son relevantes y cuáles no, así como imputar los valores ausentes de las que sí son relevantes. En segundo lugar, se pretende emplear varios escaladores, así como una variedad de algoritmos de: regresión, clasificación, *ensembles* y *clustering* que permitan calcular una serie de métricas que ayuden a interpretar los datos y conocer si los datos contienen sesgos. 
# 
# 
# Por último, comparar las métricas de los algoritmos de un mismo tipo y decidir cuál de ellos se ajusta mejor a nuestro conjunto de datos.
# 
# 
# 1. Preprocesamiento del dataset (limpieza, imputación, etc)
# 2. Uso de técnicas de Regresión, Clasificación, Ensembles y Clustering.
# 3. Cálculo de métricas (matriz de confusión, Curva ROC, etc), comparación de resultados e interpretación de los valores obtenidos, así como de los sesgos que se producen.
# 
# Fuente: [Garmin Connect Data Analysis
# ](https://www.kaggle.com/code/kmader/garmin-connect-data-analysis/data)

# ## 1. Preprocesamiento de los datos
# 
# En este primer apartado vamos a realizar un preprocesamiento de los datos. Este consistirá en seleccionar y extraer características que aporten información relevante, imputar los valores ausentes de las distintas características, definir la variable objetivo y discretizar sus valores, así como asegurarse que las clases de esta variable objetivo se encuentran balanceadas.

# ### 1.1 Importamos las librerías necesarias
# 
# Primero, importamos las librerías que necesitatremos para realizar el preprocesamiento de los datos.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from itertools import cycle, islice
from scipy import stats
import random
from typing import Optional
import time

# Importamos el modelo de regresión Decision Tree
from sklearn.tree import DecisionTreeRegressor

# Importamos las métricas R2, MAE y MSE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Importamos para aplicar la técnica de Cross Validation
from sklearn.model_selection import KFold

# Importamos para los ensembles
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,StackingClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_curve, auc,confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize,StandardScaler

# Importamos para el clustering
from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture

from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings('ignore')


# ### 1.2. Definimos una plantilla
# 
# En este apartado hemos creado una plantilla con los estilos que se emplearán para todas las representaciones gráficas.

# In[ ]:


pio.templates["Tema"] = go.layout.Template(
   layout = {

        'title':
            {'font': {'size':30 }, "x":0.5
            },
        'font': { 'size':18}
    },
    
    data = {
        'bar': [go.Bar(texttemplate = '%{value:.2s}',
                       textposition='outside',
                       textfont={'size': 23  }
                       )]
    }
)

pio.templates.default = "plotly_white+Tema"

colores = ["#8f58bf","#55c95c","#2fbaaf","#ad2234","#899D78","#adc244","#4d5294","#dd5234","#9d3254","#64bd20","#845550"]
colors = ['#1F77B4', '#FF7F0E']


# ### 1.3. Importamos los datos
# 
# Leemos el archivo CSV con los datos de actividad de una pulsera de Garmin del [siguiente enlace](https://www.kaggle.com/code/kmader/garmin-connect-data-analysis/data), cargamos los datos en el DataFrame `user_df`, y mostramos las primeras 5 filas del DataFrame mediante la función `head()`.

# In[ ]:


user_df = pd.read_csv("../input/garmin-connect-data-analysis/user_data.csv",
                      parse_dates=['calendarDate', 'restingHeartRateTimestamp', 'wellnessEndTimeGmt',
                                   'wellnessEndTimeLocal', 'wellnessStartTimeGmt', 'wellnessStartTimeLocal'])
user_df.head()


# ### 1.4. Dimensiones de los datos
# 
# Tras cargar el archivo CSV y visualizarlo, empleamos la propiedad `shape` para obtener el número de instancias y de características de los datos.

# In[ ]:


# Mostramos las dimensiones originales del dataset
print("Dimensiones del dataset:", user_df.shape)
print("Número de instancias:", user_df.shape[0])
print("Número de características:", user_df.shape[1])


# ### 1.5. Tipos de datos
# 
# En este punto analizamos el tipo de los datos del dataset para ver la variedad que existe.

# In[ ]:


tipos_datos = user_df.dtypes.value_counts()
tipos_datos = tipos_datos.sort_index()
labels = tipos_datos.index.astype(str).tolist()
values = tipos_datos.tolist()

fig = px.pie(names=labels, values=values,
             title="Tipos de datos: float64 es el más común", color_discrete_sequence=colores)
fig.update_layout(legend_title_text='Tipo')
fig.show()


# Como se observa en el gráfico sectorial anterior, en este conjunto de datos existen 5 tipos: `object`, `bool`, `int64`, `datetime64[ns]` y `float64`, siendo este último el tipo de datos más frecuente, pues un 68.9% de las características son de este tipo.

# ### 1.6. Cantidad y proporción de valores ausentes
# 
# En este apartado se analiza la cantidad y la proporción de valores ausentes que posee cada una de las características del dataset.

# In[ ]:


total_na = user_df.isna().sum()
prop_na = total_na / len(user_df)

for col in user_df.columns:
    print("Columna '{}'. Valores nulos: {} ({:.2f} %)".format(col, total_na[col], prop_na[col] * 100))


# Puesto que el dataset original posee un gran número de características, resulta complejo representar la proporción de valores ausentes que posee cada una de las columnas. Como alternativa, tomamos un umbral de 0.4 y mostramos en un gráfico sectorial la cantidad de características cuya proporción de valores ausentes es menor o igual a un 40% y la cantidad de características cuya proporción es mayor.

# In[ ]:


men_ig_umb = prop_na <= 0.4
labels = men_ig_umb.value_counts().index.astype(str).tolist()
values = men_ig_umb.value_counts().tolist()

fig = px.pie(names=labels, values=values, title='El 37.8% de las características tienen una proporción de nulos > 40%')
fig.update_layout(legend_title_text='Proporción de nulos <= 40%')
fig.show()


# Según se observa en el gráfico sectorial anterior, si establecemos como umbral una proporción de valores ausentes del 40%, nos encontramos con que el 37.8% de las características de nuestro dataset superan este umbral. Dicho de otra forma, si eliminamos todas aquellas características que posean una proporción de valores ausentes superior al 40%, entonces eliminamos 17 de las 45 características que posee nuestro dataset, quedándonos con las otras 28.
# 
# Puesto que una proporción de valores ausentes superior al 40% supone que casi la mitad de los registros de una característica tengan que ser imputados, consideramos que este es un buen criterio para seleccionar aquellas columnas que son verdaderamente útiles. Por ello, hemos empleado este como un criterio para eliminar características con escasa utilidad.

# In[ ]:


# Nos quedamos con las características que posean una proporción de valores ausentes <= 40%
user_df = user_df.loc[:, prop_na <= 0.4]
user_df.head()


# ### 1.7. Eliminación de características inútiles
# 
# Aunque hemos reducido la dimensión de los datos, todavía quedan muchas características que no son de utilidad, por tanto, hemos eliminado algunas más en base a nuestro propio criterio.
# 
# En primer lugar, comparamos las columnas `calendarDate`, `wellnessStartTimeLocal` y `wellnessEndTimeLocal` con las columnas `wellnessStartTimeGmt` y `wellnessEndTimeGmt`, respectivamente.

# In[ ]:


user_df[['calendarDate', 'wellnessStartTimeLocal', 'wellnessStartTimeGmt', 'wellnessEndTimeLocal', 'wellnessEndTimeGmt']]


# Como se aprecia en la celda anterior, todas estas columnas representan fechas utilizando distintos formatos. Por tanto, la información que proporcionan es semejante. Por ello, hemos decidido quedarnos con la columna `calendarDate` y eliminar las demás.

# In[ ]:


# Eliminamos las columnas wellnessStartTimeLocal, wellnessStartTimeGmt, wellnessEndTimeLocal y wellnessEndTimeGmt
user_df.drop(['wellnessStartTimeLocal', 'wellnessStartTimeGmt',
              'wellnessEndTimeLocal', 'wellnessEndTimeGmt'],
             axis=1, inplace=True)


# A continuación, analizamos las columnas `rulePk` y `uuid`.

# In[ ]:


user_df[['rulePk', 'uuid']]
print("Varianza rulePk:", user_df['rulePk'].var())


# Tal y como se observa en la celda anterior, la columna `rulePk` siempre tiene el valor 1 y no varía (varianza = 0). Por otra parte, la columna `uuid` es una cadena de texto que representa un identificador, por lo tanto, ninguna de las dos aporta información relevante para medir la actividad del usuario.

# In[ ]:


# Eliminamos las columnas rulePk y uuid
user_df.drop(['rulePk', 'uuid'], axis=1, inplace=True)


# A continuación, analizamos las columnas `includesActivityData`, `includesCalorieConsumedData`, e `includesWellnessData`.

# In[ ]:


user_df[['includesActivityData', 'includesCalorieConsumedData', 'includesWellnessData']]


# Las tres columnas indicadas en la celda anterior son de tipo `bool` e indican si el registro actual incluye la información sobre un campo determinado o no. Esto tampoco aporta información relevante para medir la actividad del usuario, por tanto, podemos descartarlas.

# In[ ]:


# Eliminamos las columnas includesActivityData, includesCalorieConsumedData, includesWellnessData
user_df.drop(['includesActivityData', 'includesCalorieConsumedData', 'includesWellnessData'], axis=1, inplace=True)


# Seguidamente, vamos a analizar las columnas `version` y `userProfilePK` para ver si es de utilidad conservarlas o si es mejor eliminarlas.

# In[ ]:


user_df[['version', 'userProfilePK']]
print("Varianza userProfilePK:", user_df['userProfilePK'].var())


# Los valores de la columna `version` no aportan información relevante. Por otro lado, los valores la columna `userProfilePK` son constantes y su varianza es igual a 0, por tanto, podemos descartar ambas columnas.

# In[ ]:


# Eliminamos las columnas version y userProfilePK
user_df.drop(['version', 'userProfilePK'], axis=1, inplace=True)


# Respecto a las columnas `netCalorieGoal` y `remainingKilocalories`, la primera hace referencia al objetivo de calorías netas, pero esta no sirve si no se conoce las calorías ingeridas. Por otra parte, la segunda columna analizada indica las calorías restantes, pero esta no nos sirve si no tenemos las calorías netas. Por tanto, hemos decididos eliminarlas también.

# In[ ]:


# Eliminamos las columnas netCalorieGoal y remainingKilocalories
user_df.drop(['netCalorieGoal', 'remainingKilocalories'], axis=1, inplace=True)


# Por último, vamos a analizar las columnas `dailyStepGoal` y `durationInMilliseconds`.

# In[ ]:


user_df[['dailyStepGoal', 'durationInMilliseconds']]


# La primera columna contiene los objetivos de pasos establecidos diariamente, por lo tanto, no afecta a la actividad de la persona y no aporta información relevante. Por otra parte, la información de la columna `durationInMilliseconds` es similar a la columna `activeSeconds` pero con distinto formato, con lo cual, es una columna redundante. En conclusión, podemos eliminar ambas columnas, pues no aportan información de interés.

# In[ ]:


# Eliminamos las columnas durationInMilliseconds y activeSeconds
user_df.drop(['dailyStepGoal', 'durationInMilliseconds'], axis=1, inplace=True)


# Una vez eliminadas todas las características que no aportan información, volvemos a mostrar las dimensiones del dataset.

# In[ ]:


# Mostramos las dimensiones del dataset después de eliminar las características que no aportan información
print("Dimensiones del dataset:", user_df.shape)
print("Número de instancias:", user_df.shape[0])
print("Número de características:", user_df.shape[1])


# Después de analizar las características del dataset, nos hemos quedado con 13 de las 28 que teníamos después de eliminar aquellas con gran proporción de valores nulos. Esto supone que hemos eliminado un 53.57% de las características. Además, de las 45 columnas que tenía el dataset original, sólo nos hemos quedado con el 28.89% de las mismas.

# ### 1.8. Imputación de valores nulos
# 
# Aunque hemos eliminado una gran cantidad de características que no aportaban información, todavía quedan características con valores nulos. Por ello, primeramente vamos a analizar qué columnas todavía contienen valores nulos.

# In[ ]:


total_na = user_df.isna().sum()
prop_na = total_na / len(user_df)

for col in user_df.columns:
    print("Columna '{}'. Valores nulos: {} ({:.2f} %)".format(col, total_na[col], prop_na[col] * 100))


# Como se observa en la celda anterior, la grann mayoría de las columnas todavía contienen valores nulos. Esto es un inconveniente, pues los modelos de *Machine Learning* no permiten valores nulos. Para resolver este inconveniente hemos empleado la técnica de imputación. En concreto, hemos empleado dos estrategias distintas para imputar los valores nulos:
# 
# - La columna `burnedKilocalories` la hemos imputado con el valor constante 0.
# - El resto de columnas las hemos imputado mediante un imputador KNN con `n_neighbors=5`.

# In[ ]:


# Importamos el imputador KNN
from sklearn.impute import KNNImputer

# Imputamos los valores NaN de la columna burnedKilocalories con el valor constante 0
user_df['burnedKilocalories'].fillna(value=0, inplace=True)

# Seleccionamos sólo aquellas columnas que son de tipo numérico (int64 o float64)
num_cols = user_df.select_dtypes('number')

# Creamos el imputador KNN con 5 vecinos y pesos uniformes
knnImp = KNNImputer(n_neighbors=5, weights='uniform')

# Imputamos los valores nulos de las columnas de tipo numérico
imp_cols = knnImp.fit_transform(num_cols)

# Los valores imputados se almacenan en un array auxiliar,
# por tanto, hay que crear un nuevo DataFrame
dfAux = pd.DataFrame(imp_cols, columns=num_cols.columns)

# A este DataFrame auxiliar hay que añadirle las columnas no numéricas
dfAux.insert(0, 'calendarDate', user_df.calendarDate.values)

# Asignamos a nuestro antiguo DataFrame el contenido del nuevo
user_df = dfAux

# Mostramos el contenido del nuevo DataFrame
user_df.head()


# Tras visualizar los datos, vemos que la columna `calendarDate` no se encuentra ordenada. Para corregirlo, procedemos a ordenar los datos por la columna `calendarDate` de forma ascendente.

# In[ ]:


# Ordenamos los datos por la columna calendarDate de forma ascendente
user_df = user_df.sort_values(by='calendarDate')

# Reseteamos los índices del DataFrame
user_df.reset_index(drop=True, inplace=True)

# Mostramos el contenido del DataFrame tras ordenar los datos
user_df.head()


# ### 1.9. Extracción de características
# 
# En esta sección partiremos de la columna `calendarDate` para extraer como características las columnas `day`, `month`, `year` y `weekday`, por último, eliminaremos la columna original `calendarDate`.

# In[ ]:


# Extraemos el año de la columna calendarDate
user_df['year'] = user_df['calendarDate'].dt.year

# Extraemos el mes de la columna calendarDate
user_df['month'] = user_df['calendarDate'].dt.month

# Extraemos el día de la columna calendarDate
user_df['day'] = user_df['calendarDate'].dt.day

# Extraemos el día de la semana de la columna calendarDate
user_df['weekday'] = user_df['calendarDate'].dt.dayofweek

# Eliminamos la columna original
user_df.drop(['calendarDate'], axis=1, inplace=True)


# In[ ]:


# Mostramos el contenido del DataFrame tras la extracción de características
user_df.head()


# ### 1.10. Creación de la columna objetivo
# 
# En este apartado vamos a crear la columna `Perfil`, la cual será una variable categórica multilabel que podrá tomar los siguientes valores:
# 
#  - `0`. Indica que la actividad física de ese día ha sido **Baja**.
#  
#  - `1`. Indica que la actividad física de ese día ha sido **Media**.
#  
#  - `2`. Indica que la actividad física de ese día ha sido **Alta**.
# 
# Esta nueva variable es la suma ponderada de las variables `totalKilocalories`, `totalSteps` y `activeSeconds`, siendo sus ponderaciones 0.5, 0.3 y 0.2, respectivamente. De esta forma, el cálculo de la variable `Perfil` será el siguiente:
# 
# $$Perfil = 0.5 · totalKilocalories + 0.3 · totalSteps + 0.2 · activeSeconds$$
# 
# > **Nota**: Aunque pueda parecer que se trata de una variable numérica, es de tipo categórico. El único motivo por el cuál se representa con un valor numérico del 0 al 2 y no con una palabra es que así se puede usar la misma etiqueta tanto para clasificación como para regresión.

# In[ ]:


# Creación de la variable Perfil
user_df['Perfil'] = 0.5 * user_df['totalKilocalories'] + 0.3 * user_df['totalSteps'] + 0.2 * user_df['activeSeconds']


# Una vez creada la variable `Perfil` esta tendrá valores de tipo continua, por lo que habrá que discretizar la variable. Para ello, primero debemos extraer tantos intervalos como clases queramos que tenga la variable discreta. En nuestro caso, queremos obtener 3 clases, por lo que tendremos que dividir el conjunto en 3 intervalos. Además, estos intervalos tendrán que asegurarse que las clases no están desbalanceadas.

# In[ ]:


# Creamos 3 intervalos con un número de muestras balanceado
res, bins = pd.qcut(user_df.Perfil, 3, precision=2, retbins=True)

perfiles = []

# Función que discretiza los valores de la variable Perfil
def convert_values():
    for perfil in user_df.Perfil:
        if perfil >= bins[0] and perfil <= bins[1]:
            perfil = 0
        elif perfil >= bins[1] and perfil <= bins[2]:
            perfil = 1
        else:
            perfil = 2
            
        perfiles.append(perfil)
    
    return perfiles

user_df['Perfil'] = convert_values()


# ### 1.11. Distribución de las clases de la variable objetivo
# 
# Para comprobar que las clases están balanceadas, representamos mediante un gráfico de barras el número de instancias de cada una de las clases de la variable `Perfil`.

# In[ ]:


# Representamos la distribución de las instancias de las distintas clases
intervalos = user_df['Perfil'].value_counts().sort_index()

values = intervalos.values.tolist()
labels = intervalos.index.astype(str).tolist()

fig = px.bar(res, x=labels, y=values, height=400, width=600, text_auto=True,
             labels={ # replaces default labels by column name
                 "x": 'Clases', "y": 'Número instancias'})
fig.update_layout(title_text="Distribución equilibrada de las clases")
fig.show()


# ## 2. Técnicas de Regresión
# 
# En este apartado vamos a ...

# ### 2.1. División del conjunto de datos 
# !!!!Y SI PONEMOS ESTE APARTADO FUERA EN EL PREPROCESADO O EN UN PUNTO A PARTE YA QUE SE VA A USAR EN TODOS LOS ALGORITMOS NO SOLO EN ESTE!!!
# 
# En esta sección vamos a generar los subconjuntos de datos de entrenamiento y testeo.

# Primero, vamos a separar el conjunto de datos en las variables `X` e `y`, donde la variable `X` contendrá todas las características menos la variable objetivo, y la variable `y` contendrá sólo la variable objetivo.

# In[ ]:


X = user_df.drop(['Perfil'], axis=1)
y = user_df['Perfil']


# Después, vamos a dividir el conjunto de datos en los subconjuntos de entrenamiento y testeo, de esta forma, podremos entrenar los distintos modelos de regresión con un subconjunto de datos y medir su capacidad de predicción con el otro subconjunto. El subconjunto de entrenamiento representará el 75% de los datos, mientras que el de testeo representará el 25% restante.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25, random_state=42)


# Una vez generados los distintos subconjuntos, mostramos las dimensiones de estos.

# In[ ]:


print("Dimensiones X_train:", X_train.shape)
print("Dimensiones X_test:", X_test.shape)
print("Dimensiones y_train:", y_train.shape)
print("Dimensiones y_test:", y_test.shape)


# ### 2.2. Selección del valor óptimo del parámetro `max_depth`
# Tras generar los distintos subconjuntos, vamos a evaluar el desempeño de un árbol de decisión con distintos valores del parámetro `max_depth` mediante las métricas **R2**, **MAE** y **MSE**.

# In[ ]:


# Evalúa el desempeño del árbol de decisión en los subconjuntos de
# entrenamiento y testeo con diferentes valores del parámetro max_depth
train_scores_r2, test_scores_r2 = [], []
train_scores_mae, test_scores_mae = [], []
train_scores_mse, test_scores_mse = [], []

# Valores del parámetro max_depth
values = list(range(1, 21))

# Evaluamos el árbol de decisión para cada uno de los valores
for value in values:
    # Creamos el modelo
    model = DecisionTreeRegressor(max_depth=value)
    
    # Entrenamos el modelo
    model.fit(X_train, y_train)
    
    # Evaluamos los conjuntos de datos de entrenamiento y testeo
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Métrica R2 para los conjuntos de entrenamiento y testeo
    train_r2 = round(r2_score(y_train, train_predict), 3)
    test_r2 = round(r2_score(y_test, test_predict), 3)
    train_scores_r2.append(train_r2)
    test_scores_r2.append(test_r2)
    
    # Métrica MAE para los conjuntos de entrenamiento y testeo
    train_mae = round(mean_absolute_error(y_train, train_predict), 3)
    test_mae = round(mean_absolute_error(y_test, test_predict), 3)
    train_scores_mae.append(train_mae)
    test_scores_mae.append(test_mae)
    
    # Métrica MSE para los conjuntos de entrenamiento y testeo
    train_mse = round(mean_squared_error(y_train, train_predict), 3)
    test_mse = round(mean_squared_error(y_test, test_predict), 3)
    train_scores_mse.append(train_mse)
    test_scores_mse.append(test_mse)


# Una vez calculadas las distintas métricas, las visualizamos en varias gráficas. El objetivo de las siguientes gráficas es determinar de forma visual el valor óptimo del parámetro `max_depth` sin llegar al sobreentrenamiento (*overfitting*). Para determinar el valor óptimo, debemos visualizar las curvas de entrenamiento y testeo y seleccionar el último valor antes de que las curvas empiecen a diverger.

# In[ ]:


fig = make_subplots(rows=3, cols=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_r2,
                            mode='lines+markers', name='R2 Train',
                            line_color=colors[0]),
                 row=1, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_r2,
                            mode='lines+markers', name='R2 Test',
                            line_color=colors[1]),
                 row=1, col=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_mae,
                            mode='lines+markers', name='MAE Train',
                            line_color=colors[0]),
                 row=2, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_mae,
                            mode='lines+markers', name='MAE Test',
                            line_color=colors[1]),
                 row=2, col=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_mse,
                            mode='lines+markers', name='MSE Train',
                            line_color=colors[0]),
                 row=3, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_mse,
                            mode='lines+markers', name='MSE Test',
                            line_color=colors[1]),
                 row=3, col=1)

fig.update_xaxes(title_text='max_depth', title_font={"size": 14})
fig.update_layout(height=700, width=600,
                  title={'x': 0.5, 'y': 0.9, 'font': {'size': 16},
                         'text': 'Métricas para distintos valores de max_depth',
                         'xanchor': 'center', 'yanchor': 'top'},
                  yaxis=dict(title="R2", titlefont={"size": 14}),
                  yaxis2=dict(title="MAE", titlefont={"size": 14}),
                  yaxis3=dict(title="MSE", titlefont={"size": 14}))
fig.show()


# Tal y como se observa en las gráficas de las 3 métricas, el valor óptimo es `max_depth=4`, pues a partir de este valor los resultados de entrenamiento y testeo empiezan a distanciarse, lo cual indica que se está produciendo sobreentrenamiento.

# ### 2.3. Selección del valor óptimo del parámetro `min_samples_split`
# Una vez que hemos determinado el valor óptimo del parámetro `max_depth`, volvemos a repetir el procedimiento para determinar el valor óptimo del parámetro `min_samples_split` mediante las métricas **R2**, **MAE** y **MSE**.

# In[ ]:


# Evalúa el desempeño del árbol de decisión en los subconjuntos de
# entrenamiento y testeo con diferentes valores del parámetro min_samples_split
train_scores_r2, test_scores_r2 = [], []
train_scores_mae, test_scores_mae = [], []
train_scores_mse, test_scores_mse = [], []

# Valores del parámetro max_depth
values = list(range(2, 21))

# Evaluamos el árbol de decisión para cada uno de los valores
for value in values:
    # Creamos el modelo
    model = DecisionTreeRegressor(max_depth=4, min_samples_split=value)
    
    # Entrenamos el modelo
    model.fit(X_train, y_train)
    
    # Evaluamos los conjuntos de datos de entrenamiento y testeo
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Métrica R2 para los conjuntos de entrenamiento y testeo
    train_r2 = round(r2_score(y_train, train_predict), 3)
    test_r2 = round(r2_score(y_test, test_predict), 3)
    train_scores_r2.append(train_r2)
    test_scores_r2.append(test_r2)
    
    # Métrica MAE para los conjuntos de entrenamiento y testeo
    train_mae = round(mean_absolute_error(y_train, train_predict), 3)
    test_mae = round(mean_absolute_error(y_test, test_predict), 3)
    train_scores_mae.append(train_mae)
    test_scores_mae.append(test_mae)
    
    # Métrica MSE para los conjuntos de entrenamiento y testeo
    train_mse = round(mean_squared_error(y_train, train_predict), 3)
    test_mse = round(mean_squared_error(y_test, test_predict), 3)
    train_scores_mse.append(train_mse)
    test_scores_mse.append(test_mse)


# Tras calcular las distintas métricas, volvemos a visualizarlas en varias gráficas. En esta ocasión, el objetivo es determinar de forma visual el valor óptimo del parámetro `min_samples_split` sin llegar al sobreentrenamiento (*overfitting*).

# In[ ]:


fig = make_subplots(rows=3, cols=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_r2,
                            mode='lines+markers', name='R2 Train',
                            line_color=colors[0]),
                 row=1, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_r2,
                            mode='lines+markers', name='R2 Test',
                            line_color=colors[1]),
                 row=1, col=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_mae,
                            mode='lines+markers', name='MAE Train',
                            line_color=colors[0]),
                 row=2, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_mae,
                            mode='lines+markers', name='MAE Test',
                            line_color=colors[1]),
                 row=2, col=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_mse,
                            mode='lines+markers', name='MSE Train',
                            line_color=colors[0]),
                 row=3, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_mse,
                            mode='lines+markers', name='MSE Test',
                            line_color=colors[1]),
                 row=3, col=1)

fig.update_xaxes(title_text='min_samples_split', title_font={"size": 14})
fig.update_layout(height=700, width=600,
                  title={'x': 0.5, 'y': 0.9, 'font': {'size': 16},
                         'text': 'Métricas para distintos valores de min_samples_split',
                         'xanchor': 'center', 'yanchor': 'top'},
                  yaxis=dict(title="R2", titlefont={"size": 14}),
                  yaxis2=dict(title="MAE", titlefont={"size": 14}),
                  yaxis3=dict(title="MSE", titlefont={"size": 14}))
fig.show()


# En esta ocasión, se observa que los valores de entrenamiento y testeo son muy diferentes y nunca convergen. Por este motivo, hemos determinado que el valor óptimo es `min_samples_split=2`. En este caso, no resulta útil seleccionar un valor mayor, pues esto sólo añadiría complejidad, y produciría *overfitting* en el modelo de predicción.

# ### 2.3. Selección del valor óptimo del parámetro `min_samples_leaf`
# Una vez que hemos determinado el valor óptimo del parámetro `min_samples_split`, volvemos a repetir el procedimiento para determinar el valor óptimo del parámetro `min_samples_leaf` mediante las métricas **R2**, **MAE** y **MSE**.

# In[ ]:


# Evalúa el desempeño del árbol de decisión en los subconjuntos de
# entrenamiento y testeo con diferentes valores del parámetro min_samples_split
train_scores_r2, test_scores_r2 = [], []
train_scores_mae, test_scores_mae = [], []
train_scores_mse, test_scores_mse = [], []

# Valores del parámetro max_depth
values = list(range(1, 21))

# Evaluamos el árbol de decisión para cada uno de los valores
for value in values:
    # Creamos el modelo
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=value)
    
    # Entrenamos el modelo
    model.fit(X_train, y_train)
    
    # Evaluamos los conjuntos de datos de entrenamiento y testeo
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Métrica R2 para los conjuntos de entrenamiento y testeo
    train_r2 = round(r2_score(y_train, train_predict), 3)
    test_r2 = round(r2_score(y_test, test_predict), 3)
    train_scores_r2.append(train_r2)
    test_scores_r2.append(test_r2)
    
    # Métrica MAE para los conjuntos de entrenamiento y testeo
    train_mae = round(mean_absolute_error(y_train, train_predict), 3)
    test_mae = round(mean_absolute_error(y_test, test_predict), 3)
    train_scores_mae.append(train_mae)
    test_scores_mae.append(test_mae)
    
    # Métrica MSE para los conjuntos de entrenamiento y testeo
    train_mse = round(mean_squared_error(y_train, train_predict), 3)
    test_mse = round(mean_squared_error(y_test, test_predict), 3)
    train_scores_mse.append(train_mse)
    test_scores_mse.append(test_mse)


# Tras calcular las distintas métricas, volvemos a visualizarlas en varias gráficas. En esta ocasión, el objetivo es determinar de forma visual el valor óptimo del parámetro `min_samples_leaf` sin llegar al sobreentrenamiento (*overfitting*).

# In[ ]:


fig = make_subplots(rows=3, cols=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_r2,
                            mode='lines+markers', name='R2 Train',
                            line_color=colors[0]),
                 row=1, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_r2,
                            mode='lines+markers', name='R2 Test',
                            line_color=colors[1]),
                 row=1, col=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_mae,
                            mode='lines+markers', name='MAE Train',
                            line_color=colors[0]),
                 row=2, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_mae,
                            mode='lines+markers', name='MAE Test',
                            line_color=colors[1]),
                 row=2, col=1)

fig.append_trace(go.Scatter(x=values, y=train_scores_mse,
                            mode='lines+markers', name='MSE Train',
                            line_color=colors[0]),
                 row=3, col=1)

fig.append_trace(go.Scatter(x=values, y=test_scores_mse,
                            mode='lines+markers', name='MSE Test',
                            line_color=colors[1]),
                 row=3, col=1)

fig.update_xaxes(title_text='min_samples_leaf', title_font={"size": 14})
fig.update_layout(height=700, width=600,
                  title={'x': 0.5, 'y': 0.9, 'font': {'size': 16},
                         'text': 'Métricas para distintos valores de min_samples_leaf',
                         'xanchor': 'center', 'yanchor': 'top'},
                  yaxis=dict(title="R2", titlefont={"size": 14}),
                  yaxis2=dict(title="MAE", titlefont={"size": 14}),
                  yaxis3=dict(title="MSE", titlefont={"size": 14}))
fig.show()


# Al igual que sucedió con el parámetro `min_samples_split`, se puede ver que los valores de entrenamiento y testeo del parámetro `min_samples_leaf` son muy diferentes y nunca convergen. Por este motivo, hemos seleccionado como valor óptimo `min_samples_leaf=1`. Como en el caso de `min_samples_split`, no es de utilidad elegir un valor mayor, pues esto sólo añade complejidad y produce *overfitting* en el modelo.

# ### 2.4. Selección de parámetros óptimos mediante GridSearchCV
# 
# ...

# ### 2.5. Comprobación de los parámetros óptimos elegidos mediante Cross Validation
# 
# Tras probar distintos valores con los parámetros `max_depth`, `min_samples_split` y `min_samples_leaf`, hemos determinado que los valores óptimos para nuestro conjunto de datos son los siguientes:
# 
#  - `max_depth=4`
#  - `min_samples_split=2`
#  - `min_samples_leaf=1`
# 
# Llegados a este punto, nos interesa saber si con estos parámetros nuestro modelo se comporta igual, independientemente del conjunto de datos de entrenamiento y testeo que le asignemos. Para ello, nos vamos a apoyar en la técnica de Cross Validation.

# In[ ]:


# Declaramos Cross Validation con 5 particiones (folds)
cv = KFold(n_splits=5, random_state=50, shuffle=True)

scores_r2 = []
scores_mae = []
scores_mse = []
models = []

# Para cada fold
# Variable acumuladora
mean_score = 0.0

for train_index, test_index in cv.split(X):
    # Seleccionamos los datos
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    
    # Declaramos el modelo
    model = DecisionTreeRegressor(max_depth=4,
                                  min_samples_split=2,
                                  min_samples_leaf=1)
    # Entrenamos el modelo
    model = model.fit(X_train, y_train)
    
    # Guardamos el modelo en una lista
    models.append(model)
    
    # Evaluamos el modelo
    predict = model.predict(X_test)
    
    # Métrica R2
    r2 = round(r2_score(y_test, predict), 3)
    scores_r2.append(r2)
    
    # Métrica MAE
    mae = round(mean_absolute_error(y_test, predict), 3)
    scores_mae.append(mae)
    
    # Métrica MSE
    mse = round(mean_squared_error(y_test, predict), 3)
    scores_mse.append(mse)


# Una vez que hemos calculado las métricas para cada una de las 5 particiones, mostramos un gráfico de barras por cada métrica con la puntuación de cada partición.

# In[ ]:


fig = make_subplots(rows=1, cols=3)

labels=['1', '2', '3', '4', '5']

fig.append_trace(go.Bar(x=labels, y=scores_r2, name="R2 score",
                        textposition="none"), row=1, col=1)

fig.append_trace(go.Bar(x=labels, y=scores_mae, name="MAE score",
                        textposition="none"), row=1, col=2)

fig.append_trace(go.Bar(x=labels, y=scores_mse, name="MSE score",
                        textposition="none"), row=1, col=3)

fig.update_xaxes(title_text='Partición', title_font={"size": 14})
fig.update_layout(height=400, width=1200,
                  title={'x': 0.5, 'y': 0.9, 'font': {'size': 16},
                         'text': 'Métricas para cada partición de CV',
                         'xanchor': 'center', 'yanchor': 'top'},
                  yaxis=dict(title="R2", titlefont={"size": 14}),
                  yaxis2=dict(title="MAE", titlefont={"size": 14}),
                  yaxis3=dict(title="MSE", titlefont={"size": 14}))
fig.show()


# Como se observa en las gráficas anteriores, los valores de las distintas métricas en cada partición son muy parecidos. Esto nos permite afirmar que estos parámetros son óptimos para nuestro modelo, pues este se comporta igual de bien independientemente del conjunto de entrenamiento asignado.

# ### 2.6. Relevancia de cada una de las características
# 
# Hasta ahora hemos estado entrenando el modelo con todas las características posibles. Sin embargo, esto no es muy beneficioso pues, al igual que hay características que aportan mucha información, hay otras que hacen todo lo contrario y lo único que introducen es ruido en el modelo.
# 
# Por ello, es conveniente entrenar el modelo sólo con aquellas características que sean más importantes. Para determinar cuáles son realmente importantes y cuáles no, existe la propiedad `feature_importances_`.

# In[ ]:


# DataFrame que contendrá la importancia de las variables del modelo para cada conjunto de Cross Validation
df_coeficientes = pd.DataFrame()

for i in range(0, len(models)):
    coeficientes = pd.DataFrame([models[i].feature_importances_], columns=X.columns)
    df_coeficientes = df_coeficientes.append(coeficientes)
    
# Ordenamos los coeficientes de mayor a menor importancia
ord_coef = df_coeficientes.mean().sort_values(ascending=False)
labels = ord_coef.index.astype(str).tolist()
values = ord_coef.values.tolist()

fig = px.pie(names=labels[:3], values=[round(v, 4) for v in values][:3],
             title="3 características más importantes para el modelo",
             color_discrete_sequence=colores,
             height=400, width=700)

fig.update_layout(title={'x': 0.5, 'y': 0.9, 'font': {'size': 16},
                         'xanchor': 'center', 'yanchor': 'top'},
                  legend={'title': 'Características', 'font': {'size': 16}})

fig.update_traces(textinfo='value')
fig.show()


# Como se observa en el gráfico sectorial anterior, las 3 características más relevantes del dataset son: `totalSteps`, `activeSeconds` y `wellnessKilocalories`. Puesto que la importancia de las demás características es inferior al 1%, podemos considerar que no aportan información al modelo sino más bien ruido. Por ello, es recomendable volver a entrenar el modelo solo con estas 3 características y, posteriormente, comparar los resultados obtenidos con los anteriores.

# In[ ]:


# Seleccionamos las tres variables más importantes
X = X[['totalSteps', 'activeSeconds', 'activeKilocalories']].copy()

scores_rw = []
scores_mae = []
scores_mse = []
models = []

# Para cada fold
# Variable acumuladora
mean_score = 0.0

for train_index, test_index in cv.split(X):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    
    # Declaramos el modelo
    model = DecisionTreeRegressor(max_depth=4,
                                  min_samples_split=2,
                                  min_samples_leaf=1)
    
    # Entrenamos el modelo
    model = model.fit(X_train, y_train)
    
    # Guardamos el modelo en una lista
    models.append(model)

    # Evaluamos el modelo
    predict = model.predict(X_test)

    # Métrica R2
    r2 = round(r2_score(y_test, predict), 3)
    scores_r2.append(r2)
    
    # Métrica MAE
    mae = round(mean_absolute_error(y_test, predict), 3)
    scores_mae.append(mae)
    
    # Métrica MSE
    mse = round(mean_squared_error(y_test, predict), 3)
    scores_mse.append(mse)
    
df_coeficientes = pd.DataFrame()

for i in range(0, len(models)):
    coeficientes = pd.DataFrame([models[i].feature_importances_], columns=X.columns)
    df_coeficientes = df_coeficientes.append(coeficientes)
    
# Ordenamos los coeficientes de mayor a menor importancia
ord_coef = df_coeficientes.mean().sort_values(ascending=False)
labels = ord_coef.index.astype(str).tolist()
values = ord_coef.values.tolist()

fig = px.pie(names=labels[:3], values=[round(v, 4) for v in values][:3],
             title="3 características más importantes para el modelo",
             color_discrete_sequence=colores,
             height=400, width=700)

fig.update_layout(title={'x': 0.5, 'y': 0.9, 'font': {'size': 16},
                         'xanchor': 'center', 'yanchor': 'top'},
                  legend={'title': 'Características', 'font': {'size': 16}})

fig.update_traces(textinfo='value')
fig.show()


# 

# Tras volver a entrenar el modelo, observamos que los coeficientes de relevancia son muy similares a los obtenidos anteriormente. La diferencia es que, al haber eliminado las variables que no son relevantes, el algoritmo se ajusta ligeramente mejor. Además, al utilizar muchas menos variables, el modelo es más sencillo de utilizar y el coste temporal de entrenamiento también es menor.

# In[ ]:





# ## 3. Técnicas de clasificación

# ## 4. Técnicas de *ensembles*

# Los *ensembles* son unas estrategias de combinación de distintos algoritmos, que combinando sus predicciones, dan lugar a una predicción final. Existen un gran número de métodos de ensembles que se pueden utilizar y, a continuación, crearemos un modelo con cada uno de los más relevantes para, posteriormente, evaluar su funcionamiento. 

# ### 4.1. Selección de parámetros óptimos mediante GridSearchCV
# 
# Para establecer los parámetros óptimos para cada modelo a utilizar, usaremos la clase `gridSearcCV`de `sckitLearn`, mediante la cuál, indicandole los modelos a utilizar y sus distintos parámetros, establecerá cuales de estos últimos son los óptimos para cada parámetro a partir de la validación cruzada. 
# 
# En primer lugar, debemos de establecer los modelos a evaluar, en nuestro caso serán *Bagging*, *Random Forest*, *AdaBoost*, *GradientBoost* y *Stacking*. 

# In[ ]:


estim = [('knn', KNeighborsClassifier(n_neighbors=3)),
              ('cart', DecisionTreeClassifier(random_state=0)),
              ('svm', SVC(random_state=0)),
              ('lr', LogisticRegression(random_state=0))]

ensembles = [
    BaggingClassifier(random_state=0), 
    RandomForestClassifier(random_state=0), 
    AdaBoostClassifier(random_state=0),
    GradientBoostingClassifier(random_state=0),
    StackingClassifier(estimators=estim)
]


# A continuación, para cada uno de los modelos, estableceremos qué parámetros vamos a probar e incluiremos todos en una única lista:

# In[ ]:


bagging_parameters = {
    'base_estimator'      : [DecisionTreeClassifier(random_state=0), SVC(random_state=0), KNeighborsClassifier(n_neighbors=3), SGDClassifier(max_iter=1000, tol=1e-3, random_state=0)],
    'n_estimators' : [1, 5, 10, 20],
    'max_samples'    : [0.25, 0.5, 0.75, 1],
    'max_features'   : [0.25, 0.5, 0.75, 1]
}

randomforest_parameters = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth'   : [None, 1, 2, 5],
    'max_features':  ['auto', 'log2'],
    'max_samples' : [None, 0.25, 0.5, 0.75, 1 ]
}

adaboost_parameters = {
    'base_estimator'      : [DecisionTreeClassifier(random_state=0), SVC(random_state=0), KNeighborsClassifier(n_neighbors=3), SGDClassifier(max_iter=1000, tol=1e-3, random_state=0)],
    'n_estimators' : [1, 5, 10, 20],
    'learning_rate'  : [0.25, 0.5, 1, 5]
}

gradientboost_parameters = {
    'learning_rate'  : [0.25, 0.5, 1, 5],
    'n_estimators' : [1, 5, 10, 20],
    'max_depth'      : [None, 1, 2, 5],
}

stacking_parameters = {
    'final_estimator' : [LogisticRegression(random_state=0), GradientBoostingClassifier(random_state=0)]
}

parameters = [
    bagging_parameters, 
    randomforest_parameters, 
    adaboost_parameters,
    gradientboost_parameters,
    stacking_parameters
]


# Finalmente almacenaremos el modelo elegido para cada algoritmo, su precisión y tiempo de ejecución para posteriormente evaluarlo:

# In[ ]:


estimators = [] # Para almacenar los modelos
accuracies = [] # Para almacenar su precisión y tiempo
times = []

# Iteramos para cada uno de los modelos de ensembles
for i, ensemble in enumerate(ensembles):
    start_time = time.time()

    clf = GridSearchCV(ensemble,          # Modelo
              param_grid = parameters[i], # Parámetro
              scoring='accuracy',         # Métrica de evaluación
              cv=10)                      # Número de folds para el CV
    print('************', ensemble.__class__.__name__, '************')
    clf.fit(X_train, y_train)
    print("Parámetros :", clf.best_params_)
    acc = (clf.predict(X_test) == y_test).mean()*100
    print("Accuracy :", (acc))
    sec = (time.time() - start_time)
    print("Time of tunning and training :", (sec))
    accuracies.append((ensemble.__class__.__name__, acc))
    times.append((ensemble.__class__.__name__, sec))
    estimators.append((ensemble.__class__.__name__, clf))


# Mostramos los resultados en una gráfica.

# In[ ]:


df = pd.DataFrame(accuracies, columns=['Model', 'Accuracy (%)'])
df2 = pd.DataFrame(times, columns=['Model', 'Time (secs)'])

fig = make_subplots(rows=1, cols=2)

fig.append_trace(go.Bar(x=df['Model'], y=df['Accuracy (%)'], name="Accuracy (%)",
                        textposition="none"), row=1, col=1)

fig.append_trace(go.Bar(x=df2['Model'], y=df2['Time (secs)'], name="Time (secs)",
                        textposition="none"), row=1, col=2)

fig.update_layout(height=400, width=1200,
                  title={'x': 0.5, 'y': 0.9, 'font': {'size': 16},
                         'text': 'Precisión y tiempo de ejecución por modelo',
                         'xanchor': 'center', 'yanchor': 'top'}
                 )
fig.show()


# Como podemos observar todos los modelos presentan una precisión muy elevada con el conjunto de validación, pasando del 90%. Con estos valores podríamos determinar que cualquiera de los modelos creados, tendría un buen funcionamiento como clasificador en este conjunto de datos, aunque si tuvieramos que elegir uno, atendiendo a tiempo de ajuste y entrenamiento que ha tomado cada uno, el **Stacking** sería aquel que con un requerimiento de tiempo de entrenamiento muy pequeño nos proporcionaria grandes resultados.

# Para corroborar este correcto funcionamiento, evaluaremos otros aspectos del modelo, creando una función auxiliar que nos ayude y que mostraá la precisión de clasificación del modelo para cada una de las clases gracias a la matriz de confusión y la curva ROC, para cada uno de los modelos.

# In[ ]:


# Función auxiliar que evalúa los resultados de una clasificación
def evaluate_model(y_test, y_pred, n_classes):
  """
    Evalúa el modelo e imprime por pantalla las estadísitcas
  """
  print('==== Sumario de la clasificación ==== ')
  print(classification_report(y_test, y_pred))

  print('Accuracy -> {:.2%}\n'.format(accuracy_score(y_test, y_pred)))

  # Ploteamos la matriz de confusión
  display_labels = sorted(unique_labels(y_test, y_pred), reverse=True)
  cm = confusion_matrix(y_test, y_pred, labels=display_labels)

  z = cm[::-1]
  x = display_labels
  y =  x[::-1].copy()
  z_text = [[str(y) for y in x] for x in z]

  fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

  fig_cm.update_layout(
      height=400, width=400,
      showlegend=True,
      margin={'t':150, 'l':0},
      title={'text' : 'Matriz de Confusión', 'x':0.5, 'xanchor': 'center'},
      xaxis = {'title_text':'Valor Real', 'tickangle':45, 'side':'top'},
      yaxis = {'title_text':'Valor Predicho', 'tickmode':'linear'},
  )
  fig_cm.show()

  y_test

  # Ploteamos la curva ROC
  y_test_enc = label_binarize(y_test, classes=np.arange(n_classes))
  y_pred_enc = estimator[1].predict_proba(X_test)

  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(3):
      fpr[i], tpr[i], _ = roc_curve(y_test_enc[:, i], y_pred_enc[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  fpr["micro"], tpr["micro"], _ = roc_curve(y_test_enc.ravel(), y_pred_enc.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  mean_tpr /= n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  plt.figure()
  plt.plot(
      fpr["micro"],
      tpr["micro"],
      label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
      color="deeppink",
      linestyle=":",
      linewidth=4,
  )

  plt.plot(
      fpr["macro"],
      tpr["macro"],
      label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
      color="navy",
      linestyle=":",
      linewidth=4,
  )

  colors = cycle(["aqua", "darkorange", "cornflowerblue"])
  for i, color in zip(range(n_classes), colors):
      plt.plot(
          fpr[i],
          tpr[i],
          color=color,
          lw=lw,
          label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
      )

  plt.plot([0, 1], [0, 1], "k--", lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Some extension of Receiver operating characteristic to multiclass")
  plt.legend(loc="lower right")
  plt.show()


# Ejecutamos la evaluación para cada uno de los modelos

# In[ ]:


lw = 2
n_classes = len(np.unique(y_test))

for estimator in estimators:
    print('****************************************************************')
    print('Evaluación del modelo ', estimator[0])
    print('****************************************************************')
    y_pred = estimator[1].predict(X_test)

    evaluate_model(y_test, y_pred, n_classes)


# Gracias a la matriz de confusión no solo podemos comprobar el número de elementos clasificados erróneamente, si no que podemos apreciar en que categoría han sido identificados cada uno. La curva ROC a su vez nos permite saber la precisión de cada modelo, evaluando el área bajo la curva, a mayor área, mayor será esta precisión.
# 
# Como podemos observar por la curva ROC, el método de *Stacking* tiene una clasificación casi perfecta, fallando únicamente 7 elementos del total de 275. De ese total, únicamente seha clasificado incorrectamente un elmento de la clase alta, que se ha clasificado como medio; dos elementos de la clase baja que se han clasificado como medios y cuatro elementos de la clase media.

# ## 5. Técnicas de *clustering*

# El *clustering* es una técnica gracias a la cual, podemos organizar un conjunto de datos, en subconjuntos más pequeños denominados *clústers*. Estos clústers tendrán unas características muy similares enre ellos pero diferentes con otros elementos de clusteres distintos.

# ## 5.1 Selección del cluster óptimo
# 
# Para elegir un clústering óptimo en nuestro conjunto de datos, crearemos distintos modelos y los aplicaremos sobre el conjunto de datos para posteriormente evaluar su desempeño. Los tipos de clustering que probaremos serán: *KMeans*, *Clústering jerárquico*, *BIRHC*, *DBSCAN*, *OPTICS* y un modelo de *mixtura gaussiano*
# 
# En primer lugar creamos una lista con los nombres de los tipos de clustering que crearemos:

# In[ ]:


clusters = ['kmeans', 'agglomerative', 'dbscan', 'optics', 'birch', 'gaussian']


# Para cada uno de estos tipos de clustering crearemos varios modelos, probando distintas combinaciones de sus posible parámetros. Para ello iteraremos por cada uno de los métodos y crearemos los modelos.

# In[ ]:


results = []

for cluster in clusters:
  if cluster == 'kmeans':
    clust = MiniBatchKMeans(n_clusters=3).fit(X)
    results.append(clust)
    
  if cluster == 'agglomerative':
    linkage = ['ward', 'complete', 'average', 'single']
    for item in linkage:
      clust =  AgglomerativeClustering(n_clusters=3, linkage = item).fit(X)
      results.append(clust)

  if cluster == 'dbscan':
    eps = [0.3, 0.5, 0.7]
    min_samples = [5, 10, 50]

    for item in eps:
      for item2 in min_samples:
        clust = DBSCAN(eps = item, min_samples = item2).fit(X)
        results.append(clust)

  if cluster == 'optics':
    eps = [0.3, 0.5, 0.7]
    min_samples = [5, 10, 50]

    for item in eps:
      for item2 in min_samples:
        clust = OPTICS(eps = item, min_samples = item2).fit(X)
        results.append(clust)

  if cluster == 'birch':
    threshold = [0.25, 0.5, 1]
    branching_factor = [25, 50, 100]

    for item in threshold:
      for item2 in branching_factor:
        clust = Birch(n_clusters=3, threshold = item, branching_factor = item2).fit(X)
        results.append(clust)

  if cluster == 'gaussian':
    covariance_type = ['full', 'tied', 'diag', 'spherical']
    init_params = ['kmeans', 'random']

    for item in covariance_type:
          for item2 in init_params:
            clust = GaussianMixture(n_components=3, covariance_type = item, init_params = item2).fit(X)
            results.append(clust)


# In[ ]:


for res in results:

  if hasattr(res, 'labels_'):
      y_pred = res.labels_.astype(int)
  else:
      y_pred = res.predict(X)

  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                        '#f781bf', '#a65628', '#984ea3',
                                        '#999999', '#e41a1c', '#dede00']),
                                int(max(y_pred) + 1))))
  # add black color for outliers (if any)
  colors = np.append(colors, ["#000000"])
  plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

  plt.xlim(-2.5, 2.5)
  plt.ylim(-2.5, 2.5)
  plt.xticks(())
  plt.yticks(())    

  plt.show()

