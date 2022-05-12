#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://drive.google.com/uc?export=download&id=1nv5uGKO9BLD9Y19LnZZH35nnQghZsPdD" />
# 
# # Feature Engineering and Feature Selection
# 
# Para empezar, vamos a revisar tres tareas similares pero diferentes: 
# 
# * **feature extraction** and **feature engineering**: Transformación de data(raw) en características adecuadas para modelado;
# * **feature transformation**: Transformación de data para mejorar la precisión de los algoritmos;
# * **feature selection**: Removiendo características innecesarias.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Temas
# 
# 1. [Feature Extraction](#1.-Feature-Extraction)
#  - [Texts](#Texts)
#  - [Geospatial data](#Geospatial-data)
#  - [Date and time](#Date-and-time)
#  - [Time series, web, etc.](#Time-series,-web,-etc.)
# 2. [Feature transformations](#Feature-transformations)
#  - [Normalization and changing distribution](#Normalization-and-changing-distribution)

# ## 1. Feature Extraction
# 
# 
# En la práctica, los datos rara vez se presentan en forma de matrices listas para usar. Es por eso que cada tarea comienza con la extracción de características. A veces, puede ser suficiente leer el archivo .csv y convertirlo en `numpy.array`, pero esta es una rara excepción. Veamos algunos de los tipos populares de datos de los que se pueden extraer características.
# 

# In[ ]:


import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ### Texts
# 
# El texto es un tipo de datos que puede venir en diferentes formatos, revisaremos los más populares.
# 
# Antes de trabajar con texto, hay que tokenizarlo. La tokenización implica dividir el texto en unidades (tokens). Los tokens son sólo las palabras. Pero el dividir por palabras nos puede llevar a perder parte del significado-- "Santa Bárbara" es un token, no dos, pero "rock'n'roll" no debe dividirse en dos token. Hay tokenizadores listos para usar que tienen en cuenta las peculiaridades del lenguaje, pero también cometen errores, especialmente cuando trabajas con fuentes de texto específicas (periódicos, jerga, errores ortográficos, errores tipográficos).
# 
# Después de la tokenización, normalizamos los datos. Para el texto, se trata de la derivación y/o lematización; Estos son procesos similares utilizados para procesar diferentes formas de una palabra. Se puede leer sobre la diferencia entre ellos [aqui](http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html).
# Entonces, ahora que hemos convertido el documento en una secuencia de palabras, podemos representarlo con vectores. El enfoque más fácil se llama Bag of Words: creamos un vector con la longitud del diccionario, calculamos el número de ocurrencias de cada palabra en el texto y colocamos ese número de ocurrencias en la posición apropiada en el vector. El proceso descrito parece más simple en el código:
# 
# 

# In[ ]:


from functools import reduce 
import numpy as np

# definicion de corpus
texts = [['i', 'have', 'a', 'cat'], 
        ['he', 'have', 'a', 'dog'], 
        ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(list(reduce(lambda x, y: x + y, texts)))))
print(dictionary)
def vectorize(text): 
    vector = np.zeros(len(dictionary)) 
    for i, word in dictionary: 
        num = 0 
        for w in text: 
            if w == word: 
                num += 1 
        if num: 
            vector[i] = num 
    return vector

for t in texts: 
    print(vectorize(t))


# Esta es una ilustración del proceso:
# <img src="https://drive.google.com/uc?export=download&id=18dEqfTKT10i5_EJz4MLy_hywpQ_IusUJ" />
# 
# Esta es una implementación extremadamente ingenua. En la práctica, debe considerar palabras de parada, la longitud máxima del diccionario, estructuras de datos más eficientes (generalmente los datos de texto se convierten en un matrices esparsa), etc.
# 
# Cuando utilizamos algoritmos como Bag of Words, perdemos el orden de las palabras en el texto, lo que significa que los textos "i have no cows" y "no, i have cows" aparecerán idénticos después de la vectorización cuando, de hecho, tienen el significado opuesto. Para evitar este problema, podemos volver a visitar nuestro paso de tokenización y usar N-grams (la *secuencia* de N tokens consecutivos) en su lugar.
# 
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1,1))
vect.fit_transform(['i have no cows','no, i have cows']).toarray()


# In[ ]:


vect.vocabulary_ 


# In[ ]:


vect = CountVectorizer(ngram_range=(1,2))
vect.fit_transform(['i have no cows','no, i have cows']).toarray()


# In[ ]:


vect.vocabulary_


# También tenga en cuenta que uno no tiene que usar sólo palabras. En algunos casos, es posible generar N-gram de caracteres. Este enfoque podría dar cuenta de la similitud de palabras relacionadas o manejar errores tipográficos.

# In[ ]:


from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(3,3), analyzer='char_wb')

n1, n2, n3, n4 = vect.fit_transform(['andersen', 'petersen', 'petrov', 'smith']).toarray()


euclidean(n1, n2), euclidean(n2, n3), euclidean(n3, n4)


# Agregando a la idea de Bag of Words: las palabras que rara vez se encuentran en el corpus (en todos los documentos del dataset) pero que están presentes en un documento en particular podrían ser más importantes. Entonces tiene sentido aumentar el peso de más palabras específicas del dominio para separarlas de las palabras comunes. Este enfoque se llama TF-IDF (term frequency-inverse document frequency), que no se puede escribir en unas pocas líneas, por lo que debe consultar los detalles en referencias como [wiki](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). La opción predeterminada es la siguiente:
# 
# <img src="https://drive.google.com/uc?export=download&id=1zRnAL7xslzRl3odfsLa3yzbSR9SMw0CM" />

# Usando estos algoritmos, es posible obtener una solución para un problema simple, que puede servir como línea base. Sin embargo, para aquellos a quienes no les gustan los clásicos, hay nuevos enfoques. Un método popular es Word2Vec, pero también hay algunas alternativas (GloVe, Fasttext, etc.).
# 
# Word2Vec es un caso especial de los algoritmos word embedding. Usando Word2Vec y modelos similares, no sólo podemos vectorizar palabras en un espacio de alta dimensión (típicamente unos pocos cientos de dimensiones) sino también comparar su similitud semántica. Este es un ejemplo clásico de operaciones que se pueden realizar en conceptos vectorizados: **king - man + woman = queen.**
# 
# ![image](https://cdn-images-1.medium.com/max/800/1*K5X4N-MJKt8FGFtrTHwidg.gif)

# Vale la pena señalar que este modelo no comprende el significado de las palabras, simplemente trata de posicionar los vectores de manera que las palabras utilizadas en el contexto común estén cerca unas de otras.
# 
# Dichos modelos necesitan ser entrenados en data sets muy grandes para que las coordenadas del vector capturen la semántica. Se puede descargar un modelo previamente entrenado para sus propias tareas
# [aquí](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models).

# ### Geospatial data
# 
# Los datos geográficos no se encuentran tan a menudo en los problemas, pero sigue siendo útil dominar las técnicas básicas para trabajar con ellos, especialmente porque hay bastantes soluciones listas para usar en este campo.
# 
# Los datos geoespaciales a menudo se presentan en forma de direcciones o coordenadas de (Latitud, Longitud). Dependiendo de la tarea, es posible que necesite dos operaciones mutuamente inversas: geocodificación (recuperar un punto de una dirección) y geocodificación inversa (recuperar una dirección de un punto). Ambas operaciones son accesibles en la práctica a través de API externas de Google Maps u OpenStreetMap. Los diferentes geocodificadores tienen sus propias características, y la calidad varía de una región a otra. Afortunadamente, hay bibliotecas universales como [geopy] (https://github.com/geopy/geopy) que actúan como encapsuladores para estos servicios externos.
# 
# * Si tiene **MUCHOS DATOS**, alcanzará rápidamente los límites de la API externa. Además, no siempre es el más rápido para recibir información a través de HTTP. Por lo tanto, es necesario considerar usar una versión local de OpenStreetMap.
# 
# * Si tiene una **PEQUEñA CANTIDAD DE DATOS**, tiempo suficiente y no desea extraer características sofisticadas, puede usar reverse_geocoder en lugar de OpenStreetMap:

# In[ ]:


## Algunos ejemplos utilizarán el dataset de la compañía Renthop, que se usa en la competencia 
## Two Sigma Connect: Consultas de listado de alquileres de Kaggle. 
## En esta tarea, debe predecir la popularidad de un nuevo listado de alquiler, es decir, 
### clasificar el listado en tres clases: `['low', 'medium' , 'high']`. Para evaluar las soluciones, 
### 
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Let's load the dataset from Renthop right away
with open('../input/twosigmaconnect/renthop_train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)


# In[ ]:


df.tail()


# In[ ]:


get_ipython().system('pip install reverse_geocoder')
import reverse_geocoder as revgc


# In[ ]:


revgc.search([df.latitude[1], df.longitude[2]])


# Al trabajar con geocodificación, no debemos olvidar que las direcciones pueden contener errores tipográficos, lo que hace que la limpieza de datos sea necesario. La posición de las coordenadas pueden ser incorrectas debido al ruido del GPS o la mala precisión en lugares como túneles, áreas del centro, etc. Si la fuente de datos es un dispositivo móvil, la ubicación geográfica no puede determinarse por GPS sino por redes WiFi en el área, que conduce a agujeros en el espacio y la teletransportación. Mientras viaja por Arequipa, de repente puede haber una ubicación WiFi desde Lima.
# 
# 
# Un punto generalmente se ubica entre una infraestructura. Aquí, puede liberar su imaginación e inventar características basadas en su experiencia de vida y conocimiento de dominio: la proximidad de un punto a una estación de servicio, la distancia a la tienda más cercana, la cantidad de cajeros automáticos alrededor, etc. Para cualquier tarea, puede crear fácilmente docenas de funciones y extraerlas de varias fuentes externas. Para problemas fuera de un entorno urbano, puede considerar características de fuentes más específicas, por ejemplo la altura sobre el nivel del mar.
# 
# Si dos o más puntos están interconectados, puede valer la pena extraer características de la ruta entre ellos. En ese caso, serán útiles las distancias ( distancia de carretera calculada por el gráfico de ruta), número de giros con la proporción de giros de izquierda a derecha, número de semáforos, cruces y puentes. 

# ### Date and time
# 
# Se podría pensar que la fecha y la hora están estandarizadas debido a su prevalencia, pero, sin embargo, persisten algunas dificultades.
# 
# 
# Comencemos con el día de la semana, que son fáciles de convertir en 7 variables ficticias utilizando one-hot encoding. Además, también crearemos una función binaria separada para el fin de semana llamada `is_weekend`.
# 
# 

# In[ ]:


df['dow'] = df['created'].apply(lambda x: pd.to_datetime(x).weekday())
df['is_weekend'] = df['created'].apply(lambda x: 1 if pd.to_datetime(x).weekday() in (5, 6) else 0)


# In[ ]:


#df['is_weekend']


# Algunas tareas pueden requerir funciones de calendario adicionales. Por ejemplo, los retiros de efectivo se pueden vincular a un día de pago. En general, cuando se trabaja con datos de series de tiempo, es una buena idea tener un calendario con días festivos, condiciones climáticas anormales y otros eventos importantes.
# 
# 
# > Q: ¿Qué tienen en común el Año Nuevo chino, la maratón de Nueva York ?
# 
# > A: Todos deben colocarse en el calendario de posibles anomalías..
# 
# Tratar con la hora (minuto, día del mes ...) no es tan simple como parece. Si utiliza la hora como una variable real, contradecimos ligeramente la naturaleza de los datos: `0 <23` mientras que `0:00:00 02.01 > 01.01 23:00:00`. Para algunos problemas, esto puede ser crítico. Al mismo tiempo, si las codifica como variables categóricas, generará una gran cantidad de características y perderá información sobre la proximidad
# 
# También existen algunos enfoques más esotéricos para tales datos, como proyectar el tiempo en un círculo y usar las dos coordenadas.

# In[ ]:


def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period 
    return np.cos(value), np.sin(value)


# In[ ]:


#import numpy as np
import matplotlib.pyplot as plt
xx=np.arange(0,24,1)
X=[]
Y=[]
for i in xx:
    x,y=make_harmonic_features(i)
    X.append(x)
    Y.append(y)
    
plt.plot(X, Y)
plt.show()


# Esta transformación preserva la distancia entre puntos, lo cual es importante para los algoritmos que estiman la distancia (kNN, SVM, k-means ...)

# In[ ]:


from scipy.spatial import distance
euclidean(make_harmonic_features(23), make_harmonic_features(1)) 


# In[ ]:


euclidean(make_harmonic_features(9), make_harmonic_features(11)) 


# In[ ]:


euclidean(make_harmonic_features(9), make_harmonic_features(21))


# ### Time series, web, etc.
# 
# Con respecto a las series de tiempo--time series--: no entraremos en demasiados detalles aquí, pero podemos recomendar [biblioteca útil que genera automáticamente características para series de tiempo](https://github.com/blue-yonder/tsfresh).
# 
# Si está trabajando con datos web, generalmente tiene información sobre el 'User-Agent'. Es una gran cantidad de información. Primero, uno necesita extraer el sistema operativo de este. En segundo lugar, crear una función `is_mobile`. Tercero, mirar el navegador.
# 

# In[ ]:


get_ipython().system('pip install -q pyyaml ua-parser user-agents')
import user_agents

ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'
ua = user_agents.parse(ua)

print('Is a bot? ', ua.is_bot)
print('Is mobile? ', ua.is_mobile)
print('Is PC? ',ua.is_pc)
print('OS Family: ',ua.os.family)
print('OS Version: ',ua.os.version)
print('Browser Family: ',ua.browser.family)
print('Browser Version: ',ua.browser.version)


# La siguiente información útil es la dirección IP, desde la cual puede extraer el país y posiblemente la ciudad, el proveedor y el tipo de conexión (móvil / estacionario). Se debe comprender que existe una variedad de bases de datos proxy y obsoletas, por lo que esta característica puede contener ruido. Los gurús de la administración de redes pueden intentar extraer características aún más sofisticadas como  [VPN](https://habrahabr.ru/post/216295/).Por cierto, los datos de la dirección IP se combinan bien con`http_accept_language`: si el usuario está sentado en los servidores proxy de Chile y la configuración regional del navegador es `ru_RU`, algo no está bien y vale la pena echarle un vistazo a la columna correspondiente de la tabla (`is_traveler_or_proxy_user`).

# ## Feature transformations
# 
# ### Normalization and changing distribution
# 
# La transformación monotónica de características es crítica para algunos algoritmos y no tiene efecto en otros. Esta es una de las razones de la mayor popularidad de los árboles de decisión `decision tree`  y todos sus algoritmos derivados (random forest, gradient boosting). No todos pueden o quieren jugar con las transformaciones, y estos algoritmos son robustos para distribuciones inusuales.
# 
# 
# 
# 

# En el siguiente ejemplo, entrenaremos un modelos que usa `LightGBM` en un conjunto de datos *toy dataset* donde sabemos que la relación entre X e Y es monótona (pero con ruido) y comparamos el modelo predeterminado y el monotono.

# In[ ]:


import numpy as np
plt.style.use('seaborn-whitegrid')
size = 100
x = np.linspace(0, 10, size) 
y = x**2 + 10 - (20 * np.random.random(size))


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(x,y,'o')


# ### Vamos a ajustar un modelo gradient boosted en estos datos, estableciendo min_child_samples = 5.

# In[ ]:


import lightgbm as lgb
overfit_model = lgb.LGBMRegressor(silent=False, min_child_samples=5)
overfit_model.fit(x.reshape(-1,1), y)
 
#predicted output from the model from the same input
prediction = overfit_model.predict(x.reshape(-1,1))


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(x,y,'o')
plt.plot(x,prediction,color='r')


# El modelo se sobreajustará ligeramente (debido valor pequeño de 'min_child_samples'), lo que podemos ver al trazar los valores de X contra los valores predichos de Y: la línea roja no es monótona como nos gustaría que fuera.

# Como sabemos que la relación entre X e Y debe ser monótona, podemos establecer esta restricción al especificar el modelo.

# In[ ]:


monotone_model = lgb.LGBMRegressor(min_child_samples=5, 
                                   monotone_constraints="1")
monotone_model.fit(x.reshape(-1,1), y)


# In[ ]:


#predicted output from the model from the same input
prediction = monotone_model.predict(x.reshape(-1,1))


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(x,y,'o')
plt.plot(x,prediction,color='r')


# El parámetro `monotone_constraints = "1″` establece que la salida debería aumentar monotónicamente con respecto a las primeras características (que en nuestro caso es la única característica). Después de entrenar el modelo monótono, podemos ver que la relación ahora es estrictamente monótona.

# ### **Y si verificamos el rendimiento del modelo, podemos ver que la restricción de monotonicidad no sólo proporciona un ajuste más natural, sino que el modelo también generaliza mejor (como se esperaba). Al medir el error cuadrático medio MSE en los nuevos datos de prueba, vemos que el error es menor para el modelo monótono.**

# In[ ]:


from sklearn.metrics import mean_squared_error as mse
 
size = 1000000
x = np.linspace(0, 10, size) 
y = x**2  -10 + (20 * np.random.random(size))
 
print ("MSE del modelo por defecto", mse(y, overfit_model.predict(x.reshape(-1,1))))
print ("MSE del modelo Monotono", mse(y, monotone_model.predict(x.reshape(-1,1))))


# También hay razones puramente de ingeniería: `np.log` es una forma de tratar con grandes números que no caben en `np.float64`. Esta es una excepción más que una regla; a menudo es impulsado por el deseo de adaptar el dataset a los requisitos del algoritmo. Los métodos paramétricos generalmente requieren un mínimo de distribución simétrica y unimodal de datos, que no siempre se da en datos reales.
# 
# Sin embargo, los requisitos de datos se imponen no sólo por métodos paramétricos; [K nearest neighbors](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-3-classification-decision-trees-and-k-nearest-neighbors-8613c6b6d2cd) predecirá salidas sin sentido completo si las características no están normalizadas, por ejemplo cuando una distribución se encuentra cerca de cero y no va más allá (-1, 1).
# 
# ### Un ejemplo simple: suponga que la tarea es predecir el costo de un apartamento a partir de dos variables: la `distancia desde el centro de la ciudad` y la `cantidad de habitaciones`. El número de habitaciones rara vez excede de 5, mientras que la distancia desde el centro de la ciudad puede ser fácilmente de miles de metros.
# 
# 
# La transformación más simple es Standard Scaling (o la normalización Z-score):
# 
# $$ \large z= \frac{x-\mu}{\sigma} $$
# 
# Tenga en cuenta que Standard Scaling no hace que la distribución sea normal en sentido estricto.

# In[ ]:


# EJEMPLO: usando Shapiro-Wilk - Test
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta
from scipy.stats import shapiro
import numpy as np

#Generamos 1000 números aleatorios [1,10]
data = beta(1, 10).rvs(1000).reshape(-1, 1)

# Realizamos la prueba de Shapiro-Wilk para la normalidad.
shapiro_stat,shapiro_p_value=shapiro(data)

#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')


# In[ ]:


#
shapiro_stat,shapiro_p_value=shapiro(StandardScaler().fit_transform(data))

# Con el valor p tendríamos que rechazar la hipótesis nula de normalidad de los datos.
#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')


# **Otra opción** bastante popular es MinMaxScaling, que reúne todos los puntos dentro de un intervalo predeterminado (típicamente (0, 1)).
# 
# $$ \large X_{norm}=\frac{X-X_{min}}{X_{max}-X_{min}} $$

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
shapiro_stat,shapiro_p_value=shapiro(MinMaxScaler().fit_transform(data))

# Con el valor p tendríamos que rechazar la hipótesis nula de normalidad de los datos.
#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')


# In[ ]:


(data - data.min()) / (data.max() - data.min()) 


# `StandardScaling` y `MinMaxScaling` tienen aplicaciones similares y, a menudo, son más o menos intercambiables. Sin embargo, si el algoritmo implica el cálculo de distancias entre puntos o vectores, la opción predeterminada es `StandardScaling`. Pero `MinMaxScaling` es útil para la visualización al incorporar características dentro del intervalo (0, 255).
# 
# Si suponemos que algunos datos no se distribuyen normalmente, sino que se describen mediante [la distribución log-normal ](https://en.wikipedia.org/wiki/Log-normal_distribution), se pueden transformar fácilmente en una distribución normal
# 
# 

# In[ ]:


from scipy.stats import lognorm

data = lognorm(s=1).rvs(1000)
shapiro_stat,shapiro_p_value=shapiro(data)
#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')


# In[ ]:


# graficando
plt.figure(figsize=(10,5))
plt.hist(data, bins=50)


# ## Pero: 

# In[ ]:


shapiro_stat,shapiro_p_value=shapiro(np.log(data))
#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')


# In[ ]:


# graficando
plt.figure(figsize=(10,5))
plt.hist(np.log(data), bins=50)


# La distribución lognormal es adecuada para describir: salarios, precios de seguro, la población urbana, la cantidad de comentarios sobre artículos en Internet, etc. Sin embargo, para aplicar este procedimiento, la distribución subyacente no necesariamente tiene que ser `lognormal`; puede intentar aplicar esta transformación a cualquier distribución con una cola derecha sesgada. Además, se puede tratar de usar otras transformaciones similares, formulando sus propias hipótesis sobre cómo aproximar la distribución disponible a una normal. Ejemplos de tales transformaciones son [transformación Box-Cox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html) (el logaritmo es un caso especial de la transformación Box-Cox ) o [transformación de Yeo-Johnson](https://gist.github.com/mesgarpour/f24769cd186e2db853957b10ff6b7a95) (extiende el rango de aplicabilidad a números negativos). Además, también puede intentar agregar una constante a la función: `np.log (x + const)`.
# 
# En los ejemplos anteriores, hemos trabajado con datos sintéticos y probamos estrictamente la normalidad utilizando la prueba de Shapiro-Wilk. Intentemos ver algunos datos reales y comprobar la normalidad utilizando un método menos formal: [Q-Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot). Para una distribución normal, se verá como una línea diagonal suave, y las anomalías visuales deben ser intuitivamente comprensibles.
# 
# 
# 
# ![image](https://habrastorage.org/webt/on/bk/qg/onbkqg1j1tdcj9kc4txfjv6soco.png)
# Q-Q plot para distribución lognormal
# 
# ![image](https://habrastorage.org/webt/cs/vq/xw/csvqxwpf023p16m4pu6zrndynvm.png)
# Q-Q plot para la misma distribución después de tomar el logaritmo

# In[ ]:


# ¡Dibujemos !
import statsmodels.api as sm


# Tomemos la característica de precio del dataset de Renthop y filtremos los valores más extremos para mayor claridad.
price = df.price[(df.price <= 20000) & (df.price > 500)]
price_log = np.log(price)

# usamos transformaciones
price_mm = MinMaxScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
price_z = StandardScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()


# ## Q-Q plot inicial

# In[ ]:


sm.qqplot(price, loc=price.mean(), scale=price.std())


# ## Q-Q plot después de StandardScaler......la forma no cambia

# In[ ]:


sm.qqplot(price_z, loc=price_z.mean(), scale=price_z.std())


# ## Q-Q plot después de MinMaxScaler... la forma tampoco cambia

# In[ ]:


sm.qqplot(price_mm, loc=price_mm.mean(), scale=price_mm.std())


# ## Q-Q plot después de tomar el logaritmo. ¡Las cosas están mejorando!

# In[ ]:


sm.qqplot(price_log, loc=price_log.mean(), scale=price_log.std())


# ### Interactions
# 
# Si las transformaciones anteriores parecían orientadas a las matemáticas, esta parte trata más sobre la naturaleza de los datos; se puede atribuir tanto a las transformaciones de características como a la creación de características.
# 
# If previous transformations seemed rather math-driven, this part is more about the nature of the data; it can be attributed to both feature transformations and feature creation.
# 
# Volvamos nuevamente al problema del dataset de Renthop: Consultas de listado de alquileres. Entre las características de este problema están la `cantidad de habitaciones` y el `precio`. La lógica sugiere que el costo por habitación individual es más sugerente que el costo total, por lo que podemos generar dicha característica.
# 

# In[ ]:


rooms = df["bedrooms"].apply(lambda x: max(x, .5))
# Evitar la división por cero; .5 se elige más o menos arbitrariamente
df["price_per_bedroom"] = df["price"] / rooms


# In[ ]:


df["price_per_bedroom"]


# ### Debes limitarte en este proceso. 
# 
# Si hay un número limitado de características, es posible generar todas las interacciones posibles y luego eliminar las innecesarias utilizando las técnicas descritas en la siguiente sección. Además, no todas las interacciones entre características deben tener un significado físico; por ejemplo, las características polinómicas (ver [sklearn.preprocessing.PolynomialFeatures](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)) a menudo se usan en modelos lineales y son casi imposibles de interpretar.
# 

# ### Rellenando valores faltantes
# 
# No muchos algoritmos pueden funcionar con valores faltantes, y data real a menudo se proporciona datos incompletos. Afortunadamente, esta es una de las tareas para las que no se necesita creatividad. Ambas bibliotecas de python para el análisis de datos proporcionan soluciones fáciles de usar: [pandas.DataFrame.fillna](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html) y [sklearn.preprocessing.Imputer](http://scikit-learn.org/stable/modules/preprocessing.html#imputation).
# 
# Estas soluciones no tienen ninguna magia detrás de escena. Los enfoques para manejar los valores faltantes son bastante sencillos:
# 
# * codificar valores faltantes con un valor en blanco separado como `"n/a"` (para variables categóricas);
# * usa el valor más probable de la característica (media o mediana para las variables numéricas, el valor más común para las variables categóricas);
# * o, por el contrario, codificar con algún valor extremo (bueno para los modelos decision-tree, ya que permite que el modelo haga una partición entre los valores faltantes y los no faltantes);
# * para los datos ordenados (por ejemplo, series de tiempo), tome el valor adyacente: siguiente o anterior.
# 
# 
# ![image](https://cdn-images-1.medium.com/max/800/0*Ps-v8F0fBgmnG36S.)
# 
# Las soluciones de biblioteca fáciles de usar a veces sugieren apegarse a algo como `df = df.fillna (0)` y no preocuparse por las brechas. Pero esta no es la mejor solución: la preparación de datos lleva más tiempo que la construcción de modelos, por lo que el relleno irreflexivo puede ocultar un error en el procesamiento y dañar el modelo.
# 
# 

# ### **Ejercicio: Clasificación de Spam usando Support Vector Machines.(SOLUCIóN)**
# #### CONTEXTO:    SMS Spam Collection es un conjunto de mensajes etiquetados SMS que se han recopilado para la investigación de SMS Spam. Contiene un conjunto de mensajes SMS en inglés 5,574 mensajes etiquetados de acuerdo a su contenido 'ham' (legítimo) o 'spam'.
# 
# AGRADECIMIENTO:  El dataset original se puede encontrar [aquí](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). Los creadores desean tener en cuenta que en caso de que encuentre útil el conjunto de datos, haga referencia al documento anterior y la página web: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
# 
# CONTENIDO: Los archivos contienen un mensaje por línea. Cada línea está compuesta por dos columnas: v1 contiene la etiqueta (ham o spam) y v2 contiene el texto sin formato. Este corpus se ha recopilado de forma gratuita.

# In[ ]:





# ### **Librerias**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ### **Explorando el Dataset**

# In[ ]:


data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
data.head(n=10)


# ### **Análisis de texto**
# 1.  Graficar y encontrar la frecuencia de las palabras en los mensajes spam y no-spam(ham)
# 2.  Describir cada gráfico
# 

# 

# In[ ]:


count1 = Counter(" ".join(data[data['v1']=='ham']["v2"]).split()).most_common(10)
df1 = pd.DataFrame.from_dict(count1)
print(df1.head())
df1 = df1.rename(columns={0: "palabras non-spam", 1 : "count"})
count2 = Counter(" ".join(data[data['v1']=='spam']["v2"]).split()).most_common(10)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "palabras spam", 1 : "count_"})


# In[ ]:


df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["palabras non-spam"]))
plt.xticks(y_pos, df1["palabras non-spam"])
plt.title('Palabras frecuentes en mensajes no-spam')
plt.xlabel('Palabras')
plt.ylabel('Numero')
plt.show()


# In[ ]:


df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["palabras spam"]))
plt.xticks(y_pos, df2["palabras spam"])
plt.title('Palabras frecuentes en mensajes spam ')
plt.xlabel('Palabras')
plt.ylabel('numero')
plt.show()


# ### **feature extraction and feature engineering **
# 
# 1. Preprocesamiento de texto 
# 2. Creación de tokens y el filtrado de palabras clave 
# (puede usar un componente de alto nivel como: CountVectorizer que puede crear un diccionario de características y transformar documentos en vectores de características.)

# In[ ]:


#usando CountVectorizer
#f = feature_extraction.text.CountVectorizer(stop_words = 'english')
#feat = feature_extraction.text.CountVectorizer(stop_words = 'english', ngram_range=(1,2)) ## usando n_gram?
feat = feature_extraction.text.CountVectorizer(stop_words = 'english')

X = feat.fit_transform(data["v2"])

np.shape(X)


# ### **Análisis Predictivo**
# 
# 1. El objetivo es predecir si un nuevo sms es spam o no-spam. usando SVM
# 2. validar: Matriz de confusión 

# In[ ]:


data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.2, random_state=42)
print([np.shape(X_train), np.shape(X_test)])


# In[ ]:


# usamos Support Vector Machine de :https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svc = svm.SVC()
svc.fit(X_train, y_train)
score_train = svc.score(X_train, y_train)
score_test = svc.score(X_test, y_test)


# In[ ]:


# para validar debe usar una matriz de confusión usando el siguiente código:
matr_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
pd.DataFrame(data = matr_confusion_test, columns = ['Prediccion spam', 'Prediccion no-spam'],
            index = ['Real spam', 'Real no-spam'])

