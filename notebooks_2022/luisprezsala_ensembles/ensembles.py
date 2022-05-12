#!/usr/bin/env python
# coding: utf-8

# # Repositorio

# Se puede consultar la metodología de desarrollo del proyecto y el histórico de imágenes en el siguiente repositorio:
# 
# - https://github.com/luperezsal/DM-Classification-Tree

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time


# # Directory and version specifications

# In[ ]:


from datetime import datetime

MODEL_TIMESTAMP = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

DATA_PATH = '../input/atp-matches/'

REPORTS_PATH = 'reports/ensembles/'
SAMPLE_GRAPH_RESULTS_PATH  = 'sample_graph_result/ensembles/'
TREE_PATH = 'tree/'

# Resolución de imágenes
resolution = 300
random_state = 2


# # Download and Store Data

# In[ ]:


# for index in range(0,22):
#     index_str = str(index)

#     print(index_str)
    
#     if len(index_str) == 1:
#         index_str = '0' + index_str

#     print(index_str)

#     url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_20{}.csv".format(index_str)
#     print(url)

#     FILE_NAME = "atp_matches_20{}.csv".format(index_str)

#     df = pd.read_csv(url, index_col=0, parse_dates=[0])
#     df.to_csv(DATA_PATH + FILE_NAME)

# # data_frame = pd.read_csv(DATA_PATH + FILE_NAME)


# # Load Data

# In[ ]:


atp = pd.DataFrame()

years_index_20_22 = range(0,22)

for index in years_index_20_22:
    index_str = str(index)

    if len(index_str) == 1:
        index_str = '0' + index_str

    FILE_NAME = "atp_matches_20{}.csv".format(index_str)

    data_frame_iter = pd.read_csv(DATA_PATH + FILE_NAME)
    atp = pd.concat([atp, data_frame_iter])

pd.set_option('display.max_columns', None)
atp


# In[ ]:


# COLUMNS_TO_REMOVE = ["tourney_id", "tourney_name", "tourney_date",
#                      "match_num",
#                      "winner_id", "loser_id",
#                      "winner_seed", "loser_seed",
#                      "winner_name", "loser_name",
#                      "winner_ioc", "loser_loc",
#                      "winner_rank", "loser_rank",
#                      "winner_rank_points", "loser_rank_points",
#                      "round"]


# # Clean Dataset

# In[ ]:


# Vamos a eliminar las variables que son identificadores, nombres etc
# Incluimos en el drop las siguientes variables que tienen muchos registros NaN
# quitaremos las columna de score
df_regression = atp


COLUMNS_TO_REMOVE = ['tourney_id', 'tourney_name', 'tourney_date',
                     'winner_name', 'loser_name',
                     'winner_entry', 'loser_entry',
                     'winner_seed', 'loser_seed',
                     'winner_id', 'loser_id',
                     'score']

df_regression = df_regression.drop(COLUMNS_TO_REMOVE, axis = 1) 
df_regression = df_regression.dropna()
df_regression = df_regression.drop_duplicates()

# Crearemos dos formulas para calculos del ganador y el perdedor para evitar la correlación de estas variables, tambien haremos un drop de estas variables.
df_regression['w_calculation'] = df_regression['w_svpt'] + df_regression['w_1stIn'] + df_regression['w_1stWon'] + df_regression['w_2ndWon'] + df_regression['w_SvGms']
df_regression['l_calculation'] = df_regression['l_svpt'] + df_regression['l_1stIn'] + df_regression['l_1stWon'] + df_regression['l_2ndWon'] + df_regression['l_SvGms']

df_regression = df_regression.drop(['w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms'], axis = 1) 

df_regression = df_regression._get_numeric_data() #drop non-numeric cols


# # Split Data

# In[ ]:


from sklearn.model_selection import train_test_split

X = df_regression.drop('minutes', axis = 1) 
y = df_regression['minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)


# # Ensembles

# Los ensembles son técnicas que permiten combinar las predicciones de distintos modelos con el objetivo de aumentar la prediccción global de los resultados.
# En esta sección aplicaremos distintas técnicas de ensembles y ejecutaremos cada una de ellas al dataset, con el objetivo de predecir la duración de los partidos como se hizo en el apartado de Regresión:
# 
# - Ensemble Bagging
# - Ensembles Boosting
#     - AdaBoost
#     - Gradient Boosting Regressor
# - Ensemble Stacking

# Lo primero será buscar el Árbol de regresión que más accuracy nos dé mediante cross-validation para usarlo en el proceso de Bagging.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


best_params = {'criterion': 'absolute_error',
               'max_depth': 7,
               'max_features': None,
               'min_weight_fraction_leaf': 0.0,
               'splitter': 'best'}

decision_tree = DecisionTreeRegressor(random_state = random_state)
decision_tree.set_params(**best_params)


# Inicializamos un diccionario `info` donde almacenaremos toda la información relacionada con los ensembles, sus resultados, su tiempo de ejecución, el nombre del ensemble, etc.

# In[ ]:


info = {}


# ## Bagging

# Bootstrap Aggregation o Bagging es una técnica que permite utilizar el resampling Bootstrap para consturir ensembles y poder utilizar varios conjuntos de datos para cada uno de los modelos pertenecientes a la arquitectura ensemble diseñada. Los modelos que se utilzarán en esta arquitectura serán todos el mismo, con la única diferencia de haberlos entrenado con distintos conjuntos de datos.
# 
# En el caso de la regresión, esta técnica permite calcular la media de las prediccciones de cada uno de los modelos entrenados mediante Bootstrap para obtener un resultado final que dependa de todos los modelos.
# 
# 
# [Referencia](https://machinelearningmastery.com/bagging-ensemble-with-python/)

# ### Definition

# Crearemos un Ensemble de tipo Bagging con 10 estimadores, es decir, se entrenarán diez árboles de decisión que predecirán las muestras del conjunto de test y el resultado que ofrecerá será la media de todas las prediciones.

# In[ ]:


from sklearn.ensemble import BaggingRegressor

MODEL_NAME = 'bagging'
info[MODEL_NAME] = {}
info[MODEL_NAME]['model_name'] = MODEL_NAME

num_models = 10
bagging = BaggingRegressor(decision_tree,
                           n_estimators = num_models,
                           random_state = random_state)


# ### Training

# Entrenamos el ensemble Bagging y almacenamos el tiempo de ejecución para analizar en apartados posteriores los rendimientos de los modelos.

# In[ ]:


start = time.time()

bagging_regressor = bagging.fit(X_train, y_train)

end = time.time()

ellapsed_time = round(end - start, 2)

info[MODEL_NAME]['time'] = ellapsed_time

print(f"Done in {ellapsed_time} (s)")


# ### Metrics

# Las métricas que utilizaremos para medir los errores del modelos serán:
# - **Mean Squared Error (MSE)**:  error cuadrático medio, es la suma al cuadrado de los residuos dividido entre el número de muestras totales. Es deseable minimizar este estadístico.
# - **Root Mean Squared Error (RMSE)**: raíz del error cuadrático medio, es la raíz cuadrada de la suma de los residuos al cuadrado entre el número de muestras totales. Es deseable minimizar este estadístico.
# - **R-Squared**: indica la cantidad de varianza en los datos explicada por el modelo actual. Es deseable maximizar este estadístico.

# In[ ]:


y_pred = bagging_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
info[MODEL_NAME]['mse'] = mse

print("MSE: ", info[MODEL_NAME]['mse'])

rmse = mean_squared_error(y_true  = y_test,
                          y_pred  = y_pred,
                          squared = False
                        )
info[MODEL_NAME]['rmse'] = rmse

print("RMSE: ", info[MODEL_NAME]['rmse'])

score = bagging_regressor.score(X_test, y_test)
info[MODEL_NAME]['score'] = score

print("R-squared:", info[MODEL_NAME]['score']) 


# ### Graphic Results

# A continuación graficaremos un ejemplo de cómo el modelo se ajusta a los valores verdaderos sobre las muestras del conjunto de test. Es deseable tener un modelo que prediga valores lo más cercanos posibles a las muestras verdaderas.

# In[ ]:


n_samples = range(len(y_test[:50]))

info[MODEL_NAME]['y_pred'] = y_pred[:50]

plt.figure(figsize=(15,10))

plt.scatter(n_samples, y_test[:50], s = 15, color = 'red', label = "original")
plt.plot(n_samples, y_pred[:50], linewidth = 1.1, color = 'blue', label = "predicted")

plt.title("Y-true / Y-predicted (minutes)")
plt.xlabel('Sample')
plt.ylabel('Minutes')

plt.legend(loc = 'best',
           fancybox = True,
           shadow = True)

GRAPH_PATH = f"{SAMPLE_GRAPH_RESULTS_PATH}{MODEL_NAME}/"
FILE_NAME  = f"{MODEL_NAME}_{MODEL_TIMESTAMP}.png"

plt.grid(True)
# plt.savefig(GRAPH_PATH + FILE_NAME, dpi = resolution)
plt.show()


# ## Boosting

# ### ADABoost

# AdaBoost utiliza múltiples weak learners (árboles de decisión de un nivel) que son agregados secuencialmente al conjunto de modelos, con el objetivo de que cada uno de estos árboles minimice el error producido por el anterior modelo.
# 
# Esto se consigue asignando una serie de pesos (ponderación) a cada una de las muestras que estén clasificadas erróneamente (clasificación) o que tengan un error alto (regresión).
# 
# [Referencia](https://machinelearningmastery.com/adaboost-ensemble-in-python/)

# #### Definition

# Crearemos un Ensemble de tipo AdaBoost con 10 estimadores, es decir, se entrenarán diez árboles de decisión que minimizarán el error de los anteriores árboles.

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression

MODEL_NAME = 'adaboost'
info[MODEL_NAME] = {}
info[MODEL_NAME]['model_name'] = MODEL_NAME

num_models = 10

ada_boosting_regresor = AdaBoostRegressor(random_state = random_state)


# #### Training

# Entrenamos el ensemble Ada Boost y anotamos el tiempo de ejecución.

# In[ ]:


start = time.time()

ada_boosting_regresor.fit(X, y)

end = time.time()

ellapsed_time = round(end - start, 2)
info[MODEL_NAME]['time'] = ellapsed_time

print(f"Done in {ellapsed_time} (s)")


# #### Metrics

# Calculamos los estadísticos **MSE**, **RMSE** y **R-Squared**.

# In[ ]:


y_pred = ada_boosting_regresor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
info[MODEL_NAME]['mse'] = mse

print("MSE: ", info[MODEL_NAME]['mse'])

rmse = mean_squared_error(y_true  = y_test,
                          y_pred  = y_pred,
                          squared = False
                        )
info[MODEL_NAME]['rmse'] = rmse

print("RMSE: ", info[MODEL_NAME]['rmse'])

score = ada_boosting_regresor.score(X_test, y_test)
info[MODEL_NAME]['score'] = score

print("R-squared:", info[MODEL_NAME]['score']) 


# #### Graphic Results

# Graficamos un interavalo de predicciones de 50 muestras de test.

# In[ ]:


n_samples = range(len(y_test[:50]))

info[MODEL_NAME]['y_pred'] = y_pred[:50]

plt.figure(figsize=(15,10))

plt.scatter(n_samples, y_test[:50], s = 15, color = 'red', label = "original")
plt.plot(n_samples, y_pred[:50], linewidth = 1.1, color = 'blue', label = "predicted")

plt.title("Y-true / Y-predicted (minutes)")
plt.xlabel('Sample')
plt.ylabel('Minutes')

plt.legend(loc='best',fancybox = True, shadow = True)

GRAPH_PATH = f"{SAMPLE_GRAPH_RESULTS_PATH}boosting/{MODEL_NAME}/"
FILE_NAME  = f"{MODEL_NAME}_{MODEL_TIMESTAMP}.png"

plt.grid(True)
# plt.savefig(GRAPH_PATH + FILE_NAME, dpi = resolution)
plt.show()


# ### Gradient Boosting Regressor

# Al igual que en el caso del AdaBoost, utiliza la técnica Boosting para entrenar los modelos, es decir, trata de minimizar los residuos de los modelos anteriores. Sin embargo, utiliza el método de Descenso por Gradiente en lugar de la asignación de pesos como en el caso anterior.
# 
# [Referencia](https://www.cienciadedatos.net/documentos/py09_gradient_boosting_python.html)

# #### Definition

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

MODEL_NAME = 'gradientboost'
info[MODEL_NAME] = {}
info[MODEL_NAME]['model_name'] = MODEL_NAME

num_models = 10

gradient_boosting = GradientBoostingRegressor(criterion = best_params['criterion'],
                                              max_depth = best_params['max_depth'],
                                              n_estimators  = num_models,
                                              random_state = random_state)


# #### Training

# Entrenamos el ensemble Gradient Boosting y anotamos el tiempo de ejecución.

# In[ ]:


start = time.time()

gradient_boosting_regressor = gradient_boosting.fit(X_train, y_train)

end = time.time()

ellapsed_time = round(end - start, 2)
info[MODEL_NAME]['time'] = ellapsed_time

print(f"Done in {ellapsed_time} (s)")


# #### Metrics

# Calculamos los estadísticos **MSE**, **RMSE** y **R-Squared**.

# In[ ]:


y_pred = gradient_boosting.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
info[MODEL_NAME]['mse'] = mse

print("MSE: ", info[MODEL_NAME]['mse'])

rmse = mean_squared_error(y_true  = y_test,
                          y_pred  = y_pred,
                          squared = False
                        )
info[MODEL_NAME]['rmse'] = rmse

print("RMSE: ", info[MODEL_NAME]['rmse'])

score = gradient_boosting.score(X_test, y_test)
info[MODEL_NAME]['score'] = score

print("R-squared:", info[MODEL_NAME]['score']) 


# #### Graphic Results

# Graficamos un interavalo de predicciones de 50 muestras de test.

# In[ ]:


n_samples = range(len(y_test[:50]))

info[MODEL_NAME]['y_pred'] = y_pred[:50]

plt.figure(figsize=(15,10))

plt.scatter(n_samples, y_test[:50], s = 15, color = 'red', label = "original")
plt.plot(n_samples, y_pred[:50], linewidth = 1.1, color = 'blue', label = "predicted")

plt.title("Y-true / Y-predicted (minutes)")
plt.xlabel('Sample')
plt.ylabel('Minutes')

plt.legend(loc='best',fancybox = True, shadow = True)

GRAPH_PATH = f"{SAMPLE_GRAPH_RESULTS_PATH}boosting/{MODEL_NAME}/"
FILE_NAME  = f"{MODEL_NAME}_{MODEL_TIMESTAMP}.png"

plt.grid(True)
# plt.savefig(GRAPH_PATH + FILE_NAME, dpi = resolution)
plt.show()


# ## Stacking

# Stacking permite utilizar distintas tipologías de modelos para crear una arquitectura que combine las predicciones de éstos.
# 
# A diferencia del Bagging, el Stacking nos da la flexibilidad de explorar distintos modelos en lugar de siempre el mismo.
# 
# Es por esto que en esta sección crearemos un Ensemble Stacking con los siguientes modelos:
# - **KNeighborsRegressor**: configurado con tres vecinos cercanos para calcular la regresión. 
# - **DecisionTreeRegressor**: con los parámetros óptimos calculados en el apartado de Regresión.
# - **SVR**: con los paráemtros por defecto.
# 
# Debido a motivos de tiempo de ejecución, no ha sido posible buscar los mejores hiperparámetros para cada uno de los modelos que pertenecen a la arquitectura de Stacking, por lo que un posible trabajo a futuro sería encontrar estos parámetros que logren minimizar los residuos.
# 
# [Referencia](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)

# ### Definition

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor


MODEL_NAME = 'stacking'
info[MODEL_NAME] = {}
info[MODEL_NAME]['model_name'] = MODEL_NAME

base_models = list()
base_models.append(('knn', KNeighborsRegressor(n_neighbors = 3)))
base_models.append(('cart', DecisionTreeRegressor(criterion = best_params['criterion'],
                                                  max_depth = best_params['max_depth'],
                                                  random_state = random_state)))
base_models.append(('svm', SVR()))


meta_learner = DecisionTreeRegressor(criterion = best_params['criterion'],
                                     max_depth = best_params['max_depth'],
                                     random_state = random_state)

stacking = StackingRegressor(estimators = base_models,
                             final_estimator = meta_learner,
                             cv = 5)


# ### Training

# Entrenamos el ensemble Stacking y anotamos el tiempo de ejecución.

# In[ ]:


start = time.time()

stacking_regressor = stacking.fit(X_train, y_train)

end = time.time()

ellapsed_time = round(end - start, 2)
info[MODEL_NAME]['time'] = ellapsed_time

print(f"Done in {ellapsed_time} (s)")


# ### Metrics

# Calculamos los estadísticos **MSE**, **RMSE** y **R-Squared**.

# In[ ]:


from sklearn.metrics import mean_squared_error

y_pred = stacking_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
info[MODEL_NAME]['mse'] = mse

print("MSE: ", info[MODEL_NAME]['mse'])

rmse = mean_squared_error(y_true  = y_test,
                          y_pred  = y_pred,
                          squared = False
                        )
info[MODEL_NAME]['rmse'] = rmse

print("RMSE: ", info[MODEL_NAME]['rmse'])


score = stacking_regressor.score(X_test, y_test)
info[MODEL_NAME]['score'] = score

print("R-squared:", info[MODEL_NAME]['score'])


# ### Graphic Results

# Graficamos un interavalo de predicciones de 50 muestras de test.

# In[ ]:


n_samples = range(len(y_test[:50]))

info[MODEL_NAME]['y_pred'] = y_pred[:50]

plt.figure(figsize=(15,10))

plt.scatter(n_samples, y_test[:50], s = 15, color = 'red', label = "original")
plt.plot(n_samples, y_pred[:50], linewidth = 1.1, color = 'blue', label = "predicted")

plt.title("Y-true / Y-predicted (minutes)")
plt.xlabel('Sample')
plt.ylabel('Minutes')

plt.legend(loc = 'best',
           fancybox = True,
           shadow = True)

GRAPH_PATH = f"{SAMPLE_GRAPH_RESULTS_PATH}{MODEL_NAME}/"
FILE_NAME  = f"{MODEL_NAME}_{MODEL_TIMESTAMP}.png"

plt.grid(True)
# plt.savefig(GRAPH_PATH + FILE_NAME, dpi = resolution)
plt.show()


# # Reports

# A continuación analizaremos los resultados de los ensembles obtenidos, centrándonos concretamente en:
# 
# - **Tiempo**: tiempo empleado en entrenar el ensemble.
# - **Mean Squared Error (MSE)**: Error cuadrático medio de las predicciones respecto a su valor verdadero.
# - **Root-mean-square deviation (RMSE)**: Raíz cuadrada del error cuadrático medio o MSE.
# - **Score**: Media de la precisión de los modelos.

# In[ ]:


FEATURES = ['model_name', 'time', 'mse', 'rmse', 'score']
summary_dataframe = pd.DataFrame(columns = FEATURES)

for key in info:
    row = info[key]
    fields = []
    for feature in row:
        if (feature in FEATURES):
            fields.append(row[feature])

    row_series = pd. Series(fields, index = summary_dataframe.columns)
    summary_dataframe = summary_dataframe.append(row_series, ignore_index = True)

SAVE_PATH =  f"{REPORTS_PATH}{MODEL_TIMESTAMP}.csv"

# summary_dataframe.to_csv(SAVE_PATH, index = True)
summary_dataframe.style.highlight_min(subset = ['time', 'mse', 'rmse'], color = 'green')                       .highlight_max(subset = ['score'], color = 'green')                       .highlight_max(subset = ['time', 'mse', 'rmse'], color = 'red')                       .highlight_min(subset = ['score'], color = 'red')


# Podemos observar que cada uno de los modelos emplea un tiempo de entrenamiento notablemente diferente. Esto, junto a la distinta precisión entre los modelos, genera la necesidad de analizar cada modelo individualmente.
# 
# Como vemos en el reporte, el modelo que mejor resultados nos ofrece es el Bagging, además con un tiempo de ejecución razonable con respecto a los modelos Gradient Boosting y Stacking, no siendo así con respecto al Ensemble AdaBoost, sin embargo, debido a la diferencia de precisión, este modelo queda descartado para este problema. 
# 
# Por lo que, con esta configuración de hiperparámetros con los modelos actuales, eligiríamos el modelo Ensemble Bagging como solución a este problema.

# In[ ]:


plt.figure(figsize=(15,10))

plt.scatter(n_samples, y_test[:50], s = 20, color = 'red', label = "original")

for key in info:
    y_pred = info[key]['y_pred']
    model_name = info[key]['model_name']
    plt.plot(n_samples, y_pred, linewidth = 1.1, label = model_name)

plt.title("Y-true / Y-predicted (minutes)")
plt.xlabel('Sample')
plt.ylabel('Minutes')

plt.legend(loc = 'best',
           fancybox = True,
           shadow = True)

GRAPH_PATH = f"{SAMPLE_GRAPH_RESULTS_PATH}/"
FILE_NAME  = f"{MODEL_TIMESTAMP}.png"

plt.grid(True)
# plt.savefig(GRAPH_PATH + FILE_NAME, dpi = resolution)
plt.show()


# Observando la gráfica podemos apreciar dos casos contrapuestos en lo que a valores predichos se refiere.
# 
# Si ponemos el foco en el ensemble Stacking (rojo) y Bagging (azul), podemos comprobar que siguen una predicción bastante cercana a los valores reales en comparación a los métodos de Ensembles AdaBoost (naranja) y GradientBoost (verde).
# 
# Visualizando los valores de los resultados de la sección anterior comprobamos que tanto el modelo Bagging como el modelo Stacking tienen mayores precisiones con respecto a AdaBoost y GraidentBoost.

# # Conclusiones

# Como hemos podido comprobar, es posible aplicar distintas técnicas de ensembles a un problema, concretamente:
# - Bagging
# - Boosting
#     - AdaBoost:
#     - Gradient Boosting Regressor
# - Stacking
# 
# Después de haber probado las distintas arquitecturas, los mejores resultados se han obtenido en Bagging con un R-Squared de 0.8586 con respecto al 0.849 del ensemble Stacking, 0.752 del Gradient Boost y 0.723 del AdaBoost respectivamente.
#     
# 
# Por motivos de tiempo de ejecución no se han podido testar tantas combinaciones distintas de modelos e hiperparámetros deseadas a la hora de construir los ensembles.
# 
# Por lo que se propone como trabajos futuros la implementación y el estudio de distintos modelos en las arquitecturas de los ensembles y realizar una búsqueda más profunda de hiperparámetros que logren minimizar aún más el error producido en la regresión y, por ende, mejorar la precisión de los resultados.
