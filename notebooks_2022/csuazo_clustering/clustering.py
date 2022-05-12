#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install sweetviz
#!pip install pandas-profiling
#!pip install imbalanced-learn

import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import plotly.express as px
from datetime import datetime
import time
import matplotlib.cm as cm

# Importamos algunas librerías para análisis de datos
import sweetviz as sv

from sklearn.tree import DecisionTreeClassifier
#from sklearn import cluster
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, AgglomerativeClustering, SpectralClustering
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


# # Directory and Version Specification
# 

# In[ ]:


MODEL_TIMESTAMP = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

DATA_PATH = 'data/'

# Resolución de imágenes
resolution = 300

MODEL_TIMESTAMP


# In[ ]:


def silhoutte_analysis(model):
    num_clusters = range(2,10)
    for k in num_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(cluster_data) + (k + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = model(n_clusters=k, random_state=10) #KMeans(n_clusters=k, random_state=10)
        cluster_labels = clusterer.fit_predict(cluster_data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(cluster_data, cluster_labels)
        print("For n_clusters =", k,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(cluster_data, cluster_labels)

        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values =                 sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
        ax2.scatter(np.array(cluster_data)[:, 0], np.array(cluster_data)[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % k),
                     fontsize=14, fontweight='bold')

    plt.show()

def elbow_method(cluster_data, _model):
    sum_of_squared_distances = []
    num_clusters = range(1,10)

    for i in num_clusters:
        model = _model(n_clusters = i)#, random_state = 42)
        model.fit(cluster_data)
        sum_of_squared_distances.append(model.inertia_)

    plt.plot(num_clusters, sum_of_squared_distances, 'bx-')
    plt.xlabel('Valor de k (número de clusters)')
    plt.ylabel('Suma de las distancias al cuadrado')
    plt.title('Método del codo para buscar una k optima')
    plt.show()


# > Aquí definimos la función para tratar los datos del clustering.

# In[ ]:


def handle_non_numerical_data(df):
    
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            # great, found them. 
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string. 
            df[column] = list(map(convert_to_int,df[column]))

    return df


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

# Mostramos todas las columnas, con este comando evitamos que se oculten cuando son muchas.
pd.set_option('display.max_columns', None) 
atp


# In[ ]:


print('=================================================================')
print(atp.info())
print('=================================================================')


# # Clean Dataset
# > Realizamos la limpieza de algunos datos, realizamos algunas transformaciones en algunas variables categóricas.
# > 

# In[ ]:


df_clustering = atp.copy()
round_replace = {'R128': 128,
                  'R64': 64,
                  'R32': 32,
                  'R16': 16,
                  'QF': 4,
                  'SF': 2,
                  'F': 1
}

# Eliminamos las Round Robin (RR y ER)
df_clustering['round'].replace(round_replace, inplace = True)
df_clustering

df_pca = atp.copy()
df_pca['round'].replace(round_replace, inplace = True)


# In[ ]:


accident_type_replace = {}
for index,accident_type in enumerate(df_clustering.tourney_name.unique()):
    if not pd.isna(accident_type): accident_type_replace[accident_type] = int(index)
    
df_clustering['tourney_name'].replace(accident_type_replace, inplace = True)
df_clustering

df_pca['tourney_name'].replace(accident_type_replace, inplace = True)
df_pca.info()


# In[ ]:


COLUMNS_TO_GET = [
                  "surface",
                  "minutes",
                  "winner_ht", "loser_ht",
                  "w_ace", "l_ace",
                  "w_svpt", "l_svpt", # service points
                  "w_1stWon", "l_1stWon",
                  "w_2ndWon", "l_2ndWon",
                  "w_bpSaved", "l_bpSaved",
                  "w_bpFaced", "l_bpFaced",
                  "w_SvGms", "l_SvGms", # service games won
                  "winner_rank_points", "loser_rank_points",
                  "round",
                 ]
# Parámetros de los winners y losers con los que se realizan cálculos y se debn hacer drop.
UNNECESSARY_ATTR = ['tourney_id', 'tourney_name', 'winner_name', 'loser_name', 'winner_entry', 'winner_seed', 'loser_entry', 'loser_seed','tourney_date', 'winner_id', 'loser_id', 'score']

WL_DROP = [ 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms',  'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms']


# In[ ]:


df_clustering = df_clustering[df_clustering['round'] != 'RR']
df_clustering = df_clustering[df_clustering['round'] != 'ER']

df_clustering = df_clustering.drop(UNNECESSARY_ATTR, axis = 1) 
df_clustering = df_clustering.dropna()
df_clustering = df_clustering.drop_duplicates()

# Crearemos dos formulas para calculos del ganador y el perdedor para evitar la correlación de estas variables, tambien haremos un drop de estas variables.
df_clustering['w_calculation'] = df_clustering['w_svpt'] + df_clustering['w_1stIn'] + df_clustering['w_1stWon'] + df_clustering['w_2ndWon'] + df_clustering['w_SvGms']
df_clustering['l_calculation'] = df_clustering['l_svpt'] + df_clustering['l_1stIn'] + df_clustering['l_1stWon'] + df_clustering['l_2ndWon'] + df_clustering['l_SvGms']
df_clustering = df_clustering.drop(WL_DROP, axis = 1)

df_clustering = handle_non_numerical_data(df_clustering)

# Eliminamos los outlier para minutes
df_clustering = df_clustering[df_clustering['minutes'] < 400]

Counter(df_clustering['surface'])


# In[ ]:


#df_clustering[df_clustering['match'] > 400]


# # Resampling
# > Vamos a realizar un undersampling de la variable categorica a predecir **surface** vamos a reducir las muestras a la categoría minoritaria.

# In[ ]:


# Podemos observar que hay un desbalanceo en las varaibles a predecir.
df_without_under_sampling = df_clustering
print(Counter(df_clustering['surface']))


# In[ ]:


df_clustering.head()

X = df_clustering.drop('surface', axis=1)
y = df_clustering['surface']

# Hacemos undersampling a la categoría con menor cantidad de datos.
undersample = RandomUnderSampler(sampling_strategy='not minority')

X_over, y_over = undersample.fit_resample(X, y)
print(Counter(y_over))

df_clustering = pd.concat([X_over, y_over], axis=1)

# Eliminamos los outlier para minutes, se investigó los partidos más extensos de la historia del tennis y ninguno duró mas de 
# 660 minutos según el record mundial de los partidos de tennis más largos.
df_clustering = df_clustering[df_clustering['minutes'] < 400]

print(df_clustering.head())


# In[ ]:


atp_report = sv.analyze(df_clustering)
atp_report.show_html()


# # Clustering

# ## Principal Components Analysis **(PCA)** y KMeans
# 
# > Realizaremos un análisis de componentes principales y los graficaremos para ver si es posible ver algún patrón en los datos, esto se realizará previamente a la aplicación de técnicas de clustering.

# In[ ]:


# Graficaremos PCA con 3 componentes
scaler = StandardScaler()
df_clustering_pca = df_clustering.copy()
X = df_clustering_pca
y = df_clustering_pca['minutes'].copy()

pca = PCA(n_components = 3) 
components = pca.fit_transform(X)#,y)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, color=y, z=2,
    title=f'Varianza Total Explicada: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show()


# In[ ]:


wcss = []
for i in range(1, 21):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(components)
    wcss.append(kmeans_pca.inertia_)


# In[ ]:


plt.figure(figsize = (10, 8))
plt.plot(range(1,21), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Valor de k (número de clusters)')
plt.ylabel('Suma de las distancias al cuadrado')
plt.title('Método del Codo')
plt.show()


# In[ ]:


kmeans_pca = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
kmeans_pca.fit(components)


# In[ ]:


df_pca_kmeans = pd.concat([df_clustering_pca.reset_index(drop = True), pd.DataFrame(components)], axis = 1)
df_pca_kmeans.columns.values[-3: ] = ['Componente 1', 'Componente 2', 'Componente 3']
df_pca_kmeans['ATP KMeans PCA'] = kmeans_pca.labels_


# In[ ]:


df_pca_kmeans


# In[ ]:


# df_pca_kmeans['clase'] = df_pca_kmeans['ATP KMeans PCA'].map({0: 'Primero',
#                                                                  1: 'Segundo',
#                                                                  2: 'Tercero',
#                                                                  3: 'Cuarto'})


# In[ ]:


# x_axis = df_pca_kmeans['Componente 2']
# y_axis = df_pca_kmeans['Componente 1']
# plt.figure(figsize = (10, 8))
# sns.scatterplot(x_axis, y_axis, hue = df_pca_kmeans['clase'], palette = ['g', 'r', 'c', 'm'])
# plt.title('Cluster Utilizando Análisis de Componentes Principales')
# plt.show()


# > Podemos observar que aplicando el análisis de componentes principales PCA utilizando 3 componentes principales se puede explicar el 99.44 de la varianza de los datos. Esto podría ser un muy buen indicador ya que nos permitiría utilizar los datos filtrados para entrenar nuestros modelos, incluso PCA es una muy buena opción, aunque se pierda información del resto de las componentes.
# 
# > Pero podemos notar que al graficar las 3 componentes principales, no somos capaces de ver clústeres de datos.
# 

# ## KMeans

# In[ ]:


loser_rank_points = df_clustering.loser_rank_points.to_list()
w_calculation = df_clustering.w_calculation.to_list()
w_bpSaved = df_clustering.w_bpSaved.to_list()
cluster_data = list(zip(loser_rank_points, w_calculation , w_bpSaved))

elbow_method(cluster_data, KMeans)


# Haciendo un análsisi de la `Suma de las distancias acummuladas` o método del codo, podemos observar que `k = 2` y `k = 3` son buenas opciones, pero cómo podemos verificar cúal es la mejor? Para esto utilizaremos el análisis de silueta que se encuentra en `sklearn`.

# ### Silhoutte Analysis
# 
# [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#:~:text=Silhouette%20analysis%20can%20be%20used,like%20number%20of%20clusters%20visually)
# 

# In[ ]:


silhoutte_analysis(KMeans)


# Del análisis de silueta del cluster utilizando KMeans, podemos identificar que el mejor score es para `k = 2`, también es una buenas opción `k = 3`.

# In[ ]:


fig = px.scatter_3d(
    df_clustering, x = 'loser_rank_points', y = 'w_calculation', z = 'w_bpSaved',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show()


# ## MiniBatchKMeans

# In[ ]:


loser_rank_points = df_clustering.loser_rank_points.to_list()
w_calculation = df_clustering.w_calculation.to_list()
w_bpSaved = df_clustering.w_bpSaved.to_list()
cluster_data = list(zip(loser_rank_points, w_calculation , w_bpSaved))

elbow_method(cluster_data, MiniBatchKMeans)


# ### Silhoutte Analysis

# In[ ]:


silhoutte_analysis(MiniBatchKMeans)


# ## Agglomerative Clustering

# In[ ]:


# X = np.array(df_clustering.drop('surface', 1).astype(float))
# X = preprocessing.scale(X)
# y = np.array(df_clustering['surface'])

# clf = AffinityPropagation()
# clf.fit(X)
# cluster_centers_indices = clf.cluster_centers_indices_

# labels = clf.labels_

# n_clusters_ = len(cluster_centers_indices)

# print("Estimated number of clusters: %d" % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y, labels))
# # print(
# #     "Adjusted Mutual Information: %0.3f"
# #     % metrics.adjusted_mutual_info_score(y, labels)
# # )
# # print(
# #     "Silhouette Coefficient: %0.3f"
# #     % metrics.silhouette_score(X, labels, metric="sqeuclidean")
# # )

loser_rank_points = df_clustering.loser_rank_points.to_list()
w_calculation = df_clustering.w_calculation.to_list()
w_bpSaved = df_clustering.w_bpSaved.to_list()
cluster_data = list(zip(loser_rank_points, w_calculation , w_bpSaved))

elbow_method(cluster_data, SpectralClustering)


# ## Mean Shift

# In[ ]:


# original_df = pd.DataFrame.copy(df_clustering)

# df_ms = handle_non_numerical_data(df_clustering.copy())
# #df_ms.drop(['ticket','home.dest'], 1, inplace=True)

# X = np.array(df_ms.drop(['surface'], 1).astype(float))
# X = preprocessing.scale(X)
# y = np.array(df_ms['surface'])

# clf = MeanShift()
# clf.fit(X)

# labels = clf.labels_
# cluster_centers = clf.cluster_centers_

# original_df['cluster_group']=np.nan

# for i in range(len(X)):
#     original_df['cluster_group'].iloc[i] = labels[i]
    
# n_clusters_ = len(np.unique(labels))
# surface_rates = {}

# for i in range(n_clusters_):
#     temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
#     #print(temp_df.head())

#     surface_cluster = temp_df[  (temp_df['surface'] == 1) ]

#     surface_rate = len(surface_cluster) / len(temp_df)
#     #print(i,survival_rate)
#     surface_rates[i] = surface_rate
    
# print(surface_rates)

# print(original_df[ (original_df['cluster_group']==1) ])

# print(original_df[ (original_df['cluster_group']==0) ].describe())

# print(original_df[ (original_df['cluster_group']==2) ].describe())

# cluster_0 = (original_df[ (original_df['cluster_group']==0) ])

# #cluster_0_fc = (cluster_0[ (cluster_0['pclass']==1) ])

# #print(cluster_0_fc.describe())

loser_rank_points = df_clustering.loser_rank_points.to_list()
w_calculation = df_clustering.w_calculation.to_list()
w_bpSaved = df_clustering.w_bpSaved.to_list()
cluster_data = list(zip(loser_rank_points, w_calculation , w_bpSaved))

elbow_method(cluster_data, MeanShift)


# # Bibliografía
# 
# https://medium.com/analytics-vidhya/clustering-on-mixed-data-types-in-python-7c22b3898086
# 
# https://pythonprogramming.net/mean-shift-titanic-dataset-machine-learning-tutorial/?completed=/hierarchical-clustering-mean-shift-machine-learning-tutorial/
# 
# 
# 
