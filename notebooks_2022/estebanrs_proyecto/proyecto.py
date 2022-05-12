#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np


# In[ ]:





# In[ ]:


pip install xlrd


# In[ ]:


import xlrd


# # Cargamos Datos

# In[ ]:


data2015 = pd.read_csv("../input/marginacion-municipal/Base_Indice_de_marginacion_municipal_90-15.csv", encoding='latin1')


# In[ ]:


data2015.head()


# In[ ]:


data2020 = pd.read_excel("../input/marginacion-municipal/IMM_2020.xls", sheet_name = "IMM_2020", engine="xlrd")


# In[ ]:


data2020.head()


# In[ ]:


data2020.isnull().sum().sum()


# In[ ]:


del data2020['CVE_ENT']
del data2020['CVE_MUN']


# In[ ]:


# Extraemos los nombres por estado
names = sorted(list(set(data2020['NOM_ENT'])))
len(names)


# In[ ]:


# Extraemos datafame de un estado particular
ag = data2020[data2020['NOM_ENT'] == names[0]]


# In[ ]:


data2020[data2020['NOM_ENT'] == 'Chiapas']


# # Dataset 2020

# In[ ]:


len(set(data2020['NOM_ENT']))


# In[ ]:


total_data = [sum(data2020['NOM_ENT'] == name) for name in names]
dict_data = {names[i] : total_data[i] for i in range(len(names))}

 # Revisamos estados con más datos disponibles
dict(sorted(dict_data.items(), key=lambda item: item[1]))


# ## Seleccionamos Estados

# In[ ]:


# Seleccionamos estados del dataframe
oaxaca   = data2020[data2020['NOM_ENT'] == 'Oaxaca']
puebla   = data2020[data2020['NOM_ENT'] == 'Puebla']
veracruz = data2020[data2020['NOM_ENT'] == 'Veracruz de Ignacio de la Llave']


# In[ ]:


# Removemos columna de estado
del oaxaca['NOM_ENT']
del puebla['NOM_ENT']
del veracruz['NOM_ENT']


# In[ ]:


# Revisamos categoría de índice de pobreza
set(oaxaca['GM_2020']), set(puebla['GM_2020']), set(veracruz['GM_2020'])


# # PCA

# In[ ]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plot


# ### Oaxaca

# In[ ]:


def do_pca(dataframe, num_components, show_names = False, size = 30) :
    dataframe_pca = dataframe[['POB_TOT','ANALF','SBASC','OVSDE','OVSEE','OVSAE','OVPT','VHAC','PL.5000','PO2SM']]
    # Normalize dataframe
    dataframe_pca=(dataframe_pca - dataframe_pca.mean()) / dataframe_pca.std()
    
    # Do pca 
    pca                 = PCA(n_components = num_components)
    principalComponents = pca.fit_transform(dataframe_pca)

    # Reformat and view results
    loadings = pd.DataFrame(pca.components_.T,
    columns  = ['PC%s' % _ for _ in range(num_components)],
    index    = dataframe_pca.columns)
    print(loadings)

    plot.plot(pca.explained_variance_ratio_)
    plot.ylabel('Explained Variance')
    plot.xlabel('Components')
    plot.show()
    
    
    # Keep two components
    finalDf = pd.DataFrame(pca.transform(dataframe_pca)[:, :2])
    # Agrego categorias
    finalDf['target'] = dataframe['GM_2020'].to_numpy()
    
    
    # Plot projections
    fig = plot.figure(figsize = (size,size))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = list(set(finalDf['target']))
    colors = ['r', 'g', 'b', 'c', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 0]
                   , finalDf.loc[indicesToKeep, 1]
                   , c = color
                   , s = 50)
    if show_names :
        for name, cor in zip(dataframe['NOM_MUN'], finalDf[[0, 1]].to_numpy()):
            ax.annotate(name, cor)
    ax.legend(["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"])
    ax.grid()
    
    
    return loadings, finalDf


# In[ ]:


oaxaca.loc[oaxaca["GM_2020"] == "Muy alto", "GM_2020"] = 4
oaxaca.loc[oaxaca["GM_2020"] == "Alto"    , "GM_2020"] = 3
oaxaca.loc[oaxaca["GM_2020"] == "Medio"   , "GM_2020"] = 2
oaxaca.loc[oaxaca["GM_2020"] == "Bajo"    , "GM_2020"] = 1
oaxaca.loc[oaxaca["GM_2020"] == "Muy bajo", "GM_2020"] = 0

oaxaca


# In[ ]:


oaxaca.isnull().sum().sum()


# In[ ]:


_, _, = do_pca(oaxaca, 2, show_names = True, size = 30)


# ### Veracruz

# In[ ]:


veracruz.loc[veracruz["GM_2020"] == "Muy alto", "GM_2020"] = 4
veracruz.loc[veracruz["GM_2020"] == "Alto"    , "GM_2020"] = 3
veracruz.loc[veracruz["GM_2020"] == "Medio"   , "GM_2020"] = 2
veracruz.loc[veracruz["GM_2020"] == "Bajo"    , "GM_2020"] = 1
veracruz.loc[veracruz["GM_2020"] == "Muy bajo", "GM_2020"] = 0

veracruz


# In[ ]:


do_pca(veracruz, 2, show_names = True, size = 30)


# ### Puebla

# In[ ]:


puebla.loc[puebla["GM_2020"] == "Muy alto", "GM_2020"] = 4
puebla.loc[puebla["GM_2020"] == "Alto"    , "GM_2020"] = 3
puebla.loc[puebla["GM_2020"] == "Medio"   , "GM_2020"] = 2
puebla.loc[puebla["GM_2020"] == "Bajo"    , "GM_2020"] = 1
puebla.loc[puebla["GM_2020"] == "Muy bajo", "GM_2020"] = 0

puebla


# In[ ]:


do_pca(puebla, 2, show_names = True, size = 30)


# # Isomap

# In[ ]:


from sklearn import manifold


# In[ ]:


def do_isomap(dataframe, num_components, n_neighbors, show_names = False, size = 30):
    dataframe_iso = dataframe[['POB_TOT','ANALF','SBASC','OVSDE','OVSEE','OVSAE','OVPT','VHAC','PL.5000','PO2SM']]
    # Normalize dataframe
    dataframe_iso=(dataframe_iso - dataframe_iso.mean()) / dataframe_iso.std()
    
    # Do isomap
    iso = manifold.Isomap(n_neighbors=n_neighbors, n_components = num_components)
    iso.fit(dataframe_iso)
    manifold_2Da = iso.transform(dataframe_iso)
    manifold_2D  = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])
   
    
    # Keep two components
    finalDf = pd.DataFrame(manifold_2D)
    # Agrego categorias
    finalDf['target'] = dataframe['GM_2020'].to_numpy()
       
    
    # Plot projections
    fig = plot.figure(figsize = (size,size))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Fist Component 1', fontsize = 15)
    ax.set_ylabel('Second Component 2', fontsize = 15)
    ax.set_title('2 component isomap', fontsize = 20)
    targets = list(set(finalDf['target']))
    colors = ['r', 'g', 'b', 'c', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Component 1']
                   , finalDf.loc[indicesToKeep, 'Component 2']
                   , c = color
                   , s = 50)
    if show_names :
        for name, cor in zip(dataframe['NOM_MUN'], finalDf[['Component 1', 'Component 2']].to_numpy()):
            ax.annotate(name, cor)
    ax.legend(["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"])
    ax.grid()
    

    return manifold_2D, finalDf


# ### Veracruz

# In[ ]:


do_isomap(veracruz, 2, 6, show_names = True, size = 30)


# ### Oaxaca

# In[ ]:


do_isomap(oaxaca, 2, 6, show_names = True, size = 30)


# ## Puebla

# In[ ]:


do_isomap(puebla, 2, 6, show_names = True, size = 30)


# # TSNE

# In[ ]:


from sklearn.manifold import TSNE
import seaborn as sns


# In[ ]:


def do_tsne(dataframe, num_components, perplexity, show_names = False, size = 30):
    MACHINE_EPSILON = np.finfo(np.double).eps
    
    dataframe_tsne = dataframe[['POB_TOT','ANALF','SBASC','OVSDE','OVSEE','OVSAE','OVPT','VHAC','PL.5000','PO2SM']]
    # Normalize dataframe
    dataframe_tsne=(dataframe_tsne - dataframe_tsne.mean()) / dataframe_tsne.std()
    
    # Do isomap
    m             = TSNE(n_components = num_components, perplexity = perplexity,  verbose = 0)
    tsne_features = m.fit_transform(dataframe_tsne)
    
    # Keep two components
    finalDf       = pd.DataFrame(tsne_features, columns=['Component 1', 'Component 2'])    
    # Agrego categorias
    finalDf['target'] = dataframe['GM_2020'].to_numpy()
       
    
    # Plot projections
    fig = plot.figure(figsize = (size,size))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Fist Component 1', fontsize = 15)
    ax.set_ylabel('Second Component 2', fontsize = 15)
    ax.set_title('2 component isomap', fontsize = 20)
    targets = list(set(finalDf['target']))
    colors = ['r', 'g', 'b', 'c', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Component 1']
                   , finalDf.loc[indicesToKeep, 'Component 2']
                   , c = color
                   , s = 50)
    if show_names :
        for name, cor in zip(dataframe['NOM_MUN'], finalDf[['Component 1', 'Component 2']].to_numpy()):
            ax.annotate(name, cor)
    ax.legend(["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"])
    ax.grid()
    

    return tsne_features, finalDf


# In[ ]:


do_tsne(veracruz, 2, 5, show_names = True, size = 30)


# In[ ]:


do_tsne(oaxaca, 2, 5, show_names = True, size = 30)


# In[ ]:


do_tsne(puebla, 2, 5, show_names = True, size = 30)


# In[ ]:


do_tsne(puebla, 2, 5, show_names = True, size = 30)


# # LLE

# In[ ]:


from sklearn.manifold import LocallyLinearEmbedding


# In[ ]:


def do_lle(dataframe, num_components, n_neighbors, show_names = False, size = 30):
    MACHINE_EPSILON = np.finfo(np.double).eps
    
    dataframe_lle = dataframe[['POB_TOT','ANALF','SBASC','OVSDE','OVSEE','OVSAE','OVPT','VHAC','PL.5000','PO2SM']]
    # Normalize dataframe
    dataframe_lle=(dataframe_lle - dataframe_lle.mean()) / dataframe_lle.std()
    
    # Do lle
    m             = LocallyLinearEmbedding(n_neighbors = n_neighbors, n_components = num_components)
    lle_features = m.fit_transform(dataframe_lle)
    
    # Keep two components
    finalDf       = pd.DataFrame(lle_features, columns=['Component 1', 'Component 2'])    
    # Agrego categorias
    finalDf['target'] = dataframe['GM_2020'].to_numpy()
       
    
    # Plot projections
    fig = plot.figure(figsize = (size,size))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Fist Component 1', fontsize = 15)
    ax.set_ylabel('Second Component 2', fontsize = 15)
    ax.set_title('2 component LLE', fontsize = 20)
    targets = list(set(finalDf['target']))
    colors = ['r', 'g', 'b', 'c', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Component 1']
                   , finalDf.loc[indicesToKeep, 'Component 2']
                   , c = color
                   , s = 50)
    if show_names :
        for name, cor in zip(dataframe['NOM_MUN'], finalDf[['Component 1', 'Component 2']].to_numpy()):
            ax.annotate(name, cor)
    ax.legend(["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"])
    ax.grid()
    

    return lle_features, finalDf


# In[ ]:


do_lle(puebla, 2, 20, show_names = True, size = 30)


# In[ ]:


do_lle(veracruz, 2, 20, show_names = True, size = 30)


# In[ ]:


do_lle(oaxaca, 2, 20, show_names = True, size = 30)


# In[ ]:




