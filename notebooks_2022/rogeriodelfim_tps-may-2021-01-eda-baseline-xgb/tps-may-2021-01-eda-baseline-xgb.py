#!/usr/bin/env python
# coding: utf-8

# <h1 div class='alert alert-info alert-dismissible'><center> Ponto de partida (EDA)</center></h1>
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/26480/logos/header.png?t=2021-04-09-00-57-05)

# A edição de maio do problema de classificação binária da série Tabular Playground de 2022 que inclui várias interações de recursos diferentes. Esta competição é uma oportunidade para explorar vários métodos para identificar e explorar essas interações de recursos. 

# # <div class="alert alert-info alert-dismissible">  OBJETIVO </div> 

# Neste notebook vamos fazer uma análise (EDA) para conhencer os dados e estabelecer uma linha de base, mostrarei as etapas iniciais de uma competição do Kaggle - desde a compreensão do conjunto de dados até a preparação dos dados para serem usados em um modelo machine learning. Vamos passar pelas seguintes tarefas:
# 
# - Leitura no conjunto de dados
# - Calculando estatísticas sobre o conjunto de dados
# - Visualização univariada
# - Visualizando multivariada
# - Pré-processamento 
# 
# 
# 

# ---

# # <div class="alert alert-info alert-dismissible">  1. IMPORTAÇÕES </div> 

# ## 1.1. Instalações

# In[ ]:


# https://pub.towardsai.net/use-google-colab-like-a-pro-39a97184358d
COLAB = 'google.colab' in str(get_ipython()) 

if COLAB:        
    get_ipython().system('pip install --q scikit-plot')
    get_ipython().system('pip install --q category_encoders')
    get_ipython().system('pip install --q shap')
    get_ipython().system('pip install --q inflection    ')
    #!pip install --q pycaret

    from google.colab import drive
    drive.mount('/content/drive')


# ## 1.2. Bibliotecas 

# In[ ]:


import warnings
import random
import os
import gc
import torch
import sklearn.exceptions
import datetime
import shap


# In[ ]:


import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt 
import seaborn           as sns
import joblib            as jb
import xgboost           as xgb
import scipy.stats       as stats


# In[ ]:


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing   import StandardScaler, MinMaxScaler, power_transform
from sklearn.preprocessing   import PowerTransformer, RobustScaler, Normalizer
from sklearn.preprocessing   import MaxAbsScaler, QuantileTransformer, LabelEncoder
from sklearn                 import metrics
from sklearn.metrics         import ConfusionMatrixDisplay, confusion_matrix


# In[ ]:


from datetime                import datetime


# ---

# ## 1.3. Funções
# Aqui centralizamos todas as funções desenvolvidas durante o projeto para melhor organização do código.

# In[ ]:


def jupyter_setting():
    
    get_ipython().run_line_magic('matplotlib', 'inline')
      
    #os.environ["WANDB_SILENT"] = "true" 
    #plt.style.use('bmh') 
    #plt.rcParams['figure.figsize'] = [20,15]
    #plt.rcParams['font.size']      = 13
     
    pd.options.display.max_columns = None
    #pd.set_option('display.expand_frame_repr', False)

    warnings.filterwarnings(action='ignore')
    warnings.simplefilter('ignore')
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category= sklearn.exceptions.UndefinedMetricWarning)

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', None)

    icecream = ["#00008b", "#960018","#008b00", "#00468b", "#8b4500", "#582c00"]
    #sns.palplot(sns.color_palette(icecream))
    
    colors = ["lightcoral", "sandybrown", "darkorange", "mediumseagreen",
          "lightseagreen", "cornflowerblue", "mediumpurple", "palevioletred",
          "lightskyblue", "sandybrown", "yellowgreen", "indianred",
          "lightsteelblue", "mediumorchid", "deepskyblue"]
    
    # Colors
    dark_red   = "#b20710"
    black      = "#221f1f"
    green      = "#009473"
    myred      = '#CD5C5C'
    myblue     = '#6495ED'
    mygreen    = '#90EE90'    
    color_cols = [myred, myblue,mygreen]
    
    return icecream, colors, color_cols

icecream, colors, color_cols = jupyter_setting()


# In[ ]:


def missing_zero_values_table(df):
        mis_val         = df.isnull().sum()
        mis_val_percent = round(df.isnull().mean().mul(100), 2)
        mz_table        = pd.concat([mis_val, mis_val_percent], axis=1)
        mz_table        = mz_table.rename(columns = {df.index.name:'col_name', 
                                                     0 : 'Valores ausentes', 
                                                     1 : '% de valores totais'})
        
        mz_table['Tipo de dados'] = df.dtypes
        mz_table                  = mz_table[mz_table.iloc[:,1] != 0 ].                                      sort_values('% de valores totais', ascending=False)
        
        msg = "Seu dataframe selecionado tem {} colunas e {} " +               "linhas. \nExistem {} colunas com valores ausentes."
            
        print (msg.format(df.shape[1], df.shape[0], mz_table.shape[0]))
        
        return mz_table.reset_index()


# In[ ]:


def reduce_memory_usage(df, verbose=True):
    
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    
    for col in df.columns:
        
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
        
    return df


# In[ ]:


def graf_label(ax, total):
    
     for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        width, height = i.get_width() -.2 , i.get_height()
        
        x, y  = i.get_xy()  
        color = 'white'
        alt   = .5
        soma  = 0 

        if height < 70:
            color = 'black'
            alt   = 1
            soma  = 10

        ax.annotate(str(round((i.get_height() * 100.0 / total), 1) )+'%', 
                    (i.get_x()+.3*width, 
                     i.get_y()+soma + alt*height),
                     color   = color,
                     weight = 'bold',
                     size   = 14)


# In[ ]:


def graf_bar(df, col, title, xlabel, ylabel, tol = 0):
    
    #ax    = df.groupby(['churn_cat'])['churn_cat'].count()
    ax     = df    
    colors = col
    
    if tol == 0: 
        total  = sum(ax)
        ax = (ax).plot(kind    ='bar',
                       stacked = True,
                       width   = .5,
                       rot     = 0,
                       color   = colors, 
                       grid    = False)
    else:
        total  = tol     
        
        ax = (ax).plot(kind    ='bar',
                       stacked = True,
                       width   = .5,
                       rot     = 0,
                       figsize = (10,6),
                       color   = colors,
                       grid    = False)

    #ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    #y_fmt = tick.FormatStrFormatter('%.0f') 
    #ax.yaxis.set_major_formatter(y_fmt)

    title   = title #+ ' \n'
    xlabel  = '\n ' + xlabel 
    ylabel  = ylabel + ' \n'
    
    ax.set_title(title  , fontsize=22)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)    

    min = [0,23000000]
    #ax.set_ylim(min)
    
    graf_label(ax, total)


# In[ ]:


def calc_erro(y, y_pred, outros=True, ruturn_score=False):
    erro   = smape(y, y_pred)    
      
    
    if outros:        
        rmse = metrics.mean_squared_error(y, y_pred, squared=False)
        mape = metrics.mean_absolute_percentage_error(y, y_pred)
        mae  = metrics.mean_absolute_error(y, y_pred)
        
        print('RMSE : {:2.5f}'.format(rmse))
        print('MAE  : {:2.5f}'.format(mae))
        print('MAPE : {:2.5f}'.format(mape))
        
        
    if ruturn_score: 
        return erro
    else: 
        print('SMAPE: {:2.5f}'.format(erro))


# In[ ]:


def df_corr(df, annot_=False, method_='pearson'):
    
    df = df.corr(method=method_).round(5)

    # Máscara para ocultar a parte superior direita do gráfico, pois é uma duplicata
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True

    # Making a plot
    plt.figure(figsize=(15,12))
    ax = sns.heatmap(df, annot=annot_, mask=mask, cmap="RdBu", annot_kws={"weight": "bold", "fontsize":13})

    ax.set_title("Mapa de calor de correlação das variável", fontsize=17)

    plt.setp(ax.get_xticklabels(), 
             rotation      = 90, 
             ha            = "right",
             rotation_mode = "anchor", 
             weight        = "normal")

    plt.setp(ax.get_yticklabels(), 
             weight        = "normal",
             rotation_mode = "anchor", 
             rotation      = 0, 
             ha            = "right");


# In[ ]:


def describe(df):
    var = df.columns

    # Medidas de tendência central, média e mediana 
    ct1 = pd.DataFrame(df[var].apply(np.mean)).T
    ct2 = pd.DataFrame(df[var].apply(np.median)).T

    # Dispensão - str, min , max range skew, kurtosis
    d1 = pd.DataFrame(df[var].apply(np.std)).T
    d2 = pd.DataFrame(df[var].apply(min)).T
    d3 = pd.DataFrame(df[var].apply(max)).T
    d4 = pd.DataFrame(df[var].apply(lambda x: x.max() - x.min())).T
    d5 = pd.DataFrame(df[var].apply(lambda x: x.skew())).T
    d6 = pd.DataFrame(df[var].apply(lambda x: x.kurtosis())).T
    d7 = pd.DataFrame(df[var].apply(lambda x: (3 *( np.mean(x) - np.median(x)) / np.std(x) ))).T

    # concatenete 
    m = pd.concat([d2, d3, d4, ct1, ct2, d1, d5, d6, d7]).T.reset_index()
    m.columns = ['attrobutes', 'min', 'max', 'range', 'mean', 'median', 'std','skew', 'kurtosis','coef_as']
    
    return m


# In[ ]:


def graf_outlier(df, feature):
    col = [(0,4), (5,9)]

    df_plot = ((df[feature] - df[feature].min())/
               (df[feature].max() - df[feature].min()))

    fig, ax = plt.subplots(len(col), 1, figsize=(15,7))

    for i, (x) in enumerate(col): 
        sns.boxplot(data = df_plot.iloc[:, x[0]:x[1] ], ax = ax[i]); 


# In[ ]:


def diff(t_a, t_b):
    from dateutil.relativedelta import relativedelta
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


# In[ ]:


def free_gpu_cache():
    
    # https://www.kaggle.com/getting-started/140636
    #print("Initial GPU Usage")
    #gpu_usage()                             

    #cuda.select_device(0)
    #cuda.close()
    #cuda.select_device(0)   
    
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


def graf_eval():

    results     = model.evals_result()
    ntree_limit = model.best_ntree_limit

    plt.figure(figsize=(20,7))

    for i, error in  enumerate(['mlogloss', 'merror']):#
        
        plt.subplot(1,2,i+1)
        plt.plot(results["validation_0"][error], label="Treinamento")
        plt.plot(results["validation_1"][error], label="Validação")

        plt.axvline(ntree_limit, 
                    color="gray", 
                    label="N. de árvore ideal {}".format(ntree_limit))
                    
        
        title_name ='\n' + error.upper() + ' PLOT \n'
        plt.title(title_name)
        plt.xlabel("Número de árvores")
        plt.ylabel(error)
        plt.legend();


# In[ ]:


#define the smape function
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)


# In[ ]:


def linear_fit_slope(y):
    """Return the slope of a linear fit to a series."""
    y_pure = y.dropna()
    length = len(y_pure)
    x = np.arange(0, length)
    slope, intercept = np.polyfit(x, y_pure.values, deg=1)
    return slope


# In[ ]:


def linear_fit_intercept(y):
    """Return the intercept of a linear fit to a series."""
    y_pure = y.dropna()
    length = len(y_pure)
    x = np.arange(0, length)
    slope, intercept = np.polyfit(x, y_pure.values, deg=1)
    return intercept


# In[ ]:


def cromer_v(x, y):
    cm       = pd.crosstab(x, y).to_numpy()        
    n        = cm.sum()
    r, k     = cm.shape
    chi2     = stats.chi2_contingency(cm)[0]
    chi2corr = max(0, chi2 - (k-1) * (r-1) /(n-1))
    kcorr    = k - (k-1) **2/(n-1)
    rcorr    = r - (r-1) **2/(n-1)    
    v        = np.sqrt((chi2corr/n) / (min(kcorr-1, rcorr-1)))        
    return v  


# In[ ]:


def generate_category_table(data):

    cols    = data.select_dtypes(include='object').columns
    dataset = pd.DataFrame()

    for i in cols:
        corr = []
        for x in cols: 
            corr.append(cromer_v(data[i],data[x]))

        aux     = pd.DataFrame({i:corr})
        dataset = pd.concat([dataset, aux], axis=1) 

    return dataset.set_index(dataset.columns)


# In[ ]:


def graf_feature_corr(df, annot_=False, threshold=.8, print_var=False):
    
    df = df.corr(method ='pearson').round(5)

    # Máscara para ocultar a parte superior direita do gráfico, pois é uma duplicata
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True

    # Making a plot
    ax = sns.heatmap(df, annot=annot_, mask=mask, cmap="RdBu", annot_kws={"weight": "bold", "fontsize":13})

    ax.set_title("Mapa de calor de correlação das variável", fontsize=17)

    plt.setp(ax.get_xticklabels(), 
             rotation      = 90, 
             ha            = "right",
             rotation_mode = "anchor", 
             weight        = "normal")

    plt.setp(ax.get_yticklabels(), 
             weight        = "normal",
             rotation_mode = "anchor", 
             rotation      = 0, 
             ha            = "right");
    
    if print_var: 
        print('Variáveis autocorrelacionadas threshold={:2.2f}'.format(threshold))
        df_corr = df[abs(df)>threshold][df!=1.0].unstack().dropna().reset_index()
        df_corr.columns =  ['var_1', 'var_2', 'corr']
        display(df_corr)


# In[ ]:


def plot_roc_curve(fpr, tpr, label=None):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, "r-", label=label)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for FLAI 08')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    plt.grid(True)


# In[ ]:


def feature_engineering(df_):
    
    var_f27 = ''
    for col in df_['f_27']: 
        var_f27 +=col

    var_f27 = list(set(var_f27))
    var_f27.sort()
    
    df_["fe_f_27_unique"] = df_["f_27"].apply(lambda x: len(set(x)))
    
    for letra in var_f27:             
        df_['fe_' + letra.lower() + '_count'] = df2_train["f_27"].str.count(letra)
        
    return df_ 


# ---

# ## 1.4. Dataset

# ### 1.4.1. Descrição de dados
# 
# Para este desafio, temos um conjunto de dados de controle de fabricação (simulados) e temos a tarefa de prever se a máquina está em estado 0 ou estado 1, os dados têm várias interações de variáveis que podem ser importantes na determinação do estado da máquina, o qual devemos identificar para uma boa tareja de previsão. 
# 
# 
# ### 1.4.2. Arquivos
# - **treino.csv**: conjunto de treinamento, que incluem dados contínuos normalizados e dados categóricos;
# - **teste.csv**: conjunto de teste, a tarefa é prever um binário para target que representa o estado de um processo de fabricação
# - **sample_submission.csv**: um arquivo de envio de amostra no formato correto

# ### 1.4.3. Estrutura de pasta
# A finalidade é criar um estrutura de pasta para armazenar os artefatos criados no processo de análise e modelagem.

# In[ ]:


paths = ['img', 'Data', 'Data/pkl', 'Data/submission', 'Data/tunning', 
         'model', 'model/preds', 'model/optuna','model/preds/test', 
         'model/preds/test/n1', 'model/preds/test/n2', 'model/preds/test/n3', 
         'model/preds/train', 'model/preds/train/n1', 'model/preds/train/n2', 
         'model/preds/train/n3', 'model/preds/param']

for path in paths:
    try:
        os.mkdir(path)       
    except:
        pass  


# ### 1.4.4. Carrega dados

# In[ ]:


path        = '/content/drive/MyDrive/kaggle/Tabular Playground Series/05 - Maio/' if COLAB else ''   
path        = '../input/tabular-playground-series-may-2022/'
path_data   = ''  
target      = 'target'
path_automl = 'automl/'


# In[ ]:


df1_train     = pd.read_csv(path + path_data + 'train.csv')
df1_test      = pd.read_csv(path + path_data + 'test.csv')
df_submission = pd.read_csv(path + path_data + 'sample_submission.csv')

df1_train.shape, df1_test.shape, df_submission.shape


# In[ ]:


df1_train.head()


# In[ ]:


df1_test.head()


# ---

# ### 1.3.1. Redução dos datasets

# In[ ]:


df1_train = reduce_memory_usage(df1_train)
df1_test  = reduce_memory_usage(df1_test)


# # <div class="alert alert-info alert-dismissible">2. Análise Exploratória de Dados (EDA)  </div> 

# In[ ]:


df2_train = df1_train.copy()
df2_test  = df1_test.copy()


# ## 2.1. Dimensão do DataSet

# In[ ]:


print('TREINO')
print('Number of Rows: {}'.format(df2_train.shape[0]))
print('Number of Columns: {}'.format(df2_train.shape[1]), end='\n\n')

print('TESTE')
print('Number of Rows: {}'.format(df2_test.shape[0]))
print('Number of Columns: {}'.format(df2_test.shape[1]))


# ---

# ## 2.2. Tipo de dados

# In[ ]:


df2_train.info()


# In[ ]:


df2_test.info()


# In[ ]:


print(f'{3*"="} For Pandas {10*"="}\n{(df2_train.dtypes).value_counts()}')
print(f'\n{3*"="} For Datatable {7*"="}\n{(df2_test.dtypes).value_counts()}')


# Vamos dar uma olhada nas variáveis do tipo int8.  

# In[ ]:


for col in df2_train.select_dtypes(np.int8).columns.drop(target):   
    num = df2_train[col].unique().tolist()
    num.sort()
    print('-'* 70)
    print('{} unique: {}'.format(col, num))    
    print('-'* 70)
    print()


# <div class="alert alert-info" role="alert"> 
#     
# **`NOTA:`** <br>
# Podemos observar que as variáveis acima segue um ordem, provavelmente foram transformadas em ordinal no processo de geração dos datasets, sendo assim, vamos transformá-las em categóricas para as nossas análises.
#     
# </div>

# In[ ]:


for col in df2_train.select_dtypes(np.int8).columns.drop(target):   
    df2_train[col] = df2_train[col].astype(object)   
    df2_test[col]  = df2_test[col].astype(object)    


# In[ ]:


df2_train.info()


# In[ ]:


df2_test.info()


# ---

# ## 2.3. Identificar NA

# In[ ]:


missing = missing_zero_values_table(df2_train)
missing[:].style.background_gradient(cmap='Reds')


# In[ ]:


missing = missing_zero_values_table(df2_test)
missing[:].style.background_gradient(cmap='Reds')


# <div class="alert alert-info" role="alert"> 
#     
# **`NOTA:`** <br>
# 
# Não temos valores ausentes. 
#     
# </div>

# ## 2.4 Estatística Descritiva
# Abaixo estão as estatísticas básicas para cada variável que contém informações sobre contagem, média, desvio padrão, mínimo, 1º quartil, mediana, 3º quartil e máximo.

# In[ ]:


feature_float = df2_test.select_dtypes(np.number).columns.to_list()
feature_cat   = df2_test.select_dtypes(object).columns.to_list()

feature_float.remove('id')

msg = 'Temos {} variávies numéricas e {} categóricas.'
print(msg.format(len(feature_float), len(feature_cat)))


# - Train

# In[ ]:


df2_train.drop([target, 'id'], axis=1).describe().T.style.background_gradient(cmap='YlOrRd')


# <div class="alert alert-info" role="alert"> 
#     
# **`NOTA:`** <br>
# Podemos observar que a maioria  das variáveis tem desvio padrão igual a zero, exceto a variável **f_28**, sendo assim, tem uma baixa variância.
# 
# Outro ponto em relação a variáveil **f_28** são os altos valores negativo e positivo nas estatísticas mínimo e  máximo, o interessante é que o valor mínimo e máximo são iguais e os valores de 25% e 75% são guase iguais.   
#     
# </div>

# ---

# ## 2.6. Distribuição

# ### 2.6.1. Train / Test

# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))

pie = ax.pie([len(df2_train), len(df2_test)],
             labels   = ["Train dataset", "Test dataset"],
             #colors   = ["teal", "b"],
             textprops= {"fontsize": 15},
             autopct  = '%1.1f%%')

ax.axis("equal")
ax.set_title("Comparação de comprimento do conjunto de dados \n", fontsize=18)
fig.set_facecolor('white')
plt.show();


# ---

# ### 2.6.2. Distribuição Train x Test

# In[ ]:


lines   = int(len(feature_float)/2)
fig, ax = plt.subplots(lines,2 ,figsize=(20,20))

for i,feature in enumerate(feature_float):
    plt.subplot(lines,2,i+1)
    sns.histplot(data=df2_train, x=df2_train[feature],color='blue', alpha=0.5, label='train', bins=1000)
    sns.histplot(data=df2_test , x=df2_test[feature] ,color='teal', alpha=0.5, label='test' , bins=1000)     
    plt.xlabel(feature, fontsize=12)
    plt.legend()
         
plt.suptitle('DistPlot: train & test data', fontsize=20);


# <div class="alert alert-info" role="alert"> 
#     
# **`NOTA:`** <br>
# Ambos os datasets seguem a mesma distribuição.
# 
# </div>

# ### 2.6.3. Proporção de variáveis

# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))

plt.pie([ len(feature_cat), len(feature_float)], 
        labels=['Categorical', 'Continuos' ],
        textprops={'fontsize': 13},
        autopct='%1.1f%%')

ax.set_title("Comparação variáveis continuas/categóricas \n Dataset Treino/Teste", fontsize=18)
fig.set_facecolor('white')
plt.show()


# ### 2.6.4. Distribuição da target

# In[ ]:


plt.figure(figsize=(7,5))    

graf_bar(df2_train.groupby([target])[target].count() , 
         icecream, 
         'Distribuição da variável preditora', 
         target, 
         'Quantidade pessoas');


# <div class="alert alert-info" role="alert"> 
#     
# **`NOTA:`** <br>
#     
# Não temos desbalanceamento nos dados. 
#                                                        
# </div>

# ## 2.7. Dados Categóricas
# 
# Vamos fazer uma contagem das observações e em cada variável categórica, para termos uma noção de como nosso conjunto de dados estar distribuído.

# In[ ]:


for i in feature_cat:
    print("Coluna: ",i)
    print(df2_train[[i]].value_counts(), "\n")


# <div class="alert alert-info" role="alert"> 
#     
# **`NOTA:`** <br>
#     
# A variável **f_27** precisa ser transformada, pois temos 741354 elementos únicos, isto é, criar novas variáveis apartir dessa variável, sendo assim, vamos excluí-la das análises neste primeiro momento e na parte de feature engineering vamos tratá-la.     <br>    
# Agora temos uma noção de como estão distribuídas as categorias, temos colunas apresentam uma pequena concentração em uma única categoria, vamos fazer uma análise gráfica para tirar mais algumas informações.
#     
# </div>

# In[ ]:


feature_cat.remove('f_27')
print(feature_cat)


# In[ ]:


plt.figure(figsize=(20,40))

for i, col in enumerate(feature_cat):
    plt.subplot(int(len(feature_cat)/2)+1, 2, i+1)
    ax = sns.countplot(data=df2_train, y=col, hue=target)    


# <div class="alert alert-info" role="alert"> 
#  
# **`NOTA:`** <br>
#  
# Observando os gráficos acima, as variáveis com numeração de  17 à 18 tem pouca quantidade nas ultimas classe, sendo assim, podemos fazer um junção desses classes em outras classas, assim podemos reduzir o ruído na modelagem, outra abordagem que podemos implementar é a transformação **Target Encoding**.      
# <br>
# Vamos dar uma olhada na correlação das variaveis categóricas, para isso foi criado uma função que calcula a matriz de coeficientes de Cramer, que permite entender a correlação entre duas variáveis categóricas em um conjunto de dados.
# </div>

# In[ ]:


plt.figure(figsize=(20,15))

df         = df2_train[feature_cat].copy()
df[target] = df2_train[target].astype(object)

corr = generate_category_table(df)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, mask= mask).          set_title('Mapa de calor de correlação das variável categóricas', fontsize=17);

del df


# <div class="alert alert-info" role="alert"> 
#  
# **`NOTA:`** <br>
# Temos uma correlação baixa entre as variáveis categóricas e principalmente em relação a variável alvo.
#     
# </div>

# ## 2.8. Dados Númericos

# ### 2.8.1. Correlação
# Vamos examinar a correlação entre as variáveis.

# In[ ]:


plt.figure(figsize=(20,15))
graf_feature_corr(df=df2_train.copy().drop('id', axis=1), annot_=False, threshold=.7, print_var=False)


# <div class="alert alert-info" role="alert"> 
#  
# **`NOTA:`** <br>
#     
# Como podemos observar, a correlação fica entre -0.2 e 0.3%, sendo assim, não temos variáveis autocorrelacionadas, mais a frente com a criação de novas variáveis podemos ter variáveis autocorrelacionadas e voltaremos a fazer essa análise de autocorrelação.
# 
#     
# </div>

# ### 2.8.2. Histograma

# In[ ]:


plt.subplots(figsize=(20, 15))

for i, col in enumerate(feature_float):    
    plt.subplot(int(len(feature_float)/3 +1),3,i+1)
    sns.kdeplot(data=df2_train, x=col, hue=target, legend=True, shade=True, multiple='stack');  


# <div class="alert alert-info" role="alert"> 
#  
# **`NOTA:`** <br>
# - A distribuição das variáveis parecem ser bem comportadas, igual a uma distribuição normal.   <br> 
# 
# Vamos dar uma olhda nos outliers dessas variáveis.     
# </div>

# ---

# ### 2.8.3. Outliers

# In[ ]:


f, ax = plt.subplots(figsize=(20, 20))

for i, col in enumerate(feature_float): 
    plt.subplot(int(len(feature_float)/3 +1),3,i+1)
    sns.boxplot(data=df2_train, x=target, y=col)    


# <div class="alert alert-info" role="alert"> 
#  
# **`NOTA:`** <br>
# - Todas variáveis tem outliers, o bom é que todas as mediana são iguais a zero, provavelmente os modelos de arvosres podem ter melhor desempenho, podemos tentar alguma técnica que identifique os outliers para as variáveis com maior importancia, por exemplo criando uma nova variável que identifique as amostras com outliers.  
#     
# </div>
# 

# 
# # <div class="alert alert-info alert-dismissible">3. Modelagem (baseline)  </div> 
# 

# Vamos excluir a variável **f_27** neste primeiro momento, no próximo notebook vamos tratar essa variável.  

# In[ ]:


df3_train = df2_train.copy()
df3_test  = df2_test.copy()

df3_train.drop(['f_27'], axis=1 , inplace=True)
df3_test.drop(['f_27'], axis=1 , inplace=True)

for col in df3_train.select_dtypes(object).columns: 
    df3_train[col] = df3_train[col].astype(np.int32)
    df3_test[col] = df3_test[col].astype(np.int32)    


# ## 3.1. Split Train/Test

# In[ ]:


X      = df3_train.drop([target, 'id'], axis=1)
y      = df3_train[target]
X_test = df3_test.drop(['id'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      test_size    = 0.29,
                                                      shuffle      = True, 
                                                      stratify     = y, 
                                                      random_state = 12359)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape , X_test.shape


# ## 3.2. Parametros do modelo

# In[ ]:


seed   = 12359
params = {'objective'        : 'binary:logistic',   
          'eval_metric'      : 'auc',  
          'n_estimators'     : 1000,                
          'random_state'     : seed}

if torch.cuda.is_available():           
    params.update({'tree_method': 'gpu_hist','predictor': 'gpu_predictor'})
    
params


# ## 3.4. Seleção do Scaler

# In[ ]:


path=''


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nscalers = [StandardScaler(),\n           RobustScaler(), \n           MinMaxScaler(), \n           MaxAbsScaler(),            \n           QuantileTransformer(output_distribution='normal', random_state=0)]\n\nmodel_baseline = xgb.XGBClassifier(**params)\nscaler_best    = None\nmodel_best     = None\ncols           = X_test.columns\nf1_best        = 0 \nauc_best       = 0 \n\nfor scaler in scalers: \n    \n    X_train_s = X_train.copy() \n    X_valid_s = X_valid.copy()\n    \n    if scaler!=None:                      \n        X_train_s = pd.DataFrame(scaler.fit_transform(X_train_s), columns=cols)\n        X_valid_s = pd.DataFrame(scaler.transform(X_valid_s), columns=cols)\n        X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=cols)\n        \n    model_baseline.fit(X_train_s, y_train, verbose=False)\n\n    y_pred_prob_tr = model_baseline.predict_proba(X_train_s)[:,1]\n    y_pred_prob_vl = model_baseline.predict_proba(X_valid_s)[:,1]\n    y_pred_prob_ts = model_baseline.predict_proba(X_test_s)[:,1]\n    \n    y_pred_tr = (y_pred_prob_tr>.5).astype(int) \n    y_pred_vl = (y_pred_prob_vl>.5).astype(int)\n\n    f1     = metrics.f1_score(y_valid, y_pred_vl)\n    auc_vl = metrics.roc_auc_score(y_valid, y_pred_prob_vl)\n    auc_tr = metrics.roc_auc_score(y_train, y_pred_prob_tr)\n        \n    print('AUC Trn: {:2.5f} - AUC Val: {:2.5f} - F1: {:2.5f} => {}'.format(auc_tr, auc_vl, f1, scaler))\n    \n    if auc_vl>auc_best:\n        f1_best     = f1    \n        auc_best    = auc_vl\n        scaler_best = scaler\n        model_best  = model_baseline\n        \n    # Gera arquivo de submissão\n    name_file_sub         = 'xgb_base_line_01_score_{:2.5f}_{}.csv'.format(auc_vl,str(scaler).lower()[:4])\n    df_submission[target] = y_pred_prob_ts\n    \n    df_submission.to_csv(path + 'Data/submission/' + name_file_sub, index=False)\n\ndel scaler, f1, auc_vl\n    \nprint()\nprint('The Best')  \nprint('Scaler: {}'.format(scaler_best))    \nprint('AUC   : {:2.5f}'.format(auc_best))\nprint()")


# <div class="alert alert-info" role="alert"> 
#  
# **`NOTA:`** <br>
#     
# Podemos observar que a maioria dos scalers não tem efeito neste conjunto de dados utilizando o XGB, no nosso caso o melhor scaler foi **QuantileTransformer**. 
#         
# </div>

# ### 3.3.2. Featuere Importance

# In[ ]:


features       = pd.Series(model_baseline.feature_importances_)
features.index = cols

features.sort_values(ascending=True, inplace=True)
features.plot(kind ='barh', figsize=(15,15));


# <div class="alert alert-info" role="alert"> 
#  
# **`NOTA:`** <br>
# Olhando as variáveis mais importante, obeservamos que a maioria das variáveis que identificamos como categoricas aparecem entre as mais importantes.  
#     
# </div>

# ## 3.4. Validação Cruzada

# In[ ]:


def cross_val_model(model_, model_name_, X_, y_, X_test_, target_, scalers_, lb_, fold_=5, path_='',  
                    seed_=12359, feature_scaler_=None, print_report_=False, save_submission_=False):
    
    n_estimators = model_.get_params()['n_estimators']
             
    valid_preds     = {}
    taco            = 76 
    acc_best        = 0
   # col_prob        = y_.sort_values().unique()
    df_proba        = pd.DataFrame()
    feature_imp     = pd.DataFrame()
    test_preds      = []
    test_pred_proba = np.zeros((1, 1))
    preds           = []
    model           = []
    
    for i, scaler_ in enumerate(scalers_): 

        time_start = datetime.now()
        score      = []        
                
        if scaler_!=None:            
            string_scaler = str(scaler_)        
            string_scaler = string_scaler[:string_scaler.index('(')]
        else:
            string_scaler = None 
            
        y_pred_test = np.zeros(len(X_test_))

        folds = KFold(n_splits=fold_, shuffle=True, random_state=seed_)
        folds = StratifiedKFold(n_splits=fold_, shuffle=True, random_state=seed_)
        
        print('='*taco)
        print('Scaler: {} - n_estimators: {}'.format(string_scaler, n_estimators))
        print('='*taco)
        
        pred_test=0 
        
        for fold, (trn_idx, val_idx) in enumerate(folds.split(X_, y_, groups=y_)): 

            time_fold_start = datetime.now()

            # ---------------------------------------------------- 
            # Separar dados para treino 
            # ----------------------------------------------------     
            X_trn, X_val = X_.iloc[trn_idx], X_.iloc[val_idx]
            y_trn, y_val = y_.iloc[trn_idx], y_.iloc[val_idx] 
            
            # ---------------------------------------------------- 
            # Processamento 
            # ----------------------------------------------------     
            if scaler_!=None: 
                X_tst = X_test_.copy()
                if feature_scaler_!=None:                     
                    X_trn[feature_scaler_] = scaler_.fit_transform(X_trn[feature_scaler_])
                    X_val[feature_scaler_] = scaler_.transform(X_val[feature_scaler_])                      
                    X_tst[feature_scaler_] = scaler_.transform(X_tst[feature_scaler_])
                else:            
                    X_trn = scaler_.fit_transform(X_trn)
                    X_val = scaler_.transform(X_val)
                    X_tst = scaler_.transform(X_test_.copy())
                
            # ---------------------------------------------------- 
            # Treinar o modelo 
            # ----------------------------------------------------            
            model_.fit(X_trn, y_trn,
                       eval_set              = [(X_trn, y_trn), (X_val, y_val)],          
                       early_stopping_rounds = int(n_estimators*.1),
                       verbose               = False)
            
            # ---------------------------------------------------- 
            # Predição 
            # ----------------------------------------------------     
            y_pred_val_prob = model_.predict_proba(X_val, ntree_limit=model_.best_ntree_limit)[:,1]    
            y_pred_val      = (y_pred_val_prob>.5).astype(int)

            preds.append(model_.predict(X_tst))    
            
            pred_test += model_.predict_proba(X_tst)[:, 1] / folds.n_splits
                
            df_prob_temp    = pd.DataFrame(y_pred_val_prob)
            #y_pred_pbro_max = df_prob_temp.max(axis=1)

            df_prob_temp['fold']    = fold+1
            df_prob_temp['id']      = val_idx            
            df_prob_temp['y_val']   = y_val.values
            df_prob_temp['y_pred']  = y_pred_val            
            df_prob_temp['y_proba'] = y_pred_val_prob
            df_prob_temp['scaler']  = str(string_scaler)
                        
            # ---------------------------------------------------- 
            # Score 
            # ---------------------------------------------------- 
            acc   = metrics.accuracy_score(y_val, y_pred_val)
            auc   = metrics.roc_auc_score(y_val, y_pred_val_prob)
            f1    = metrics.f1_score(y_val, y_pred_val) 
            prec  = metrics.log_loss (y_val, y_pred_val)
            
            score.append(auc)     
            
            # ---------------------------------------------------- 
            # Feature Importance
            # ----------------------------------------------------             
            feat_imp = pd.DataFrame(index   = X_.columns,
                                    data    = model_.feature_importances_,
                                    columns = ['fold_{}'.format(fold+1)])

            feat_imp['auc_'+str(fold+1)] = auc
            feature_imp = pd.concat([feature_imp, feat_imp], axis=1)
            
            # ---------------------------------------------------- 
            # Print resultado  
            # ---------------------------------------------------- 
            time_fold_end = diff(time_fold_start, datetime.now())
            msg = '[Fold {}] AUC: {:2.5f} - F1-score: {:2.5f} - L. Loss: {:2.5f}  - {}'
            print(msg.format(fold+1, auc, f1, prec, time_fold_end))
            
            # ---------------------------------------------------- 
            # Salvar o modelo 
            # ---------------------------------------------------- 
            dic_model = {'scaler' : scaler_, 
                         'fold'   : fold+1, 
                         'model'  : model_ }
            
            model.append(dic_model)

        score_mean = np.mean(score) 
        score_std  = np.std(score)

        if score_mean > acc_best:     
            acc_best    = score_mean           
            model_best  = model_    
            scaler_best = scaler_

        time_end = diff(time_start, datetime.now())   

        print('-'*taco)
        print('[Mean Fold] AUC: {:2.5f} std: {:2.5f} - {}'.format(score_mean, score_std, time_end))
        print('='*taco)
        print()
               
        if save_submission_:
            name_file_sub  = model_name_ + '_' + str(i+1) + '_' + str(scaler_).lower()[:4] + '.csv'
            name_file_sub  = path_ + 'Data/submission/' + name_file_sub.format(score_mean)        
            df_sub         = df_submission.copy()
            df_sub[target] = pred_test
            df_sub.to_csv(name_file_sub, index=False)
            
        if print_report_:
            y_pred = df_prob_temp[df_prob_temp['scaler']==str(string_scaler)]['y_pred']
            y_vl   = df_prob_temp[df_prob_temp['scaler']==str(string_scaler)]['y_val']
            print(metrics.classification_report(y_vl,y_pred))

    print('-'*taco)
    print('Scaler Best: {}'.format(scaler_best))
    print('Score      : {:2.5f}'.format(acc_best))
    print('-'*taco)
    print()

    return model, df_prob_temp.sort_values(by=['scaler','id']) , feature_imp 


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nseed        = 12359\neval_metric = ['auc', 'error']                 \nscalers     = [StandardScaler(), \n               QuantileTransformer(output_distribution='normal', random_state=0)]\n\nparams = {'objective'        : 'binary:logistic',   \n          'eval_metric'      : eval_metric,  \n          'n_estimators'     : 1000,   \n          'random_state'     : seed}\n\nif torch.cuda.is_available():           \n    params.update({'tree_method' : 'gpu_hist',                    \n                   'predictor'   : 'gpu_predictor'})\n\nmodel, df_proba, feature_imp = \\\n    cross_val_model(model_           = xgb.XGBClassifier(**params),\n                    model_name_      = 'xgb_bs_vc_score_02_{:2.5f}',\n                    X_               = X,\n                    y_               = y,\n                    X_test_          = X_test,\n                    target_          = target,\n                    scalers_         = scalers,\n                    fold_            = 5, \n                    lb_              = None,\n                    path_            = path,\n                    seed_            = seed, \n                    feature_scaler_  = None, \n                    print_report_    = True, \n                    save_submission_ = True)\nprint()")


# ## 3.5. Análise do Modelo 
# Vamos fazer o treinamento novamente do melhor modelo, sendo que agora vamos utilizar a divisão de treino/teste, isto é, vamos o treino em um conjunto de dados e fazer a previsões em dados que o modelo não viu no treino. 

# In[ ]:


scalers     = [StandardScaler()]

model, df_proba, feature_imp =     cross_val_model(model_           = xgb.XGBClassifier(**params),
                    model_name_      = 'xgb_bs_vc_score_02_{:2.5f}',
                    X_               = X_train,
                    y_               = y_train,
                    X_test_          = X_test,
                    target_          = target,
                    scalers_         = scalers,
                    fold_            = 5, 
                    lb_              = None,
                    path_            = path,
                    seed_            = seed, 
                    feature_scaler_  = None, 
                    print_report_    = True, 
                    save_submission_ = False)
print()


# ### 3.5.1. Feature Importances  

# In[ ]:


plt.figure(figsize=(15,12))
for fold, col in enumerate(feature_imp.filter(regex=r'fold').columns):            
    col_acc = 'auc_' + str(fold+1)
    df_fi = feature_imp.sort_values(by=col, ascending=False).reset_index().iloc[:15]
    df_fi = df_fi[['index', col, col_acc]]
    df_fi.columns = ['Feature', 'score', col_acc]
    plt.subplot(3,2, fold+1)
    sns.barplot(x='score', y='Feature', data=df_fi)    
    plt.title('Fold {} - score: {:2.5f}'.format(fold+1, df_fi[col_acc].mean()), 
              fontdict={'fontsize':18})    

plt.suptitle('Feature Importance XGB', y=1.05, fontsize=24);
plt.tight_layout(h_pad=3.0); 


# ### 3.5.2. Shap

# In[ ]:


mdl = model[1]['model']
sc  = model[1]['scaler']

X_valid_sc = sc.transform(X_valid)#, columns=X_valid.columns)
y_pred_val = mdl.predict_proba(X_valid_sc)[:,1]
score      = metrics.roc_auc_score(y_valid, y_pred_val)

print('AUC em dados não viscto: {:2.5f}'.format(score))


# In[ ]:


explainer = shap.TreeExplainer(mdl)
shap_values = explainer.shap_values(X_valid_sc)

shap.summary_plot(shap_values, X_valid_sc, plot_type="bar")


# In[ ]:


shap.summary_plot(shap_values, X_valid_sc, max_display=15)


# ### 3.1.1. Erro e  número de Estimadores

# In[ ]:


for erro in eval_metric:
    plt.figure(figsize=(15,10))

    for i in range(len(model)):
        results     = model[i]['model'].evals_result() # merror
        ntree_limit = model[i]['model'].best_ntree_limit

        plt.subplot(2,3,i+1)
        plt.plot(results["validation_0"][erro], label="Treinamento")
        plt.plot(results["validation_1"][erro], label="Validação")

        plt.axvline(ntree_limit, 
                    color="gray", 
                    label="N. de árvore ideal {}".format(ntree_limit))

       # plt.xlabel('Número de árvores')
        plt.ylabel(erro)
        plt.legend();

    plt.suptitle('Performance XGB - {}'.format(erro), y=1.05, fontsize=24);
    plt.tight_layout(h_pad=3.0);


# <div class="alert alert-info" role="alert"> 
#     
# **`NOTA:`** <br>
# Acima recuperamos as informações de treinamento do nosso modelo, podemos observar que o número de 1000 estimadores é mais que suficiente para o treinamento do modelo, já na predição utilizamos 500 estimadores que torna as predições mais performáticas, esse parametro pode ajuda na tunning do XGB.  
#     
# </div>

# ### 3.1.2. Previsão 

# In[ ]:


scaler_tr  = model[1]['scaler']
model_tr   = model[1]['model']
X_valid_sc = pd.DataFrame(scaler_tr.transform(X_valid), columns=X_valid.columns)

y_pred_prob = model_tr.predict_proba(X_valid_sc.values, ntree_limit=model_tr.best_ntree_limit)[:,1]
y_pred      = (y_pred_prob >.5).astype(int)
auc         = metrics.roc_auc_score(y_valid, y_pred_prob)

print('AUC: {:2.5f}'.format(auc))


# ### 3.1.3. Matriz de Confusão

# In[ ]:


def plot_cm(preds,true,ax=None):
    cm = confusion_matrix(preds.round(), true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False, values_format = '.6g')
    plt.title('Confusion matrix \n')
    plt.grid(False)
    return disp

plot_cm(y_pred, y_valid);


# In[ ]:


threshold = .46

y_pred_threshold     = (y_pred_prob>threshold).astype(int)
fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred_threshold)

plot_roc_curve(fpr, tpr, label="XGB")
plt.show()

print('AUC     : {:2.5f} '.format(metrics.roc_auc_score(y_valid, y_pred_prob) ))
print('F1-score: {:2.5f}'.format(metrics.f1_score(y_valid, y_pred)))
print('F1-score: {:2.5f} threshold({:2.2f})'.format(metrics.f1_score(y_valid, y_pred_threshold), threshold))


# <div class="alert alert-info" role="alert"> 
#   
# CONCLUSÃO: <BR>
# Agora temos uma noção do conjunto de dados para esse competição e uma linha de base com o classificador XGB, que se mostrou promissor nas submissões, obtive os sequintes resultados: 0.93628 e 0.93935 todos melhores que o treinamento do modelo, isso demostra que o modelo é robusto e teve uma boa generalização nos dados de teste que tem aproximadamente 29% dos dados, esse é um ponto importante que temos que considerar, pois no final da competição nosso modelo terá que ter um performance em todo o conjunto de dados. <br>
#     
#     
# Os próximos passos são: <br>
# 
# - Feature Engineering <br>    
# Ainda há muito a explorar dentro do conjunto de dados, recomendo explorar as variáveis que se relacionam com a variável de destino. Com base nessas relações, podemos criar mais variáveis que são combinações de dois ou mais variaveis. Um exemplo seria a variável C que é formada pelas variáveis A x B. Além disso, a variável f_27 precisa ser examinada com mais detalhes. Pode haver informações importantes escondidas em todos essas variáveis aparentemente sem sentido; 
# <br>
#     
# - Tunning de hyperparametros   <br>  
# Neste notebook não nos aprofundamos no tunning de hyperparametro do classificado XGB, é outro ponto importante mesmo que o ganho seja pouco, o tunning faz com que os modelos se tornem mais robusto e qualquer ganho pode melhora a nosso colocação na competição;
#     
#    
#     
# </div>

# In[ ]:




