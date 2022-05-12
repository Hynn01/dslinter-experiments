#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import VarianceThreshold
#warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
from hyperopt import space_eval
import seaborn as sns
import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (16,7)
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,roc_auc_score,precision_score,recall_score,confusion_matrix,make_scorer
from sklearn.ensemble import RandomForestClassifier
from scikitplot.helpers import binary_ks_curve
from matplotlib import pyplot
from sklearn.model_selection import validation_curve
from category_encoders import OneHotEncoder,TargetEncoder
from yellowbrick.model_selection import RFECV
import shap
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import VarianceThreshold
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,7)
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,roc_auc_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from scikitplot.helpers import binary_ks_curve
from matplotlib import pyplot
from sklearn.model_selection import validation_curve
from category_encoders import OneHotEncoder,TargetEncoder
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import shap
from hyperopt import fmin, tpe, hp,Trials


# ### Lendo dataset disponivel

# In[ ]:


df_train = pd.read_csv("../input/santander-customer-satisfaction/train.csv")
df_train.head()


# ## Definindo features
# 
# Aqui estamos dividindo o dataframe em 3 grupos:
# 
# #### 1 - Colunas de identificação
# 
# - Colunas que representam chaves ID 
# 
# #### 2 Target
# 
# - Coluna que identifica a target do problema
# 
# #### 3 Variáveis dependentes/explicativas
# 
# - Aqui são todas as features disponiveis, esse grupo foi dividido em 2 subgrupos que são variaveis categoricas e variaveis continuas

# In[ ]:


id_columns = ['ID']
target_column = ['TARGET']

num_vars = df_train.select_dtypes(include=['float64','int64'])
cat_vars = df_train.select_dtypes(include=['object'])

print('initial numerical vars =',len(num_vars.columns))
print('initial categorical vars =',len(cat_vars.columns))

y = df_train[target_column]
x = df_train.drop(columns=id_columns + target_column).fillna(0)


# ### Qual o percentual de target do nosso problema?
# 
# Ao olharmos a distribuição de target da nossa base toda temos um problema que concentra apenas 3,95% de clientes com target positiva

# In[ ]:


y.value_counts()/len(y)


# ### Separação em treino e teste 
# 
# - Foram usados 25% dos dados para teste e 75% para treino
# - O split foi feito de forma estratificada para preservarmos a distribuição da target em treino e teste

# In[ ]:


seed = 18051996

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=seed,stratify=y)

print("Número de linhas no treino= ",len(x_train))
print('---------------------------')
print('Verificando distribuição da target no treino')
print(y_train.value_counts()/len(y_train)*100)
print('---------------------------')
print('Verificando distribuição da target no teste')
print("Número de linhas no teste = ",len(x_test))
print('---------------------------')
print(y_test.value_counts()/len(y_test)*100)


# ## **Case A**
# Um falso positivo ocorre quando classificamos um cliente como insatisfeito, mas ela não se comporta como tal. Neste caso, o custo de preparar e executar uma ação de retenção é um valor fixo de 10 reais por cliente. Nada é ganho pois a ação de retenção não é capaz de mudar o comportamento do cliente. Um falso negativo ocorre quando um cliente é previsto como satisfeito, mas na verdade ele estava insatisfeito. Neste caso, nenhum dinheiro foi gasto e nada foi ganho. Um verdadeiro positivo é um cliente que estava insatisfeito e foi alvo de uma ação de retenção. O benefício neste caso é o lucro da ação (RS 100) menos os custos relacionados à ação de retenção (RS 10). Por fim, um verdadeiro negativo é um cliente insatisfeito e que não é alvo de nenhuma ação. O benefício neste caso é zero, isto é, nenhum custo, mas nenhum lucro. A primeira tarefa deste case é maximizar o lucro esperado por cliente considerando o contexto descrito no parágrafo acima.

# ### Definição da fução que devemos otimizar
# 
# - Aqui por mais que estamos trabalhando com m problema de classificação, devemos ter o cuidado pois o problema possui uma série de condições que devemos tomar cuidado na modelagem que são:
# 
# - **Beneficio do verdadeiro positivo (cliente que estava insatisfeito e foi alvo de ação) = 100RS - 10RS**

# In[ ]:


def lucro(modelo, x, y_true,matriz = False):
    
    if modelo == "aleatorio":
        x['y_pred'] = 0
        y_pred = x['y_pred']
        x.drop(columns='y_pred',inplace=True)
    else:
        y_pred = modelo.predict(x)
    
    #Cria matriz de confusao para nos ajudar capturar os TP e FP
    matriz_confusao = confusion_matrix(y_true, y_pred)
      
    #capturando dados de verdadeiro positivo e falso positivo na matriz de confusao
    verdadeiro_positivo = matriz_confusao[1][1]
    falso_positivo = matriz_confusao[0][1]

    #Calcula o lucro com base na formula do problema
    lucro_total = 90*verdadeiro_positivo - 10*falso_positivo 

    if matriz == True:
        print('Matriz de confusão:')
        print(matriz_confusao)      
        print('\nLucro obtido com a ação: R$ ' + str(lucro_total))
        print('Porcentagem do lucro ideal: ' + str(round(lucro_total/(int(y_true.sum())*90),2)) + '%')

    
    return lucro_total


# ### Modelo baseline
# 
# 1) Temos um modelo totalmente desbalanceado 3,95% de targte se chutarmos todo mundo como target 0, acertamos 97% da base. Qual o lucro esperado de uma previsao totalmente aleatoria como essa? Será que teremos lucro?
# 
# 2) E se treinarmos uma random forest sem restrições e sem tratamento na base de treino qual o lucro esperado para essa previsão?

# In[ ]:


lucro("aleatorio", x_test, y_test,True)


# In[ ]:


baseline = RandomForestClassifier(random_state=seed).fit(x_train,y_train)
lucro(baseline, x_test, y_test,True)


# ### Modelo otimizado

# In[ ]:


max_profit_scorer = make_scorer(lucro, greater_is_better=True)

def train_model(x_train,y_train,x_test,y_test, objetivo):

    def objective(params):
        params = {'n_estimators': int(params['n_estimators'])
              ,'max_depth': int(params['max_depth'])
              ,'max_features': params['max_features']
              ,'min_samples_split': int(params['min_samples_split'])}
        clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', **params,random_state=seed)
        score = cross_val_score(clf, x_train, y_train, scoring=objetivo, cv=StratifiedKFold()).mean()
        print("Lucro {:.3f} params {}".format(score, params))
        score = -1*(score)
        return score

    space = {
    'n_estimators': hp.quniform('n_estimators', 50, 1000, 50),
    'max_depth': hp.quniform('max_depth', 3, 8, 1),
    'max_features': hp.choice('max_features', ['auto', 'sqrt','log2']),
    "min_samples_split" : hp.quniform('min_samples_split',10,200,10)
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=10)

    # Print best parameters
    best_params = space_eval(space, best)

    print("Start training the best model")

    model = RandomForestClassifier(n_jobs=-1,random_state=seed
                                          , class_weight='balanced'
                                          , max_features=best_params['max_features']
                                          , max_depth = int(best_params['max_depth'])
                                          , min_samples_split = int(best_params['min_samples_split'])
                                          , n_estimators = int(best_params['n_estimators'])).fit(x_train,y_train)
    #Scoring the best model in train dataset
    lucro(model, x_test, y_test)
    print("Scoring the train data")
    #Scoring the best model in train dataset
    predict_train_entire = model.predict(x_train)
    proba_train_entire = model.predict_proba(x_train)[:,1]

    print("Scoring the test data")
    #Scoring the best model in test dataset
    predict_test_entire = model.predict(x_test)
    proba_test_entire = model.predict_proba(x_test)[:,1]

    print("Getting metrics")
    # calculate scores
    auc_train = roc_auc_score(y_train, proba_train_entire)
    auc_test = roc_auc_score(y_test, proba_test_entire )

    print('AUC_TRAIN', auc_train)
    print('AUC_Test', auc_test)
    # calculate roc curves
    fpr_train, tpr_train, _ = roc_curve(y_train, proba_train_entire)
    fpr_test, tpr_test, _ = roc_curve(y_test, proba_test_entire)

    # plot the roc curve for the model
    pyplot.plot(fpr_train, tpr_train, linestyle='--', label='Train')
    pyplot.plot(fpr_test, tpr_test, linestyle='--', label='Test')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    return pyplot.show()


# ### Treinando um modelo de random forest
# 
# - Tuning de alguns parametros
# 
# Aqui já obtemos um resultado melhor do que simplemente chutarmos aleatoriamente um valor para nossa target ou se treinarmos uma random forest sem restrições
# 
# **LUCRO 5686R$**

# In[ ]:


train_model(x_train,y_train,x_test,y_test, objetivo = lucro)


# ### Redução do número de features
# 
# Com intuito de minimizar o custo de altas dimensões, ajudando a diminuir a complexidade computacional de todos algoritmos que serão criados e diminuir a chance de problemas de sobreajuste devido ao alto numero de dimensoes iremos fazer uma análise para diminuir o espaco de possibilidades. Todo processo foi divididos nessas etapas:
# 
# - Eliminação de variaveis constantes ou que possuam pouca variação (variance threshould)
# - Eliminação de multicolinearidade em variaveis continuas

# In[ ]:


def variance_threshold(df,threshold):
    vt = VarianceThreshold(threshold=threshold)

    vt.fit(df)

    mask = vt.get_support()

    num_vars_reduced = df.iloc[:, mask]
    return num_vars_reduced

def correlation(df, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in df.columns:
                    del df[colname] # deleting the column from the dataset

    return df

#Aplicando variance e correlação
num_vars_vt = variance_threshold(x_train.filter(num_vars),threshold = 0.01)
num_vars_vt_corr = correlation(x_train.filter(num_vars_vt), threshold = 0.8)
           
#Select important features
x_train = x_train.filter(list(num_vars_vt_corr.columns)+list(cat_vars)+list(id_columns)).fillna(0)
x_test  =  x_test.filter(list(num_vars_vt_corr.columns)+list(cat_vars)+list(id_columns)).fillna(0)

print('Número total de features =', len(x_train.columns))


# ### Recursive Feature elimination (Feature selection)
# 
# Com intuito de selecionar as variaveis que melhor irão nos ajudar a discriminar nosso problema iremos utilizar um método chamado recursive feature elimination. esse método funciona da seguinte forma:
# 
# - 1) Treina um modelo com todas as features
# - 2) Elimina as features com feature_importances_ menores
# - 3) Retreina um novo modelo com as features restantes
# - 4) Repete passo 2 e 3
# - 5) Avalia o número de features selecionadas versus a métrica de sucesso do seu modelo
# 
# Aqui escolhemos o algoritmo de Random forest e a métrica de avaliação é o nosso lucro máximo

# In[ ]:


from yellowbrick.model_selection import RFECV
fs_model = RandomForestClassifier(max_depth=7,random_state=seed,n_jobs=-1,n_estimators=100,class_weight='balanced')

# Instantiate RFECV visualizer with a linear Random forest classifier
visualizer = RFECV(fs_model,scoring=lucro,cv=3,step=0.1)

# Fit the data to the visualizer
visualizer.fit(x_train, y_train)

# Finalize and render the figure
visualizer.show()

print('Optimal number of features :', visualizer.n_features_)
best_features = list(x_train.columns[visualizer.support_])
print('features selecionadas: ', best_features)

x_train = x_train[best_features]
x_test = x_test[best_features]


# ### Validation curve 
# 
# Dado que aplicamos o feature selection e outras técnicas para reduzirmos o número de features, iremos novamente treinar um novo modelo random forest passando por toda etapa de tuning de hiperparametros. 
# Para termos um **melhor direcionamento** na etapa de tuning e termos um bom balanco entre **Viés e variancia** iremos utilizar o validation curve para testarmos como nossa métrica de performance se comporta dado um intervalo de busca dos parâmetros que serão tunados na random forest.

# In[ ]:


def plot_validation_curve(x,y,modelo,parametro,param_range,metrica):

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(modelo, 
                                                 x, 
                                                 y, 
                                                 param_name=parametro, 
                                                 param_range=param_range,
                                                 cv=3, 
                                                 scoring=metrica, 
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="red")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="blue")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gray")

    # Create plot
    plt.title("Validation Curve With Random Forest")
    plt.xlabel("Parameter")
    plt.ylabel("retorno")
    plt.tight_layout()
    plt.ylim(ymin=0)
    plt.legend(loc="best")
    plt.show()


# ### Max depth

# In[ ]:


plot_validation_curve(x = x_train,
                      y = y_train.values.reshape(-1,),
                      modelo = RandomForestClassifier(class_weight= 'balanced'),
                      parametro = "max_depth",
                      param_range = np.arange(3, 10, 1),
                      metrica = lucro)


# ### N estimators
# 
# Iremos iterar no número máximo de árvores que nossa random forest pode receber

# In[ ]:


plot_validation_curve(x = x_train,
                      y = y_train.values.reshape(-1,),
                      modelo = RandomForestClassifier(class_weight= 'balanced' ,max_depth=6),
                      parametro = "n_estimators",
                      param_range = np.arange(50, 300, 50),
                      metrica = lucro)


# ### Class weight

# In[ ]:


plot_validation_curve(x = x_train,
                      y = y_train.values.reshape(-1,),
                      modelo = RandomForestClassifier(class_weight= 'balanced' ,max_depth=6),
                      parametro = "class_weight",
                      param_range = np.arange(50, 300, 50),
                      metrica = lucro)


# ### Tuning de hiperparametros dado o validation curve

# In[ ]:


def objective(params):
    
    params = {'n_estimators': int(params['n_estimators'])
              ,'max_depth': int(params['max_depth'])
              ,'max_features': params['max_features']
              ,'min_samples_split': int(params['min_samples_split'])}
    
    clf = RandomForestClassifier(n_jobs=-1
                                 , class_weight='balanced'
                                 , **params
                                 ,random_state=seed)
    
    score = cross_val_score(clf, x_train, y_train, scoring=lucro, cv=StratifiedKFold()).mean()
    
    print("Lucro {:.3f} params {}".format(score, params))
    score = -1*(score)
    return score

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 1000, 50),
    'max_depth': hp.quniform('max_depth', 3, 8, 1),
    'max_features': hp.choice('max_features', ['auto', 'sqrt','log2']),
    "min_samples_split" : hp.quniform('min_samples_split',10,200,10)
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)

# Print best parameters
best_params = space_eval(space, best)

print("Start training the best model")

model = RandomForestClassifier(n_jobs=-1,random_state=seed
                                      , class_weight='balanced'
                                      , max_features=best_params['max_features']
                                      , max_depth = int(best_params['max_depth'])
                                      , min_samples_split = int(best_params['min_samples_split'])
                                      , n_estimators = int(best_params['n_estimators'])).fit(x_train,y_train)
print("Scoring the train data")
#Scoring the best model in train dataset
lucro(model, x_test, y_test)
print("Scoring the train data")
#Scoring the best model in train dataset
predict_train_entire = model.predict(x_train)
proba_train_entire = model.predict_proba(x_train)[:,1]

print("Scoring the test data")
#Scoring the best model in test dataset
predict_test_entire = model.predict(x_test)
proba_test_entire = model.predict_proba(x_test)[:,1]

print("Getting metrics")
# calculate scores
auc_train = roc_auc_score(y_train, proba_train_entire)
auc_test = roc_auc_score(y_test, proba_test_entire )

# calculate roc curves
fpr_train, tpr_train, _ = roc_curve(y_train, proba_train_entire)
fpr_test, tpr_test, _ = roc_curve(y_test, proba_test_entire)

# plot the roc curve for the model
pyplot.plot(fpr_train, tpr_train, linestyle='--', label='Train')
pyplot.plot(fpr_test, tpr_test, linestyle='--', label='Test')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# summarize scores
print('ROC AUC Train =%.3f' % (auc_train))
print('ROC AUC Test =%.3f' % (auc_test))

print('-----------------------------------------------------')

ks_stat_train = binary_ks_curve(y_train, predict_train_entire)[3]
print('ks train =',ks_stat_train)

ks_stat_test = binary_ks_curve(y_test, predict_test_entire)[3]
print('ks test =',ks_stat_test)

print('-----------------------------------------------------')

recall_train = recall_score(y_train, predict_train_entire)
print('recall train =',recall_train)

recall_test = recall_score(y_test, predict_test_entire)
print('recall test =',recall_test)

print('-----------------------------------------------------')
print("------------------------")
print("Calculate decil in train")
print("------------------------")

avg_tgt = y_train.sum()/len(y_train)
df_data = x_train.copy()
X_data = df_data.copy()
df_data['Actual'] = y_train
df_data['Predict']= model.predict(X_data)
y_Prob = pd.DataFrame(model.predict_proba(X_data))
df_data['Prob_1']=list(y_Prob[1])
df_data.sort_values(by=['Prob_1'],ascending=False,inplace=True)
df_data.reset_index(drop=True,inplace=True)
df_data['Decile']=pd.qcut(df_data.index,5,labels=False)
output_df = pd.DataFrame()
grouped = df_data.groupby('Decile',as_index=False)
output_df['Qtd']=grouped.count().Actual
output_df['Sum_Target']=grouped.sum().Actual
output_df['Per_Target'] = (output_df['Sum_Target']/output_df['Sum_Target'].sum())*100
output_df['Per_Acum_Target'] = output_df.Per_Target.cumsum()
output_df['Max_proba']=grouped.max().Prob_1
output_df['Min_proba']=grouped.min().Prob_1
output_df["Per_Pop"] = (output_df["Qtd"]/len(y_train))*100
output_df["Lift"] = output_df["Per_Acum_Target"]/output_df.Per_Pop.cumsum()
output_df= output_df.drop(columns='Per_Pop')
print(round(output_df,3))

print("------------------------")
print("Calculate decil in test")
print("------------------------")
Avg_tgt = y_test.sum()/len(y_test)
df_data = x_test.copy()
X_data = df_data.copy()
df_data['Actual'] = y_test
df_data['Predict']= model.predict(X_data)
y_Prob = pd.DataFrame(model.predict_proba(X_data))
df_data['Prob_1']=list(y_Prob[1])
df_data.sort_values(by=['Prob_1'],ascending=False,inplace=True)
df_data.reset_index(drop=True,inplace=True)
df_data['Decile']=pd.qcut(df_data.index,5,labels=False)
output_df = pd.DataFrame()
grouped = df_data.groupby('Decile',as_index=False)
output_df['Qtd']=grouped.count().Actual
output_df['Sum_Target']=grouped.sum().Actual
output_df['Per_Target'] = (output_df['Sum_Target']/output_df['Sum_Target'].sum())*100
output_df['Per_Acum_Target'] = output_df.Per_Target.cumsum()
output_df['Max_proba']=grouped.max().Prob_1
output_df['Min_proba']=grouped.min().Prob_1
output_df["Per_Pop"] = (output_df["Qtd"]/len(y_test))*100
output_df["Lift"] = output_df["Per_Acum_Target"]/output_df.Per_Pop.cumsum()
output_df= output_df.drop(columns='Per_Pop')
print(round(output_df,3))


# ### Batemos 8920 RS de lucro, agora vamos procura pelo melhor threshould para obtermos um máximo lucro

# In[ ]:


import itertools
def plot_confusion_matrix(y_test, y_pred, title='Confusion matrix'):
    
    cm = confusion_matrix(y_test, y_pred)
    classes = ['Não satisfeito', 'Satisfeito']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, )
    plt.title(title, fontsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    #capturando dados de verdadeiro positivo e falso positivo na matriz de confusao
    verdadeiro_positivo = cm[1][1]
    falso_positivo = cm[0][1]

    #Calcula o lucro com base na formula do problema
    lucro_total = 90*verdadeiro_positivo - 10*falso_positivo 
    print(round(lucro_total,2))
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def train_clf_threshold(x_train,y_train,x_test,y_test,modelo):
    thresholds = np.arange(0.1, 1, 0.1)
    
            
    model.fit(x_train, y_train)
    y_proba = modelo.predict_proba(x_test)
    
    plt.figure(figsize=(10,10))

    j = 1
    for i in thresholds:
        y_pred = y_proba[:,1] > i

        plt.subplot(4, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test,y_pred)
        np.set_printoptions(precision=2)

        print(f"Threshold: {round(i, 1)} | Test Precision: {round(precision_score(y_test, y_pred), 2)} ")

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, title=f'Threshold >= {round(i, 1)}')


# In[ ]:


train_clf_threshold(x_train,y_train,x_test,y_test,modelo=model)


# ## **B**
# A segunda tarefa consiste em dar uma nota de 1 a 5 para cada cliente da base teste, respeitando a variável ‘TARGET’, isto é, o seu nível de satisfação, sendo 1 o mais insatisfeito e 5 o mais satisfeito. Ao dar essa nota deve-se ter em mente que somente os clientes com nota 1 serão alvos de uma ação de retenção e que o objetivo dessa ação é maximizar o lucro esperado por cliente (usando os mesmos valores da primeira questão).

# In[ ]:


print("melhor modelo",model)
proba_test = model.predict_proba(x_test)[:,1]
threshold = 0.7

score_nps = pd.cut(proba_test, bins=list(np.linspace(0,threshold,5))+[1.0], labels=[5,4,3,2,1])

table_fim = pd.concat((pd.Series(score_nps, name='score_nps')
           , y_test.reset_index(drop=True))
          , axis=1, ignore_index=False)

table_fim.groupby(['TARGET','score_nps']).size().unstack('TARGET').plot.bar(stacked=True,figsize = (16,6))
plt.xticks(rotation=0)
plt.ylabel('Volume de clientes no conjunto teste')
plt.xlabel('SCORE NPS')
plt.title('Volume de clientes por nota com volume real de clientes insatisfeitos')


# In[ ]:


y_pred_fim = table_fim[table_fim.score_nps==1]['score_nps']
y_true_fim = table_fim[table_fim.score_nps==1]['TARGET']
y_true_total = table_fim['TARGET']


# In[ ]:


matriz_confusao = confusion_matrix(y_true_fim, y_pred_fim)
      
#capturando dados de verdadeiro positivo e falso positivo na matriz de confusao
verdadeiro_positivo = matriz_confusao[1][1]
falso_positivo = matriz_confusao[0][1]

#Calcula o lucro com base na formula do problema
lucro_total = 90*verdadeiro_positivo - 10*falso_positivo 
print('\nLucro obtido com a ação: R$ ' + str(lucro_total))
print('Porcentagem do lucro ideal: ' + str(round(lucro_total/(int(y_true_total.sum())*90),2)) + '%')


# ## *C*
# Todo conjunto de dados é passível de ser dividido em grupos coesos, conhecidos como agrupamentos naturais. A terceira tarefa é encontrar os três grupos naturais que possuem os maiores lucros esperados por cliente (usando os mesmos valores da primeira questão).

# In[ ]:


def clip_outliers(df,drop_columns,q_min,q_max):
    
    num_vars_name = df.select_dtypes(include=['float64']).columns     
    for i in num_vars_name:
        
        #get max and min quantile
        min_value = df.loc[:,i].quantile(q_min)
        max_value = df.loc[:,i].quantile(q_max)

        #replace values with max and min quantile value
        df.loc[:,i] = np.where(df.loc[:,i] < min_value, min_value,df.loc[:,i])
        df.loc[:,i] = np.where(df.loc[:,i] > max_value, min_value,df.loc[:,i])
        
    return df

#Reduce outliers for clustering analyses
x_train = clip_outliers(x_train,drop_columns=id_columns,q_min=0.05,q_max=0.95)
x_test = clip_outliers(x_test,drop_columns=id_columns,q_min=0.05,q_max=0.95)


# In[ ]:


min = 4
max = 30
wcss = []
silhouette= []

train_kmeans = x_train.select_dtypes(include="float64")

for i in range(min, max):
    
    ##Training a kmeans model
    model = KMeans(n_clusters = i, random_state = seed,n_init=20)
    model.fit(train_kmeans)
    
    #Scoring
    pred = model.predict(train_kmeans)
    
    #Get silhouette score
    score = silhouette_score(train_kmeans, pred)
    
    # inertia method returns wcss for that model
    wcss.append(model.inertia_)
    print('Silhouette Score for k = {}: {:<.3f}'.format(i, score))
    
plt.figure(figsize=(10,max))
sns.lineplot(range(min, max), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


from sklearn.mixture import GaussianMixture
#training
best_k = 9

model = KMeans(n_clusters = best_k, random_state = 42)
# model = GaussianMixture(n_components = best_k, random_state = 42)
model.fit(train_kmeans)

#Scoring
cluster = model.predict(x_test.filter(train_kmeans.columns))

df_cluster = pd.DataFrame({'cluster': cluster}).join(y_test)
df_cluster.columns = ['labels','true_target']

# # Create crosstab: ct
data = pd.crosstab(df_cluster['labels'],df_cluster['true_target'])

data ['volumetria'] = data [0]+data [1]
data ['lucro'] = data [1]*90 - data[0]*10
data.sort_values(by='lucro', ascending=False)


# ### Avaliação de cada cluster em relação ao lucro

# In[ ]:




