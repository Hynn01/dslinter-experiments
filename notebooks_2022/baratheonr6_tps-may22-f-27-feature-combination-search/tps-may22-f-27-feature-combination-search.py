#!/usr/bin/env python
# coding: utf-8

# <div style="background:#2b6684   ;font-family:'Times';font-size:35px;color:  #F0CB8E" >&ensp;TPS - MAY2022</div>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.style as style
import heapq
import scipy
from imblearn.under_sampling import (
            RandomUnderSampler,
            OneSidedSelection,
            InstanceHardnessThreshold,
        )
import lightgbm as lgbm
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif,f_classif
style.use('fivethirtyeight')


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')


# In[ ]:


train['f_27']


# In[ ]:


import string
mapper = dict(zip(string.ascii_uppercase,np.arange(0,len(string.ascii_lowercase))))
for i in range(10):
    train[str(i) +'_27']=train.f_27.str[i]
    train[str(i) +'_27_num']=train[str(i) +'_27'].map(mapper)
    test[str(i) +'_27']=test.f_27.str[i]
    test[str(i) +'_27_num']=test[str(i) +'_27'].map(mapper)


# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:18px; color:#A20404">Observations: 
# <li>We observe that not just the combination but the permutations can perform differently as well ! .</li>
# <li>8_27_num+_7_27_num performs better than 7_27_num+_8_27_num.</li>
# </ul>
# </div>

# In[ ]:


train['7_27_num+_8_27_num']=(train['7_27_num'].astype(str) + train['8_27_num'].astype(str)).astype(int)
test['7_27_num+_8_27_num'] = (test['7_27_num'].astype(str) + test['8_27_num'].astype(str)).astype(int)

train['8_27_num+_7_27_num']=(train['8_27_num'].astype(str) + train['7_27_num'].astype(str)).astype(int)
test['8_27_num+_7_27_num'] = (test['8_27_num'].astype(str) + test['7_27_num'].astype(str)).astype(int)


# In[ ]:


categorical_columns = [str(x)+'_27_num' for x in range(0,10)] +['7_27_num+_8_27_num','8_27_num+_7_27_num']
fs = SelectKBest(score_func=chi2, k='all')
fs.fit(train[categorical_columns], train['target'])
plt.figure(figsize=(10,7))
sns.barplot(x='feat',y='imp',data=pd.DataFrame({'feat':categorical_columns,'imp':fs.scores_}).sort_values(['imp'],ascending=False))
plt.xticks(rotation=70)
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
feature_list = [f for f in train.columns if train[f].dtype in ['float64','int64'] and f not in ['target','id','7_27_num+_8_27_num','8_27_num+_7_27_num']]
X_train, X_test, y_train, y_test = train_test_split( train[feature_list], train['target'], test_size=0.2, random_state=42, stratify=train['target'])


# <div style="background:#2b6684   ;font-family:'Times';font-size:35px;color:  #F0CB8E" >&ensp;f_27 Feature combination search using Genetic algo</div>
# <div class="alert alert-warning" role="alert">
# <p style="font-family:cursive;font-size:20px;color:  #A20404"> We can use two modes of fitness calculation</p>
# 
# <li style="font-family:cursive;font-size:18px; color:#A20404">Chi2 value as fitness.</li>
# <li style="font-family:cursive;font-size:18px; color:#A20404">Feature importance using a model fit on Undersampled Data</li>

# In[ ]:


from typing import Union,TypeVar

class GeneticAlgo:
    def __init__(
        self,
        populationsize:int,
        genelength:int,
        elitism:int,
        X_train:pd.DataFrame,
        y_train:Union[pd.DataFrame,pd.Series],
        data:pd.DataFrame,
        mutation_rate:float,
        n_generations:int,
        non_num_columns:list,
    ):
        """__init__ 

        Parameters
        ----------
        populationsize : int
            population size 
        genelength : int
            number of combinations you want to work with 
        elitism : int
            number of fittest individuals to consider
        X_train : pd.DataFrame
            training data , required for feature importance method
        y_train : Union[pd.DataFrame,pd.Series]
            data , required for feature importance method
        data : pd.DataFrame
            data , required for chi2 method
        mutation_rate : float
            rate of mutation required for genetic algo
        n_generations : int
            number of iterations
        non_num_columns : list
            columns excluding f_27
        """
        self.populationsize = populationsize
        self.population = []
        self.genelength = genelength # if i want feature combinations up to 3 eg.[4,5,6],then genelength =3
        self.elitism = elitism
        self.data = data
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.non_num_columns = non_num_columns
        
        # heap will get us best k features
        self.li = [(-np.inf, [0])] * 10
        heapq.heapify(self.li)

        # Undersampling for feature importance method
        from imblearn.under_sampling import RandomUnderSampler
        nm = RandomUnderSampler(sampling_strategy={0: 100000, 1: 100000})
        self.X_res, self.y_res = nm.fit_resample(X_train, y_train)

    def _check_individuals(self):
        """ -1 represent empty choice.
            This functions checks and replaces all individuals which are completely empty """
        all_empty_individuals = np.where(self.population.sum(axis=1) == -self.genelength)[0]
        for inx in all_empty_individuals:
            pop = np.random.choice(
                np.arange(0, 10),
                size=np.random.choice(np.arange(1, self.genelength + 1)),
                replace=False,
            )
            pop = np.concatenate([pop, np.array([-1] * (self.genelength - len(pop)))])
            self.population[inx] = pop

    def initialize_population(self):
        """ initialize the first population"""
        for _ in range(self.populationsize):
            # we choose without replacement to avoid repetition of same feature
            pop = np.random.choice(
                np.arange(0, 10, dtype=np.int64),
                size=np.random.choice(np.arange(1, self.genelength + 1, dtype=np.int64)),
                replace=False,
            )
            pop = np.concatenate([pop, np.array([-1] * (self.genelength - len(pop)))])
            self.population.append(pop)
        self.population = np.array(self.population)

    def fitness_via_model_split(self):
        """ Calculate fitness value via feature importance """
        self._all = pd.Series(dtype="float64")
        for individual in self.population:
            self._all = pd.concat(
                [
                    self._all,
                    self.X_res[[str(int(x)) + "_27_num" for x in individual if x != -1]]
                    .astype(str)
                    .sum(axis=1)
                    .astype(int)
                    .rename( '+'.join([str(int(x))+'_27_num' for x in individual if x!=-1])),
                ],
                axis=1,
            )

        self._all = self._all.drop(columns=[0])
        all_tomap = list(self._all.columns)

        self._all = self._all.loc[:,~self._all.columns.duplicated()]
        self._all = pd.concat([self._all, self.X_res[self.non_num_columns]], axis=1)

        params = {
            "n_estimators": 1000,
            "reg_lambda": 0.0015,
            "learning_rate": 0.09,
            "max_depth": 11,
            "min_child_weight": 135,
        }
        model = lgbm.LGBMClassifier(**params)
        model.fit(self._all, self.y_res)
        #dd=classification_report(self.y_res, model.predict(self._all),output_dict=True)
        
        # iteration to only include fitness of our generated features
        rank_variable = 1/(1 + len(model.feature_importances_) - scipy.stats.rankdata(model.feature_importances_))
        _dict = dict(zip(self._all.columns, rank_variable ) )
        self.population_fitness = [ _dict[key] for key in all_tomap ]

        for individual, _fit in zip(self.population, self.population_fitness):
            if list(individual) in [_iter[1] for _iter in self.li]:
                ix  = [_iter[1] for _iter in self.li].index(list(individual))
                if _fit>self.li[ix][0]:
                    self.li[ix]=(_fit,list(individual))
                    heapq.heapify(self.li)
            else:   
                # push new individual to heap
                heapq.heappushpop(self.li, (_fit, list(individual)))

        self.ranks = scipy.stats.rankdata(self.population_fitness, method="average")
        self.fitness_ranks = 2 * self.ranks

        if np.max(self.population_fitness) > self.best_fitess:
            self.best_fitess = np.max(self.population_fitness)
            self.best_individual = self.population[np.argmax(self.population_fitness)]

    def fitness(self):
        self._all = pd.Series(dtype="float64")
        for individual in self.population:
            self._all = pd.concat(
                [
                    self._all,
                    self.data[[str(int(x)) + "_27_num" for x in individual if x != -1]]
                    .astype(str)
                    .sum(axis=1)
                    .astype(int),
                ],
                axis=1,
            )

        self._all.columns = np.arange(0, self._all.shape[1])
        self._all = self._all.drop(columns=[0])
        fs = SelectKBest(score_func=chi2, k="all")
        fs.fit(self._all, self.data["target"])
        self.population_fitness = fs.scores_

        for individual, _fit in zip(self.population, self.population_fitness):
            if (_fit, list(individual)) not in self.li:
                heapq.heappushpop(self.li, (_fit, list(individual)))

        self.ranks = scipy.stats.rankdata(self.population_fitness, method="average")
        self.fitness_ranks = 2 * self.ranks

        if np.max(self.population_fitness) > self.best_fitess:
            self.best_fitess = np.max(fs.scores_)
            self.best_individual = self.population[np.argmax(fs.scores_)]

    def _select_individuals(self):
        # self.fitness()
        self.fitness_via_model_split()

        sorted_individuals_fitness = sorted(
            zip(self.population, self.fitness_ranks), key=lambda x: x[1], reverse=True
        )
        elite_individuals = np.array(
            [individual for individual, fitness in sorted_individuals_fitness[: self.elitism]]
        )
        non_elite_individuals = np.array(
            [individual[0] for individual in sorted_individuals_fitness[self.elitism :]]
        )

        non_elite_individuals_fitness = [
            individual[1] for individual in sorted_individuals_fitness[self.elitism :]
        ]
        selection_probability = non_elite_individuals_fitness / np.sum(
            non_elite_individuals_fitness
        )

        selected_indices = np.random.choice(
            range(len(non_elite_individuals)), self.populationsize // 2, p=selection_probability
        )
        selected_individuals = non_elite_individuals[selected_indices, :]
        self.fit_individuals = np.vstack((elite_individuals, selected_individuals))

    def _mutate(self, array):
        mutated_array = np.copy(array)
        for idx, gene in enumerate(array):
            if np.random.random() < self.mutation_rate:
                array[idx] = np.random.choice(np.arange(-1, 10))

        return mutated_array

    def fix_repeatition(self, ind):
        s = set()
        ind_copy = ind
        for i, ix in enumerate(ind):
            if ix in s:
                ind_copy[i] = -1
            s.add(ix)
        return ind_copy

    def _produce_next_generation(self):
        new_population = np.empty(shape=(self.populationsize, self.genelength), dtype=np.float64)

        for i in range(0, self.populationsize, 2):
            parents = self.fit_individuals[
                np.random.choice(self.fit_individuals.shape[0], 2, replace=False), :
            ]

            crossover_index = np.random.randint(0, len(self.population[0]))
            new_population[i] = np.hstack(
                (parents[0][:crossover_index], parents[1][crossover_index:])
            )

            new_population[i + 1] = np.hstack(
                (parents[1][:crossover_index], parents[0][crossover_index:])
            )

            new_population[i] = self.fix_repeatition(self._mutate(new_population[i]))
            new_population[i + 1] = self.fix_repeatition(self._mutate(new_population[i + 1]))
        self.population = new_population

    def fit(self):
        self.initialize_population()

        self.best_fitess = -np.inf
        self.best_individual = [-1] * self.genelength
        for i in range(self.n_generations):
            self._check_individuals()

            self._select_individuals()

            self._produce_next_generation()

            print(
                "Iteration-->",
                i,
                " Best feature-->",
                self.best_individual,
                " Best fitness-->",
                self.best_fitess,
            )

colz = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07',
       'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16',
       'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25',
       'f_26', 'f_28', 'f_29', 'f_30']

obj = GeneticAlgo(
    populationsize=50,
    genelength=3,
    elitism=2,
    X_train=X_train,
    y_train=y_train,
    data=train,
    mutation_rate=0.1,
    n_generations=10,
    non_num_columns=colz,
)
obj.fit()


# In[ ]:


sorted(obj.li, key=lambda x:x[0],reverse=True)


# In[ ]:


ll=[]
for i,ix in enumerate(obj.li):
    if '+'.join([str(int(x))+'_27_num' for x in obj.li[i][1] if x!=-1]) not in ll:
        ll.append('+'.join([str(int(x))+'_27_num' for x in obj.li[i][1] if x!=-1]))
        X_train['+'.join([str(int(x))+'_27_num' for x in obj.li[i][1] if x!=-1])]=X_train[[str(int(x))+'_27_num' for x in obj.li[i][1] if x!=-1]].astype(str).sum(axis=1).astype(int)
        X_test['+'.join([str(int(x))+'_27_num' for x in obj.li[i][1] if x!=-1])]=X_test[[str(int(x))+'_27_num' for x in obj.li[i][1] if x!=-1]].astype(str).sum(axis=1).astype(int)
        test['+'.join([str(int(x))+'_27_num' for x in obj.li[i][1] if x!=-1])]=test[[str(int(x))+'_27_num' for x in obj.li[i][1] if x!=-1]].astype(str).sum(axis=1).astype(int)


# In[ ]:


feature_list = [f for f in X_train.columns if X_train[f].dtype in ['float64','int64'] and f not in ['target','id']]
import lightgbm as lgbm
params = {'n_estimators': 10000,
          'lambda_l2': 0.0015, 
          'alpha': 9.82, 
          'learning_rate': 0.02, 
          'max_depth': 11, 
          'min_child_weight': 135}
model = lgbm.LGBMClassifier(**params)
model.fit(X_train[colz+ll], y_train)

print(classification_report(y_test, model.predict(X_test[colz+ll])))
print(classification_report(y_train, model.predict(X_train[colz+ll])))


# In[ ]:


f, ax = plt.subplots(figsize=(10, 10))
lgbm.plot_importance(model,ignore_zero=False,ax=ax)

