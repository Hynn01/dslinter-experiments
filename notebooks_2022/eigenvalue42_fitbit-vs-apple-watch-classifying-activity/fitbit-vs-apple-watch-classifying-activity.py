#!/usr/bin/env python
# coding: utf-8

#  <font size="6.5"> Predicting Activity From Fitbit and Apple Watch Data</font> 

# The Goal of this project is to see how accurately various popular classifiers can predict the activity of the participant given both the Fitbit and Apple Watch data. Along the way I will conduct exploratory data analysis and look at the predictor space using K-Means Clustering and Principle Component Analysis. Finally, I will perform a grid search with a 5 fold cross validation to tune various classifiers and then test the best models on a validation set and display the classification report as well as the confusion matrix.

#  <font size="5"> Imports</font> 

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
from itertools import product
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import  GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Set Style for Seaborn Plots
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


#  <font size="5"> Read in Data</font> 

# In[ ]:


df=pd.read_csv('/kaggle/input/apple-watch-and-fitbit-data/aw_fb_data.csv')
df.drop(['Unnamed: 0','X1'],inplace = True , axis =1 )
df.rename({'hear_rate':'heart_rate'},inplace = True, axis =1 )


#  <font size="5"> Feature Engineering</font> 

# In[ ]:


#Create unique ID to match with person
unique_id_list = ['age','gender','height','weight']
df_person = df[unique_id_list].drop_duplicates()
df_person.reset_index(drop=True,inplace=True)
df_person['person'] = list(range(df_person.shape[0]))
df = pd.merge(df,df_person,on=unique_id_list)
# Calculate Bmi
df['bmi'] = df['weight']/(df['height']**2) * 10_000 
df['calories_bmi'] =df['calories']/df['bmi']


#  <font size="5"> Helper Classes And Functions</font> 

# In[ ]:


# Class to wrap grid_search and create pipeline steps
class ClassifierPipeLine:
    """Class to run gridsearch pipeline for
    classifiers and display confusion matrix and
    classification report.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        clf,
    ):
        """Initialize Object

        Args:
            X_train (_type_): training predictors
            y_train (_type_): training response
            X_test (_type_): test predictors
            y_test (_type_): test response
            clf (_type_): classifer
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf = clf

        self.pipeline = None
        self.grid_cv_fit = None

    def create_pipeline(
        self, steps: Union[List[Tuple[str, str]], None] = None, output: bool = False
    ) -> Union[None, Pipeline]:
        """Takes in steps for pipeline default is set to None
        If no steps are given will use classifier as last step
        automatically.

        Args:
            steps (Union[List[Tuple[str,str]],None], optional): _description_. Defaults to None.
            output (bool, optional): _description_. Defaults to False.

        Returns:
            Union[None,Pipeline]: _description_
        """
        if steps:
            steps.append(("clf", self.clf))
        else:
            steps = [("clf", self.clf)]
        self.pipeline = Pipeline(steps)
        return self.pipeline if output else None

    def create_grid_search(
        self, param_grid: Dict[str, str], cv: int, scoring: str, output: bool = False
    ) -> Union[GridSearchCV, None]:
        """will create a grid search object given params

        Args:
            param_grid (_type_): _description_
            cv (_type_): _description_
            scoring (_type_): _description_
            output (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        model_obj = (
            self.pipeline if self.pipeline else self.create_pipeline(output=True)
        )
        grid_cv = GridSearchCV(model_obj, param_grid=param_grid, cv=cv, scoring=scoring)
        self.grid_cv_fit = grid_cv.fit(self.X_train, self.y_train)
        return self.grid_cv_fit if output else None

    def display_confusion_matrix(
        self, font_scale: float, fig_size: Tuple[int, int]
    ) -> None:
        """Takes in font_scale and fig_size and displays confusion matrix
        of y_test and clf.predict(X_test)

        Args:
            font_scale (float): the font scale
            fig_size (Tuple[int,int]): figsize
        """

        sns.set(font_scale=font_scale)
        fig, ax = plt.subplots(figsize=fig_size)
        cmp = ConfusionMatrixDisplay(
            confusion_matrix(self.y_test, self.grid_cv_fit.predict(self.X_test)),
            display_labels=self.y_test.sort_values().unique(),
        )
        cmp.plot(ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"Confusion Matrix for {type(self.clf).__name__}")
        plt.tight_layout()

    def display_classification_report(self) -> None:
        """Prints Classification Report"""
        print(classification_report(self.y_test, self.grid_cv_fit.predict(self.X_test)))


# strip_box plot
def grid_strip_box(
    rows: int, cols: int, figsize: Tuple[int, int], var_list: List[str], **params
) -> None:
    """Creates a Strip plot inside a box plot
    Used to analyze multiple predictors across target
    levels of activity in facetgrid

    Args:
        rows (int): number of rows
        cols (int): number of columns
        figsize (Tuple[int,int]): figsixe
        var_list (List[str]): list of predictors
    """
    fig, ax = plt.subplots(rows, cols, figsize=figsize)

    # if statement to take care of edge case where ax will not be a tuple
    if (rows > 1) & (cols > 1):
        l1 = list(range(rows))
        l2 = list(range(cols))

        axes_list = list(product(l1, l2))
    else:
        axes_list = list(range(max([rows, cols])))

    for y_var, ax_tup in zip(var_list, axes_list):

        sns.stripplot(
            y=y_var,
            size=5,
            jitter=0.35,
            palette="tab10",
            edgecolor="black",
            linewidth=1,
            ax=ax[ax_tup],
            **params,
        )
        sns.boxplot(y=y_var, palette=["#D3D3D3"], linewidth=5, ax=ax[ax_tup], **params)
        ax[ax_tup].set_title(f"Distribution of {(y_var.replace('_',' ')).title()} by Activity")
        ax[ax_tup].tick_params(axis="x", labelrotation=45)
        plt.tight_layout()
    plt.show()

    # Standard Scaler function


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function to scale a dataframe

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: scaled dataframe
    """
    scaler = StandardScaler()
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df))


# Return inertia for K Means Elbow Plot


def calculate_inertia(k: int, df: pd.DataFrame) -> float:
    """Calculate the intertia from a k means
    used to iterate over array of k values
    for elbow plot

    Args:
        k (int): K in KMeans
        df (pd.DataFrame): Data

    Returns:
        float: inertia
    """
    scaled_df = scale_data(df)
    model = KMeans(n_clusters=k)
    model.fit(scaled_df)
    return model.inertia_


#  <font size="5"> EDA Apple Watch</font> 

# In[ ]:


df_a = df[df['device']=='apple watch'].copy()
df_a.reset_index(drop=True,inplace=True)


# In[ ]:


g = sns.FacetGrid(data = df_a,col='activity', col_wrap=3,height=3,aspect=1.5,hue = 'person',palette = 'tab10')
g.map(sns.scatterplot,'bmi','calories' )
plt.tight_layout()


# For the Apple Watch data it appears each person tend to show consistent calories burned across all activities as we can see the snowman like stacking of the dots.

# In[ ]:


sns.set(font_scale=1)
params = dict(data=df_a,
              x='activity',            
              order=['Lying', 'Sitting', 'Self Pace walk', 'Running 3 METs','Running 5 METs', 'Running 7 METs'])
# List of Variables to Compare Distributions
distribution_vars = ['steps', 'heart_rate',
                     'calories','distance',
                     'entropy_heart','calories_bmi',
                     'corr_heart_steps','norm_heart', 
                     'intensity_karvonen','sd_norm_heart', ] 
grid_strip_box(2,2,(15,10),var_list=distribution_vars[0:4],**params)


# In[ ]:


# List of Variables to Compare Distributions
grid_strip_box(2,2,(15,10),var_list=distribution_vars[4:9],**params)


# In[ ]:


grid_strip_box(1,2,(15,5),var_list=distribution_vars[8:],**params)


# For Apple Watch Data it looks like heart rate, sd norm hear, intensity karvonen all show separation by activity. Next I will look at some histograms for norm heart and the calories/bmi feature.

# In[ ]:


g = sns.FacetGrid(data = df_a,col='activity', col_wrap=3,height=3,aspect=1.5,)
g.map(sns.histplot,'norm_heart', kde =True , color = '#ff7f0e', stat='probability')
plt.tight_layout()


# In[ ]:


g = sns.FacetGrid(data = df_a,col='activity', col_wrap=3,height=3,aspect=1.5,)
g.map(sns.histplot,'calories_bmi', kde =True , color = '#1f77b4', stat='probability',bins=45)
plt.tight_layout()


# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize = (15,10))
sns.heatmap(df_a.corr(),annot=True,fmt='.2f')
plt.tight_layout()


# Steps and Distance show a somewhat strong correlation. However, one would expect it to be stronger.

# In[ ]:



plt.figure(figsize=(15,10))
sns.regplot(data = df_a, x='steps', y='distance')
plt.title('Distance Vs. Steps')
plt.tight_layout()


# The Regplot shows there is a strong linear relationship between distance and steps, however the regression line is weighted down by high leverage noise in the data.
# 

#  <font size="5"> Training and Validation Apple Watch</font> 

# In[ ]:


# Break into Test Train Split
# predictor_list = ['calories','distance','norm_heart', 'intensity_karvonen','sd_norm_heart','hear_rate']
predictor_list = [col for col in df.columns if col not in ['activity','age', 'gender', 'height', 'weight','person','bmi','device','resting_heart']]
X = df_a[predictor_list]
y = df_a['activity']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state = 42 , stratify=y)


#  <font size="5"> PCA Apple Watch</font> 

# In[ ]:


X_train_scaled = scale_data(X_train)
plt.figure(figsize = (15,10))
pca = PCA()
pca.fit(scale_data(X_train_scaled)) 
components = range(pca.n_components_)
plot_comp_df = pd.DataFrame({'Components':components, 'Variance':pca.explained_variance_})
bp = sns.barplot(x = 'Components',y = 'Variance',data = plot_comp_df , palette = 'tab10')
variance_ratios = [round(float(ratio),3) for ratio in pca.explained_variance_ratio_]
# font = {'family': 'serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 16,
#         }

font = {'color':  'black',
        'weight': 'normal',
        'size': 16,
        }


for component,variance,ratio in zip(components , pca.explained_variance_ ,variance_ratios):
    bp.text(x=component,y=variance,s=ratio,ha="center",fontdict = font)

plt.title('Explained Variance',fontdict = font)
plt.tight_layout()


# In[ ]:


np.cumsum(pca.explained_variance_ratio_)


# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize = (13,10))
loadings = pca.components_
n_comp = pca.n_features_
pc_list = ["PC"+str(i) for i in range(1,n_comp+1)]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = X_train.columns.values
loadings_df = loadings_df.set_index('variable')
sns.heatmap(loadings_df ,annot=True,fmt='.2f')
plt.tight_layout()


# PCA shows that 6 PCs cover 95% of the variance in the predictor space. 
# The PCA correlation plot shows a few mild correlations between the norm_heart predictor, intensity karvonen and the first PC.

#  <font size="5"> K-Means Clustering Apple Watch</font> 

# In[ ]:


k_s = list(range(2,len(y_train.unique() ) + 3))
inertias = [calculate_inertia(k,X_train_scaled) for k in k_s]


# In[ ]:


plt.figure(figsize = (13,10))
sns.pointplot(x=k_s,y=inertias , linestyles ='--',markers = 'o',scale=1.5, color='#2ca02c')
sns.set(font_scale=1.3)
plt.title('Inertia Vs K')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.tight_layout()


# The inertia elblow plot suggests that 4 would be an ideal number for k on the predictor space.

# In[ ]:


model_kmeans = KMeans(n_clusters = 4 , random_state=42)
model_kmeans.fit_predict(X_train_scaled)
clusters = pd.Series( model_kmeans.fit_predict(X_train_scaled)).rename('clusters',inplace= True)


# In[ ]:


plt.figure(figsize = (15,10))
cross_tab = pd.crosstab(y_train,clusters,normalize=False)
cross_tab.values
row_sums = cross_tab.values.sum(axis=1)
proportions = cross_tab / row_sums.reshape((6,1))
sns.heatmap(proportions,annot=True,fmt='.2f')
plt.title('Proportions of Activities in clusters')
plt.tight_layout()


# There does not appear to be a strong relationship between the activities and the clusters from k means as the frequency of each level of activity does not favor any 1 particular cluster.

#  <font size="5"> Random Forest Pipeline Apple Watch</font> 

# In[ ]:


parameters = {'clf__criterion':['entropy','gini'] , 'clf__max_depth':[80,100] , 'clf__n_estimators':[100,120],"clf__max_features":['sqrt', 'log2']}
rf_model  = ClassifierPipeLine( X_train, y_train ,  X_test, y_test,RandomForestClassifier())
grid_cv = rf_model.create_grid_search(param_grid = parameters , cv =5 , scoring = 'accuracy',output = True)


# In[ ]:


grid_cv.best_params_
rf_params = {key[5:]:val for key,val in grid_cv.best_params_.items()}
rf = RandomForestClassifier()
rf.set_params(**rf_params)
rf.fit(X_train,y_train)

importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

importances_sorted = importances.sort_values()
plt.figure(figsize = (15,10))
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='#2ca02c')
plt.title('Features Importance')
plt.show()


# In[ ]:


rf_model.display_classification_report()


# In[ ]:


rf_model.display_confusion_matrix(font_scale=1.5,fig_size=(15,10))


# The Random Forest had an 82 percent accuracy and had 0.90 precision and 0.88 recall on the Running 7 Mets category.
# Additionally, for Apple Watch data the variable importance favored the variables related to heart rate. It will be interesting to see if this holds for Fitbit data as well.

#  <font size="5"> KNN Pipeline Apple Watch</font> 

# In[ ]:


parameters = {'clf__n_neighbors':list(range(3,15,1)),'clf__metric':['euclidean']}
steps = [('scale',StandardScaler()),('KNN',KNeighborsClassifier())]

knn_model  = ClassifierPipeLine( X_train, y_train ,  X_test, y_test,KNeighborsClassifier())
knn_model.create_pipeline()
grid_cv = knn_model.create_grid_search(param_grid = parameters , cv =5 , scoring = 'accuracy',output = True)


# In[ ]:


knn_model.display_classification_report()


# In[ ]:


knn_model.display_confusion_matrix(font_scale=1.5,fig_size=(15,10))


# Knn shows a 61 percent accuracy which is much lower than the Random Forest.

#  <font size="5"> NB Pipeline Apple Watch</font> 

# In[ ]:


parameters = {'clf__var_smoothing':[1e-9,1e-8,1e-7]}
steps = [('scale',StandardScaler())]
nb_model  = ClassifierPipeLine( X_train, y_train ,  X_test, y_test,GaussianNB())
nb_model.create_pipeline(steps = steps)
grid_cv = nb_model.create_grid_search(param_grid = parameters , cv =5 , scoring = 'accuracy',output = True)


# In[ ]:


nb_model.display_classification_report()


# In[ ]:


nb_model.display_confusion_matrix(font_scale=1.5,fig_size=(15,10))


# The Naive Bayes Classifier shows a mere 32% accuracy which is worse than both KNN and Random Forest. It appears this data does not do as well with traditional statistics based approaches. Next I will try a Gradient Boost Classifier.

#  <font size="5"> Gradient Boost Pipeline Apple Watch</font> 

# In[ ]:


parameters = {'clf__n_estimators':[200,250] , 'clf__learning_rate':[0.5,0.1],'clf__max_features':["sqrt"], 'clf__max_depth':[10,5]}
g_boost_clf = GradientBoostingClassifier()
g_boost_model = ClassifierPipeLine( X_train, y_train ,  X_test, y_test,g_boost_clf)
g_boost_model.create_pipeline()
grid_cv = g_boost_model.create_grid_search(param_grid = parameters , cv =5 , scoring = 'accuracy',output = True)


# In[ ]:


g_boost_model.display_classification_report()


# In[ ]:


g_boost_model.display_confusion_matrix(font_scale=1.5,fig_size=(13,10))


# In[ ]:


grid_cv.best_params_
gb_params = {key[5:]:val for key,val in grid_cv.best_params_.items()}
gb = GradientBoostingClassifier()
gb.set_params(**gb_params)
gb.fit(X_train,y_train)

importances = pd.Series(data=gb.feature_importances_,
                        index= X_train.columns)

importances_sorted = importances.sort_values()
plt.figure(figsize = (13,10))
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='#2ca02c')
plt.title('Features Importance')
plt.show()


# Gradient Boost shows an 84% accuracy with a 0.93 Precision and 0.89 recall on Running 7 Mets. 
# Feature Importance also shows the heart rate variables as the ones with the highest feature importance.

#  <font size="5"> EDA Fitbit</font> 

# In[ ]:


df_f = df[df['device'] == 'fitbit'].copy()
df_f.reset_index(drop=True,inplace=True)


# In[ ]:


g = sns.FacetGrid(data = df_f,col='activity', col_wrap=3,height=3,aspect=1.5,hue = 'person',palette = 'tab10')
g.map(sns.scatterplot,'bmi','calories' )
plt.tight_layout()


# In[ ]:


sns.set(font_scale=1)
params = dict(data=df_f,
              x='activity',            
              order=['Lying', 'Sitting', 'Self Pace walk', 'Running 3 METs','Running 5 METs', 'Running 7 METs'])
# List of Variables to Compare Distributions
distribution_vars = ['steps', 'heart_rate',
                     'calories','distance',
                     'entropy_heart','calories_bmi',
                     'corr_heart_steps','norm_heart', 
                     'intensity_karvonen','sd_norm_heart', ] 
grid_strip_box(2,2,(15,10),var_list=distribution_vars[0:4],**params)


# In[ ]:


grid_strip_box(2,2,(15,10),var_list=distribution_vars[4:9],**params)


# In[ ]:


grid_strip_box(1,2,(15,5),var_list=distribution_vars[8:],**params)


# Interestingly, the Fitbit data seems to show more separation on calories, distance and steps across the activities where Apple Watch showed heart rate to be a stronger separator.

# In[ ]:


g = sns.FacetGrid(data = df_f,col='activity', col_wrap=3,height=3,aspect=1.5,)
g.map(sns.histplot,'norm_heart', kde =True , color = '#ff7f0e', stat='probability')
plt.tight_layout()


# In[ ]:


g = sns.FacetGrid(data = df_f,col='activity', col_wrap=3,height=3,aspect=1.5,)
g.map(sns.histplot,'calories', kde =True , color = '#1f77b4', stat='probability',bins=45)
plt.tight_layout()


# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize = (15,10))
sns.heatmap(df_f.corr(),annot=True,fmt='.2f')
plt.title('Correlation Matrix Fitbit')
plt.tight_layout()


# Distance and steps show a weaker correlation on the Fitbit data. Looking at the data, it also appears the units on distance and steps are different between the 2 devices.

# In[ ]:


g = sns.FacetGrid(data = df_f,col='activity', col_wrap=3,height=3,aspect=1.5,hue = 'person',palette = 'tab10')
g.map(sns.scatterplot,'steps','distance' )
plt.tight_layout()


# It looks like in some of the Fitbit data there is some of the positive correlation we would expect to see, but it also looks like there is a lot of noise.

#  <font size="5"> Training and Validation Fitbit</font> 

# In[ ]:


predictor_list = [col for col in df_f.columns if col not in ['activity','age', 'gender', 'height', 'weight','bmi','device','person','resting_heart']]

X = df_f[predictor_list]
y = df_f['activity']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state = 42 , stratify=y)


#  <font size="5"> PCA Fitbit</font> 

# In[ ]:


X_train_scaled = scale_data(X_train)
from sklearn.decomposition import PCA
plt.figure(figsize = (13,10))
pca = PCA()
pca.fit(scale_data(X_train_scaled)) 
components = range(pca.n_components_)
plot_comp_df = pd.DataFrame({'Components':components, 'Variance':pca.explained_variance_})
bp = sns.barplot(x = 'Components',y = 'Variance',data = plot_comp_df , palette = 'tab10')
variance_ratios = [round(float(ratio),3) for ratio in pca.explained_variance_ratio_]

font = {'color':  'black',
        'weight': 'normal',
        'size': 16,
        }


for component,variance,ratio in zip(components , pca.explained_variance_ ,variance_ratios):
    bp.text(x=component,y=variance,s=ratio,ha="center",fontdict = font)

plt.title('Explained Variance',fontdict = font)
plt.tight_layout()


# In[ ]:


np.cumsum(pca.explained_variance_ratio_)


# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize = (15,10))
loadings = pca.components_
n_comp = pca.n_features_
pc_list = ["PC"+str(i) for i in range(1,n_comp+1)]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = X_train.columns.values
loadings_df = loadings_df.set_index('variable')
sns.heatmap(loadings_df ,annot=True,fmt='.2f')
plt.tight_layout()


# From PCA we see about 95% of the variance in the predictor space is taken care of in the first 6 PCs. We also see norm_hear and intensity karvonen correlate strongest with PC1

#  <font size="5"> K Means Fitbit</font> 

# In[ ]:


k_s = list(range(2,len(y_train.unique() ) + 3))
inertias = [calculate_inertia(k,X_train_scaled) for k in k_s]


# In[ ]:


plt.figure(figsize = (13,10))
sns.pointplot(x=k_s,y=inertias , linestyles ='--',markers = 'o',scale=1.5, color='#2ca02c')
sns.set(font_scale=1.3)
plt.title('Inertia Vs K')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.tight_layout()


# In[ ]:


model_kmeans = KMeans(n_clusters = 5 , random_state=40)
model_kmeans.fit_predict(X_train_scaled)
clusters = pd.Series( model_kmeans.fit_predict(X_train_scaled)).rename('clusters',inplace= True)


# In[ ]:


plt.figure(figsize = (15,10))
cross_tab = pd.crosstab(y_train,clusters,normalize=False)
row_sums = cross_tab.values.sum(axis=1)
proportions = cross_tab / row_sums.reshape((6,1))
sns.heatmap(proportions,annot=True,fmt='.2f')
plt.title('Proportions of Activities in clusters')
plt.tight_layout()


# From the Elbow plot of the inertia score it looks like 5 cluster is ideal. From the cluster crosstab It does look like the Lying activity lines up slightly more with cluster 1.

#  <font size="5"> Random Forest Pipeline Fitbit</font> 

# In[ ]:


parameters = {'clf__criterion':['entropy','gini'] , 'clf__max_depth':[80,100] , 'clf__n_estimators':[100,120],"clf__max_features":['sqrt', 'log2']}
rf_model  = ClassifierPipeLine( X_train, y_train ,  X_test, y_test,RandomForestClassifier())
grid_cv = rf_model.create_grid_search(param_grid = parameters , cv =5 , scoring = 'accuracy',output = True)


# In[ ]:


grid_cv.best_params_
rf_params = {key[5:]:val for key,val in grid_cv.best_params_.items()}
rf = RandomForestClassifier()
rf.set_params(**rf_params)
rf.fit(X_train,y_train)

importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

importances_sorted = importances.sort_values()
plt.figure(figsize = (15,10))
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='#2ca02c')
plt.title('Features Importance')
plt.show()


# In[ ]:


rf_model.display_classification_report()


# In[ ]:


rf_model.display_confusion_matrix(1.5 , (13,10))


# The Random Forest showed an 89% accuracy on the validation set with a 0.95 precision and 0.95 recall on the 7 Mets activity on the fitbit data. Interestingly, the fitbit data showed steps and calories to be the most important features where as Apple Watch showed variables related to heart rate.

#  <font size="5"> KNN Pipeline Fitbit</font> 

# In[ ]:


parameters = {'clf__n_neighbors':list(range(3,15,1)),'clf__metric':['euclidean']}
steps = [('scale',StandardScaler()),('KNN',KNeighborsClassifier())]

knn_model  = ClassifierPipeLine( X_train, y_train ,  X_test, y_test,KNeighborsClassifier())
knn_model.create_pipeline()
grid_cv = knn_model.create_grid_search(param_grid = parameters , cv =5 , scoring = 'accuracy',output = True)


# In[ ]:


knn_model.display_classification_report()


# In[ ]:


knn_model.display_confusion_matrix(1.5, (15,10))


# On the Fitbit data KNN achieved an accuracy of 80% on the validation data set.

#  <font size="5"> Naive Bayes Pipeline Fitbit</font> 

# In[ ]:


parameters = {'clf__var_smoothing':[.1,.15,0.5]}
steps = [('scale',StandardScaler())]
nb_model  = ClassifierPipeLine( X_train, y_train ,  X_test, y_test,GaussianNB())
nb_model.create_pipeline(steps = steps)
grid_cv = nb_model.create_grid_search(param_grid = parameters , cv =5 , scoring = 'accuracy',output = True)


# In[ ]:


nb_model.display_classification_report()


# In[ ]:


nb_model.display_confusion_matrix(1.5, (15,10))


# The Naive Bayes does very poorly on the Fitbit data as well

#  <font size="5"> Gradient Boost Pipeline Fitbit</font> 

# In[ ]:


parameters = {'clf__n_estimators':[200,250] , 'clf__learning_rate':[0.5,0.1],'clf__max_features':["sqrt"], 'clf__max_depth':[10,5]}
g_boost_clf = GradientBoostingClassifier()
g_boost_model = ClassifierPipeLine( X_train, y_train ,  X_test, y_test,g_boost_clf)
g_boost_model.create_pipeline()
grid_cv = g_boost_model.create_grid_search(param_grid = parameters , cv =5 , scoring = 'accuracy',output = True)


# In[ ]:


g_boost_model.display_classification_report()


# In[ ]:


g_boost_model.display_confusion_matrix(1.5,(15,10))


# In[ ]:


grid_cv.best_params_
gb_params = {key[5:]:val for key,val in grid_cv.best_params_.items()}
gb = RandomForestClassifier()
gb.set_params(**rf_params)
gb.fit(X_train,y_train)

importances = pd.Series(data=gb.feature_importances_,
                        index= X_train.columns)

importances_sorted = importances.sort_values()
plt.figure(figsize = (15,10))
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='#2ca02c')
plt.title('Features Importance')
plt.show()


# The Gradient Boost again seems to perform the best on the Fitbit data with a 90% accuracy and high precision and recall on the Running 7 Mets Category. Again, the feature importance shows steps and calories to be the most important which is not what we saw with the Apple Watch data.

#  <font size="5"> Final Thoughts</font> 

# The EDA showed a lot of noise in many of the variables in the predictor space
# 
# Interestingly, the ensemble models Random Forest and Gradient Boosting performed well on accurately predicting the activity in the validation data set. On both the Fitbit and Apple Watch data the best performance was from the Gradient Boost model with around 90 and 84 percent accuracy respectively. The plus side of using the Stochastic Gradient Boost (when each weak classifier is limited in the predictors it can choose) is that we get to look at the variable importance feature. This also true of the Random Forest model.
# 
# We saw that the feature importance varied between the data collected on Apple Watch and Fitbit. It seems the ensemble models preferred the steps and calories on the Fitbit data and the heart rate related variables on the Apple Watch data. 
# 
# It would be interesting to see how these models would fair if the same data was collected using more current versions of these devices.
#  
# 
