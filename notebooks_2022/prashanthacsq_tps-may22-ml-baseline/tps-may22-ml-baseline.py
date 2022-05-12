#!/usr/bin/env python
# coding: utf-8

# ### **Library Imports**

# In[ ]:


import os
import pickle
import numpy as np
import pandas as pd
import random as r
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,                              ExtraTreesClassifier,                              GradientBoostingClassifier,                              RandomForestClassifier 

from sklearn.metrics import accuracy_score,                             precision_recall_fscore_support,                             roc_auc_score,                             classification_report,                             confusion_matrix


# ### **Utilities and Constants**

# In[ ]:


def breaker(num: int=50, char: str="*") -> None:
    print("\n" + num*char + "\n")

    
def print_scores(accuracy: float, auc: float, precision: np.ndarray, recall: np.ndarray, f_score: np.ndarray) -> None:
    print(f"Accuracy  : {accuracy:.5f}")
    print(f"ROC-AUC   : {auc:.5f}")
    print(f"Precision : {precision}")
    print(f"Recall    : {recall}")
    print(f"F-Score   : {f_score}")
    

def get_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    accuracy = accuracy_score(y_pred, y_true)
    auc = roc_auc_score(y_pred, y_true)
    precision, recall, f_score, _ = precision_recall_fscore_support(y_pred, y_true)

    return accuracy, auc, precision, recall, f_score

    
SEED = 42


# ### **Configuration**

# In[ ]:


class CFG(object):
    def __init__(self,
                 seed: int = 42,
                 n_splits: int = 5,
                 show_info: bool = False,
                 ):

        self.seed = seed
        self.n_splits = n_splits
        
        self.tr_path = "../input/tabular-playground-series-may-2022/train.csv"
        self.ts_path = "../input/tabular-playground-series-may-2022/test.csv"
        self.ss_path = "../input/tabular-playground-series-may-2022/sample_submission.csv"

        self.model_save_path = "models"
        if not os.path.exists(self.model_save_path): os.makedirs(self.model_save_path)


cfg = CFG(seed=SEED)


# ### **Model**

# In[ ]:


class Pipelines(object):
    def __init__(self, model_name: str, preprocessor, seed: int):
        self.model_name = model_name

        if self.model_name == "lgr":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", LogisticRegression(random_state=seed)),
                ]
            )
        
        elif self.model_name == "knc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", KNeighborsClassifier()),
                ]
            )

        
        elif self.model_name == "dtc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", DecisionTreeClassifier(random_state=seed)),
                ]
            )

        elif self.model_name == "etc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", ExtraTreeClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "rfc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "gbc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", GradientBoostingClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "abc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", AdaBoostClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "etcs":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", ExtraTreesClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "gnb":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", GaussianNB()),
                ]
            )


# ### **Train**

# In[ ]:


names = ["lgr", "knc", "gnb", "dtc", "etc", "abc", "gbc", "etcs", "rfc"]

df = pd.read_csv(cfg.tr_path)
df = df.drop(columns=["id", "f_27"])

breaker()
for val in set(df.target):
    print(f"Class {val} count : {df[df.target == val].shape[0]}")

X = df.iloc[:, :-1].copy().values
y = df.iloc[:, -1].copy().values

features = [i for i in range(X.shape[1])]

feature_transformer = Pipeline(
    steps=[
        ("Simple_Imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("Standard_Scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("features", feature_transformer, features),
    ]
)

best_auc = 0.0
for name in names:
    fold = 1
    breaker()
    for tr_idx, va_idx in KFold(n_splits=cfg.n_splits, random_state=cfg.seed, shuffle=True).split(X):
        X_train, X_valid, y_train, y_valid = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
        my_pipeline = Pipelines(name, preprocessor, cfg.seed)
        my_pipeline.model.fit(X_train, y_train)

        y_pred = my_pipeline.model.predict(X_valid)
        acc, auc, pre, rec, f1 = get_scores(y_valid, y_pred)
        print(f"{my_pipeline.model_name}, {fold}\n")
        print_scores(acc, auc, pre, rec, f1)
        print("")

        if auc > best_auc:
            best_auc = auc
            model_fold_name = f"{name}_{fold}"
            
            with open(os.path.join(cfg.model_save_path, f"best_model.pkl"), "wb") as fp:
                pickle.dump(my_pipeline.model, fp)
        fold += 1
    

breaker()
print(f"Best Model : {model_fold_name.split('_')[0]}, Best Fold : {model_fold_name.split('_')[1]}")
breaker()


# In[ ]:


# cols = [col for col in df.columns]
# cat_cols = [col for col in df.select_dtypes(include="object").columns] # f_27

