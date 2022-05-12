#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install econml')
import econml


# ## ForestDML, ForestDRLearner, OrthoForest and CausalForest: Basic Example

# We depict the performance of our ForestDML, ForestDRLearner, OrthoForest and CausalForest estimators on the same data generating process as the one used in the tutorial page of the grf package (see https://github.com/grf-labs/grf#usage-examples). This is mostly for qualitative comparison and verification purposes among our implementation of variants of Causal Forests and the implementation in the grf R package.

# In[ ]:


# Helper imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Generating Process (DGP)
# 
# We'll generate data for two types of models T and Y

# In[ ]:


import numpy as np
import scipy.special
np.random.seed(123)
n = 2000
p = 10
X = np.random.normal(size=(n, p))
true_propensity = lambda x: .4 + .2 * (x[:, 0] > 0)
true_effect = lambda x: (x[:, 0] * (x[:, 0] > 0))
true_conf = lambda x: x[:, 1] + np.clip(x[:, 2], - np.inf, 0)



T = np.random.binomial(1, true_propensity(X))
Y =  true_effect(X) * T + true_conf(X) + np.random.normal(size=(n,))


# ## Cross-Validated Forest Nuisance Models
# 
# We use forest based estimators (Gradient Boosted Forests or Random Forests) as nuisance models. For the meta-learner versions of our forest based estimators, we also use a generic forest estimator even as a final model. The hyperparameters of the forest models (e.g. number of estimators, max depth, min leaf size) is chosen via cross validation. We also choose among Gradient or Random Forests via cross validation

# In[ ]:


from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from econml.sklearn_extensions.linear_model import WeightedLasso

def first_stage_reg():
    return GridSearchCVList([Lasso(),
                             RandomForestRegressor(n_estimators=100, random_state=123),
                             GradientBoostingRegressor(random_state=123)],
                             param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
                                               {'max_depth': [3, None],
                                               'min_samples_leaf': [10, 50]},
                                              {'n_estimators': [50, 100],
                                               'max_depth': [3],
                                               'min_samples_leaf': [10, 30]}],
                             cv=5,
                             scoring='neg_mean_squared_error')

def first_stage_clf():
    return GridSearchCVList([LogisticRegression(),
                             RandomForestClassifier(n_estimators=100, random_state=123),
                             GradientBoostingClassifier(random_state=123)],
                             param_grid_list=[{'C': [0.01, .1, 1, 10, 100]},
                                              {'max_depth': [3, 5],
                                               'min_samples_leaf': [10, 50]},
                                              {'n_estimators': [50, 100],
                                               'max_depth': [3],
                                               'min_samples_leaf': [10, 30]}],
                             cv=5,
                             scoring='neg_mean_squared_error')

def final_stage():
    return GridSearchCVList([WeightedLasso(),
                             RandomForestRegressor(n_estimators=100, random_state=123)],
                             param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
                                              {'max_depth': [3, 5],
                                               'min_samples_leaf': [10, 50]}],
                             cv=5,
                             scoring='neg_mean_squared_error')


# In[ ]:


model_y = clone(first_stage_reg().fit(X, Y).best_estimator_)
model_y


# In[ ]:


model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
model_t


# ## DML Estimators

# In[ ]:


from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier

n_samples, n_features = X.shape
subsample_fr_ = (n_samples/2)**(1-1/(2*n_features+2))/(n_samples/2)
est = CausalForestDML(model_y=model_y,
                      model_t=model_t,
                      discrete_treatment=True,
                      cv=3,
                      n_estimators=4000,
                      random_state=123)
est.tune(Y, T, X=X).fit(Y, T, X=X, cache_values=True)


# In[ ]:


from econml.dml import NonParamDML
est2 = NonParamDML(model_y=model_y,
                   model_t=model_t,
                   cv=3,
                   discrete_treatment=True,
                   model_final=final_stage())
est2.fit(Y, T, X=X)


# In[ ]:


X_test = np.zeros((100, p))
X_test[:, 0] = np.linspace(-2, 2, 100)


# In[ ]:


pred = est.effect(X_test)
lb, ub = est.effect_interval(X_test, alpha=0.01)


# In[ ]:


pred2 = est2.effect(X_test)


# In[ ]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(X_test[:, 0], true_effect(X_test), '--')
plt.plot(X_test[:, 0], pred2, label='nonparamdml')
plt.plot(X_test[:, 0], pred, label='forestdml (causal forest)')
plt.fill_between(X_test[:, 0], lb, ub, alpha=.4, label='honestrf_ci')
plt.legend()
plt.show()


# In[ ]:


np.mean((true_effect(X) - est.effect(X))**2)


# In[ ]:


np.mean((true_effect(X) - est2.effect(X))**2)


# In[ ]:


cf_mse = (true_effect(X) - est.effect(X))**2
np_mse = (true_effect(X) - est2.effect(X))**2


# In[ ]:


plt.plot(np_mse,'go',label="Non-Parametric")
plt.plot(cf_mse,'bo',label="Causal Forest")
plt.legend(loc="upper left")


# Comparing the non-parametric Double ML model (est2) and the Causal Forest Double ML model (est) we observe both of them produce very accurate estimates, but the Causal Forest model displays a lower MSE showing it to be the most efficient and apparent lower dispertion of Squared Errors.

# ## First Stage Learned Models

# In[ ]:


# Model T
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('honestrf')
for mdls in est.models_t:
    for mdl in mdls:
        plt.plot(X_test[:, 0], mdl.predict_proba(X_test)[:, 1])
plt.plot(X_test[:, 0], true_propensity(X_test), '--', label='truth')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('rf')
for mdls in est2.models_t:
    for mdl in mdls:
        plt.plot(X_test[:, 0], mdl.predict_proba(X_test)[:, 1])
plt.plot(X_test[:, 0], true_propensity(X_test), '--', label='truth')
plt.legend()
plt.show()


# In[ ]:


# Model Y
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('honestrf')
for mdls in est.models_y:
    for mdl in mdls:
        plt.plot(X_test[:, 0], mdl.predict(X_test))
plt.plot(X_test[:, 0], true_effect(X_test) * true_propensity(X_test) + true_conf(X_test), '--', label='truth')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('rf')
for mdls in est2.models_y:
    for mdl in mdls:
        plt.plot(X_test[:, 0], mdl.predict(X_test))
plt.plot(X_test[:, 0], true_effect(X_test) * true_propensity(X_test) + true_conf(X_test), '--', label='truth')
plt.legend()
plt.show()


# ## Interpretability of CATE Model of NonParamDML with SHAP

# In[ ]:


import shap
import string

feature_names = list(string.ascii_lowercase)[:X.shape[1]]
# explain the model's predictions using SHAP values
shap_values = est.shap_values(X[:100], feature_names=feature_names)


# In[ ]:


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(shap_values["Y0"]["T0_1"][0], matplotlib=True)


# In[ ]:


shap.summary_plot(shap_values["Y0"]["T0_1"])


# If the dots on one side of the central line are increasingly red or blue, that suggests that increasing values or decreasing values, respectively, move the predicated output in that direction. For instance, lower "a" values (blue dots) are associated with lower predicted output.
# 
# Therefore, feature "a" seems to be very predictive of the output level

# # DRLearner

# 

# In[ ]:


model_regression = clone(first_stage_reg().fit(np.hstack([T.reshape(-1, 1), X]), Y).best_estimator_)
model_regression


# In[ ]:


from econml.dr import ForestDRLearner
from sklearn.dummy import DummyRegressor, DummyClassifier

est = ForestDRLearner(model_regression=model_y,
                      model_propensity=model_t,
                      cv=3,
                      n_estimators=4000,
                      min_samples_leaf=10,
                      verbose=0,
                      min_weight_fraction_leaf=.005)
est.fit(Y, T, X=X)


# In[ ]:


from econml.dr import DRLearner
est2 = DRLearner(model_regression=model_y,
                 model_propensity=model_t,
                 model_final=final_stage(),
                 cv=3)
est2.fit(Y, T.reshape((-1, 1)), X=X)


# In[ ]:


X_test = np.zeros((100, p))
X_test[:, 0] = np.linspace(-2, 2, 100)


# In[ ]:


pred = est.effect(X_test)
lb, ub = est.effect_interval(X_test, alpha=0.01)


# In[ ]:


pred2 = est2.effect(X_test)


# In[ ]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(X_test[:, 0], true_effect(X_test), '--')
plt.plot(X_test[:, 0], pred2, label='nonparamdml')
plt.plot(X_test[:, 0], pred, label='forestdml (causal forest)')
plt.fill_between(X_test[:, 0], lb, ub, alpha=.4, label='honestrf_ci')
plt.legend()
plt.show()


# ## First stage nuisance models

# In[ ]:


# Model T
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('honestrf')
for mdls in est.models_propensity:
    for mdl in mdls:
        plt.plot(X_test[:, 0], mdl.predict_proba(X_test)[:, 1])
plt.plot(X_test[:, 0], true_propensity(X_test), '--', label='truth')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('rf')
for mdls in est2.models_propensity:
    for mdl in mdls:
        plt.plot(X_test[:, 0], mdl.predict_proba(X_test)[:, 1])
plt.plot(X_test[:, 0], true_propensity(X_test), '--', label='truth')
plt.legend()
plt.show()


# In[ ]:


# Model Y
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('honestrf')
for mdls in est.models_regression:
    for mdl in mdls:
        plt.plot(X_test[:, 0], mdl.predict(np.hstack([X_test, np.ones((X_test.shape[0], 1))])))
plt.plot(X_test[:, 0], true_effect(X_test) + true_conf(X_test), '--', label='truth')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('rf')
for mdls in est2.models_regression:
    for mdl in mdls:
        plt.plot(X_test[:, 0], mdl.predict(np.hstack([X_test, np.ones((X_test.shape[0], 1))])))
plt.plot(X_test[:, 0], true_effect(X_test) + true_conf(X_test), '--', label='truth')
plt.legend()
plt.show()


# ## Interpretability of CATE Model of DRLearner with SHAP

# In[ ]:


# explain the model's predictions using SHAP values
shap_values = est.shap_values(X[:100], feature_names=feature_names)


# In[ ]:


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(shap_values["Y0"]["T0_1"][0], matplotlib=True)


# In[ ]:


shap.summary_plot(shap_values["Y0"]["T0_1"])


# # OrthoForest

# In[ ]:


from econml.orf import DROrthoForest
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from econml.sklearn_extensions.linear_model import WeightedLassoCV

est3 = DROrthoForest(model_Y=Lasso(alpha=0.01),
                     propensity_model=LogisticRegression(C=1),
                     model_Y_final=WeightedLassoCV(cv=3),
                     propensity_model_final=LogisticRegressionCV(cv=3),
                     n_trees=1000, min_leaf_size=10)
est3.fit(Y, T, X=X)


# In[ ]:


pred3 = est3.effect(X_test)


# In[ ]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(X_test[:, 0], true_effect(X_test), '--')
plt.plot(X_test[:, 0], pred, label='forestdr')
plt.plot(X_test[:, 0], pred2, label='nonparamdr')
plt.plot(X_test[:, 0], pred3, label='discreteorf')
plt.fill_between(X_test[:, 0], lb, ub, alpha=.4, label='forest_dr_ci')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




