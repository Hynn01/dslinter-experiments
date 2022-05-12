#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction

# ### Data Dictionary
# 
# - **state**: state
# - **account length**: account length (number of days account has been active)
# - **area code**: area code 
# - **phone number**: phone number
# - **international plan**: international plan (yes/no)
# - **voice mail plan**: voice mail plan(yes/no)
# - **number vmail messages**: number of voice mail messages
# - **total day minutes**: day = along the day which is from 8:00 am to 6:00 pm 
# - **total day calls**: total day calls
# - **total day charge**: total day charge
# - **total eve minutes**: total evening minutes
# - **total eve calls**: total evening calls
# - **total eve charge**: total evening charge
# - **total night minutes**: total night minutes
# - **total night calls**: total night calls
# - **total night charge**: total night charge
# - **total intl minutes**: total international minutes
# - **total intl calls**: total international calls
# - **total intl charge**: total international charge
# - **number customer service calls**: number of customer service calls
# - **Churn Flag**: True or Flase (Target Variable)

# In[ ]:


# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import pandas_profiling

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 5 decimal points
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Library to split data
import sklearn
from sklearn.model_selection import train_test_split

#  For preprocessing
from sklearn.preprocessing import (
    LabelEncoder,
    RobustScaler,
)
from mlxtend.preprocessing import minmax_scaling

# To build models for prediction
import statsmodels.stats.api
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier
import optuna

from sklearn.linear_model import LogisticRegression

# To select models
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# To select features
from sklearn.feature_selection import SelectKBest, chi2

# To tune different models
from sklearn.model_selection import GridSearchCV

# Libraries to get different metric scores
from sklearn import metrics
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    plot_confusion_matrix,
    precision_recall_curve,
    classification_report,
    make_scorer,
)
import warnings

warnings.filterwarnings("ignore")


# In[ ]:


# Importing data
data = pd.read_csv('../input/telecom-churn-dataset/churn.csv')


# In[ ]:


df = data.copy()


# In[ ]:


df.head()


# #### Removing Whitespaces
# There are whitespaces in column names, we will be removing them

# In[ ]:


# remove trailing whitespaces
df.columns = df.columns.str.rstrip()


# In[ ]:


# stripping whitespaces in target column
df["Churn Flag"] = [x.strip(" ") for x in df["Churn Flag"]]


# In[ ]:


df.info()


# #### Observations:
# - There are no missing/null values
# - There are 5000 rows across 20 variables
# 

# In[ ]:


# getting count of unique values accross each column
print(df.nunique(axis=0))


# #### Observations:
# - There are 51 different states
# - 3 different area codes
# - Each row in 'phone number' is a unique value
# - There are 3 columns wih Y/N observations

# In[ ]:


# dropping 'phone number' column as it contains all unique values
df.drop(["phone number"], axis=1, inplace=True)


# In[ ]:


# adding total charge column
df["total charge"] = (
    df["total day charge"]
    + df["total eve charge"]
    + df["total night charge"]
    + df["total intl charge"]
)


# In[ ]:


# viewing a sample of the data
df.sample(n=10, random_state=1)


# In[ ]:


df.describe().T


# #### Observations:
# - The average account length is 100 days
# - Average day minutes are 180
# - Average evening minutes are 200
# - Average night minutes 200
# - Average spend on international calls is \\$2.7
# - The most calls made to customer care by a customer was 9
# - The users spent an average of \\$59 in total charges, the least amount is \\$22 and the highest is\\$ 96

# # EDA

# In[ ]:


pandas_profiling.ProfileReport(df)


# ## Univariate Analysis
# 

# In[ ]:


# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="BrBG_r",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# In[ ]:


labeled_barplot(df, "state", perc=True, n=20)


# #### Observations:
# - Percentage of users across states are almost normally distributed at around 2%, with WV having a slightly higher share at 3.2%
# 

# In[ ]:


labeled_barplot(df, "area code", perc=True, n=None)


# #### Observations:
# - The users are spread across 3 different area codes

# In[ ]:


labeled_barplot(df, "international plan", perc=True, n=None)


# #### Observations:
# - Less than 10% of the customers have an International Plan

# In[ ]:


labeled_barplot(df, "voice mail plan", perc=True, n=None)


# #### Observations:
# - 26% of users have a voice mail plan

# In[ ]:


labeled_barplot(df, "total intl calls", perc=True, n=None)


# In[ ]:


labeled_barplot(df, "number customer service calls", perc=True, n=None)


# #### Observations:
# - 56% of users made either one or no calls to customer service
# - Customer care calls made thrice or more represent customers that may be having unresolved issues

# In[ ]:


labeled_barplot(df, "Churn Flag", perc=True, n=None)


# #### Observations:
# - 14% of the customers churnedOf the users in the dataset, 14.1% churned
# 

# In[ ]:


# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, palette="BrBG"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, facecolor='midnightblue'
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, facecolor='midnightblue'
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="black", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="white", linestyle="-"
    )  # Add median to the histogram



# In[ ]:


histogram_boxplot(df, "account length", figsize=(12, 7), kde=False, bins=None)


# In[ ]:


histogram_boxplot(df, "total charge", figsize=(12, 7), kde=False, bins=None)


# In[ ]:


histogram_boxplot(df, "number vmail messages", figsize=(12, 7), kde=False, bins=None)


# In[ ]:


histogram_boxplot(df, "total intl charge", figsize=(12, 7), kde=False, bins=None)


# In[ ]:


histogram_boxplot(df, "total intl calls", figsize=(12, 7), kde=False, bins=None)


# In[ ]:


histogram_boxplot(df, "total day charge", figsize=(12, 7), kde=False, bins=None)


# In[ ]:


histogram_boxplot(df, "total day calls", figsize=(12, 7), kde=False, bins=None)


# #### Observations:
# - The distributions for total charge and account lengths are normal.
# - Mean account length is 100 days
# - The users spent an average of \\$59 in total charges, the least amount is \\$22 and the highest is \\$/96
# - Average customer spend on international calls is \\$2.7
# - Account length has outliers toward the right side, and total charge has outliers on both sides
# - Total International calls has many outliers towards the right.
# 

# ## Bivariate Analysis

# In[ ]:


# heatmap of correlation between variables
cols_list = df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(10, 5))
sns.heatmap(df[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="BrBG")
plt.show()


# #### Observations:
# - Total day, evening, and night minutes have a correlation coefficient of 1 with their respective Total charge columns.

# In[ ]:


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# In[ ]:


stacked_barplot(df, "area code", "Churn Flag")


# #### Observations:
# - The churn rate across all area codes is almost the same.

# In[ ]:


sns.boxplot(x="total day charge", y="Churn Flag", data=df, palette="crest")


# In[ ]:


stacked_barplot(df, "international plan", "Churn Flag")


# #### Observations:
# * Churn rate among customers with an International plan is at around 40%
# 

# In[ ]:


stacked_barplot(df, "number customer service calls", "Churn Flag")


# #### Observations:
# - The number of customer care calls made appears to be inversely related to the customer staying.
# - It is minor among customers who made 0-3 calls
# - It starts to increase from 4 calls, and peaks at 9, which is the highest number of calls made

# In[ ]:


sns.displot(df, x="account length", hue="Churn Flag", multiple="stack")


# #### Observations:
# The churn rate is significant among customers with account lengths falling between 50 and 150 days, and peaks between 100 and 120.

# In[ ]:


sns.displot(df, x="total charge", hue="Churn Flag", multiple="stack")


# #### Observations:
# - The churn rate is significant among customers that spent between $45 and $60, and between 70 and 80, after which it declines.
# - The churn rate peaks at customers spending around $73
# 

# In[ ]:


sns.boxplot(x="account length", y="Churn Flag", data=df, palette="crest")


# In[ ]:


sns.boxplot(x="total charge", y="Churn Flag", data=df, palette="crest")


# In[ ]:


### function to plot distributions wrt target


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="green",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="magma")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="magma",
    )

    plt.tight_layout()
    plt.show()


# In[ ]:


distribution_plot_wrt_target(df, "account length", "Churn Flag")


# In[ ]:


distribution_plot_wrt_target(df, "number customer service calls", "Churn Flag")


# # Model Building

# ### Model evaluation criterion
# 
# ### Possible errors:
# 
# 1. Model predicts that the customer will churn, but in reality, the customer stays:  A False Positive.
# 2. Model predicts that the customer will not churn, but in reality, the customer leaves: A False Negatvive.
# 
# ### Which case is more important? 
# * For us, it is important to prevent the second error
# 
# * If a customer is incorrectly predicted to churn, the company might extend an offer to him/her although it may not have been necessary. This wouldn't put significant pressure on the resources, and hence impact is negligible.
# * But if the opposite happens, an unsatisfied customer may be left unattended and will churn, and a valuable customer will have been lost
# 
# 
# ### How to reduce the losses?
# 
# * `Recall` can be used a the metric for evaluation of the model, greater the Recall score, higher are the chances of minimizing False Negatives.

# # Data Preparation for modeling

# In[ ]:


# creating a copy for model building
df1 = df.copy()


# In[ ]:


# encoding target variable
df1["Churn Flag"] = df1["Churn Flag"].apply(lambda x: 1 if x == "True" else 0)

# encoding
df1["international plan"].replace([" no", " yes"], [0, 1], inplace=True)
df1["voice mail plan"].replace([" no", " yes"], [0, 1], inplace=True)

# Label encoding
encoder = LabelEncoder()
coded_state = encoder.fit_transform(df1["state"])
df1["state"] = coded_state


# In[ ]:


# heatmap of correlation between variables
cols_list = df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(10, 5))
sns.heatmap(df[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="BrBG")
plt.show()


# #### Observations:
# - Total day,evening, night and international minutes have a correlation coefficient of 1 with their repective charges columns, so we'll drop those.
# ##### Correlations with target variable (Churn Flag):
# - International Plan has a 0.26 correlation coefficient
# - Voice mail Plan has a correlation coefficient of negative 0.11
# - Total minutes and charge has a 0.24 p
# - Number of customer service calls has a p of 0.24
# 
# 

# In[ ]:


# Dropping variables with correlation coefficient of 1
df2 = df1.drop(
    [
        "total day minutes",
        "total eve minutes",
        "total night minutes",
        "total intl minutes",
    ],
    axis=1,
)


# In[ ]:


# checking shape of modified dataframe
df2.shape


# In[ ]:


# dividing data into X and Y
X = df2.drop(["Churn Flag"], axis=1)
Y = df2["Churn Flag"]


# In[ ]:


X.shape


# In[ ]:


# scaling the data between 0 and 1
X_scaled = RobustScaler().fit_transform(X)
# X_test = minmax_scaling(X_test_init, columns=features)
# X_test


# In[ ]:


X_scaled


# In[ ]:


# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1, stratify=Y
)


# In[ ]:


X_train


# In[ ]:


# Checking shapes
print("X_train=", X_train.shape)
print("y_train=", y_train.shape)
print("X_test=", X_test.shape)
print("y_test=", y_test.shape)


# In[ ]:


def train_recall(ve):
    return recall_score(y_train, ve.predict(X_train))


def test_recall(we):
    return recall_score(y_test, we.predict(X_test))


def train_accuracy(v):
    return accuracy_score(y_train, v.predict(X_train))


def test_accuracy(w):
    return accuracy_score(y_test, w.predict(X_test))


def class_report(c):
    print("Classification_report:", classification_report((y_test, c.predict(X_test))))


def f_score(f):
    return f1_score(y_test, f.predict(X_test))


def f1_w(i):
    labels = [0, 1]
    f = f1_score(y_test, i.predict(X_test), average=None, labels=labels)
    return round(
        ((f[0] * y_test.value_counts()[0]) + (f[1] * y_test.value_counts()[1]))
        / (y_test.value_counts()[0] + y_test.value_counts()[1]),
        4,
    )


score_card = pd.DataFrame(
    columns=[
        "Model_Name",
        "Train_Accuracy",
        "Test_Accuracy",
        "Train_Recall",
        "Test_Recall",
        "f1_weighted_avg",
    ]
)


def update_score_card(algorithm_name, model):

    global score_card

    score_card = score_card.append(
        {
            "Model_Name": algorithm_name,
            "Train_Accuracy": train_accuracy(model),
            "Test_Accuracy": test_accuracy(model),
            "Train_Recall": train_recall(model),
            "Test_Recall": test_recall(model),
            "f1_weighted_avg": f1_w(model),
        },
        ignore_index=True,
    )


# In[ ]:


def confusion_matrix_sdf(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def scores(a):
    print("Train_Recall_Score:", round(recall_score(y_train, a.predict(X_train)), 3))
    print("Test_Recall_Score:", round(recall_score(y_test, a.predict(X_test)), 3))
    print(
        "Train_Accuracy_Score:", round(accuracy_score(y_train, a.predict(X_train)), 3)
    )
    print("Test_Accuracy_Score:", round(accuracy_score(y_test, a.predict(X_test)), 3))
    print("Train_F1_Score:", round(f1_score(y_train, a.predict(X_train)), 3))
    print("Test_F1_Score:", round(f1_score(y_test, a.predict(X_test)), 3))
    print("Classification_report:\n", classification_report(y_test, a.predict(X_test)))

    confusion_matrix_sdf(a, X_test, y_test)


# # Models

# In[ ]:



logreg_model = LogisticRegression(random_state=25).fit(X_train, y_train)

# Getting the scores
scores(logreg_model)
update_score_card("LogisticRegression", logreg_model)

# Getting probability scores
probability_scores_logreg_model = logreg_model.predict_proba(X_test)
print(probability_scores_logreg_model)


# In[ ]:


knn_model = KNeighborsClassifier().fit(X_train, y_train)

# Getting the scores
scores(knn_model)
update_score_card("KNeighborsClassifier", knn_model)

# Getting probability scores
probability_scores_knn_model = knn_model.predict_proba(X_test)
print(probability_scores_knn_model)


# In[ ]:


dt_model = DecisionTreeClassifier().fit(X_train, y_train)

# Getting the scores
scores(dt_model)
update_score_card("DecisionTreeClassifier", dt_model)

# Getting probability scores
probability_scores_dt_model = dt_model.predict_proba(X_test)
print(probability_scores_dt_model)


# In[ ]:


rfc_model = RandomForestClassifier().fit(X_train, y_train)

# Getting the scores
scores(rfc_model)
update_score_card("RandomForestClassifier", rfc_model)

# Getting probability scores
probability_scores_rfc_model = rfc_model.predict_proba(X_test)
print(probability_scores_rfc_model)


# In[ ]:


adb_model = AdaBoostClassifier().fit(X_train, y_train)

# Getting the scores
scores(adb_model)
update_score_card("AdaBoostClassifier", adb_model)

# Getting probability scores
probability_scores_adb_model = adb_model.predict_proba(X_test)
print(probability_scores_adb_model)


# In[ ]:


LGB_model = lgb.LGBMClassifier().fit(X_train, y_train)

# Getting the scores
scores(LGB_model)
update_score_card("LightGBMClassifier", LGB_model)

probability_scores_LGB_model = LGB_model.predict_proba(X_test)
print(probability_scores_LGB_model)


# # Comparing Model Performances

# In[ ]:


score_card


# #### Models selected for tuning:
# - DecisionTree Classifier
# - LightGBM Classifier

# # Model Tuning

# ### Tuning Decision Tree Classifier

# In[ ]:


dtree_tuned = DecisionTreeClassifier(class_weight="balanced", random_state=1)

# Grid of parameters
parameters = {
    "max_depth": np.arange(10, 30, 5),
    "min_samples_leaf": [3, 5, 7],
    "max_leaf_nodes": [2, 3, 5],
    "min_impurity_decrease": [0.0001, 0.001],
}
# Type of scoring to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(dtree_tuned, parameters, scoring=scorer, n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
dtree_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
dtree_tuned.fit(X_train, y_train)

# Getting the scores
scores(dtree_tuned)
update_score_card("DecisionTreeClassifier-Tuned", dtree_tuned)

# Getting probability scores
probability_scores_dtree_tuned = dtree_tuned.predict_proba(X_test)
print(probability_scores_dtree_tuned)


# ### Tuning LightGBM

# In[ ]:


LGB_tuned = lgb.LGBMClassifier(
    boosting_type="gbdt",
    num_leaves=128,
    class_weight="balanced",
    min_child_samples=38,
    importance_type="split",
).fit(X_train, y_train)

# Getting the scores
scores(LGB_tuned)
update_score_card("LGB-Tuned", LGB_tuned)

# Getting probability scores
probability_scores_LGB_tuned = LGB_tuned.predict_proba(X_test)
print(probability_scores_LGB_tuned)


# In[ ]:


score_card


# #### Finding the best parameters to tune LightGBM using Optuna

# In[ ]:


def objective(trial):
    X_train, X_test, y_train, y_test
    # = train_test_split(X, y, test_size=0.25)
    dtrain = lgb.Dataset(X_train, label=y_train)

    param = {
        "objective": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)
    recall = sklearn.metrics.recall_score(y_test, pred_labels)
    return recall


# In[ ]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)


# In[ ]:


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# #### Tuning LightGBM using the suggested parameters

# In[ ]:


LGB_tuned_optuna = lgb.LGBMClassifier(
    boosting_type="gbdt",
    num_leaves=128,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=100,
    subsample_for_bin=200000,
    objective="binary",
    class_weight="balanced",
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=38,
    feature_fraction=(0.8764512783052837),
    bagging_fraction=(0.9130552028137798),
    bagging_freq=(2),
    lambda_l1=(1.2903804436192665e-06),
    lambda_l2=(2.201288025144258e-06),
    random_state=None,
    n_jobs=-1,
    silent=True,
    importance_type="split",
).fit(X_train, y_train)

# Getting the scores
scores(LGB_tuned_optuna)
update_score_card("LIghtGBM_Tuned-Optuna", LGB_tuned_optuna)

# Getting probability scores
probability_scores_LGB_tuned_optuna = LGB_tuned_optuna.predict_proba(X_test)
print(probability_scores_LGB_tuned_optuna)


# # Comparing Tuned-Model Performances

# In[ ]:


score_card


# #### Of all the models built, LightGBM gives the best Train and Test Recall score, with the highest F1 weighted average.

# In[ ]:


# Plotting feature importances for selected model
feature_names = X_train.columns
importances = LGB_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(
    range(len(indices)), importances[indices], color="midnightblue", align="center"
)
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# # Feature Importances
# - Total charge has the highest importance, followed by a breakdown of the same into international and day.
# - This is followed closely by evening calls and charge, and night charge.
# - These comprise the top 6 important factors.
# - This is then followed by account length.
# - Voice mail plan and usage seems to have very little importance in determining if a customer will churn.
# - Presence of international plan also seems to have very little importance in determining churn probability
# - The least importance was shown by area code of the users
# 

# # Business Insights
# - The average account length is 100 days, which is just 37% of the 9-month period of the dataset
# - Customers that made 4 or more calls to customer service were much likely to churn than those who made 3 calls or less.
# - The most calls made to customer care by a customer was 9, and such customers had a 100% churn rate.
# - The churn rate is significant among customers with account lengths falling between 50 and 150 days, and peaks between 100 and 120.
#  - This means that customers are likely to switch between 2 months of usage and 5 months
#  - This could be because the first set of customers were unsatisfied after one month of the service, and just waited to leave before their next billing cycle.
#  - Highest churn is in customers of between 3 and 4 months. This needs to be investigated, as to what factors are causing this.
# - Total charge:
#   - The churn rate is significant among customers that spent between \\$45 and \\$60, and between \\$70 and \\$80, after which it declines.
#  - The churn rate is highest among customers spending around \\$73.
#   - This insight needs to be investigated further, it could be that these customers are holding a particular plan which costs around $70,  and this particular plan might be causing some issues that most unsatisfied customers are facing.
# 

# # Business Recommendations
# - Customers making 3 calls or more to Customer Service should be flagged as important and these cases should be tracked for immediate resolution of their issues.
# - As customers are most likely to switch between 2-5 months of usage, this signifies that the first 5 months is the important stage where the brand loyalty needs to be established
# - Highest churn rate is found in customers of between 3 and 4 months. This needs to be investigated, as to what factors are causing this.
# - The churn rate is highest among customers spending around \\$73.
# - This insight needs to be investigated further, as it could be that these customers are holding a particular plan which costs \\$73,  and this particular plan might be having some issues that most unsatisfied customers are facing.
# - A system needs to be brought in place to record the customer satisfaction after each call made to customer care. - This will not only help improve resolution efficacy, but also help better predict churn probability of any account

# In[ ]:




