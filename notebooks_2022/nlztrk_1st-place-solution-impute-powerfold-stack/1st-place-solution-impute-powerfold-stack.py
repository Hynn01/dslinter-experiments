#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=DeprecationWarning) 


# # NaN Values
# The data given by this competition has many NaN values and upper outliers in it. It was crucial to use an imputer to handle these values. We used the imputer implementation in here and edited it slightly:
# 
# https://github.com/analokmaus/kuma_utils/blob/master/preprocessing/imputer.py
# 
# Thanks [@analokamus](https://www.kaggle.com/analokamus) for the neat implementation!
# 
# But to reduce notebook runtime, this notebook will use pre-imputed dataframes. You can also run your own tests with them.
# 
# You can access the imputed data: https://www.kaggle.com/datasets/nlztrk/imputed-dataset-enerjisa-retim

# In[ ]:


IMPUTE = False


# In[ ]:


def analyze_column(input_series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(input_series):
        return 'numeric'
    else:
        return 'categorical'
    


class LGBMImputer:
    '''
    Regression imputer using LightGBM
    '''

    def __init__(self, cat_features=[], n_iter=15000, verbose=False):
        self.n_iter = n_iter
        self.cat_features = cat_features
        self.verbose = verbose
        self.n_features = None
        self.feature_names = None
        self.feature_with_missing = None
        self.imputers = {}
        self.offsets = {}
        self.objectives = {}
        
    def fit_transform(self, X, y=None):
        output_X = X.copy()
        self.n_features = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(self.n_features)]
            X = pd.DataFrame(X, columns=self.feature_names)
        self.feature_with_missing = [col for col in self.feature_names if X[col].isnull().sum() > 0]

        for icol, col in enumerate(self.feature_with_missing):
            if icol in self.cat_features:
                nuni = X[col].dropna().nunique()
                if nuni == 2:
                    params = {
                        'objective': 'binary'
                    }
                elif nuni > 2:
                    params = {
                        'objective': 'multiclass',
                        'num_class': nuni + 1
                    }
            else: # automatic analyze column
                if analyze_column(X[col]) == 'numeric':
                    params = {
                        'objective': 'regression'
                    }
                else:
                    nuni = X[col].dropna().nunique()
                    if nuni == 2:
                        params = {
                            'objective': 'binary'
                        }
                    elif nuni > 2:
                        params = {
                            'objective': 'multiclass',
                            'num_class': nuni + 1
                        }
                    else:
                        print(f'column {col} has only one unique value.')
                        continue
          
            params['verbosity'] = -1
            
            null_idx = X[col].isnull()
            x_train = X.loc[~null_idx].drop(col, axis=1)
            x_test = X.loc[null_idx].drop(col, axis=1)
            y_offset = X[col].min()
            y_train = X.loc[~null_idx, col].astype(int) - y_offset
            dtrain = lgb.Dataset(
                data=x_train,
                label=y_train
            )

            early_stopping_rounds = 50
            model = lgb.train(
                params, dtrain, valid_sets=[dtrain], 
                num_boost_round=self.n_iter,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )

            y_test = model.predict(x_test)
            if params['objective'] == 'multiclass':
                y_test = np.argmax(y_test, axis=1).astype(float)
            elif params['objective'] == 'binary':
                y_test = (y_test > 0.5).astype(float)
            y_test += y_offset
            output_X.loc[null_idx, col] = y_test
            if params['objective'] in ['multiclass', 'binary']:
                output_X[col] = output_X[col].astype(int)
            self.imputers[col] = model
            self.offsets[col] = y_offset
            self.objectives[col] = params['objective']
            if self.verbose:
                print(f'{col}:\t{self.objectives[col]}...iter{model.best_iteration}/{self.n_iter}')
        
        return output_X    


# In[ ]:


if IMPUTE:
    FEATURE_CSV_PATH = "../input/enerjisa-uretim-hackathon/features.csv"
    POWER_CSV_PATH = "../input/enerjisa-uretim-hackathon/power.csv"

    feature_df = pd.read_csv(FEATURE_CSV_PATH)
    power_df = pd.read_csv(POWER_CSV_PATH)
    feature_df = feature_df.merge(power_df, how="left", on="Timestamp")
    
    feature_df = feature_df.replace(99999.0,np.nan)

    imputer = LGBMImputer(verbose=True)
    feature_df.loc[:, feature_df.columns[1:-1]] = imputer.fit_transform(feature_df[feature_df.columns[1:-1]])

    feature_df["month"] = pd.to_datetime(feature_df["Timestamp"]).dt.month
    feature_df["year"] = pd.to_datetime(feature_df["Timestamp"]).dt.year
    feature_df["hour"] = pd.to_datetime(feature_df["Timestamp"]).dt.hour
    feature_df["week"] = pd.to_datetime(feature_df["Timestamp"]).dt.week

    test_mask = pd.to_datetime(feature_df["Timestamp"]) > "2021-08-14 23:50:00"

    train_df = feature_df[~test_mask].copy().reset_index(drop=True)
    test_df = feature_df[test_mask].copy().reset_index(drop=True)
    
else:
    TRAIN_CSV_PATH = "../input/imputed-dataset-enerjisa-retim/train_imputed.csv"
    TEST_CSV_PATH = "../input/imputed-dataset-enerjisa-retim/test_imputed.csv"
    SUBM_CSV_PATH = "../input/enerjisa-uretim-hackathon/sample_submission.csv"

    train_df = pd.read_csv(TRAIN_CSV_PATH).set_index("Timestamp")
    test_df = pd.read_csv(TEST_CSV_PATH).set_index("Timestamp")
    subm_df = pd.read_csv(SUBM_CSV_PATH).drop(columns="Power(kW)", axis=1)


# ### Creating training and test dataframes

# In[ ]:


label = "Power(kW)"
except_cols = ["year", "month", "week", "hour"]
X_train = train_df.drop(labels = except_cols + [label], axis=1)
y_train = train_df[label]


# ### Creating CV splits
# We are being evaluated with RMSE in this competition. So how far our predicted values are from the ground truths are crucial. We need to keep that in mind while executing our cross-validations. Let's look at the distribution of our label.

# In[ ]:


plt.figure(figsize=(12,6))
_ = sns.histplot(data=y_train, kde=True)


# We can see that most values are clustered at two opposite ends of the distribution. It may be advantageous to maintain this distribution in all our folds.

# In[ ]:


def create_cont_folds(df, n_s=8, n_grp=1000):
    
    skf = StratifiedKFold(n_splits=n_s, shuffle=True, random_state=1337)
    grp = pd.cut(df, n_grp, labels=False)
    target = grp
    
    fold_nums = np.zeros(len(df))
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        fold_nums[v] = fold_no
        
    return fold_nums


# In[ ]:


num_of_folds = 8
num_of_groups = 40

foldnums = create_cont_folds(y_train, n_s=num_of_folds, n_grp=num_of_groups)
cv_splits = []

for i in range(num_of_folds):
    test_indices = np.argwhere(foldnums==i).flatten()
    train_indices = list(set(range(len(y_train))) - set(test_indices))
    cv_splits.append((train_indices, test_indices))


# In[ ]:


PLOT_FOLD_NUM = 2

fig, axs = plt.subplots(2, figsize=(12,10))
sns.histplot(data=y_train.iloc[cv_splits[PLOT_FOLD_NUM][0]], kde=True, ax=axs[0]).set(title='Train Set Distribution')
sns.histplot(data=y_train.iloc[cv_splits[PLOT_FOLD_NUM][1]], kde=True, ax=axs[1]).set(title='Validation Set Distribution')
plt.show()


# It seems like we are preserving the label distribution over all folds! You can also try different fold IDs to validate that behaviour. Doing this didn't improved our CV or Kaggle scores, but it definitely is a more robust CV splitting method.

# ### Level 1: Prediction with LGBM & CatBoost
# We ran predictions for the Kaggle test set with each CV models separately.

# In[ ]:


cv_models = []
rms_errs = []

cat_preds = []
lgb_preds = []

cat_final_preds = []
lgb_final_preds = []

lgb_params = {
    "learning_rate": 0.02,
    "num_leaves": 64,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "verbosity": -1,
    "n_estimators": 7000,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "objective": "regression",
    "metric": "rmse",
}

for split_train, split_val in tqdm(cv_splits):
    split_train = X_train.index[split_train]
    split_val = X_train.index[split_val]

    model1 = CatBoostRegressor(
        iterations=5000,
        random_state=42,
        early_stopping_rounds=50,
    )

    model2 = lgb.LGBMRegressor(**lgb_params)
 
    train_x, train_y = X_train.loc[split_train], y_train.loc[split_train]
    test_x, test_y = X_train.loc[split_val], y_train.loc[split_val]

    model1.fit(
        train_x,
        train_y,
        eval_set=(test_x, test_y),
        verbose=500,
    )
    model2.fit(
        train_x,
        train_y,
        eval_set=(test_x, test_y),
        verbose=500,
    )

    preds = (model1.predict(test_x) + model2.predict(test_x)) / 2
    rms = mean_squared_error(test_y, preds, squared=False)
    rms_errs.append(rms)

    cat_preds.append(model1.predict(test_x))
    lgb_preds.append(model2.predict(test_x))
    
    cat_final_preds.append(
        model1.predict((test_df.drop(labels=except_cols + [label], axis=1)))
    )
    
    lgb_final_preds.append(
        model2.predict((test_df.drop(labels=except_cols + [label], axis=1)))
    )

    print("RMSE:", rms)


# We stored the prediction of the models for both CV folds and the Kaggle test sets for the next step.

# In[ ]:


cat_final_preds = [pd.DataFrame(cat_final_preds).mean(axis=0).values]
lgb_final_preds = [pd.DataFrame(lgb_final_preds).mean(axis=0).values]


# In[ ]:


two_stage_feats = (
    pd.DataFrame(
        {"cat_pred": np.concatenate(cat_preds), "lgb_pred": np.concatenate(lgb_preds)},
        index=pd.concat(
            [
                pd.Series(X_train.loc[X_train.index[test_indices]].index)
                for (_, test_indices) in cv_splits
            ]
        ),
    )
    .reset_index()
    .set_index("Timestamp")
)

two_stage_preds = (
    pd.DataFrame(
        {
            "cat_pred": np.concatenate(cat_final_preds),
            "lgb_pred": np.concatenate(lgb_final_preds),
        },
        index=(test_df.drop(labels=except_cols + [label], axis=1)).index,
    )
    .reset_index()
    .set_index("Timestamp")
)


# ### Level 2: Stacking with LGBM & CatBoost
# We also used stacking and used LGBM and CatBoost as meta-regressors. We used the equally blended predictions of them as our final predictions. We have seen that this approach improved both our CV and Kaggle score. ***(Public 19.46 -> 18.33)***

# In[ ]:


rms_errs = []

final_preds = []
cat_models = []
lgb_models = []

lgb_params = {
    "learning_rate": 0.02,
    "num_leaves": 64,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "verbosity": -1,
    "n_estimators": 7000,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "objective": "regression",
    "metric": "rmse",
}


for split_train, split_val in tqdm(cv_splits):
    split_train = X_train.index[split_train]
    split_val = X_train.index[split_val]

    model1 = CatBoostRegressor(
        iterations=5000,
        random_state=42,
        early_stopping_rounds=50,
    )

    model2 = lgb.LGBMRegressor(**lgb_params)

    train_x, train_y = (
        pd.concat([X_train.loc[split_train], two_stage_feats.loc[split_train]], axis=1),
        y_train.loc[split_train],
    )
    test_x, test_y = (
        pd.concat([X_train.loc[split_val], two_stage_feats.loc[split_val]], axis=1),
        y_train.loc[split_val],
    )

    model1.fit(
        train_x,
        train_y,
        eval_set=(test_x, test_y),
        verbose=500,
    )
    
    model2.fit(
        train_x,
        train_y,
        eval_set=(test_x, test_y),
        verbose=500,
    )

    preds = (model1.predict(test_x) + model2.predict(test_x)) / 2
    rms = mean_squared_error(test_y, preds, squared=False)
    rms_errs.append(rms)

    final_preds.append(
        (
            model1.predict(
                pd.concat(
                    [
                        (test_df.drop(labels=except_cols + [label], axis=1)),
                        two_stage_preds,
                    ],
                    axis=1,
                )
            )
            + model2.predict(
                pd.concat(
                    [
                        (test_df.drop(labels=except_cols + [label], axis=1)),
                        two_stage_preds,
                    ],
                    axis=1,
                )
            )
        )
        / 2
    )

    print("RMSE:", rms)


# In[ ]:


print("Current Submission CV:", rms_errs)
print("Current Submission CV Mean:", np.array(rms_errs).mean())
print("Current Submission CV Std:", np.array(rms_errs).std())


# ### Creating the submission file

# In[ ]:


pd.DataFrame(final_preds, columns=test_df.index).mean(axis=0).to_frame(
    "Power(kW)"
).reset_index().to_csv("submission.csv", index=False)

