#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

import gc
from joblib import dump

import warnings
warnings.filterwarnings("ignore")


# # Config

# In[ ]:


class CONFIG:
    use_lb = False
    kaggle = True
    kaggle_path = "../input/jpx-tokyo-stock-exchange-prediction/"
    local_path = ""
    random_seed = 69420


# # Preprocessing

# In[ ]:


get_ipython().run_cell_magic('time', '', 'if CONFIG.use_lb:\n    if CONFIG.kaggle:\n        prices = pd.concat([\n            pd.read_csv(CONFIG.kaggle_path+"train_files/stock_prices.csv"),\n            pd.read_csv(CONFIG.kaggle_path+"supplemental_files/stock_prices.csv")\n        ])\n    else:\n        prices = pd.concat([\n            pd.read_csv(CONFIG.local_path+"train_files/stock_prices.csv", engine="pyarrow"),\n            pd.read_csv(CONFIG.local_path+"supplemental_files/stock_prices.csv", engine="pyarrow")\n        ])\nelse:\n    if CONFIG.kaggle:\n        prices = pd.concat([\n            pd.read_csv(CONFIG.kaggle_path+"train_files/stock_prices.csv"),\n        ])\n    else:\n        prices = pd.concat([\n            pd.read_csv(CONFIG.local_path+"train_files/stock_prices.csv", engine="pyarrow"),\n        ])')


# In[ ]:


prices


# In[ ]:


from decimal import ROUND_HALF_UP, Decimal
def adjust_price(price):
    """
    We will generate AdjustedClose using AdjustmentFactor value. 
    This should reduce historical price gap caused by split/reverse-split.
    
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        
        # generate AdjustedClose
        df.loc[:, "Close"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["Close"] == 0, "Close"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "Close"] = df.loc[:, "Close"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)

    price.set_index("Date", inplace=True)
    return price


# In[ ]:


get_ipython().run_cell_magic('time', '', 'prices = adjust_price(prices)')


# In[ ]:


prices.head()


# In[ ]:


prices = prices.drop(
    [
        "ExpectedDividend", "RowId", "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag", "CumulativeAdjustmentFactor",
    ],
    axis=1
)


# In[ ]:


target = prices.pop("Target")


# In[ ]:


prices.head()


# In[ ]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(10)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# In[ ]:


prices = prices.fillna(method='pad')


# In[ ]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(10)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# # Feature Engineering

# In[ ]:


single_stonk = prices[prices['SecuritiesCode'] == 1301]


# In[ ]:


single_stonk['Close'].plot()


# In[ ]:


def feature_engineer(df):
    df['feature-avg_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    df['feature-median_price'] = df[['Open', 'High', 'Low', 'Close']].median(axis=1)
    
    df['feature-median/avg'] = df['feature-median_price'] / df['feature-avg_price']
    df['feature-median-avg'] = df['feature-median_price'] - df['feature-avg_price']
    
    df['feature-high/close'] = df['High'] / df['Close']
    df['feature-low/close'] = df['Low'] / df['Close']
    df['feature-open/close'] = df['Open'] / df['Close']

    df['feature-high/volume'] = df['High'] / df['Volume']
    df['feature-low/volume'] = df['High'] / df['Volume']
    df['feature-open/volume'] = df['Open'] / df['Volume']
    df['feature-close/volume'] = df['Close'] / df['Volume']

    df = df.replace([np.inf, -np.inf], 0)

    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "prices = prices.groupby('SecuritiesCode').apply(feature_engineer)\nprices")


# In[ ]:


prices.shape


# In[ ]:


prices['Target'] = target


# In[ ]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(10)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# In[ ]:


prices = prices.dropna()


# In[ ]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(10)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# In[ ]:


prices.head()


# In[ ]:


features = prices.columns.drop('Target').to_list()
features


# In[ ]:


dump(features, "features.joblib")


# ## Cross Validation Split

# In[ ]:


import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold

class GroupTimeSeriesSplit(_BaseKFold):
    """
    Time Series cross-validator for a variable number of observations within the time 
    unit. In the kth split, it returns first k folds as train set and the (k+1)th fold 
    as test set. Indices can be grouped so that they enter the CV fold together.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    """
    def __init__(self, n_splits=5, *, max_train_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is 
            the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into 
            train/test set.
            Most often just a time feature.

        Yields
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_splits = self.n_splits
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_folds = n_splits + 1
        indices = np.arange(n_samples)
        group_counts = np.unique(groups, return_counts=True)[1]
        groups = np.split(indices, np.cumsum(group_counts)[:-1])
        n_groups = _num_samples(groups)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of groups: {1}.").format(n_folds, n_groups))
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        for test_start in test_starts:
            if self.max_train_size:
                train_start = np.searchsorted(
                    np.cumsum(
                        group_counts[:test_start][::-1])[::-1] < self.max_train_size + 1, 
                        True)
                yield (np.concatenate(groups[train_start:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))
            else:
                yield (np.concatenate(groups[:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))


# In[ ]:


def setup_cv(df, splits=5):
    df['fold'] = -1

    kf = GroupTimeSeriesSplit(n_splits=splits)
    for f, (t_, v_) in enumerate(kf.split(X=df, groups=df.index)):
        df.iloc[v_, -1] = f
        
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'prices = setup_cv(prices)')


# In[ ]:


prices['fold'].value_counts()


# In[ ]:


# prices['fold'].plot()


# In[ ]:


prices.head()


# ## Data check

# In[ ]:


prices[np.isinf(prices.values)]


# In[ ]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(10)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# # Train Models

# In[ ]:


def set_rank(df):
    """
    Args:
        df (pd.DataFrame): including predict column
    Returns:
        df (pd.DataFrame): df with Rank
    """
    # sort records to set Rank
    df = df.sort_values("predict", ascending=False)
    # set Rank starting from 0
    df.loc[:, "Rank"] = np.arange(len(df["predict"]))
    return df

# https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


# # LGBM

# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


def train_lgbm(prices, folds):
    models = []
    scores = []
    feature_importance = []
    
    for f in range(folds):
        print(f"{'='*25} Fold {f} {'='*25}")
        
        X_train = prices[prices.fold != f][features]
        y_train = prices[prices.fold != f][["Target"]]
        X_valid = prices[prices.fold == f][features]
        y_valid = prices[prices.fold == f][["Target"]]
        
        model = LGBMRegressor(
            objective="rmse",
            metric="rmse",
            learning_rate=0.01,
            n_estimators=50000,
            device="gpu",
            random_state=CONFIG.random_seed,
            extra_trees=True,
            # categorical_feature=[0]
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=50,
            verbose=1000
        )
        
        feature_importance.append(model.feature_importances_)
        
        oof_preds = model.predict(X_valid)
        oof_score = np.sqrt(mean_squared_error(y_valid, oof_preds))
        
        print(f"RMSE: {round(oof_score, 4)}")
        models.append(model)
        dump(model, f"lgbm_model_{f}.joblib", compress=3)
        
        result = prices[prices.fold == f]
        result.loc[:, "predict"] = oof_preds
        result.loc[:, "Target"] = y_valid
        result = result.sort_values(["Date", "predict"], ascending=[True, False])
        result = result.groupby("Date").apply(set_rank)
        
        sharpe_scores = calc_spread_return_sharpe(result, portfolio_size=200)
        scores.append(sharpe_scores)
        print('Validation sharpe = {:.4f}'.format(sharpe_scores))
        
        del X_train, y_train, X_valid, y_valid, result, model
        _ = gc.collect()

    return models, scores, feature_importance


# In[ ]:


get_ipython().run_cell_magic('time', '', 'models, scores, feature_importance = train_lgbm(prices, 5)')


# In[ ]:


print(f"Mean Score: {np.mean(scores):.6}")


# In[ ]:


mean_importance = np.mean(feature_importance, axis=0)
mean_importance = pd.DataFrame(mean_importance, index=features)


# In[ ]:


mean_importance.sort_values(by=0, ascending=True).plot(
    kind='barh',
    figsize=[10, 8],
)


# # Make Predictions & Submit

# In[ ]:


import jpx_tokyo_market_prediction
from tqdm.auto import tqdm
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (prices, options, financials, trades, secondary_prices, sample_prediction) in tqdm(iter_test):
    
    print("Adjusting Price...")
    prices = adjust_price(prices)
    print("Adding Features...")
    prices = prices.groupby('SecuritiesCode').apply(feature_engineer)    
    prices.fillna(method='pad')
    
    prices = prices[features]
    
    print("Predicting Model...")
    lgbm_preds = []
    for model in models:
        lgbm_preds.append(model.predict(prices))
        
    lgbm_preds = np.mean(lgbm_preds, axis=0)
    sample_prediction["Prediction"] = lgbm_preds
    
    print("Ranking...")
    sample_prediction = sample_prediction.sort_values(by = "Prediction", ascending=False)
    sample_prediction.Rank = np.arange(0,2000)
    sample_prediction = sample_prediction.sort_values(by = "SecuritiesCode", ascending=True)
    sample_prediction.drop(["Prediction"],axis=1)
    submission = sample_prediction[["Date","SecuritiesCode","Rank"]]
    
    display(submission)
    
    env.predict(submission)


# In[ ]:




