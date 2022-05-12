#!/usr/bin/env python
# coding: utf-8

# 公開されているコードに、疑似ラベル（Pseudo-Label）を実施。
# 
# 猫ブーストを追加し、平均値を提出。

# In[ ]:


# 各種ライブラリをimportする
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor
from tqdm.notebook import tqdm


# In[ ]:


# データセットをDataFrameとして読み込む
df_train = pd.read_csv('../input/data-science-osaka-spring-2022/train.csv', index_col=0)
df_test = pd.read_csv('../input/data-science-osaka-spring-2022/test.csv', index_col=0)

# BaseLineModelのsubmission.csvを読み込む
predict = pd.read_csv('../input/datascienceosakaspring2022/submission.csv', index_col=0)


# In[ ]:


# 結合する
df_train.shape
df_train_ = pd.merge(df_test, predict, on='ID')
df_train = pd.concat([df_train, df_train_], axis=0)


# In[ ]:


# 結合できているか確認
df_train.shape,df_train_.shape


# In[ ]:


# T,Ikeda様Discussion投稿より
df_train = df_train[df_train['year'] < 2022]
df_train = df_train[df_train['year'] > 1990]
df_train = df_train[df_train['fuelType'] != 'Electric']
df_train = df_train[df_train['price'] < 80000]
df_train = df_train[df_train['engineSize'] <= 4.5]
df_train = df_train[df_train['engineSize'] >= 1]


# In[ ]:


df_train


# In[ ]:


df_size=pd.read_csv('../input/car-dimensions/car dimensions.csv')
df_info=pd.read_csv('../input/wikipedia-info/Wikipedia_info/wikipedia_info.csv')
df_google=pd.read_csv('../input/google-info/google_info_.csv')

# df['model'] = df['model'].str.strip()
# df_size['model'] = df_size['model'].str.strip()


# In[ ]:


# df_size.info()


# In[ ]:


# GroupKFoldを後でしたいので、控えておく
groups = df_train.manufacturer.values
groups


# In[ ]:


# 一括処理したいので、学習データとテストデータを縦に結合する
df = pd.concat([df_train, df_test])
df['model'] = df['model'].str.strip()


# In[ ]:


# df_ = pd.merge(df,df_size,how='left',on='model')


# In[ ]:


# df_


# In[ ]:


# df_.isnull().sum()


# In[ ]:


# df=df_
df['CAR_NAME']=df['manufacturer']+df['model']


# In[ ]:


# df_f = pd.read_json("../input/wikipedia-info/Wikipedia_info/1.json")

# print(df_f)
#       col1 col2
# row1     1    a
# row2     2    x
# row3     3    あ


# In[ ]:


df['year_str']= df['year'].astype(str)
# df['model_year']=df['model']+df['year_str']
df['fuelType_year']=df['fuelType']+df['year_str']
df['transmission_year']=df['transmission']+df['year_str']
df['transmission_fuelType']=df['transmission']+df['fuelType']


# In[ ]:


# Count Encoding
features = ['year', 'transmission', 'fuelType', 'mpg' ,'engineSize', 'model',  'manufacturer','fuelType_year','transmission_year','transmission_fuelType']

for f in features:
    summary = df[f].value_counts()
    df['%s_count'%f] = df[f].map(summary)


# In[ ]:


# 平均年数を追加
df["model_year"] = df.groupby(['model']).year.mean()


# In[ ]:


#中古車の古さ（2022年から見て）をカラム化
df['oldness'] = 2022 - df['year']
#1年当りの走行距離をカラム化
#車の酷使の度合い？がわかる？
df['overuse'] = df['mileage'] / df['oldness']


# In[ ]:


# ここでデータフレーム(df)を確認
display(df.head())


# ## ここでダミー変数を組み込む
# ### 今回のデータの場合は、「transmission」「fuelType」をダミー変数にした

# In[ ]:


# 二値化(列のカテゴリ⇒数値)する
# 対象説明変数のリストを作成する 
dummy_list = ['transmission','fuelType']

# 二値化
df_dummy = pd.get_dummies(df[dummy_list], drop_first = True) #多重共線性の回避のためにdrop_first

# 元のデータフレームと二値化した変数を結合する
df1 = pd.concat([df, df_dummy], axis = 1)

# 元々あった列【二値化されてない列】を削除
df = df1.drop(['transmission','fuelType'] , axis=1)


# In[ ]:


# ここでデータフレーム(df)を確認
display(df.head())

#[fuelType_Electric]のような列ができていることを確認できる


# In[ ]:


# 処理が終わったので再度分割する
df_train = df.iloc[:len(df_train)]
df_test = df.iloc[len(df_train):]


# In[ ]:


df.columns.values


# In[ ]:


df.info()


# In[ ]:


# 不要なカラムを除く
drop_col = ['model',  'manufacturer','year_str','CAR_NAME','fuelType_year','transmission_year','transmission_fuelType','model_year']


df.drop(drop_col, axis=1, inplace=True)
df_train = df.iloc[:len(df_train)]
df_test = df.iloc[len(df_train):]


# In[ ]:


df.info()


# In[ ]:


# featureとtargetを分離する
y_train = df_train.price
X_train = df_train.drop(['price'], axis=1)
X_test = df_test.drop(['price'], axis=1)


# In[ ]:


X_train


# In[ ]:


# 交差検定で精度を見積もると同時にCV Averagingでテストデータに対する予測値を生成する
n_splits=5
gkf = GroupKFold(n_splits=n_splits)
scores = []
y_pred_test = np.zeros(len(df_test))
sizes = []
df_imp = DataFrame(index=X_train.columns, columns=['fold_%d'%j for j in range(n_splits)])

for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):
    
    X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]
    X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]
    
    print('Train Groups', np.unique(groups_train_))
    print('Val Groups', np.unique(groups_val))
    sizes.append(len(X_val))
    
    model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bytree=0.7,
                                             gamma=0, gpu_id=-1, importance_type='gain', learning_rate=0.1,
                                             max_depth=8, min_child_weight=10, 
                                             n_estimators=999, n_jobs=16,
                                             random_state=71, reg_alpha=0,
                                             reg_lambda=1, subsample=1.0, tree_method='exact', num_parallel_tree=1)
    
    model.fit(X_train_, y_train_, early_stopping_rounds=10, eval_metric='rmse', eval_set=[(X_val, y_val)])
    y_pred = model.predict(X_val)
    y_pred_test += model.predict(X_test)/n_splits
    score = mean_squared_log_error(y_val, y_pred)**0.5
    scores.append(score)
    
    r = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=71)
    df_imp['fold_%d'%i] = r['importances_mean']#model.feature_importances_
    
    print('CV Score of Fold_%d is %f' % (i, score))
    print('\n')


# In[ ]:


# CVスコアを確認する
print('CV Score: %1.3f　±　%1.3f'%(np.mean(scores), np.std(scores)))


# In[ ]:


# catboost追加
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool


# In[ ]:


#Catboostに渡すカテゴリ変数を定義
categorical_features = ['year', 'mileage', 'tax', 
       'year_count', 'transmission_count', 'fuelType_count',
       'mpg_count', 'engineSize_count', 'model_count',
       'manufacturer_count', 'fuelType_year_count',
       'transmission_year_count', 'transmission_fuelType_count',
       'oldness']


# In[ ]:


# 交差検定で精度を見積もると同時にCV Averagingでテストデータに対する予測値を生成する
n_splits=5
gkf = GroupKFold(n_splits=n_splits)
scores2 = []
y_pred_test2 = np.zeros(len(df_test))
sizes = []
df_imp = DataFrame(index=X_train.columns, columns=['fold_%d'%j for j in range(n_splits)])

for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):
    
    X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]
    X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]
    
    print('Train Groups', np.unique(groups_train_))
    print('Val Groups', np.unique(groups_val))
    sizes.append(len(X_val))
    
    
    train_pool = Pool(X_train_, label=y_train_,cat_features=categorical_features)  
    test_pool = Pool(X_val, y_val,cat_features=categorical_features)
    
    
    params = {
    'depth' : 6,                  # 木の深さ
    'learning_rate' : 0.16,       # 学習率
    'early_stopping_rounds' : 10,
    'iterations' : 100, 
    'random_seed' :42
    }

    model2 = CatBoost(params)
    model2.fit(train_pool, eval_set=[test_pool])
   
    y_pred2 = model2.predict(test_pool)
    
    
    y_pred_test2 += model2.predict(X_test)/n_splits
    score2 = mean_squared_log_error(y_val, y_pred2)**0.5
    scores2.append(score2)
        
    print('CV Score of Fold_%d is %f' % (i, score2))
    print('\n')


# In[ ]:


# CVスコアを確認する
print('CV Score: %1.3f　±　%1.3f'%(np.mean(scores2), np.std(scores2)))


# In[ ]:


# 提出用ファイルを作成する
df_sub = pd.read_csv('../input/data-science-osaka-spring-2022/sample_submission.csv', index_col=0)

# ２つのモデルの平均を使用する
df_sub['price'] = y_pred_test * 0.5 + y_pred_test2 * 0.5
df_sub.to_csv('submission.csv')


# In[ ]:


df_sub


# In[ ]:




