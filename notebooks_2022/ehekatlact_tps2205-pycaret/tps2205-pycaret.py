#!/usr/bin/env python
# coding: utf-8

# ## 初めに
# 
# Kaggle初心者の練習なので、何卒お手柔らかにお願いいたします。

# ## Setup

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install --user pycaret -full\n!pip install numba==0.53')


# In[ ]:


import pycaret.classification as cf


# In[ ]:


import numpy as np
import pandas as pd
import warnings


# In[ ]:


def log_mem(txt):
    import psutil
    mem = psutil.virtual_memory()
    print(txt, mem.percent, '%', mem.used//1024//1024, 'MB')


# In[ ]:


warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# データセット読み込み
train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv", index_col='id')
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv", index_col='id')
sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
display(train.head())
display(test.head())
display(sub.head())


# In[ ]:


# f_27が文字列、それ以外は数値
train.dtypes


# In[ ]:


# カーディナリィが高いカテゴリ変数、f_27が混ざっており、このままだとメモリ不足する。
# また、整数値のうちカテゴリ変数は29, 30のみであり、7～18は数値として扱うべきである。
train.nunique()


# In[ ]:


# f_27を、文字の出現回数等の数値として扱う
def c2i(c):
    return ord(c) - ord('A')

def calc_str_std(str):
    import string
    ret = 0
    for A2Z in string.ascii_uppercase:
        val = str.count(A2Z)
        ret += val * val
    return ret**0.5

def calc_str_max(str):
    import string
    ret = 0
    for A2Z in string.ascii_uppercase:
        val = str.count(A2Z)
        ret = max(ret, val)
    return ret

def calc_unique_str(str):
    return len(set(str))

def calc_freq_str(str):
    from collections import Counter
    cnt_of = Counter(str)
    ret = max(cnt_of.values())
    return ret

def calc_linear_regression(str):
    arr = [c2i(c) for c in str]
    y = np.array(arr)
    x = np.arange(len(y))
    result = np.polyfit(x, y, 1)
    return tuple(result.tolist())

def separete_str(str):
    ret = []
    for i in range(len(str)):
        ret.append(float(c2i(str[i])))
    return tuple(ret)

def fe(data):
    import string
    for c in string.ascii_uppercase:
        data['f_27_' + c] = data['f_27'].apply(lambda x: x.count(c)).astype(np.float16)
    data['f_27_std'] = data['f_27'].apply(calc_str_std)
    data['f_27_max'] = data['f_27'].apply(calc_str_max)
    data['f_27_unique'] = data['f_27'].apply(calc_unique_str).astype(np.float16)
    data['f_27_freq'] = data['f_27'].apply(calc_freq_str).astype(np.float16)
    # result_type='expand'で、複数列同時に編集できちゃうらしい。すごいね
    data[['f_27_linear_regression_a', 'f_27_linear_regression_b']] = data.apply(lambda x: calc_linear_regression(x['f_27']), axis=1, result_type='expand')
    column_list = []
    for i in range(10):
        column = 'f_27_pos_' + str(i)
        column_list.append(column)
    train[column_list] = train.apply(lambda x: separete_str(x['f_27']), axis=1, result_type = "expand")
    test[column_list] = test.apply(lambda x: separete_str(x['f_27']), axis=1, result_type = "expand")
    return data

train = fe(train)
test = fe(test)


# In[ ]:


# f_27は加工して既に突っ込んだので、不要
train.drop(['f_27'], axis=1, inplace=True)
test.drop(['f_27'], axis=1, inplace=True)


# In[ ]:


# int型の列は、カテゴリ変数ではなく数値として扱ったほうがよさそう
numeric_features = list(train.select_dtypes(include=['float64', 'int64']).columns)
numeric_features.remove('target')


# In[ ]:


train.shape


# In[ ]:


log_mem('bef_setup')


# PyCaret
# 
# 英語だけど・・・使い方
# 
# https://pycaret.gitbook.io/docs/
# 
# https://pycaret.readthedocs.io/en/latest/api/classification.html

# In[ ]:


# PyCaretで分析開始
# 初期化
cf1 = cf.setup(train,
                target = 'target',
                use_gpu = True,
                session_id = 123,
                numeric_features = numeric_features,  # 数値として与える整数値
                # 1. 欠損値の処理
                # imputation_type='iterative',  # 代入の方法、simple, iterative 詳細は不明
                # numeric_imputation = 'mean',  # 数値データの補完方法、mean, median
                # categorical_imputation = 'constant',  # カテゴリデータの補完方法、constant, mode
                # 2. 順序データのエンコード
                # ordinal_features = { 'column_name' : ['low', 'medium', 'high'] },
                # 3. 特徴量の正規化
                # normalize=True,  # 正規化を行うか否か
                # zscore->普通の正規化、minmax->[0, 1]の範囲でスケーリング、maxabs->平行移動させず、絶対値を1にする、robust->四分位を使い外れ値はそのまま残す
                # normalize_method = 'robust',  # zscore, minmax, maxabs, robust
                # 4. カテゴリ変数のレア値の統合
                # combine_rare_levels = True,  # 統合を行う
                # rare_level_threshold = 0.1  # 統合を行う閾値
                # 5. 数値データのビン化
                # bin_numeric_features = ['column1', 'column2', 'column3'],  # 数値データをカテゴリ変数化させる
                # 6. 外れ値の除去
                # remove_outliers = True
                # outliers_threshold = 0.05  # 外れ値を除去する閾値
                # 7. マルチコの除去
                # remove_multicollinearity = True,  # 多変量性を除外する
                # multicollinearity_threshold = 0.9
                # remove_perfect_collinearity = True,  # 完全相関を除外する
                # 8. クラスタリング結果の特徴量化
                # create_clusters = True,  # 各特徴量をクラスタリングし、それを新たな特徴量として追加する
                # cluster_iter = 20,
                # 9. 統計的に有意ではない分散を持つ特徴量を削除
                # ignore_low_variance = True,  # 統計的に有意ではない分散を持つ特徴量を削除
                # 10. 相互作用特徴量の生成
                # 特徴量が多いとフリーズするので注意
                # polynomial_features = True, # 多項式特徴量を作成する
                # polynomial_degree = 2,  # 多項式の次数
                # trigonometry_features = True,  # 三角関数特徴量を作成する
                # polynomial_threshold = 0.1,  # 多項式特徴量を作成する閾値　低いほど少ない
                # feature_interaction = False,  # パラメータに、変数間の積を追加する
                # feature_ratio = False,  # パラメータに、変数間の比率を追加する
                # interaction_threshold = 0.0001,  # 積や比率を追加する閾値 default:0.01
                # 11. グループ特徴量の生成
                # TODO
                # 12. 特徴量選択
                feature_selection= True, # 特徴量選択を行う。パラメータをかさましした時はTrue推奨
                # feature_selection_threshold = 0.7,  # 特徴量選択後に使われるパラメータ数、大きいほどたくさん残る default 0.8
                # 13. 高カーディナリィ特徴量の削減
                # 種類数が多すぎるカテゴリ変数をきれいにする
                # high_cardinality_features = ['column1', 'column2', 'column3'],
                # high_cardinality_method = 'frequency',  # frequency, clustering
                # 14. 特徴量スケーリング
                # 正規分布を仮定しして変換する。normalizeと何が違うの？
                # transformation = False,
                # transformation_method = 'yeo-johnson',  # yeo-johnson, quantile
                # 15. 目的変数のスケーリング
                # transcorm_target = True,
                # transform_target_method = 'box-cox',  # box-cox, yeo-johnson
                # 16. 特徴量の次元削減
                # pca = True,  # PCAを行い、パラメータの次元圧縮を行う
                # pca_method = 'kernel'  # 'linear', 'kernel', 'incremental'
                # pca_components = 0.99  # 残す特徴量の割合
                # 17. その他の設定
                fold = 3,
                silent = True  # 対話的な設定を無効化
                )


# In[ ]:


log_mem('aft_setup')


# In[ ]:


# SetUpで補正後のデータ
cf.get_config('X').head()


# In[ ]:


cf.get_config('X').dtypes


# PyCaretで、機械学習を行う
# 
# 使うモデルは、xgboost, catboost, lightgbm, rf
# (事前にcf.compare_modelsで評価が良かったことを確認している)
# 
# 4モデルを作成した後、stack_model(スタッキング)->tune_model(ハイパーパラメータ調整)->calibrate_model(較正)->finalize_model(全データを対象に学習)->predict_model(testデータを対象に予測)する。
# 
# 10行以下で出来て素敵

# In[ ]:


# model_list = cf.compare_models(n_select = 4)
model_list = [cf.create_model(model_name) for model_name in ['xgboost', 'catboost', 'lightgbm', 'rf']]
log_mem('aft_model_list')


# In[ ]:


stacker = cf.stack_models(model_list, meta_model=model_list[0])
log_mem('aft_stack')


# In[ ]:


# tuned_model = cf.tune_model(stacker)
tuned_model = stacker  # あんまり差がなさそうだから一旦飛ばす


# In[ ]:


# calib_model = cf.calibrate_model(tuned_model)
calib_model = tuned_model  # あんまり差がなさそうだから一旦飛ばす


# In[ ]:


log_mem('bef_finalize')
final_model = cf.finalize_model(calib_model)
log_mem('aft_finalize')


# In[ ]:


# 本コンペで必要なのはtargetが1になる確率なので、raw_score=Trueを設定し確率を出力する。
pred =cf.predict_model(final_model, data=test, raw_score=True)
pred.head()
log_mem('aft_pred')


# In[ ]:


# 提出用csvの作成
sub['target'] = pred.reset_index()['Score_1']
sub.to_csv('submission.csv', index=False)
sub

