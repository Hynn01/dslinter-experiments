#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import sys

import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


os.listdir('/kaggle/input/')


# # 認識資料

# In[ ]:


house_data = pd.read_csv('../input/california-housing-prices/housing.csv')
house_data.head()


# # 資料的型別，會發現，除了ocean_proximity 是 object，都是 float，ocean_proximity 有可能是分類屬性

# In[ ]:


house_data.info()


# # ocean_proximity 的分類與總數
# ## 靠海、內陸、住海邊、住小島上

# In[ ]:


house_data["ocean_proximity"].value_counts()


# # 各屬性的摘要

# In[ ]:


house_data.describe()


# # 將各個之視覺化，x軸為各個屬性(longtitude)，y軸為數量(count)
# ### median_income 為 萬/美金 為單位
# ### 大部分的直方圖都有厚重的尾部，這會造成模型學習時，無法找到模式Pattern，因此要將這些屬性轉換，來產生偏向常態分布的直方圖。

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
house_data.hist(bins=50, figsize=(20,15))
plt.show()


# # 建立訓練和測試集

# In[ ]:


# 設定亂數種子
np.random.seed(42)


# In[ ]:


import numpy as np 

def spilt_train_test(data, test_ratio):
    # 基於亂數種子以及data的長度而產生相應的隨機排序
    shuffled_indices = np.random.permutation(len(data))
    # 根據比例將資料集切割
    test_set_size = int(len(data) * test_ratio)
    # 0 ~ test_set_size 
    test_indices = shuffled_indices[:test_set_size]
    # test_set_size ~ final
    train_indices = shuffled_indices[test_set_size:]
    # 回傳切割完的資料，train、test
    return data.iloc[train_indices], data.iloc[test_indices]


# # train 資料集的總數

# In[ ]:


train_set, test_set = spilt_train_test(house_data, 0.2)
len(train_set)


# # test 資料集的總數

# In[ ]:


len(test_set)


# ## 保證每次取得更新過的資料，都不曾出現在訓練集中，使用hash當作識別碼進行驗證，當各個識別碼小於或等於最大值雜湊值的20%時，才將之放入測試集。

# In[ ]:


from zlib import crc32


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[ ]:


import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# ## 由於原先沒有識別碼，因此將 id 設為識別碼

# In[ ]:


housing_with_id = house_data.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# ## 也可將座標設為識別碼，因其在地理資訊系統的不變性

# In[ ]:


# housing_with_id["id"] = house_data["longitude"] * 1000 + house_data["latitude"]
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[ ]:


test_set.head()


# In[ ]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(house_data, test_size=0.2, random_state=42)


# ## 分層抽樣分析

# In[ ]:


test_set.head()


# In[ ]:


house_data["median_income"].hist()


# ## 確保所有的測試集能平均地分布各個種類收入的中位數，因此制定制定各個種類收入的分層
# ### 1 = 0. ~ 1.5 
# ### 2 = 1.5 ~ 3.0 
# ### 3 = 3.0 ~ 4.5 
# ### 4 = 4.5 ~ 6.0 

# In[ ]:


# 以 median_income 為底，建立income_cat
house_data["income_cat"] = pd.cut(house_data["median_income"],
                                 bins=[0.,1.5,3.0,4.5,6.,np.inf],
                                 labels=[1,2,3,4,5])


# ## 分層後的結果

# In[ ]:


house_data["income_cat"].value_counts()


# ## 透過直方圖，會發現到與之前相比，現在的 income 更趨近常態分佈。

# In[ ]:


house_data["income_cat"].hist()


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(house_data, house_data["income_cat"]):
    strat_train_set = house_data.loc[train_index]
    strat_test_set = house_data.loc[test_index]


# ## 測試集的收入中位數的比例

# In[ ]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# ## 整體的收入中位數的比例

# In[ ]:


house_data["income_cat"].value_counts() / len(house_data)


# ## 代表分層抽樣成功

# ## 比較分層抽樣與存隨機抽樣的抽樣偏差(bias)

# In[ ]:


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(house_data, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(house_data),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100


# In[ ]:


compare_props


# ## 分層完後，再將 income_cat 欄位刪除，回到最初的資料欄位

# In[ ]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# ## 視覺化

# In[ ]:


house_dataa = strat_train_set.copy()


# ## 將地理資料視覺化

# In[ ]:


house_data.plot(kind="scatter", x="longitude", y="latitude")


# ## 在將 alpha = 0.1 時，呈現高密度資料的地方。

# In[ ]:


house_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# ## 想要更清楚表達高密度人口的地方，也可這樣呈現

# In[ ]:


house_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
               s=house_data["population"]/100, label="population", figsize=(10,7),
               c="median_house_value",cmap=plt.get_cmap("jet"), colorbar=True,
               sharex=False)
plt.legend()


# ## 使用 kmeans 呈現地理座標，變數為 座標 與 收入中位數

# In[ ]:


import seaborn as sns
from sklearn.cluster import KMeans

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

X = house_data.loc[:, ["median_income", "latitude", "longitude"]]
X.head()


# 

# In[ ]:


kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()


# # 會發現也可呈現加州的地區與收入的分群

# In[ ]:


sns.relplot(
    x="longitude", y="latitude", hue="Cluster", data=X, height=6,
);


# ## 每個群的收入比例

# In[ ]:


X["median_house_value"] = house_data["median_house_value"]
sns.catplot(x="median_house_value", y="Cluster", data=X, kind="boxen", height=6);


# ## 尋找相關性 

# In[ ]:


corr_matrix = house_data.corr()


# ## 會發現到前四個的對於價格有高度的相關性

# In[ ]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[ ]:


from pandas.plotting import scatter_matrix
attributes= ["median_house_value","median_income","total_rooms",
           "housing_median_age"]
scatter_matrix(house_data[attributes], figsize=(12,8))


# ## 會發現收入中位數的上限為 500000

# In[ ]:


house_data.plot(kind="scatter", x="median_income", y="median_house_value",
               alpha=0.1)


# ## 建立新的屬性，檢查更為細緻關聯性

# In[ ]:


house_data["rooms_per_household"] = house_data["total_rooms"]/house_data["households"]

house_data["bedroom_per_household"] = house_data["total_bedrooms"]/house_data["total_rooms"]

house_data["population_per_household"] = house_data["population"]/house_data["households"]


# In[ ]:


corr_matrix = house_data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[ ]:


house_data.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()


# In[ ]:


house_data.describe()


# In[ ]:


housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


# # 清理資料

# ## 缺值 NaN 集中於 total_bedrooms

# In[ ]:


sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[ ]:


sample_incomplete_rows


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[ ]:


housing_num = housing.drop("ocean_proximity", axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])


# In[ ]:


imputer.fit(housing_num)


# In[ ]:


imputer.statistics_


# In[ ]:


housing_num.median().values


# In[ ]:


X = imputer.transform(housing_num)


# In[ ]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)


# ## 將缺值填補為其欄位的中位數

# In[ ]:


housing_tr.loc[sample_incomplete_rows.index.values]


# In[ ]:


imputer.strategy


# In[ ]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)


# In[ ]:


housing_tr.head()


# ## 處理文件與分類屬性

# In[ ]:


housing_cat = house_data[["ocean_proximity"]]
housing_cat.head(10000)


# ## 將文字轉為數值，以讓所有的資料值處於數值的狀態

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# ## 找到分類的種類

# In[ ]:


ordinal_encoder.categories_


# ## 接著，再用 one-hot encoding，將分類對應的數值轉為在每一列存有 1 的陣列位置
# ## ex: NEAR-BAY = 3 = [0,0,1,0,0]

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[ ]:


housing_cat_1hot.toarray()


# ## 第二種寫法

# In[ ]:


cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[ ]:


cat_encoder.categories_


# # 自訂轉換器

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
        


# In[ ]:


col_names = "total_rooms","total_bedrooms","population","households"

rooms_ix, bedrooms_ix, popluation_ix, households_ix = [
    house_data.columns.get_loc(c) for c in col_names]


# ## 添加之前觀察的特徵屬性，"rooms_per_household", "population_per_household"

# In[ ]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# ## 建立 pipline 對於各個屬性進行轉換與 標準化，以利讓模型運算

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[ ]:


housing_num_tr


# In[ ]:


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[ ]:


housing_prepared


# In[ ]:


housing_prepared.shape


# # 訓練模型

# ## LinearRegression

# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# ## 從訓練集抽取隨機五個資料進行預測

# In[ ]:



some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("預測值:", lin_reg.predict(some_data_prepared))
print("實際值:", list(some_labels))


# ## 目前來看，預測結果和實際結果有將近10%~20%的誤差，為了更精確解釋誤差，使用 RMSE(均方根誤差) 來衡量
# * RMSE MAE MSE 的誤差越低越好!

# In[ ]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("線性回歸的RMSE的誤差值:", lin_rmse)


# ## 由於大多數地區的 median_housing_value 介於 120000 ~ 265000 之間，因此這個結果不好，可能的原因如下:
# * 訓練資料已經欠擬(underfitting)，特徵不強
# * 此模型能力不強

# ## 另一種模型: 決策樹

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# ## 訓練評估結果誤差為 0，不太可能，可能這個發生了過擬 (overfitting) 

# In[ ]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# ## 使用交叉驗證再次評估模型

# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# # 十次交驗證，決策樹預估的結果

# In[ ]:


def display_scores(scores):
    print("RMSE 十次的誤差:", scores)
    print("RMSE 十次的總誤差平均數:", scores.mean())
    print("RMSE 十次的總標準差:", scores.std())

display_scores(tree_rmse_scores)


# ## 結果而言，誤差不減反增，也不是個好模型

# ## 來看看同樣經過交叉驗證後的線性回歸模型如何?

# In[ ]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# ## 相同條件相比，線性回歸優於決策樹的結果

# ## 來看看隨機森回歸模型效果

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[ ]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[ ]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# ## 效果比前兩者好，因此將 隨機森林回歸模型作為我的模型預測，但在這之前，先微調模型

# # 微調模型
# * n_estimators：森林中估計值的個數
# * max_features ：尋找最佳分裂點時考慮的屬性數目
# * bootstrap ： 是否採用有放回樣本的方式 [True / False]

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # 3*4 的超參數的組合
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # 2*3 的組合
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# 加上交叉驗證的組合，(12+6)*5
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[ ]:


grid_search.best_params_


# In[ ]:


print("RMSE 誤差值, 超参數{ 最佳分裂點考慮的属性數目, 參數數量 } ")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:


pd.DataFrame(grid_search.cv_results_)


# In[ ]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# ## 經過暴力法下的組合發現下，各屬性影響度高低的排序

# In[ ]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# ## 會發現到，median_income、inland、population_per_hold 以及 座標 是前四名影響的指標。

# ## 預測結果

# In[ ]:


# 使用最佳參數模型
final_model = grid_search.best_estimator_

# X_test: 將我們要預測的目標屬性(median_house_value)分離。
X_test = strat_test_set.drop("median_house_value", axis=1)
# y_test: 複製目標屬性(median_house_value)
y_test = strat_test_set["median_house_value"].copy()

# 送入 pipline 進行轉換
X_test_prepared = full_pipeline.transform(X_test)
# 預測
final_predictions = final_model.predict(X_test_prepared)

# 計算 RMSE 
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) 


# ## 製作表格比較差異

# In[ ]:


true_data=pd.DataFrame(y_test.values.round(1).reshape(-1,1),columns=['真實房價中位數'])
predict_data=pd.DataFrame(final_predictions.round(1).reshape(-1,1),columns=['預測房價中位數'])


# In[ ]:


truth_vs_prediction=pd.concat([true_data,predict_data],axis=1)
truth_vs_prediction


# In[ ]:


print("最優決策森林回歸的 RMSE誤差值:",final_rmse)


# ## RMSE誤差值的95%的信賴估計

# In[ ]:


from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) -1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))


# # 真實房價中位數 和 預測房價中位數的分布

# In[ ]:


plt.figure(figsize=(12,8))
truth_vs_prediction["真實房價中位數"].hist(bins=200,alpha=0.5,label="truth_price",
         color='blue',range=(0,499999),density=True,rwidth=0.3,histtype='stepfilled')
truth_vs_prediction["預測房價中位數"].hist(bins=200,alpha=0.5,label="predict_price",
         color='red',range=(0,499999),density=True,rwidth=0.3,histtype='stepfilled')

plt.xlabel('price_range')
plt.ylabel('cumulative_distribution')
plt.legend(fontsize=17)  
plt.show()

