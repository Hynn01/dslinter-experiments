#!/usr/bin/env python
# coding: utf-8

# 結構頑張って作成したら冗長になりました．．．．  
# 最後までお付き合い頂けたらと思います．

# もうそろそ社会人2年目になるyone-motoです!
# 過去に勉強になるコンペがたくさんあるので，過去のコンペ簡単にまとめられたらと思います． 
# # ダイハツインクラスコンペ  
# ## 第3回コンペ  
# https://www.kaggle.com/c/data-science-spring-osaka-2021  
# 体に取り付けたセンサからデータを取得して，フィットネスボクシングの動きを予測するコンペ  
# + コンペの特性
#     + 時系列のセンサデータ（生のデータ）をどうフィルタリングして予想するかのコンペ
#     + 可変長のデータにどう対応するか
#     + 分類の予測であり，学習データにない分類がテストデータにある
#     + 擬似ラベル（Pseudo-Label）を追加することである程度対応できた
# + notebook 
#     + baseline  
#     https://www.kaggle.com/code/nejumi/mean-baseline  
#     エンコーダ:LabelEncoder  
#     モデル:LGBMClassifier  
#     + EDA  
#     https://www.kaggle.com/code/mugentk/eda-explanatory-data-analysis  
#     棒グラフ・データプロット  
#     https://www.kaggle.com/code/nejumi/tsfresh  
#     tsfresh（特徴量抽出）のサンプル  
# 
# ## 第4回コンペ  
# https://www.kaggle.com/competitions/data-science-summer2-osaka   
# アニメの評価レビューを予測するコンペ  
# + コンペの特性
#     + 1つのテーブルデータだけでなくuserやアニメの情報が付随するコンペ  
#     + テキストのデータが追加（テキストの処理など)  
# + notebook  
#     + baseline  
#     https://www.kaggle.com/code/nejumi/simple-baseline  
#     エンコーダ:CountEncoder  
#     モデル:XGBRegressor  
#     タグ？の処理のサンプル例(TFIDF)  
#     taoiさんnotebook
#     https://www.kaggle.com/competitions/data-science-summer2-osaka/discussion/272533  
#     
#     + EDA  
#     https://www.kaggle.com/code/nejumi/user-x-anime 
#     
#     + 1st solution  
#     https://www.kaggle.com/competitions/data-science-summer2-osaka/discussion/272591
#     テキスト処理(fasttext)
#     
#     
# 
# ## 第5回コンペ
# https://www.kaggle.com/competitions/data-science-autumn-2021  
# 住宅価格の予測  
# + コンペの特性
#     + マルチモーダルデータ（画像データやテキストデータ・テーブルデータ）
#     +  時系列データ（1ドルの変動）を考慮する必要あり
# + notebook 
#     + baseline  
#     https://www.kaggle.com/code/nejumi/simplebaseline  
#     エンコーダ:CountEncoder  
#     モデル:XGBRegressor
#     + validation(時系列データの検証データの作り方)  
#     https://www.kaggle.com/code/nejumi/simplebaseline-validation  
#     リークしないように適切な検証データを作成する  
#     + EDA  
#     https://www.kaggle.com/code/tyonemoto/dsa2021  
#     地図上でデータの可視化を行うことができる  
#     +　TEXT処理  
#     tfidf  
#     https://www.kaggle.com/code/knakanisni/tfidf-description  
#     正規表現で特定の文字列を取得
#     kaggle.com/code/t88take/description-zestimate
#     + 1st solution  
#     https://www.kaggle.com/competitions/data-science-autumn-2021/discussion/289801  
#     BERT、fasttext，word2vec，doc2vec、tfidfのモデルで検討  
#     他にも様々参考になります。  
# 
# 
# ## 第6回コンペ
# https://www.kaggle.com/competitions/data-science-winter-osaka2
# ゲームのレビューデータからその評価結果を予測するコンペ  
# + コンペの特性
#     + 基礎的なデータ
# + notebook 
#     + baseline  
#     https://www.kaggle.com/code/hattan0523/starter-data-science-winter-osaka2-withtout-gpu  
#     https://www.kaggle.com/code/t88take/dswo2-exp030  
#     https://www.kaggle.com/competitions/data-science-winter-osaka2/discussion/305827  
#     + 1st solution 
#     https://www.kaggle.com/competitions/data-science-winter-osaka2/discussion/305818
#     https://www.kaggle.com/code/ootake/final-submit-boost-1st-solution/notebook
#     + 3st solution 
#     https://www.kaggle.com/code/knakanisni/private0-77354-submit-model-3rd
#     + GM solution  
#     https://www.kaggle.com/competitions/data-science-winter-osaka2/discussion/305827

# どのbaseline(一番シンプルなコードでこのコードを参考に改善することが一般的です．).head()でまずはデータの先頭を見ているようです．

# In[ ]:


# ライブラリのimport
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib_venn import venn2
get_ipython().run_line_magic('matplotlib', 'inline')
import re


# In[ ]:


df_train = pd.read_csv("../input/data-science-osaka-spring-2022/train.csv")
df_test = pd.read_csv("../input/data-science-osaka-spring-2022/test.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# https://www.kaggle.com/code/nejumi/user-x-anime

# In[ ]:


# ID (ID) はtrainとtestには全く重なる部分が無いようですね。
f = 'ID'
plt.figure(figsize=(3,3))
plt.title(f)
venn2(subsets=(set(df_train[f].unique()), set(df_test[f].unique())),
      set_labels=('Train', 'Test'))
plt.tight_layout()


# In[ ]:


#年代と価格の相関
plt.figure(figsize=[7,7])
plt.scatter(df_train.year, df_train.price, s=3)
plt.xlabel('year')
plt.ylabel('price')
plt.show()#2060?実際のデータなので間違いが含まれる


# In[ ]:


#走行距離と価格の相関
plt.figure(figsize=[7,7])
plt.scatter(df_train.mileage, df_train.price, s=3)
plt.xlabel('mileage')
plt.ylabel('price')
plt.show()


# In[ ]:


#https://www.kaggle.com/competitions/data-science-osaka-spring-2022/discussion/313633
#上記にあるdataをくっつけてEDAする
#https://www.kaggle.com/code/takumaaiba/corporation-info-df-employees-int/notebook


# In[ ]:


#https://www.kaggle.com/datasets/mugentk/corporation-info
df_corpo = pd.read_csv("../input/corporation-info/corporation_info.csv")
df_corpo#これをくっつける


# In[ ]:


df_all = pd.concat([df_train,df_test]).reset_index(drop = True)
df_all#testの部分のprice列はNaN値で結合されている


# In[ ]:


df_all = pd.merge(df_all,df_corpo,left_on = "manufacturer",right_on = "corporation",how = "left")
df_all["Number of employees"] = df_all["Number of employees"].str.replace(',', '').astype(int)
df_all


# In[ ]:


#https://www.kaggle.com/datasets/mugentk/google-info
df_google = pd.read_csv("../input/google-info/google_info_.csv")
df_google#なんか情報が入っているぽい


# https://www.kaggle.com/code/ootake/baseline-with-google-info-2022-spring  
# おおたけさんコードを参考にする

# In[ ]:


import json
files = df_google['FILE_NAME'].values
d2={}

for i, file in enumerate(files):
    json_open = open('../input/google-info/Google_Info/%s'%file, 'r')
    json_load = json.load(json_open)
    d2[i] = pd.Series(json_load)
temp = df_google.pop("CAR_NAME")
df_google = pd.DataFrame(d2).T
df_google["CAR_NAME"] = temp


# In[ ]:


df_google["model"] = df_google["CAR_NAME"].str.lstrip('ford mercedes bmw audi hyundai toyota')


# In[ ]:


df_google = pd.concat([df_google[["model","CAR_NAME"]],pd.json_normalize(df_google["info"])],axis = 1)
df_google


# In[ ]:


df_all['model'] = df_all['model'].str.strip()


# In[ ]:


plt.figure(figsize=(3,3))
plt.title(f)
venn2(subsets=(set(df_google["model"].unique()), set(df_all["model"].unique())),
      set_labels=('External', 'all'))
plt.tight_layout()#2個カバーできていない


# In[ ]:


set(df_google["model"].unique()) ^ set(df_all["model"].unique()) 
#被っていない車種はiがついているかいないか？のよう どうやらモデル削除の部分で一部うまくいかなかったよう


# In[ ]:


df_google['model'] = df_google['model'].replace("3","i3").replace("8","i8")


# In[ ]:


plt.figure(figsize=(3,3))
plt.title(f)
venn2(subsets=(set(df_google["model"].unique()), set(df_all["model"].unique())),
      set_labels=('External', 'all'))
plt.tight_layout()#全部できた笑


# In[ ]:


df_all = pd.merge(df_all,df_google,on = "model",how = "left",suffixes = ("","google_info"))
df_all.head()


# In[ ]:


#次これをくっつける
#https://www.kaggle.com/datasets/mugentk/wikipedia-info


# In[ ]:


from pathlib import Path
import glob
WIKI_PATH = Path("../input/wikipedia-info/Wikipedia_info")


# In[ ]:


list(WIKI_PATH.glob('*.csv'))[0]#csvファイルがあった


# In[ ]:


df_wiki_info = pd.read_csv(list(WIKI_PATH.glob('*.csv'))[0])
df_wiki_info.head()


# In[ ]:


d2={}
files = df_wiki_info['FILE_NAME'].values
for i, file in enumerate(files):
    json_open = open('../input/wikipedia-info/Wikipedia_info/%s'%file, 'r')
    json_load = json.load(json_open)
    d2[i] = pd.Series(json_load)
temp = df_wiki_info["CAR_NAME"].str.lstrip('ford mercedes bmw audi hyundai toyota').replace("3","i3").replace("8","i8")
df_wiki = pd.DataFrame(d2).T
df_wiki["model"] = temp
df_wiki


# In[ ]:


plt.figure(figsize=(3,3))
plt.title(f)
venn2(subsets=(set(df_wiki["model"].unique()), set(df_all["model"].unique())),
      set_labels=('External', 'all'))
plt.tight_layout()#全部できた笑


# In[ ]:


df_all = pd.merge(df_all,df_wiki[["info","model"]],on = "model",how = "left",suffixes = ("","_wiki"))
df_all.head()


# In[ ]:


BODY_IMG_PATH = Path("../input/body-images/body_images")
list(BODY_IMG_PATH.glob('*.csv'))#csvファイルがあった


# In[ ]:


df_body_img_info = pd.read_csv(list(BODY_IMG_PATH.glob('*.csv'))[0])
df_body_img_info


# In[ ]:


#画像系はファイル名だけくっつけておく
df_body_img_info["model"] = df_body_img_info["CAR_NAME"].str.lstrip('ford mercedes bmw audi hyundai toyota').replace("3","i3").replace("8","i8")
df_body_img_info


# In[ ]:


#ひとつのモデルで複数の画像があるのでlistにして持たせておく
temp = df_body_img_info.groupby("model")["IMG_NAME"].apply(list).reset_index()
temp


# In[ ]:


plt.figure(figsize=(3,3))
plt.title(f)
venn2(subsets=(set(temp["model"].unique()), set(df_all["model"].unique())),
      set_labels=('External', 'all'))
plt.tight_layout()#全部できた笑


# In[ ]:


df_all = pd.merge(df_all,temp[["IMG_NAME","model"]],on = "model",how = "left",suffixes = ("","_body_img"))
df_all


# In[ ]:


ENG_IMG_PATH = Path("../input/engine-images/engine_images")
list(ENG_IMG_PATH.glob('*.csv'))#csvファイルがあった
df_eng_img_info = pd.read_csv(list(ENG_IMG_PATH.glob('*.csv'))[0])
df_eng_img_info


# In[ ]:


#ひとつのモデルで複数の画像があるのでlistにして持たせておく
#画像系はファイル名だけくっつけておく
df_eng_img_info["model"] = df_eng_img_info["CAR_NAME"].str.lstrip('ford mercedes bmw audi hyundai toyota').replace("3","i3").replace("8","i8")
df_eng_img_info


# In[ ]:


df_all = pd.merge(df_all,df_eng_img_info[["IMG_NAME","model"]],on = "model",how = "left",suffixes = ("","_eng_img"))
df_all


# In[ ]:


df_train = df_all[:len(df_train)]
df_test = df_all[len(df_train):]


# In[ ]:


print(df_train.columns)
print(df_train.isnull().sum()/len(df_train))
print(df_test.columns)
print(df_test.isnull().sum()/len(df_test))


# In[ ]:


#一旦ここで保存しておきます
df_train.to_csv("train_add_data.csv",index = False)
df_test.to_csv("test_add_data.csv",index = False)


# In[ ]:


#teatデータで8割以上ない列は一旦省く
drop_col = df_test.isnull().sum()[df_test.isnull().sum()/len(df_test) >= 0.8].index[1:]
drop_col_ = df_train.isnull().sum()[df_train.isnull().sum()/len(df_train) >= 0.8].index


# In[ ]:


drop_col_


# In[ ]:


drop_col = list(set(list(drop_col) + list(drop_col_)))


# In[ ]:


df_train = df_train.drop(columns = drop_col)
df_test = df_test.drop(columns = drop_col)
df_all = df_all.drop(columns = drop_col)


# In[ ]:


df_all.columns


# In[ ]:


print(df_test.isnull().sum()/len(df_test))


# In[ ]:


print(df_train.isnull().sum()/len(df_train))


# In[ ]:


cats = []
for col in df_all.columns:
    if df_all[col].dtype == 'object':
        cats.append(col)
        try:
            print(col, df_all[col].nunique())
        except:
            print(col,type(col))


# In[ ]:


df_all[cats].head()


# In[ ]:


#MSRPとFuel economyとHorsepowerは簡単に整えたいところ


# In[ ]:


#https://www.kaggle.com/code/takoihiraokazu/duration


# In[ ]:


def MSRP2num(msrp):
    msrp = str(msrp)
    match = re.search(r"¥[\d,]+(\.\d+)?", msrp)
    if match is None:
        match = 0
    else:
        match = float(match.group(0).replace("¥",""))
    return match


# In[ ]:


df_all["MSRP"] = df_all["MSRP"].map(MSRP2num)
df_all["MSRP"]


# In[ ]:


def FuelEconomy2num(data):
    data = str(data)
    max_match = re.search(r"[\d,]+(\.\d+)? l", data)
    if max_match is None:
        max_match = 0
    else:
        max_match = float(max_match.group(0).replace("l",""))

    min_match = re.search(r"[\d,]+(\.\d+)?-", data)
    if min_match is None:
        min_match = 0
    else:
        min_match = float(min_match.group(0).replace("-",""))
    return pd.Series([max_match, min_match])


# In[ ]:


df_all[["FuelEconomy_min","FuelEconomy_max"]] = df_all["Fuel economy"].apply(FuelEconomy2num)
df_all[["FuelEconomy_min","FuelEconomy_max"]]


# In[ ]:


def Horsepower2num(data):
    data = str(data)
    max_match = re.search(r"[\d,]+(\.\d+)? to", data)
    if max_match is None:
        max_match = 0
    else:
        max_match = float(max_match.group(0).replace("to",""))

    min_match = re.search(r"[\d,]+(\.\d+)? kW", data)
    if min_match is None:
        min_match = 0
    else:
        min_match = float(min_match.group(0).replace("kW",""))
    return pd.Series([max_match, min_match])


# In[ ]:


df_all[["Horsepower_min","Horsepower_max"]] = df_all["Horsepower"].apply(Horsepower2num)
df_all[["Horsepower_min","Horsepower_max"]]


# In[ ]:


df_all["Horsepower_mean"] = df_all["Horsepower_min"] + df_all["Horsepower_max"]


# In[ ]:


#Horsepower_minと価格の相関
plt.figure(figsize=[7,7])
plt.scatter(df_all.Horsepower_mean, df_all.price, s=3)
plt.xlabel('Horsepower_mean')
plt.ylabel('price')
plt.show()#2060?実際のデータなので間違いが含まれる


# In[ ]:


#FuelEconomy_minと価格の相関
plt.figure(figsize=[7,7])
plt.scatter(df_all.FuelEconomy_min, df_all.price, s=3)
plt.xlabel('FuelEconomy_min')
plt.ylabel('price')
plt.show()#2060?実際のデータなので間違いが含まれる


# In[ ]:



#FuelEconomy_minと価格の相関
plt.figure(figsize=[7,7])
plt.scatter(df_all.MSRP, df_all.price, s=3)
plt.xlabel('MSRP')
plt.ylabel('price')
plt.show()#2060?実際のデータなので間違いが含まれる


# In[ ]:


#追加（Excel読み書き用）
get_ipython().system('pip install openpyxl # .xlsx読み書き用')
get_ipython().system('pip install xlrd # .xls読み書き用')


# In[ ]:





# In[ ]:


df_generation = pd.read_excel('../input/model-segment-and-generation-0430/2022_Spring_segment01.xlsx', index_col=0)
df_all = pd.concat([df_all, df_generation.reset_index()],axis=1)


# In[ ]:


df_all


# In[ ]:


df_train = df_all[:len(df_train)]
df_test = df_all[len(df_train):]


# In[ ]:


#一旦ここで保存しておきます
df_train.to_csv("train_add_data_.csv",index = False)
df_test.to_csv("test_add_data_.csv",index = False)


# In[ ]:




