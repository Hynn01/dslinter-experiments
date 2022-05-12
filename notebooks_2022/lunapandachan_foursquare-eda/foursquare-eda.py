#!/usr/bin/env python
# coding: utf-8

# [![](https://img.youtube.com/vi/FnV0thLS1Fs/0.jpg)](https://www.youtube.com/watch?v=FnV0thLS1Fs)

# lunana  
# last update 2022 05 01  
# ゆっくりしていってね

# version 2 +histogram

# # Data list  
# * [**train.csv**](#train.csv)  
# * [**pair.csv**](#pair.csv)  
# * [**test.csv**](#test.csv)  
# * [**sample_submission.csv**](#sample_submission.csv)

# **霊夢:今日はマッチングのコンペだね。  
# 魔理沙:まずは概略を見るぞ。**  
# 
# **Reimu: Today is a matching competition.  
# Marisa: First, let's take a look at the outline.**

# 近くのレストランを探したり、未知のエリアで用事を計画したりするときは、関連性のある正確な情報を期待します。世界中で高品質のデータを維持することは課題であり、ナビゲーションを超えた意味合いがあります。企業は、市場拡大のために新しいサイトを決定し、競争環境を分析し、位置データによって通知された関連広告を表示します。これらや他の多くの用途では、信頼できるデータが重要です。
# 
# 商用のPOI（Points-of-Interest）に関する大規模なデータセットには、実際の情報が豊富に含まれている可能性があります。最高レベルの精度を維持するには、データを複数のソースからのタイムリーな更新と照合して重複排除する必要があります。生データにはノイズ、非構造化情報、不完全または不正確な属性が含まれている可能性があるため、重複排除には多くの課題が伴います。データセットの重複排除には、機械学習アルゴリズムと厳密な人間による検証方法の組み合わせが最適です。
# 
# Foursquareは、このような手法を完成させた12年以上の経験を持ち、グローバルPOIデータの独立したプロバイダーとして第1位です。主要な独立したロケーションテクノロジーおよびデータクラウドプラットフォームであるFoursquareは、デジタル空間と物理的な場所の間に意味のあるブリッジを構築することに専念しています。アップル、マイクロソフト、サムスン、ユーバーなどの大手企業から信頼されているフォースクエアのテクノロジースタックは、場所と動きの力を利用して、顧客体験を改善し、より良いビジネス成果を推進します。
# 
# このコンテストでは、POIを一致させます。ノイズ、重複、無関係、または不正確な情報を含むように大幅に変更された150万を超えるプレイスエントリのデータセットを使用して、どのプレイスエントリが同じスポットを表すかを予測するアルゴリズムを作成します。各プレイスエントリには、名前、住所、座標などの属性が含まれています。送信が成功すると、一致が最も正確に識別されます。
# 
# POIを効率的かつ正常に照合することで、新しい店舗やビジネスが人々に最も利益をもたらす場所を簡単に特定できるようになります。  
# 
# When you look for nearby restaurants or plan an errand in an unknown area, you expect relevant, accurate information. To maintain quality data worldwide is a challenge, and one with implications beyond navigation. Businesses make decisions on new sites for market expansion, analyze the competitive landscape, and show relevant ads informed by location data. For these, and many other uses, reliable data is critical.
# 
# Large-scale datasets on commercial points-of-interest (POI) can be rich with real-world information. To maintain the highest level of accuracy, the data must be matched and de-duplicated with timely updates from multiple sources. De-duplication involves many challenges, as the raw data can contain noise, unstructured information, and incomplete or inaccurate attributes. A combination of machine-learning algorithms and rigorous human validation methods are optimal to de-dupe datasets.
# 
# With 12+ years of experience perfecting such methods, Foursquare is the #1 independent provider of global POI data. The leading independent location technology and data cloud platform, Foursquare is dedicated to building meaningful bridges between digital spaces and physical places. Trusted by leading enterprises like Apple, Microsoft, Samsung, and Uber, Foursquare’s tech stack harnesses the power of places and movement to improve customer experiences and drive better business outcomes.
# 
# In this competition, you’ll match POIs together. Using a dataset of over one-and-a-half million Places entries heavily altered to include noise, duplications, extraneous, or incorrect information, you'll produce an algorithm that predicts which Place entries represent the same point-of-interest. Each Place entry includes attributes like the name, street address, and coordinates. Successful submissions will identify matches with the greatest accuracy.
# 
# By efficiently and successfully matching POIs, you'll make it easier to identify where new stores or businesses would benefit people the most.

# **霊夢:人間と人間をマッチングさせるのかな？マッチングアプリのイメージかな？**  
# 
# **Reimu: Do you match humans with humans? Is it an image of a matching app?**

# **霊夢:次はデータを見てみよう**  
# 
# **Reimu: Let's look at the data next**

# train.csv-トレーニングセット。100万を超える場所のエントリ用の11の属性フィールドと、次のもので構成されています。  
# id-各エントリの一意の識別子。  
# point_of_interest-エントリが表すPOIの識別子。同じPOIを説明する1つまたは複数のエントリが存在する場合があります。2つのエントリは、共通のPOIを説明するときに「一致」します。  

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt


# In[ ]:


df_train=pd.read_csv('../input/foursquare-location-matching/train.csv')
df_train.head()


# In[ ]:


len(df_train)


# In[ ]:


df_train.describe()


# **霊夢:緯度、経度のデータを使って、地図上に表示してみよう。**  
# 
# **Reimu: Let's display it on a map using latitude and longitude data.**

# In[ ]:


from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame


# In[ ]:


def draw_lon_lat(df, world):
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = GeoDataFrame(df, geometry=geometry)   
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);


# In[ ]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
draw_lon_lat(df_train, world)


# In[ ]:


df_train['state'].unique()


# In[ ]:


len(df_train['state'].unique())


# In[ ]:


df_train['country'].unique()


# In[ ]:


len(df_train['country'].unique())


# In[ ]:


df_train['categories'].unique()


# In[ ]:


len(df_train['categories'].unique())


# **魔理沙:Top10をグラフで表示してみよう。**  
# 
# **Marisa: Let's display the Top 10 in a graph.**

# In[ ]:


plt.figure(figsize=(15,10))
plt.rcParams['font.size']=15
plt.bar(df_train["categories"].value_counts().sort_values(ascending=False)[:10].index,df_train["categories"].value_counts().sort_values(ascending=False)[:10])
plt.xticks(rotation=90)


# **霊夢:NANがいっぱいある気がする。  
# 魔理沙:数えるか。**  
# 
# **Reimu: I feel like there are a lot of NANs.  
# Marisa: Do you count?**

# In[ ]:


print(df_train.isnull().sum())


# **霊夢:phoneとurlはそこまで重要な要素ではない気がするが、zipは半分近くがNANなんだね。**  
# 
# **Reimu: I don't think phone and url are that important, but zip is almost half NAN.**

# # pair.csv

# **魔理沙:pair.csvを見てみよう**  
# 
# **Marisa: Let's take a look at pair.csv**

# pair.csv-train.csv一致の検出を改善するように設計された、事前に生成された場所エントリのペアのセット。POIを識別するモデルの能力を向上させるために、追加のペアを生成することをお勧めします。  
# match-エントリのペアが共通のPOIを記述しているかどうか（TrueまたはFalse）。

# In[ ]:


df_pair=pd.read_csv('../input/foursquare-location-matching/pairs.csv')
df_pair.head()


# In[ ]:


len(df_pair)


# **霊夢:matchのTrueとFalseの割合をグラフで見てみよう。**  
# 
# **Reimu: Let's look at the ratio of True and False of match in a graph.**

# In[ ]:


plt.figure(figsize=(5,5))
plt.rcParams['font.size']=20
plt.pie(df_pair['match'].value_counts().sort_values(ascending=False),labels=df_pair['match'].value_counts().index,startangle=90)


# **霊夢:Trueのときの緯度経度を地図上で見てみよう**  
# 
# **Reimu: Let's see the latitude and longitude when True on the map**

# In[ ]:


p_num=0
print(df_pair.loc[p_num,'match'])
p_df=pd.DataFrame([[df_pair.loc[p_num,'longitude_1'],df_pair.loc[p_num,'latitude_1']]],columns=['longitude','latitude'])
p_df2=pd.DataFrame([[df_pair.loc[p_num,'longitude_2'],df_pair.loc[p_num,'latitude_2']]],columns=['longitude','latitude'])
p_df=pd.concat([p_df,p_df2])
p_df


# In[ ]:


draw_lon_lat(p_df, world)


# **魔理沙:Falseのときも見てみよう。**  
# 
# **Marisa: Let's take a look at False.**

# In[ ]:


p_num=1
print(df_pair.loc[p_num,'match'])
p_df=pd.DataFrame([[df_pair.loc[p_num,'longitude_1'],df_pair.loc[p_num,'latitude_1']]],columns=['longitude','latitude'])
p_df2=pd.DataFrame([[df_pair.loc[p_num,'longitude_2'],df_pair.loc[p_num,'latitude_2']]],columns=['longitude','latitude'])
p_df=pd.concat([p_df,p_df2])
p_df


# In[ ]:


draw_lon_lat(p_df, world)


# **霊夢:pairsの距離を測ってみよう。**

# In[ ]:


from geopy.distance import geodesic


# In[ ]:


idx=0

TokyoStation = (df_pair.loc[idx,'longitude_1'],df_pair.loc[idx,'latitude_1'] )
NagoyaStation = (df_pair.loc[idx,'longitude_2'],df_pair.loc[idx,'latitude_2'] )

dis = geodesic(TokyoStation, NagoyaStation).km

print(dis)


# In[ ]:


from tqdm import tqdm
for idx in tqdm(df_pair.index):
    p1 = (df_pair.loc[idx,'latitude_1'],df_pair.loc[idx,'longitude_1'] )
    p2 = (df_pair.loc[idx,'latitude_2'],df_pair.loc[idx,'longitude_2'] )
    dis = geodesic(p1, p2).km
    df_pair.loc[idx,'distance']=dis
    
df_pair.head()


# **霊夢:distanceをTrueとFalseのそれぞれでヒストグラムにしてみよう。違いが出るかな？**  
# 
# **Reimu: Let's make the distance a histogram for True and False respectively. Will it make a difference?**

# In[ ]:


df_true=df_pair[df_pair['match']==True]
df_true.head()


# In[ ]:


df_true['distance'].max()


# **霊夢:ほぼ地球の裏側じゃん**  
# 
# **Reimu: Almost on the other side of the globe**

# In[ ]:


df_true['distance'].min()


# In[ ]:


df_true['distance'].mean()


# In[ ]:


plt.figure(figsize=(8, 5))
df_pair.query('match==True')['distance'].hist(bins=20)


# In[ ]:


df_false=df_pair.query('match==False')
df_false.head()


# In[ ]:


df_false['distance'].max()


# In[ ]:


df_false['distance'].min()


# In[ ]:


df_false['distance'].mean()


# In[ ]:


plt.figure(figsize=(8, 5))
df_pair.query('match==False')['distance'].hist(bins=20)


# **霊夢:距離はあんまり関係なさそうだな**  
# 
# **Reimu: Distance doesn't seem to matter much**

# **魔理沙:次はTestを見てみよう**  
# 
# **Marisa: Let's take a look at Test next**

# test.csv-トレーニングセットと同様に、記録された属性フィールドを持つ場所エントリのセット。

# In[ ]:


df_test=pd.read_csv('../input/foursquare-location-matching/test.csv')
df_test


# In[ ]:


len(df_test)


# **魔理沙:最後にsumple_submissionを見てみよう**  
# 
# **Marisa: Finally, let's take a look at sumple_submission**

# sample_submission.csv-正しい形式のサンプル送信ファイル。  
# id-場所エントリの一意の識別子。テストセットのエントリごとに1つ。  
# matches-指定されたIDに一致するテストセット内のエントリのIDのスペース区切りリスト。エントリは常に自己一致します。

# テストセットの場所エントリidごとに、一致する場所idのスペース区切りのリストを送信する必要があります。場所は常に自己一致するため、idの一致リストには常にそのidが含まれている必要があります。  
# 
# ファイルにはヘッダーが含まれ、名前が付けられsubmission.csv、次の形式である必要があります。  
# 
# id,matches  
# E_00001118ad0191,E_00001118ad0191  
# E_000020eb6fed40,E_000020eb6fed40  
# E_00002f98667edf,E_00002f98667edf  
# E_001b6bad66eb98,E_001b6bad66eb98 E_0283d9f61e569d  
# E_0283d9f61e569d,E_0283d9f61e569d E_001b6bad66eb98

# In[ ]:


sumple_submission=pd.read_csv('../input/foursquare-location-matching/sample_submission.csv')
sumple_submission


# **霊夢:今回はここまでです。  
# 魔理沙:随時更新予定です。**  
# 
# **Reimu: That's all for this time.
# Marisa: Will be updated from time to time.**
