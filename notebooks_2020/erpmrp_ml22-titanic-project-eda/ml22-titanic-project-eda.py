#!/usr/bin/env python
# coding: utf-8

# # Introduction
# the sinking of Titanic.
# 
# <font color = 'blue'>
# Content:
#     
# 1. [Load and Check data](#1)
# 1. [Variable description](#2)
#     * [Univariate variable analysis](#3)
#         * [Categorcal variable analysis](#4)
#         * [NUmerical variable analysis](#5)
# 1. [Basic data analysis](#6)  
# 1. [Outlier detection](#7)  
# 1. [Missing values](#8)  
#     * [Find values](#9)
#     * [Fill values](#10)
# 1. [Visualization, correlation matrix](#11)  
#     * [Correlation between Sibsp -- Parch -- Age -- Fare -- Survived](#12)
#     * [SibSp -- Survived](#13)
#     * [Parch -- Survived](#14)
#     * [Pclass -- Survived](#15)
#     * [Age -- Survived](#16)
#     * [Pclass -- Survived -- Age](#17)
#     * [Embarked -- Sex -- Pclass -- Survived](#18)
#     * [Embarked -- Sex -- Fair -- Survived](#19)
#     * [Fill missing Age feature](#20)
# 1. [Feature engineering](#21)  
#     * [Name -- Title](#22)
#     * [Family size](#23)
#     * [Embarked](#24)
#     * [Ticket](#25)
#     * [Pclass](#26)
#     * [Sex](#27)
#     * [Drop Passenger ID and Cabin](#28)
# 1.  [Modeling](#29)  
#     * [Train - Test Split](#30)
#     * [Simple Logistic Regression](#31)
#     * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#32)
#     * [Ensemble Modeling](#33)
#     * [Prediction and Submission](#34)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = '1'></a><br>
# # 1.Load and Check data

# In[ ]:





# In[ ]:


#a=[1,2,3,4]
#plt.plot(a)
#plt.show()


# In[ ]:


#çeşitleri görmek çin:

#plt.style.available


# In[ ]:


import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id = '2'></a><br>
# # 2. Variable description

# 1. PassengerId
# 1. Survived', 
# 1. Pclass', 
# 1. Name', 
# 1. Sex', 
# 1. Age', 
# 1. SibSp',
# 1. Parch', 
# 1. Ticket', 
# 1. Fare', 
# 1. Cabin', 
# 1. Embarked

# In[ ]:


train_df.info()


# * float64(2)
# * int64(5)
# * object(5)

# <a id = '3'></a><br>
# # Univariate variable analysis
# * Categorcal variable: Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, Sibsp and Parch
# * NUmerical variable: Fare, age and passengerId

# <a id = '4'></a><br>
# ## ****Categorcal variable:

# In[ ]:


def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))


# In[ ]:



    
category1 = ["Survived","Sex","Pclass","Embarked","SibSp", "Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# In[ ]:





# <a id = '5'></a><br>
# ## NUmerical variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ["Fare", "Age","PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id = '6'></a><br>
# # 3.Basic data analysis

# In[ ]:


# Plcass vs Survived
# Sex vs Survived
# Sibsp vs Survived
# Parch vs Survived


# In[ ]:


# Plcass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# In[ ]:


# Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# In[ ]:


# Sibsp vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# In[ ]:


# Parch vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# <a id = '7'></a><br>
# # 4.Outlier detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]


# In[ ]:


# drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)


# <a id = '8'></a><br>
# # 5.Missing values

# In[ ]:


#ilk 891 lik halini kaybetmeyelim
train_df_len = len(train_df) 
train_df_len


# In[ ]:


#hem train, hem de test i bir arada düzenlemek için, birleştiriyoruz:
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
train_df.head()


# <a id = '9'></a><br>
# ## Find values

# In[ ]:


# hangi kolonlarda miising value var:
train_df.columns[train_df.isnull().any()]


# In[ ]:


# bu kolonlarda, kaçar tane misisng value var:
train_df.isnull().sum()


# <a id = '10'></a><br>
# ## Fill values
# * Embark has only 1 missing value
# * Fare has ony 1

# In[ ]:


#Embarked kolondaki hangi kayıtlar null mu, bakalım:
train_df[train_df["Embarked"].isnull()]


# In[ ]:


#Pclass veya Fare değerlerine bakarak, bu gruptaki yolcuların değerleri alınabilir.
#Yani Plclass=1 olan diğer yolcuların Embarked değerini al, buraya yaz. gibi
#veya Fare=80 olan diğer yolcuların Embarked değerini al, buraya yaz. gibi

train_df.boxplot(column="Fare",by = "Embarked")
plt.show()

#boxplot a bak, y=80 e göre, C den binmiş olma olasılıkları daha yüksek


# In[ ]:


#bu yüzden bu iki null yolcunun Embarked değerlerini C olarak dolduralım:
train_df["Embarked"] = train_df["Embarked"].fillna("C")


# In[ ]:


#sonra kontrol edelim:
train_df[train_df["Embarked"].isnull()]


# In[ ]:


#Şİmdi Fare için.
#Hangi yolcumuzun Fare değeri yok null bakalım:
train_df[train_df["Fare"].isnull()]

#Embarked veya Pclass üzerinden gidilebilir.
#Bu sefer Pclass tan yola çıkalım. Pclass 3 olanlar


# In[ ]:


#Pclass 3 olanlar
train_df[train_df["Pclass"] == 3]


# In[ ]:


#Pclass 3 olanlar, Fare değerlerinin ortalamasını alalım. ne kadar ödemişler ort.
np.mean(train_df[train_df["Pclass"] == 3]["Fare"])


# In[ ]:


#Şİmfi Fare null olan kayıtları, bu değer ile dolduralım:
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))


# In[ ]:


#sonra kontrol edelim:
train_df[train_df["Fare"].isnull()]


# <a id = '11'></a><br>
# # 6.Visualization, correlation matrix

# <a id = '12'></a><br>
# ## Correlation between Sibsp -- Parch -- Age -- Fare -- Survived

# In[ ]:


list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
#heatmap:
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show()

#annot: kareler üzerinde, değerleri üzerinde gösermesi için
#fmt: virgülden sonra 2 basamak göstersin


# In[ ]:


#amacımız Survived feature le ilişkili bireyler var mı, bunu yakalamak.
#Survived ençok Fare ile yüksek korelasyona sahip gözüküyor.
#yorum: Fre bilet için çok fazla para ödeyenlerin, hayatta kalma şansı Survived daha yüksek. İlşki var gözüküyor (0.26 lık korelasyon)


# <a id = '13'></a><br>
# SibSp -- Survived

# In[ ]:


#yolcunun sahip olduğu kardeş veya eş sayısı

#factor plot ile inceleyelim:
g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()

#yorum:SibSp>2 ise, Survived probability azalıyor. Buradan yeni br feature extract edebiliriz. 0, 1 d,ye. Sonra bunu ML modleini eğitmek için kullanabilriiz.


# <a id = '14'></a><br>
# ## SibSp -- Survived

# In[ ]:


#factor plot ile inceleyelim:
g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Survived Probability")
plt.show()


# In[ ]:


# yalnız 0 gelen kişler, 1,2,3 kişi ile gelenlere göre (küçk a,,leler) daha az Surviva olasılığına sahip.
# 1 ve 2 ler aynı olalığıa sahip. 
# 3'te yakın olasığüı asahip, ama siyah çizgiye dikkat, bu std. sapma
# 3=Kırımızdaki 0.6 değeri ortalama. Srd. sapmaya bakınca 1 de olabilir, 0.2de olabilir. bu aralıkta değişken
# Sibsp ve Parch ile yeni fearurei hreshold 3 ile yapıabilir.
# <3 small ffamşlies have more chance to survive.


# <a id = '15'></a><br>
# ## Pclass -- Survived

# In[ ]:


#factor plot ile inceleyelim:
g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()


# In[ ]:


#Pclass yolcuların sınıfı
#yorum: 1.soınıftakilerin hayatta kalma olasılıpı daha yüksek gözüküyor.
#Pclass ML de kullanılabilir.


# <a id = '16'></a><br>
# ## Age -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived")
#her bir Survived tipi için, dikeyde ayıracak

#distplot:
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# In[ ]:


#Yorum:
#Survived 1 e bakalım. Age<=10 0 bebek çocuklar daha çok kurtuluyormuş, öncelikli.
#Yaşlı insanlarda da Age>80 kutulma yükse. Not:Sayıları daha az.
#20 li yaşlarda kurtulan az
#çoğu yolcu 15-35 yaşları arasında
#Age feature ML training de kullanılabilir.
#Age dağılımı, Mising value of Age de kullanılabilr.


# <a id = '17'></a><br>
# ## Pclass -- Survived -- Age

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 2)
#her bir Pclass tipi için ayrı satırda grafik oluşlturacak
#her bir Survived tipi için, dikeyde ayıracak

#histogram:
g.map(plt.hist, "Age", bins = 25)
g.add_legend()
plt.show()

# x ekseni Age. Garşfklerde yataylara bak
# 3. satır, Pclass 3'te en fazla yolcu olduğu bar dan gözüküyor.
# Ama yolcu fazla olması ile, Surve arasında doğru orantı gözükmüyor. Hatta Survived 0 daha fazla.
# Pclass, modeli eğitmede önemli bir feature olacak.


# <a id = '18'></a><br>
# ## Embarked -- Sex -- Pclass -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked", size = 2)
#her bir Embarked tipi için, ayrı satırda grafik oluşturacak..

#pointplot:
g.map(sns.pointplot, "Pclass","Survived","Sex")
# x Pclass, y Survived, renkler Sex
g.add_legend()
plt.show()


# In[ ]:


#yorum:
#Female have must better Survival rate . Sınıflandırmada kullanılanilir, feature olarak.
#Males have better Survival rate in Plclass=3 in C.
# Embarked S ile Pclass arasında ilişki gözüküyor.
#Emnberked and Sex will ve used in training.


# <a id = '19'></a><br>
# ## Embarked -- Sex -- Fair -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 2.3)
#her bir Embarked tipi için, ayrı satırda grafik oluşturacak..
#her bir Survived tipi için, dikeyde ayıracak

#barplot:
g.map(sns.barplot, "Sex", "Fare")
#bar sayısı Sex,y=Fare
g.add_legend()
plt.show()


# In[ ]:


#1.satır. S limanındn binenler, toplamda(male+female) daha fazla ödeyenler, daha fazla Survive olmuş.
#Üçü için genel değerlendirme, daha çok para ödeyenlerin, hayatta kalma durumu daha fazla.
#fare yi feature ,için 1:0-25, 2:25-50, 3:51-75 diye kategori lere çevrilebilir.
#S,C,Q: en yüksek C endüşük S haytta kalma.
#dikeyde incele, hayatta kalan Fenale, male e göre daha çok para ödemiş.


# <a id = '20'></a><br>
# ## Fill missing Age feature

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


# null Age leri neyegöre doldurabiliriz?
#mesela Sex e göre olsa. Female ort. yaşını al, Male ort. yaşını al. Bua göre Male ve Female Age lerini doldururuz.
#mesle Pclass a bakılabilir. 3.sınıftakilerin yaş ortalamasına göre..
#mesela Parc, SibSp
#ya da bumları hibrit yapabiliriz. Mesela Sex, Pclass, Parch be lli olan ve Age i bilinen yolculara bakılır.
#sonra Age e göre Median ının alırız(ortalamaya göre daha mantıklı), 


# In[ ]:


#Sex e bakalım:
sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")
plt.show()

#orta çizgi median ı gösterir.
#erkek ve kadın için median değeri aynı, Sex tahmin için ayırt edici değil. işe yaramayacak.


# In[ ]:


sns.factorplot(x = "Sex", y = "Age", hue = "Pclass",data = train_df, kind = "box")
plt.show()


# In[ ]:


#Mavi, Pclass=1 ler, medyanları 40 gibi düşünülebilir.
#Turuncu, Pclass=2, medyanları 30
#Yeşil, Pclass=3, medyanları 25
#Yani en yaşlılar Pclass 1 de. 
# Yani Age değeri olmayanları, Pclass larına göre, medyan Age doldurabiliriz.


# In[ ]:


#Parch ve SibSh ye bakaulım:
sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")
plt.show()


# In[ ]:


# Parch 0,1,2 için medyan Age 25 gibi.
# Parch > 2 için 45 gibi.

#SibSp 0,1,2 için Age 25 gibi
#SibSp > 2 için Age 10 gibi medyan.

#3 farklı analiz sonucu James in yaşı 25 olabilie diyebiliyoruz.


# In[ ]:


#Tüm feature lar arası korelasyon a bakalım.
#heatmap

sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True)
plt.show()

#Sex gözükmedi, dikkat


# In[ ]:


#heatmap te Sex i görebilmek için, sayısal bir değere çevirmek lazım. Şu an male, Female
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]

sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True)
plt.show()


# In[ ]:


#Age is not correlated with sex but it is correlated with parch, sibsp and pclass.


# In[ ]:


#Age deki boşlukları doldurmak için kod yazalım:
#ilk non value Age leri bulalım, sonra indeksini bul, bir listeye yaz:
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
index_nan_age


# In[ ]:


#bu indekslerde dolaş tek tek, SibSp .. fetaurlera balıp, agı predict edelim.
for i in index_nan_age:
    age_pred = train_df["Age"][
        ((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &
          (train_df["Parch"] == train_df.iloc[i]["Parch"])& 
          (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))
    ].median()
    
    #yine bazı Nan value lar var. median alalım. Bunun ile doldurucaz.
    age_med = train_df["Age"].median()
    
    if not np.isnan(age_pred):                #boş değil ise
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med


# In[ ]:


#Artık Age de null kalmadığnı görelim:
train_df[train_df["Age"].isnull()]


# <a id = '21'></a><br>
# # 7.Feature engineering

# <a id = '22'></a><br>
# ## Name -- Title

# In[ ]:


train_df["Name"].head(10)
#isimlerde anlam yok,ama ünvanları icelenebilir.


# In[ ]:


#split özelliğini örneke görelim. Noktaya göre ayıracak:
s = 'McCarthy, Mr. Timothy J'
s.split(".")


# In[ ]:


#0.indeksini alalım:
s.split(".")[0]


# In[ ]:


#0.indeksini alalım, bir de virgüle göre ayıralım:
s.split(".")[0].split(",")


# In[ ]:


#0.indeksini alalım, bir de virgüle göre ayıralım, bun son elemanını alalım:
s.split(".")[0].split(",")[-1]
#solda boşluk kaldı dikkate


# In[ ]:


#0.indeksini alalım, bir de virgüle göre ayıralım, bunun son elemanını alalım, boşlık atalım:
s.split(".")[0].split(",")[-1].strip()


# In[ ]:


#virgüle noktaya göre ayrulabilir.
name = train_df["Name"]

#Title diye yeni fature oluşturalım.
#List comprehension metodunu kullanalım:
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]


# In[ ]:


#sonucu görelim
train_df["Title"].head(10)


# In[ ]:


#
sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()
#Don dan sonrası rare ender.
#Mr. Master, Ms ve other diye kategorik yapılabilir.


# In[ ]:


#kategorik hale getirelim:
# convert to categorical

#rare title ları, other ile replace edelimm
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

#birleştirip, sayıdal değere çevirelim.
#master ise 0, ...
train_df["Title"] = [0 if i == "Master" else 
                     1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 
                     2 if i == "Mr" else 
                     3 for i in train_df["Title"]]

train_df["Title"].head(20)


# In[ ]:


sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[ ]:


#şimdi bu kategorilerin(0,1,2,3) Survaval oranlarına bakalım
g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")
g.set_xticklabels(["Master","Mrs","Mr","Other"])  # kategoriler: 0, 1, 2, 3
g.set_ylabels("Survival Probability")
plt.show()


# In[ ]:


# Mrs en yuksek hayatta kalma oranına sahip.
#train Df den Name feature ini kaldırabiriz artık. ML de gerek kalmayacak.Yeni Title feature yi kullanıcaz.

train_df.drop(labels = ["Name"], axis = 1, inplace = True)
#inplace = True: çıkardıktan sonra df ye oomatik eşitle

train_df.head()


# In[ ]:


#Title da 0,1,2,3 var. Title yı kaldırıp, BU değerleri kolonlara dağıtsın. 0 ve 1 lik hale getiriyor:
train_df = pd.get_dummies(train_df,columns=["Title"])
train_df.head()


# <a id = '23'></a><br>
# ## Family size

# In[ ]:


#dataset teki SibSp:Eş yada kerdeş ve Parch:Çocuk veya ebevyn aile bilgleri ile igli idi.
train_df.head()
#bu iki feture yi birleştirip, yeni ibir family size feature ı yapıcaz


# In[ ]:


#Kimsesi yok ise aile 0 çıkmasın, kişinin kendisi 1 dir. bu yüzden 1 ekliyoruz.
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1

train_df.head()


# In[ ]:


#bu yeni Fsize ın, Survival uzerindeki etkisine bakalım:
#barplot ile bakalım:
g = sns.factorplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()


# In[ ]:


#farklıkategorilere ayırabiriz.
#mesela threshold Fize=5 tan küçükler ve büyükler diye.
#1 ve 0 a indirgiyoruz. 1 küçük aile, 0 büyük aile
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]

train_df.head(10)


# In[ ]:


#kaç 0 var,kaç 1 var görelim:
sns.countplot(x = "family_size", data = train_df)
plt.show()


# In[ ]:


#bunların Survival ile ilişkisine bakalım:
g = sns.factorplot(x = "family_size", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()


# In[ ]:


#küçük ailelerin Survival olasılığı daha yüksek


# In[ ]:


#Familysize 1 ve 0lara göre ayır:
train_df = pd.get_dummies(train_df, columns= ["family_size"])
train_df.head()


# <a id = '24'></a><br>
# ## Embarked

# In[ ]:


#Kabul: Embarked feature ini, ML modelinde olduğu gibi kullanalım.
# bir feature extraction yapılmayacak.


# In[ ]:


train_df["Embarked"].head()


# In[ ]:


sns.countplot(x = "Embarked", data = train_df)
plt.show()


# In[ ]:


#en fazla S limanından binen olmuş.


# In[ ]:


#Embarked feature ni kullanılabilir hale getirelim.
#Embarked kolonundaki verileri 0,1,2 halinde dağıtalım
train_df = pd.get_dummies(train_df, columns=["Embarked"])

train_df.head()


# <a id = '25'></a><br>
# ## Ticket

# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


#arada boşluklar, bazında tek, bazısında ön ekli
#baştaki merinleri, sondakinden ayırıcaz. Başta hiçbirşey yok ise, X yazıcaz:
#sondaki sayıların bir kategori anlamıyok, baştaki gruo numaralarının bir katgorik anlamı olabilr.


# In[ ]:


#örnek:
a = "A/5. 2151"
a.replace(".","")


# In[ ]:


a = "A/5. 2151"
a.replace(".","").replace("/","")


# In[ ]:


a = "A/5. 2151"
a.replace(".","").replace("/","").strip()

#strip ile başta ve sonada extra boşluk varsa alır idi. burada yok


# In[ ]:


a = "A/5. 2151"
a.replace(".","").replace("/","").strip().split(" ")
#split: boşluğa göre ayırıyor. iki ayrı string yapıyor. bunları listeye koyuyor.


# In[ ]:


a = "A/5. 2151"
a.replace(".","").replace("/","").strip().split(" ")[0]
#bu listenin 0.elemanını al


# In[ ]:


tickets = []

for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
        
train_df["Ticket"] = tickets

train_df.head()


# In[ ]:


#
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")
#kolon balığı Ticket_ yerine T_ şeklinde olsun diye prefiz

train_df.head(10)


# <a id = '26'></a><br>
# ## Pclass

# In[ ]:


#bir extraction planlamadık
#kendi içindeki dağılıma bakalım
sns.countplot(x = "Pclass", data = train_df)
plt.show()


# In[ ]:


#3. sınıftaki yolcuların sayısı en fazla


# In[ ]:


#Pclass ı 0-1 olarak kolonlara dağıtalım. kategorik.
#önce tipini category yaptık???
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns= ["Pclass"])

train_df.head()


# <a id = '27'></a><br>
# ## Sex

# In[ ]:


#bir fature extraction yapmıycaz
#sadece ML de kullanılabşişr hale getiricez.
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"])

train_df.head()


# <a id = '28'></a><br>
# ## Drop Passenger ID and Cabin

# In[ ]:


#bu iki kolonu drop edecek, sütun olarak
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)

train_df.columns


# <a id = '29'></a><br>
# # 8.Modeling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id = '30'></a><br>
# ## Train - Test Split
# 

# In[ ]:


# train df miizn boyuru idi:
train_df_len


# In[ ]:


#test ve traini ayırmak için:
#test: _len den başla, sonuna kadar git:
test = train_df[train_df_len:]

#test içerisinde Survived olmayaxak:
test.drop(labels = ["Survived"],axis = 1, inplace = True)

test.head()


# In[ ]:


#train: baştan al, _len e kadar
train = train_df[:train_df_len]

#train için, x ve y yi ayıralım:
#feature lar:
X_train = train.drop(labels = "Survived", axis = 1)

#survived:
y_train = train["Survived"]

#
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

#
print("X_train:",len(X_train))
print("X_test:",len(X_test))
print("y_train:",len(y_train))
print("y_test:",len(y_test))
print("test:",len(test))


# <a id = '31'></a><br>
# ## Simple Logistic Regression
# 

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#
acc_log_train = round(logreg.score(X_train, y_train)*100,2) 
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))


# <a id = '32'></a><br>
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation
# * 5 farklı modeli karşılaştırıcaz. içindei en iyi parametre yi de arıycaz.
# * We will compare 5 ml classifier and evaluate mean accuracy of each of them by stratified cross validation.
# 
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[ ]:


random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

### ???

#10dan 500e 20'şer gitsin
dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]


# In[ ]:



cv_result = [] #sonuçları tutacak
best_estimators = [] #en iyilerini seçecek

for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])

    
    #njobs: kodu paralel koşturuyor. hızlı çalışıyor.
    #verbose: kod koşarken, sonuçları sürekli bize göstersin


# In[ ]:


#yukarıdaki sonuçları görselleştirelim:
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, 
                           "ML Models": 
                           ["DecisionTreeClassifier", "SVM","RandomForestClassifier",
                             "LogisticRegression", "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# In[ ]:


# 08. i threshold seçelim.bundan üstteki leri birletirelim. ensemblm yapıcaz


# <a id = '33'></a><br>
# ## Ensemble Modeling

# In[ ]:


#%80 barajını açan 3 model var idi. bunlarım indexlerini yazıyor:

votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)

#soft, hard

votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test),y_test))


# 
# 
# <a id = '34'></a><br>
# ## Prediction and Submission

# In[ ]:


#test verisi ile sonuçları depoluyor 0 ve 1 ler:
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
test_survived


# In[ ]:


#test_PassengerId aha öce kaydetmitik. ikiidni birleştiriyoruz:
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results


# In[ ]:


results.to_csv("titanic.csv", index = False)


# In[ ]:


#accuracy i artırmak için, farklı feature ler exact edip, denemeler yapabilirdin.

