#!/usr/bin/env python
# coding: utf-8

# # Doğrusal Regresyon Egzersizleri

# 50 adet Startup'ın araştırma ve geliştirmeye yönelik harcaması, yönetime yönelik harcaması, pazarlama harcaması, kazandıkları kar miktarı ve kuruldukları lokasyon bilgisi bulunmaktadır. Amaç kar miktarını tahmin etmektir. Bu bir sayısal tahmin problemidir ve bağımlı değişkenimiz "Profit".

# Numpy, matplotlib.pyplot, pandas ve seaborn kütüphanelerini çekirdeğe dahil edelim.

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# Dizinde bulunan veri çerçevemizi startups değişkenine atayalım. startups değişkenini df değişkenine kopyalayarak kullanmaya başlayalım.

# In[ ]:


df = pd.read_csv('../input/50startups/50_Startups.csv')


# İlk 5 gözlemini yazdıralım.

# In[ ]:


df.head()


# Veri çerçevesinin bilgilerini görüntüleyelim.

# In[ ]:


df.info()


# Kaç gözlem ve öznitelikten oluştuğunu görüntüleyelim.

# In[ ]:


df.shape


# Eksik verileri kontrol edelim.

# In[ ]:


df.isna()


# In[ ]:


df.isna().sum()


# Korelasyon matrisi çizdirelim.

# In[ ]:


df.corr()


# Seaborn ile korelasyon matrisinin ısı haritasını çizdirelim.

# In[ ]:


corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);


# R&D Spend ve Profit arasındaki korelasyonu daha iyi görebilmek için scatterplot çizdirelim.

# In[ ]:


sns.scatterplot(df["R&D Spend"], df["Profit"], color="purple")


# -Korelasyon ilişkisi pozitif yönde ve doğrusaldır.

# Sayısal değişkenlerin dağılımını görmek için df üzerinden histogram çizdirelim.

# In[ ]:


df.hist(color="purple")


# Veri çerçevesinin temel istatistik değerlerini görüntüleyelim.

# In[ ]:


df.describe().T


# State'a ait benzersiz değerleri görüntüleyelim.

# In[ ]:


df["State"].unique()


# get_dummies yardımıyla State'a dair kategorik öznitelik çıkarımlarında bulunalım. Çünkü State'ların birbirine üstünlüğü yok, nominaller. Ordinal değil.

# In[ ]:


df =  pd.get_dummies(df, columns=['State'])


# In[ ]:


df.head()


# State özniteliğini silip dummy olarak yaratılan State'lardan da birisini hariç tutarak veri çerçevemizi güncelleyelim.

# In[ ]:


df = df.drop(['State_Florida'],axis=1)


# 

# In[ ]:


df.head()


# Veri çerçevemizi bağımlı ve bağımsız değişkenler olmak üzere bölütleyelim.

# In[ ]:


x = df.drop('Profit',axis=1)  # > bağımsız değişkenler
y= df['Profit']   # > bağımlı değişken


# Bağımlı ve bağımsız değişkenleri kontrol edelim.

# In[ ]:


x.head()


# In[ ]:


y.head()


# Bu bağımlı ve bağımsız değişkenlerden train ve test olmak üzere 4 parça oluşturalım. Bunu yapmak için train_test_split kullanalım.

# In[ ]:


from sklearn.model_selection import train_test_split # Verilerimizi eğitim ve test olarak ikiye ayırabilmek için bu kütüphaneyi ekliyoruz.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
# test verilerini ve eğitim verilerinin oranını ayarladık. test verileri verilerin %20'sini kapsayacak.


# 4 parça değişkeni kontrol edelim.

# In[ ]:


x_train


# In[ ]:


x_test


# In[ ]:


y_train


# In[ ]:


y_test


# LinearRegression'u çekirdeğe dahil edip modeli inşa edelim.

# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# Modeli eğitmek için bağımlı bağımsız değişkenlerden oluşturulan eğitim verilerini verelim.

# In[ ]:


model.fit(x_train, y_train)


# Modele daha önce görmediği bağımlı test değişkenini tahmin ettirelim. Bu tahmin değerlerimizi y_pred değişkenine atayalım.

# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)


# Tahminleri ve gerçek değerleri bir veri çerçevesinde toplayıp üzerinde göz gezdirelim.

# In[ ]:


y_pred = pd.DataFrame(y_pred, columns = ['Predictions'])
y_pred = y_pred.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# In[ ]:


final = pd.concat([y_pred, y_test],axis=1)
final.sample(10)


# sklearn bünyesinde barınan metrics'i çekirdeğe dahil edelim ve MAE, MSE, RMSE değerlerini görüntüleyelim.

# In[ ]:


# MAE = Mean Absolute Error = Ortalama Mutlak Hata => iki sürekli değişken arasındaki farkın ölçüsüdür.
# MSE = Mean Squared Error = Ortalama Kare Hata => bir regresyon eğrisinin bir dizi noktaya ne kadar yakın olduğunu gösterir.
# RMSE = Root Mean Squared Error = Kök Ortalama Kare Hata => makine öğrenmesi modellerinde gerçek değerler ile tahmin edilen değerler araındaki uzaklığın bulunmasında kullanılır.
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

MAE  = mean_absolute_error(y_test, y_pred)
MSE  = mean_squared_error(y_test, y_pred)
RMSE =  mean_squared_error(y_test, y_pred, squared= False)

print(f" MAE ={MAE} MSE= {MSE} RMSE= {RMSE}")


# Modelin R Squared değerini eğitim verileri üzerinden yazdıralım.

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:




