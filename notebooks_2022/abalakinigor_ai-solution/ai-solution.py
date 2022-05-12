#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install geocoder')
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

import re
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

import geocoder
import geopy.distance

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = None


# In[ ]:


# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42


# In[ ]:


# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
get_ipython().system('pip freeze > requirements.txt')


# In[ ]:


# Подгрузим наши данные из соревнования
#!pip install pandas
DATA_DIR = '/kaggle/input/sf-booking/'
df_train = pd.read_csv(DATA_DIR+'/hotels_train.csv') # датасет для обучения
df_test = pd.read_csv(DATA_DIR+'hotels_test.csv') # датасет для предсказания
sample_submission = pd.read_csv(DATA_DIR+'/submission.csv') # самбмишн


# In[ ]:


df_train.shape[0], df_test.shape[0], sample_submission.shape[0] 


# # Удаление дубликатов, фильтрация, заполнение пропусков

# In[ ]:


dupl_columns = list(df_train.columns)
mask = df_train.duplicated(subset = dupl_columns)
data_duplicates = df_train[mask]
duplicates = df_train[df_train.duplicated(subset = df_train.columns)] # получение списка полных дубликатов
рр = df_train.drop_duplicates() # удаление найденных дубликатов
print(f'Число найденных и удаленных дубликатов: {data_duplicates.shape[0]}')


# In[ ]:


#меняем значения выбросов в df_test

positive_anomal_mean = df_test[df_test['review_total_positive_word_counts'] > 170]['review_total_positive_word_counts'].mean()
df_test['review_total_positive_word_counts'] = df_test['review_total_positive_word_counts'].apply(lambda x: x if x < 170 else positive_anomal_mean)

number_of_reviews_anomal_mean = df_test[df_test['total_number_of_reviews_reviewer_has_given'] > 170]['total_number_of_reviews_reviewer_has_given'].mean()
df_test['total_number_of_reviews_reviewer_has_given'] = df_test['total_number_of_reviews_reviewer_has_given'].apply(lambda x: x if x < 170 else number_of_reviews_anomal_mean)


# In[ ]:


# удаляем выросы в df_train
df_train = df_train[df_train['review_total_positive_word_counts'] < 170]

df_train = df_train[df_train['total_number_of_reviews_reviewer_has_given'] < 170]


# In[ ]:


# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['reviewer_score'] = 0 # в тесте у нас нет значения reviewer_score, мы его должны предсказать, по этому пока просто заполняем нулями


# In[ ]:


hotels = df_test.append(df_train, sort=False).reset_index(drop=True)


# In[ ]:


fill_lat = hotels.groupby(['hotel_address'], as_index = False)['lat'].median()
fill_lat.rename(columns={'lat':'fill_lat'}, inplace=True)
null_lat=fill_lat[fill_lat.isna().any(axis=1)] #получаем список отелей с пропуском географической широты
# заполняем пропуски с помощью geocoder
null_lat['fill_lat']=null_lat['hotel_address'].apply(lambda x: geocoder.opencage(x,  key='924f41fc864d4bed8c9b438bff7c67e1',timeout=None ).json['lat'])
hotels = hotels.merge(null_lat, on=['hotel_address'], how = 'left') # вовзращаем значения в hotels
hotels['lat']=hotels['lat'].fillna(hotels['fill_lat'])

# аналогично для долготы
fill_lng = hotels.groupby(['hotel_address'], as_index = False)['lng'].median()
fill_lng.rename(columns={'lng':'fill_lng'}, inplace=True)
null_lng=fill_lng[fill_lng.isna().any(axis=1)] 
null_lng['fill_lng']=null_lng['hotel_address'].apply(lambda x: geocoder.opencage(x,  key='924f41fc864d4bed8c9b438bff7c67e1', timeout=None).json['lng'])
hotels = hotels.merge(null_lng, on=['hotel_address'], how = 'left')
hotels['lng']=hotels['lng'].fillna(hotels['fill_lng'])


# In[ ]:


hotels.info()


# # Создание новых признаков

# **Анализ тональности с помощью TextBlob**

# In[ ]:


# анализ позитивного отзыва
hotels_arr_pos = hotels['positive_review'].to_numpy()
polarity_arr=[]
subjectivity_arr=[]
for a in hotels_arr_pos:
    text=a
    testimonial = TextBlob(text)
    testimonial.sentiment
    polarity_arr.append(testimonial.sentiment.polarity)
    subjectivity_arr.append(testimonial.sentiment.subjectivity)
hotels["pos_review_polarity"]=polarity_arr
hotels["pos_review_subjectivity"]=subjectivity_arr


# In[ ]:


# анализ негативного отзыва
hotels_arr_neg = hotels['negative_review'].to_numpy()
polarity_arr=[]
subjectivity_arr=[]
for a in hotels_arr_neg:
    text=a
    testimonial = TextBlob(text)
    testimonial.sentiment
    polarity_arr.append(testimonial.sentiment.polarity)
    subjectivity_arr.append(testimonial.sentiment.subjectivity)
hotels["neg_review_polarity"]=polarity_arr
hotels["neg_review_subjectivity"]=subjectivity_arr


# **Анализ тональности с помощью Vader**

# In[ ]:


analyz = SentimentIntensityAnalyzer()

hotels['negative_review_analyze'] = hotels['negative_review'].apply(lambda x: analyz.polarity_scores(x))
hotels['positive_review_analyze'] = hotels['positive_review'].apply(lambda x: analyz.polarity_scores(x))

hotels.loc[:,['neg_rev_neg_mood_score', 'neg_rev_neutral_mood_score','neg_rev_pos_mood_score','neg_rev_mood_score']] = list(hotels['negative_review_analyze'].apply(lambda x: [x['neg'], x['neu'], x['pos'], x['compound']]).values)
hotels.loc[:,['pos_rev_neg_mood_score', 'pos_rev_neutral_mood_score','pos_rev_pos_mood_score','pos_rev_mood_score']] = list(hotels['positive_review_analyze'].apply(lambda x: [x['neg'], x['neu'], x['pos'], x['compound']]).values)


# **Извлечение признаков из даты отзыва**

# In[ ]:


hotels['review_date'] = pd.to_datetime(hotels['review_date'])
hotels['year'] = hotels['review_date'].dt.year
hotels['month'] = hotels['review_date'].dt.month
hotels['dayofweek'] = hotels['review_date'].dt.dayofweek
hotels['weekend'] = hotels['dayofweek'].apply(lambda x: 0 if x < 5 else 1)


# **Извлечение признаков из адреса отеля**

# In[ ]:


# название города
def city_from_adress(col):
    res = col.split(' ')[-2]
    if res == 'United':
        res = col.split(' ')[-5]
    return res
hotels['hotel_city'] = hotels['hotel_address'].apply(city_from_adress)
city_list = hotels['hotel_city'].unique()


# In[ ]:


# координаты центра
city_lat = []
city_lng = []
for i in city_list:
    lat = geocoder.opencage(i,  key='924f41fc864d4bed8c9b438bff7c67e1',timeout=None ).json['lat']
    city_lat.append(lat)
    lng = geocoder.opencage(i,  key='924f41fc864d4bed8c9b438bff7c67e1',timeout=None ).json['lng']
    city_lng.append(lng)
city_coords = pd.DataFrame(
    {'hotel_city': city_list,
    'city_lat': city_lat,
    'city_lng': city_lng
    })

hotels = hotels.merge(city_coords, on=['hotel_city'], how = 'left')
hotels ['coords'] = list(zip(hotels['lat'],hotels['lng'], hotels['city_lat'], hotels['city_lng']))


# In[ ]:


# расстояние до центра используя geopy.distance
def distance(col):
    coords_1 = col[:2]
    coords_2 = col[-2:]
    return geopy.distance.geodesic(coords_1, coords_2).m
hotels['distance_from_center'] = hotels ['coords'].apply(distance)


# In[ ]:


# название страны
def country_from_adress(col):
    res = col.split(' ')[-1]
    if res == 'Kingdom':
        res = 'United Kingdom'
    return res
hotels['hotel_country'] = hotels['hotel_address'].apply(country_from_adress)


# In[ ]:


# отзыв резидента
hotels['reviewer_nationality'] = hotels['reviewer_nationality'].apply(lambda x: x.strip())
hotels['home_review'] = np.where((hotels['hotel_country'] == hotels['reviewer_nationality']), 1, 0)


# **Название отеля**

# In[ ]:


# топ 100
hotel_name_list =(hotels['hotel_name'].value_counts(normalize=True).nlargest(100))
hotels['hotel_name'] = hotels['hotel_name'].apply(lambda x: x.strip() if x in hotel_name_list else 'other') 


# **Национальность автора**

# In[ ]:


# отсавляем топ 10 популярных
reviewer_nationality_list =(hotels['reviewer_nationality'].value_counts(normalize=True).nlargest(10))
hotels['reviewer_nationality'] = hotels['reviewer_nationality'].apply(lambda x: x if x in reviewer_nationality_list else 'other')


# **Теги**

# In[ ]:


#представление тегов списком
def tags(col):
    res = []
    tag_split = col.split(',')
    for tag in tag_split:
        reg = re.compile('[^a-zA-Z0-9 ]')
        res.append(reg.sub('', tag).strip())
    return res
hotels['new_tags'] = hotels['tags'].apply(tags)


# In[ ]:


# продолжительность пребывания
def night_number(col):
    for tag in col:
        if tag.split(' ')[0] == 'Stayed':
            return int(tag.split(' ')[1])
hotels['night_number'] = hotels['new_tags'].apply(night_number)
hotels['night_number'] = hotels['night_number'].fillna(hotels['night_number'].median())
hotels['night_number'] = hotels['night_number'].apply(lambda x: x if x <= 7 else 10)


# In[ ]:


# теги описание поездки
conditions = ['Couple','Solo traveler','Business trip','Family with young children','Group','Family with older children','With a pet']

hotels['Couple']=''
hotels['Solo traveler']=''
hotels['Business trip']=''
hotels['Family with young children'] = ''
hotels['Group']=''
hotels['Family with older children']=''
hotels['With a pet']=''

for col in hotels[conditions].columns:
    hotels[col]=hotels['new_tags'].apply(lambda x: 1 if col in x else 0)


# In[ ]:


# тип комнаты
def room_type(col):
    for tag in col:
        if tag.split(' ')[-1] == 'Room':
            return tag[:-4]
hotels['room_type'] = hotels['new_tags'].apply(room_type)

room_type =(hotels['room_type'].value_counts(normalize=True).nlargest(15))
hotels['room_type'] = hotels['room_type'].apply(lambda x: x.strip() if x in room_type else 'other')


# In[ ]:


# количество тегов
hotels['tags_length'] = hotels['new_tags'].apply(lambda x: len(x))


# **Возраст отзыва**

# In[ ]:


#выделяем количество дней
hotels['num_days_since_review'] = hotels['days_since_review'].apply(lambda x: int(x.split(' ')[0]))


# # Нормализация данных

# In[ ]:


# список численных признаков, требующих внимания
cols = ['additional_number_of_scoring', 
        'review_total_negative_word_counts',
        'review_total_positive_word_counts',
        'total_number_of_reviews_reviewer_has_given', 
        'total_number_of_reviews',
        'distance_from_center',
        'num_days_since_review',
        'average_score'
       ]
hotels[cols].hist(figsize=(20, 8))


# In[ ]:


#логарифмируем признаки
hotels['review_total_positive_word_counts'] = np.log(hotels['review_total_positive_word_counts']+ 1)

hotels['review_total_negative_word_counts'] = np.log(hotels['review_total_negative_word_counts']+ 1)

hotels['additional_number_of_scoring'] = np.log(hotels['additional_number_of_scoring'])
score_anomal_mean = hotels[hotels['additional_number_of_scoring']<3.1]['additional_number_of_scoring'].mean()
hotels['additional_number_of_scoring']=hotels['additional_number_of_scoring'].apply(lambda x: score_anomal_mean if x == 0 else x)

hotels['total_number_of_reviews_reviewer_has_given'] = np.log(hotels['total_number_of_reviews_reviewer_has_given'])
#total_anomal_mean = hotels[hotels['total_number_of_reviews_reviewer_has_given']>5.2]['total_number_of_reviews_reviewer_has_given'].mean()
#hotels['total_number_of_reviews_reviewer_has_given']=hotels['total_number_of_reviews_reviewer_has_given'].apply(lambda x: total_anomal_mean if x >= 5.19 else x)

hotels['total_number_of_reviews'] = np.log(hotels['total_number_of_reviews'])
#review_anomal_mean = hotels[hotels['total_number_of_reviews']<3.9]['total_number_of_reviews'].mean()
#hotels['total_number_of_reviews']=hotels['total_number_of_reviews'].apply(lambda x: review_anomal_mean if x < 3.9 else x)

distance_anomal_mean = hotels[hotels['distance_from_center']<1000000]['distance_from_center'].mean()
hotels['distance_from_center'] = hotels['distance_from_center'].apply(lambda x: x if x <= 20000 else distance_anomal_mean)
hotels['distance_from_center'] = np.log(hotels['distance_from_center'])

hotels['num_days_since_review'] = np.log(hotels['num_days_since_review']+1)

#скорректируем выброс в average_score
hotels['average_score']=hotels['average_score'].apply(lambda x: 6.5 if x == 5.2 else x)


# In[ ]:


# результат
hotels[cols].hist(figsize=(20, 8))


# # Кодирование признаков

# In[ ]:


# Просто LabelEncoder
hotels['reviewer_nationality'] = LabelEncoder().fit_transform(hotels['reviewer_nationality'])
hotels['night_number'] = LabelEncoder().fit_transform(hotels['night_number'])
hotels['hotel_country'] = LabelEncoder().fit_transform(hotels['hotel_country'])
hotels['room_type'] = LabelEncoder().fit_transform(hotels['room_type'])
hotels['hotel_name'] = LabelEncoder().fit_transform(hotels['hotel_name'])


# In[ ]:


hotels.describe()


# # Очистка данных

# In[ ]:


hotels = hotels.drop(['fill_lat','fill_lng','review_date','lat','lng','city_lat','city_lng'], axis=1)

hotels_columns = [s for s in hotels.columns if hotels[s].dtypes == 'object']
hotels.drop(hotels_columns, axis = 1, inplace=True)

hotels.info()


# # Корреляция

# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(hotels.corr(), annot=True, linewidths=.5, ax=ax)


# In[ ]:


# анализ мультиколлинеарности
pivot = hotels.corr()
pivot = pivot.drop('sample', axis=0)
pivot = pivot.drop('sample', axis=1)
for col in pivot:
    pivot[col] = pivot[col].apply(lambda x: np.nan if (abs(x) < 0.72 or x == 1) else x)
for col in pivot:
    pivot = pivot.dropna(how='all')
    pivot = pivot.dropna(how='all', axis='columns')

multi_corr_list = []
for lower_bound in np.linspace(0.98, 0.72, num=14):
    for col in pivot.columns:
        if pivot[col].max() > lower_bound or pivot[col].min() < -lower_bound:
            multi_corr_list.append(col)
print(set(multi_corr_list))


# In[ ]:


hotels = hotels.drop(['additional_number_of_scoring','dayofweek','year','pos_rev_neutral_mood_score','neg_rev_neg_mood_score'], axis=1)


# # Проверка важности признаков

# In[ ]:


# перед обучением модели явно укажем категориальный тип 
categorical_features = ['hotel_country',
            'hotel_name',
            'month', 
            'night_number', 
            'reviewer_nationality', 
            'room_type', 
            'weekend',
            'home_review',
            'tags_length',
            'Couple',
            'Solo traveler',
            'Business trip',
            'Family with young children',
            'Group','Family with older children',
            'With a pet'
            ]
hotels[categorical_features] = hotels[categorical_features].astype('category')


# In[ ]:


hotels_selective = hotels.copy()
train_data = hotels_selective.query('sample == 1').drop(['sample'], axis=1)
test_data = hotels_selective.query('sample == 0').drop(['sample'], axis=1)


# In[ ]:


num_cols = ['review_total_negative_word_counts',
            'review_total_positive_word_counts',
            'total_number_of_reviews_reviewer_has_given', 
            'total_number_of_reviews',
            'average_score',
            'pos_review_polarity',
            'pos_review_subjectivity',
            'neg_review_polarity',
            'neg_review_subjectivity',
            'distance_from_center',
            'num_days_since_review',
            'neg_rev_neutral_mood_score',
            'neg_rev_pos_mood_score',
            'neg_rev_mood_score',
            'pos_rev_neg_mood_score',
            'pos_rev_pos_mood_score',
            'pos_rev_mood_score'
            ]

cat_cols = ['hotel_country',
            'hotel_name',
            'month', 
            'night_number', 
            'reviewer_nationality', 
            'room_type', 
            'weekend',
            'home_review',
            'tags_length',
            'Couple',
            'Solo traveler',
            'Business trip',
            'Family with young children',
            'Group','Family with older children',
            'With a pet'
            ]

X = train_data.drop(['reviewer_score'], axis = 1)  
y = train_data['reviewer_score'] 

y=y.astype('int')


# In[ ]:


# числовые признаки
from sklearn.feature_selection import f_classif 
plt.rcParams['figure.figsize'] = (15,10)
imp_num = pd.Series(f_classif(X[num_cols], y)[0], index = num_cols)
imp_num.sort_values(inplace = True, ascending = False)
imp_num.plot(kind = 'barh')


# In[ ]:


#удаляем два самых невнятных
imoprtant_cols = []
del_num = 2
imoprtant_cols.extend(imp_num.head(len(imp_num)-del_num).index)


# In[ ]:


#категориальные признаки
from sklearn.feature_selection import chi2 # хи-квадрат
plt.rcParams['figure.figsize'] = (15,10)
imp_cat = pd.Series(chi2(X[cat_cols], y)[0], index=cat_cols)
imp_cat.sort_values(inplace = True, ascending = False)
imp_cat.plot(kind = 'barh')


# In[ ]:


#удаляем 4 невнятных
del_num = 4 
imoprtant_cols.extend(imp_cat.head(len(imp_cat)-del_num).index)
imoprtant_cols.append('reviewer_score')
imoprtant_cols.append('sample')


# In[ ]:


imoprtant_cols


# In[ ]:


hotels = hotels[imoprtant_cols]


# In[ ]:


hotels.head(3)


# # Обучение модели

# In[ ]:


# Теперь выделим тестовую часть
train_data = hotels.query('sample == 1').drop(['sample'], axis=1)
test_data = hotels.query('sample == 0').drop(['sample'], axis=1)

y = train_data.reviewer_score.values            # наш таргет
X = train_data.drop(['reviewer_score'], axis=1)


# In[ ]:


# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
#!pip install scikit-learn
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# In[ ]:


# проверяем
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape


# In[ ]:


# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели


# In[ ]:


# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)


# In[ ]:


# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MAPE:', mean_absolute_percentage_error(y_test, y_pred))


# In[ ]:


y_pred.shape


# In[ ]:


# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')


# In[ ]:


test_data.sample(10)


# In[ ]:


test_data = test_data.drop(['reviewer_score'], axis=1)


# In[ ]:


sample_submission


# In[ ]:


predict_submission = model.predict(test_data)


# In[ ]:


predict_submission.shape


# In[ ]:


sample_submission = sample_submission.dropna(how='any', axis=0)
sample_submission.shape


# In[ ]:


sample_submission['reviewer_score'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


sample_submission.head(10)

