#!/usr/bin/env python
# coding: utf-8

# <a id="EDA"></a>
# # EDA - exploratory data analysis

# #### Основные библиотеки: seaborn, matplotlib, plotly

# 1. Посмотреть на отсутствующие значения в процентах, подумать, какие фичи можно вовсе убрать
# 2. Подумать, одинаковое ли распределение на train и test, сколько NA в них, как будем обучаться с учётом этой информации (иногда на тесте могут быть NA, которых не было в тесте), lightgbm хорошо работает с NA
# 3. Для временных рядов: есть ли тренд? Если есть, то вычесть его (запомнить), затем сделать предсказание и домножить на тренд.
# 4. Выбросы - есть ли? Посмотреть внимательно на pairplot
# 5. Посмотреть на гистограммы (либо оценки плотностей) в зависимости от категориальных данных, есть ли влияние, нужно ли его учитывать, и как?
# 6. Некоторые интересные тренды можно добавлять в качестве доп. переменных (особенно сезонные!!!)
# 7. Если каких-то особенных индивидов очень мало (небольшая неоднородность), то имеет смысл для них построить отдельную модель
# 8. Корреляции, убрать слишком сильно коррелирующие

# <a id="Валидация"></a>
# # Валидация

# ### Стратегии валидации:
# * Holdout - просто разбиваем на train и test
# * KFold - самый обычный
# * GroupKFold - в зависимости от группы, например, если сделали кластеризацию
# * StratifiedGroupKFold - пытается строить сбалансированные фолды 
# * TimeSeriesSplit
# * LOOCV 

# Корректна ли валидация? 
# Нужно попытаться сделать процедуру кросс-валидации  наиболее похожей на данный train-test

# <a id="Feature generation"></a>
# # Feature generation

# Одно из основных -- это statistical features

# #### Категориальные переменные: 
# * label encoding vs one-code-encoding? Если много категорий, то получается очень много нулей. Альтернатива - либо объединить маленькие группы, либо label encoding и подать модели, с пометкой, что это категориальная переменная.
# * numerical aggregation -- min/max/std/meadian/mean
# * Target Encoding -- закодировать средним targetа, аккуратно использовать (возможен leakage)

# #### Переменнные относящиеся ко времени:
# * месяц
# * год
# * день недели
# * выходной ли 
# * час 
# 
# И многие другие. Если замеры идут через разные промежутки времени, то можно ещё учитывать разницу во времени с предыдущим, число записей за некоторый промежуток времени, предыдущее время фиксации и так далее...

# <a id="Feature Selection"></a>
# # Feature Selection

# 1. Forward - Backward pass - постепенно прибавляем-удаляем фичи, и оставляем лучшие
# 2. Linear Regression - обучить LASSO модель, оставить фичи с ненулевыми коэффициентами (не забыть нормализовать фичи)
# 3. Genetic feature selection 
# 4. SHAP (считает "важность" фичи) + Boruta (рандомно перемешивает значения) = BorutaShap, считается весьма надежным способом
# 5. Class weight retraining - решение проблемы несбалансированных классов
# 

# * Blending - обучаем несколько моделей и потом усредняем предсказания, в случае классификации это Voting
# * Stacking - обучимся лесами или бустингом, запишем вероятности n моделей в матрицу, добавим ещё фичи (перемножить, сложить, поделить каждый с каждым), затем обучаем ещё одну модель на этом -- lightgbm или ridge регрессию 
# 
# #### Откуда брать разные модели? 
# External bagging - берем каждый к-ый объект, повторяем и усредняем

# #### Для линейных моделей
# * можно добавить новые фичи, такие как abs(x-k), sign(x-a), (x-a)**b
# 

# #### Изотононическая регрессия 
# Добавить как фичу, если переменная монотонно не убывает 

# <a id="Tips and tricks"></a>
# # Tips and tricks
# * Фиксировать random state и смотреть, стабильно ли поведение модели в зависимости от него
# * Можно усреднять не только значение метрики для кросс-валидации, но и сами результаты - вероятности и проч.
# * holdout для гиперпараметров обычно больше не делают
# * для цикличных переменных можно закодировать с помощью двух фичей - cos-sin
# * сделать get_features для train и test сразу и вообще для всего - для feature, для split, для train, и для predict
# * winsorization - всё что больше каких-то границ назначаем одно граничное значение
# * target transformation - попробовать предсказывать не только y, но и y^2, log(y+1), sqrt(y) и т. д.
# * postprocessing 
# * своя собственная модель для каждого индивида

# Не совсем понял:
# 1. SMAPE becames very close to MAE when using log of target
# 2. Threshold optimization
# 3. Rank average

# by the way: инсайд, надо подумать, как сделать фичи с окном в 7 дней
