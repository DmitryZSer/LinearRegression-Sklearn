'''
https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data

Используя данные из CarPrices.csv научите модель
предсказывать цену автомобиля.

Не учитывайте столбец марки машины "CarName"

Узнать типы данных в столбцых можно data.dtypes
Получить список уникальных значений можно с помощью data['имя столбца'].unique()

Для столбцов с нечисловыми (категоричными) значениями примените

from sklearn.preprocessing import LabelEncoder
label_encorder = LabelEncoder()
for name in name_of_object_labels:
    data[name] = label_encorder.fit_transform(data[name])
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('CarPrices.csv', index_col=0)
data = data.drop(['CarName'], axis=1)

data.head()

# data['имя столбца'].unique()
# data = pd.DataFrame({'color': ['blue', 'green', 'green', 'red']})

print(data.dtypes)

# Поиск имен столбцов с категоричными значениями
name_of_object_labels = []
for name, types in data.dtypes.items():
    if types == object:
        name_of_object_labels.append(name)
print(name_of_object_labels)

print('--------------------')

# Преобразование всех столбцов с категоричными значениями
from sklearn.preprocessing import LabelEncoder

label_encorder = LabelEncoder()
for name in name_of_object_labels:
    data[name] = label_encorder.fit_transform(data[name])
print(data.head(10))

# Из исходной таблицы берем все столбцы, кроме цены, ее удаляем
features = data.drop('price', axis=1)

# Из исходной таблицы берем только цену
target = data['price']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25)
model = LinearRegression()
model.fit(features_train, target_train)

result = model.predict(features_test)
print(result)

# Пример получения R2
from sklearn.metrics import r2_score

# До единицы, если 1 то ответ идеален, если отрицательная то качество модели плохое
r2 = r2_score(target_test, result)

print('R2:', r2)

'''

Min: -0.3614102576823883
Max: 0.9483059944390486
Ang: 0.8113739605556026

'''

