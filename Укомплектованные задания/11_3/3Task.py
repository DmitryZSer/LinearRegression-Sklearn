"""

Поситайте уравнение линейной регрессии для таблицы, все значения по умолчанию как в примере

#Названия столбцов можно взять функцией data.columns
X1 transaction date
X2 house age
X3 distance to the nearest MRT station
X4 number of convenience stores
X5 latitude
X6 longitude
Y house price of unit area
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Real estate.csv', index_col=0)

data.head()

# Из исходной таблицы берем все столбцы, кроме цены, его удаляем
features = data.drop('Y house price of unit area', axis=1)
features.head()

# Из исходной таблицы берем только цену
target = data['Y house price of unit area']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25,
                                                                            random_state=42)
model = LinearRegression()
model.fit(features_train, target_train)

result = model.predict(features_test)
# print(result)
print()

columns = data.columns[:-1]
for col in columns:
    print(col)
    print('-----------------------')
    print(data[col])
print('-------------------------------------\n')

coefficient = model.coef_
intercept = model.intercept_
linear_regression_eq = 0

print('\nSTARTSTARTSTARTSTART')
for i, coef in enumerate(coefficient):
    print('%%%%%%%%%%%')
    print(data[columns[i]])
    print('%%%%%%%%%%%')
    linear_regression_eq += coef * data[columns[i]].values
    print('----------------------------------------------------------\n')

res_linear_regression_eq = (linear_regression_eq + intercept)
print('\nRESULTRESULTRESULTRESULT')
print(res_linear_regression_eq)
print(len(res_linear_regression_eq))

from statistics import mean
print(mean(res_linear_regression_eq))

'''

Проверять по кол-ву строк, среднего значения

MIN: 37.05002077062724
MAX: 38.830346782120806
AVG: 37.97731604297083
'''
