import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Real estate.csv', index_col=0)

data.head()

#data.info()

# Из исходной таблицы берем все столбцы, кроме цены, его удаляем
features = data.drop('Y house price of unit area', axis=1)
features.head()

# Из исходной таблицы берем только цену
target = data['Y house price of unit area']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.25)
model = LinearRegression()
model.fit(features_train, target_train)

result = model.predict(features_test)
print(result)

#Пример корреляции получения
coefficient = model.coef_
intercept = model.intercept_

equation = f'y = {intercept}'
for i, coef in enumerate(coefficient):
    equation += f' + ({coef} * x{i+1})\n'
print('Уравнение линейной регрессии:', equation)
'''
Уравнение линейной регрессии: y = -12578.39520069035 + (4.153217904668599 * x1)
 + (-0.2695197671593577 * x2)
 + (-0.004574612634905828 * x3)
 + (1.0727862421636978 * x4)
 + (259.17389771709566 * x5)
 + (-18.191814432317983 * x6)
 '''
print()

#Пример получения p-value
import statsmodels.api as sm

features_train_sm = sm.add_constant(features_train)

model_sm = sm.OLS(target_train, features_train_sm).fit()

p_values = model_sm.pvalues

print('P-value для каждого параметра:')
for i, p_value in enumerate(p_values[1:]):
    print(f'Признак {i+1}: {p_value}')

print()

#Пример получения RMSE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from math import sqrt

rmse = root_mean_squared_error(target_test, result)

print('RMSE:', rmse)
print()

#Пример получения R2
from sklearn.metrics import r2_score

#До единицы, если 1 то ответ идеален, если отрицательная то качество модели плохое
r2 = r2_score(target_test, result)

print('R2:', r2)