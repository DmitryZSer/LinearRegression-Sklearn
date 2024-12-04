"""
Создание Функции для подсчета P-value

Создайте функцию выводящую все p-value для передаваемой таблицы

В функцию должны передаваться тренировочные предикторы и целевой признак вместе с самой таблицей

test_size = 0.25, random_state=42
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

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


def get_pvalues(features_train, target_train, data):
    features_train_sm = sm.add_constant(features_train)

    model_sm = sm.OLS(target_train, features_train_sm).fit()

    p_values = model_sm.pvalues

    columns = data.columns[:-1]

    print('P-value для каждого параметра:')
    for i, p_value in enumerate(p_values[1:]):
        print(f'Признак столбца "{columns[i]}": {p_value}')


get_pvalues(features_train, target_train, data)
