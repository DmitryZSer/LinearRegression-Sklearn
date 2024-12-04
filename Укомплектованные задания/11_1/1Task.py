"""
1 Задание
Создайте функцию которая будет на ввод принимать таблицу и название 1,
столбца, который является целевым признаком. Внутри функции должно выполняться деление
данных на тренировочные и тестовые с помощью train_test_split и возвращать 4 переменные

Соотношение деления 1 к 3
Также пропишите random_state = 42
"""
###########################################

import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(name_of_col, data):
    features = data.drop(name_of_col, axis=1)
    target = data[name_of_col]
    features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=42,
                                                                                test_size=0.25)
    return features_train, features_test, target_train, target_test


data = pd.read_csv('Real estate.csv', index_col=0)

features_train, features_test, target_train, target_test = get_data('Y house price of unit area', data)

print(features_train)
print(features_test)
print(target_train)
print(target_test)

###########################################
