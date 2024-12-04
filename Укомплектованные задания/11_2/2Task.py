"""
2 Задание

Создайте функцию которая на ввод принимает уже разделенные тренировочные
и тестовые выборки и возвращает на их основе предсказанные значенияя

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def get_data(name_of_col, data):
    features = data.drop(name_of_col, axis=1)
    target = data[name_of_col]
    features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=42,
                                                                                test_size=0.25)
    return features_train, features_test, target_train, target_test


data = pd.read_csv('Real estate.csv', index_col=0)
features_train, features_test, target_train, target_test = get_data('Y house price of unit area', data)


###########################################

def fit_and_predict(features_train, features_test, target_train, target_test):
    model = LinearRegression()
    model.fit(features_train, target_train)
    result = model.predict(features_test)
    return result


###########################################

result = fit_and_predict(features_train, features_test, target_train, target_test)
print(result)
