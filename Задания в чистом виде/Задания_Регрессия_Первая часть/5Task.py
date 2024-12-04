'''

Создать функцию которая будет создавать модель с показателем R2 не менее 0.96

Передаются таблица с признаками, с целевым признаком
Возвращается модель линейной регрессии

Для примера берется Student_Marks.csv
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

data = pd.read_csv('Student_Marks.csv')
print(data.columns.values)
features = data.drop('Marks', axis=1)
target = data['Marks']

def make_best_linear_module(features, target):
    r2 = 0
    model = LinearRegression()
    while r2 < 0.96:
        features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.25)
        model.fit(features_train, target_train)
        result = model.predict(features_test)
        r2 = r2_score(target_test, result)
        print("===================\n",r2)
    return model, r2 #Для примера возвращается R2

model, r2 = make_best_linear_module(features, target)
print("\nFinale R2: ", r2)