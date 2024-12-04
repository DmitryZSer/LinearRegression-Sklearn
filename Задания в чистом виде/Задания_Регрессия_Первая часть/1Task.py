'''
1 Задание
из таблицы Real estate.csv в качестве целевого признака возмите колонку
с информацией о ближайшей станции метро, т.е. "X3 distance to the nearest MRT station"
Разделите данные на тренировочную и тестовую в соотношении 4 к 1,
обучите модель и предскажите значения
'''

###########################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Real estate.csv', index_col=0)
data.head()

features = data.drop('X3 distance to the nearest MRT station', axis=1)
features.head()

target = data['X3 distance to the nearest MRT station']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.20)
model = LinearRegression()
model.fit(features_train, target_train)

result = model.predict(features_test)
print(result)
#print(len(result))
###########################################
'''
Тестировать на основе кол-во элементов, R2
'''
print(len(result))

from sklearn.metrics import r2_score

#До единицы, если 1 то ответ идеален, если отрицательная то качество модели плохое
r2 = r2_score(target_test, result)
print('R2:', r2)

r2_list = []
for _ in range(100000):
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.20)
    model = LinearRegression()
    model.fit(features_train, target_train)

    result = model.predict(features_test)

    r2_list.append(r2_score(target_test, result))

print('Min:', min(r2_list))
print('Max:', max(r2_list))
print('Ang:', sum(r2_list)/len(r2_list))

'''
83 # Кол-во записей
R2: 0.8796088706549011
Min: 0.42544307860157105 # Min: -0.038426467916139906 ?
Max: 0.9137113113005062
Ang: 0.7640714545258294
'''