"""

Создайте функцию которая сравнивает натуральное RMSE и RMSE “глупой модели”.
Такая модель просто берет среднее значение из target_test и на все будущие
значения отвечает одним числом – средним.

На вход подается target_test - истинные значения в тестовой выборке и
предсказанные регрессивной моделью значения,
возвращает разницу между натуральным RMSE и RMSE “глупой модели”

from sklearn.metrics import root_mean_squared_error
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Real estate.csv', index_col=0)
data.head()

features = data.drop('Y house price of unit area', axis=1)
features.head()
target = data['Y house price of unit area']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25,
                                                                            random_state=42)

model = LinearRegression()
model.fit(features_train, target_train)
result = model.predict(features_test)

###########################################
from sklearn.metrics import root_mean_squared_error
from statistics import mean


def difference_rmse(target_test, result):
    rmse = root_mean_squared_error(target_test, result)
    dumb_rmse = root_mean_squared_error(target_test, [mean(target_test)] * len(target_test))
    return dumb_rmse - rmse


difference = difference_rmse(target_test, result)
print(difference)
