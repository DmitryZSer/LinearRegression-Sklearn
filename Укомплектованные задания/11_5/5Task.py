"""
Создайте функцию которая будет возвращать R2 модели и RMSE

На вход подается target_test - истинные значения в тестовой выборке и
предсказанные регрессивной моделью значения,
возвращает R2 модели и RMSE

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
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error


def get_r2_rmse(target_test, results):
    r2 = r2_score(target_test, results)
    rmse = root_mean_squared_error(target_test, results)
    return r2, rmse


r2, RMSE = get_r2_rmse(target_test, result)
print('R2:', r2)
print('RMSE:', RMSE)
