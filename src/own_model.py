import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import datetime

X = read_csv('data/features.csv')
y = read_csv('data/target.csv').to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Масштабирование признаков (обязательно для градиентного спуска)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Добавление столбца единиц для смещения (w0)
X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]


def model(w, X):
    """
    Модель для угадывания цен на дома 

    :param w: вектор весов
    :param X: матрица признаков обучающей выборки
    """
    return X @ w


def loss(w, X, y):
    """
    Квадратичная функция потерь

    :param w: вектор коэфицентов весов
    :param X: матрица признаков обучающей выборки
    :param y: целевые значения
    """
    return np.mean((model(w, X) - y)**2)


def gradient(w, X, y):
    """
    Градиент функции потерь (не совсем понятно как получить градиент в матричном виде)

    :param w: веса модели
    :param X: матрица признаков
    :param y: целевые значения
    """
    n = X.shape[0]
    error = model(w, X) - y
    grad = (2 / n) * X.T @ error
    return grad


w = np.zeros(X_train_scaled.shape[1])
eta = 0.15  # шаг обучения
N = 1000  # количество шагов алгоритма
# значения коэфициента для вычисления экспоненциального скользящего среднего
lm = 0.02
Qe = loss(w, X_train_scaled, y_train)


for _ in range(N):
    Qe = lm * loss(w, X_train_scaled, y_train) + (1 - lm) * Qe
    w -= eta * gradient(w, X_train_scaled, y_train)

data = {
    'omega': w[1:].tolist(),
    'bias': w[0],
    'eta': eta,
    'Number of iterations': N,
    'lm': 0.02,
    'feature_names': X.columns.to_list(),
    'trained_on': f'{datetime.datetime.now()}',
    'mse_on_train': loss(w, X_train_scaled, y_train),
    'mse_on_test': loss(w, X_test_scaled, y_test)
}


with open('./data/model.json', 'a', encoding='utf-8') as file:
    json.dump(data, file)
