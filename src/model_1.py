import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import datetime
import os


class LinearRegressionModel:
    def __init__(self, eta=0.15, N_iterations=1000) -> None:
        self.eta = eta
        self.N_iterations = N_iterations
        self.w = None

    def model(self, w, X):
        """
        Модель для угадывания цен на дома 

        :param w: вектор весов
        :param X: матрица признаков обучающей выборки
        """
        return X @ w

    def loss(self, w, X, y):
        """
        Квадратичная функция потерь

        :param w: вектор коэфицентов весов
        :param X: матрица признаков обучающей выборки
        :param y: целевые значения
        """
        return np.mean((self.model(w, X) - y)**2)

    def gradient(self, w, X, y):
        """
        Градиент функции потерь (не совсем понятно как получить градиент в матричном виде, взял у иишки)

        :param w: веса модели
        :param X: матрица признаков
        :param y: целевые значения
        """
        n = X.shape[0]
        error = self.model(w, X) - y
        grad = (2 / n) * X.T @ error
        return grad

    def GD(self):
        self.X = self.data_reader('./data/features.csv')
        self.y = self.data_reader(
            './data/target.csv').to_numpy().ravel()  # type: ignore
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_train_split(
            self.X, self.y, 0.2)
        self.X_train_scaled, self.X_test_scaled = self.data_scaler(
            self.X_train, self.X_test)
        if self.w is None:
            self.w = np.zeros(self.X_train_scaled.shape[1])
        for _ in range(self.N_iterations):
            self.w -= self.eta * \
                self.gradient(self.w, self.X_train_scaled, self.y_train)

        self.model_report(self.X.columns.to_list())  # type: ignore

    def data_reader(self, filename: str):
        try:
            return read_csv(filename)
        except FileNotFoundError:
            return None

    def data_train_split(self, X, y, test_size: float) -> list:
        return train_test_split(X, y, test_size=test_size, random_state=1)

    def data_scaler(self, X_train, X_test) -> tuple:
        # Масштабирование признаков (обязательно для градиентного спуска)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Добавление столбца единиц для смещения (w0)
        X_train_scaled = np.c_[
            np.ones(X_train_scaled.shape[0]), X_train_scaled]
        X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]
        return X_train_scaled, X_test_scaled

    def model_report(self, columns: list, filename: str = './data/models.json') -> None:
        if self.w is None:
            return
        model_benchmark = {
            'model': f'{os.path.basename(__file__)}',
            'omega': self.w[1:].tolist(),
            'bias': self.w[0],
            'eta': self.eta,
            'Number of iterations': self.N_iterations,
            'feature_names': columns,
            'trained_on': f'{datetime.datetime.now()}',
            'mse_on_train': self.loss(self.w, self.X_train_scaled, self.y_train),
            'mse_on_test': self.loss(self.w, self.X_test_scaled, self.y_test)
        }

        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if len(data) > 1000:
                    data = []
        except FileNotFoundError:
            data = []

        with open(filename, 'w', encoding='utf-8') as file:
            data.append(model_benchmark)
            json.dump(data, file, indent=4)


if __name__ == '__main__':
    model = LinearRegressionModel()
    model.GD()
