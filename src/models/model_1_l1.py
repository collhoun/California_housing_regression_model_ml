from src.models.model_1 import LinearRegressionModel
import numpy as np
import os
import sys
import datetime
import json


class LinearRegressionModelL1(LinearRegressionModel):
    def __init__(self, eta: float = 0.15, N_iterations: int = 1000, data_filename: str = './data/features.csv', target_filename: str = './data/target.csv', test_size: float = 0.2, lm_l1=0.3) -> None:
        super().__init__(eta, N_iterations, data_filename, target_filename, test_size)
        self.lm_l1 = lm_l1

    def loss(self, w, X, y) -> float:
        """
        Квадратичная функция потерь с L1 регуляризацией

        :param w: вектор коэфицентов весов
        :param X: матрица признаков обучающей выборки
        :param y: целевые значения
        """
        return np.mean((self.model(w, X) - y)**2) + self.lm_l1 * np.sum(list(map(abs, w[1:])))

    def gradient(self, w, X, y):
        """
        Градиент функции потерь с L1 регуляризацией

        :param w: веса модели
        :param X: матрица признаков
        :param y: целевые значения
        """
        n = X.shape[0]
        error = self.model(w, X) - y
        grad = (2 / n) * X.T @ error
        w_l = w.copy()
        w_l[0] = 0
        return grad + self.lm_l1 * np.sign(w_l)

    def model_report(self, columns: list, filename: str = './data/models.json') -> None:
        if self.w is None:
            return

        current_module = sys.modules[self.__class__.__module__]
        current_filename = os.path.basename(
            current_module.__file__)  # type: ignore
        model_benchmark = {
            'model': current_filename,
            'omega': self.w[1:].tolist(),
            'bias': self.w[0],
            'eta': self.eta,
            'Number of iterations': self.N_iterations,
            'lm_l2': self.lm_l1,
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
    model = LinearRegressionModelL1()
    lm_l1 = 0.1
    step = 0.1
    while lm_l1 < 2:
        model.lm_l1 = lm_l1
        model.GD()
        lm_l1 += step
