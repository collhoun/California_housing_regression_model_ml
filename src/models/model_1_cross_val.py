from src.models.model_1 import LinearRegressionModel
import numpy as np
import json
import datetime
import sys
import os


class LinearRegressionModelCrossVal(LinearRegressionModel):
    def __init__(self, eta: float = 0.15, N_iterations: int = 1000, data_filename: str = './data/features.csv', target_filename: str = './data/target.csv', test_size: float = 0.2, k: int = 5) -> None:
        self.eta = eta
        self.N_iterations = N_iterations
        self.w = None
        self.k = k
        self.X = self.data_reader(data_filename)
        self.y = self.data_reader(
            target_filename).to_numpy().ravel()  # type: ignore
        self.X_scaled = self.data_scaler(
            self.X)

    def GD(self) -> None:
        """
        Градиентный спуск с k-fold кросс-валидацией
        """
        fold_size = int(self.X.shape[0] / self.k)  # type: ignore
        mse_scores = []

        for i in range(self.k):
            self.w = np.zeros(self.X_scaled.shape[1])  # type: ignore

            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.k - \
                1 else self.X_scaled.shape[0]  # type: ignore
            X_val = self.X_scaled[val_start:val_end]
            y_val = self.y[val_start:val_end]

            # обучающие фолды
            X_train = np.concatenate(
                [self.X_scaled[:val_start], self.X_scaled[val_end:]])
            y_train = np.concatenate([self.y[:val_start], self.y[val_end:]])

            for _ in range(self.N_iterations):
                self.w -= self.eta * self.gradient(self.w, X_train, y_train)

            # оценка на валид фолде
            val_mse = self.loss(self.w, X_val, y_val)
            mse_scores.append(val_mse)

        # обучение на всех данных с лучшими параметрами
        self.w = np.zeros(self.X_scaled.shape[1])  # type: ignore
        for _ in range(self.N_iterations):
            self.w -= self.eta * self.gradient(self.w, self.X_scaled, self.y)

        self.model_report(self.X.columns.to_list())  # type: ignore

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
            'feature_names': columns,
            'trained_on': f'{datetime.datetime.now()}',
            'mse': self.loss(self.w, self.X_scaled, self.y)
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


if __name__ == "__main__":
    model = LinearRegressionModelCrossVal()
    model.GD()
