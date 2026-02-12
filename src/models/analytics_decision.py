from src.models.model_1 import LinearRegressionModel
import numpy as np
import json
import sys
import os
import datetime


class AnalyticsDecision(LinearRegressionModel):

    def decision(self) -> None:
        self.w = np.linalg.inv(
            self.X_train_scaled.T @ self.X_train_scaled) @ self.X_train_scaled.T @ self.y_train
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


if __name__ == "__main__":
    model = AnalyticsDecision()
    model.decision()
