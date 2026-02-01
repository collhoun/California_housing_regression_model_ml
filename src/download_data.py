from sklearn.datasets import fetch_california_housing
import pandas as pd
import os


def download_and_save_data() -> bool:
    try:
        california = fetch_california_housing()
        X = pd.DataFrame(california.data, columns=california.feature_names)
        Y = pd.Series(california.target, name='MedHouseVal')
        os.makedirs('data', exist_ok=True)
        X.to_csv('data/features.csv', index=False)
        Y.to_csv('data/target.csv', index=False)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    download_and_save_data()
