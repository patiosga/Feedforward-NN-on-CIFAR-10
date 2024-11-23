import numpy as np
from sklearn.preprocessing import StandardScaler
from read_data import read_data


def normalize_data(data):
    # Create a StandardScaler object
    scaler = StandardScaler(with_mean=True, with_std=True)  # z-score normalization
    # Fit the scaler to the data
    scaler.fit(data)
    # Transform the data using the fitted scaler
    data_normalized = scaler.transform(data)
    return data_normalized



def main():
    data, _ = read_data()
    print(data)  # (50000, 3072)
    data = normalize_data(data)
    print(data)  # (50000, 3072)


if __name__ == '__main__':
    main()