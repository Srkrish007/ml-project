import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    return X, y

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
