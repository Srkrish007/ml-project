import numpy as np
from sklearn.linear_model import LinearRegression
from utils import load_data, split_data

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Model trained! Test score: {model.score(X_test, y_test):.3f}")
    return model

if __name__ == "__main__":
    model = train_model()
