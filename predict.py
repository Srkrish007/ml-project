"""
Prediction script - updated by teammate
"""
import numpy as np

def predict(model, X):
    """Make predictions with trained model"""
    predictions = model.predict(X)
    return predictions

def save_predictions(predictions, filename='predictions.csv'):
    """Save predictions to CSV"""
    np.savetxt(filename, predictions, delimiter=',')
    print(f"Predictions saved to {filename}")

if __name__ == "__main__":
    print("Prediction script ready!")
