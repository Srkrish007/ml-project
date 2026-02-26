# ML Project

Simple machine learning project for Git workflow demo.

## Files
- \`train.py\` - Model training
- \`predict.py\` - Predictions (updated by teammate)
- \`utils.py\` - Utilities
- \`config.py\` - Config (to be added)

## Usage
```bash
pip install numpy scikit-learn
python train.py
python predict.py





c
git status


clear
cat > predict.py << 'EOF'
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
