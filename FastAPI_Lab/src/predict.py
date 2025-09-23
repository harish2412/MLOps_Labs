from pathlib import Path
import joblib

MODEL_PATH = (Path(__file__).resolve().parent.parent / "model" / "breast_rf.pkl")

def predict_data(X):
    """
    Predict class probabilities using the trained RandomForest model.

    Parameters
    ----------
    X : array-like of shape (n_samples, 30)

    Returns
    -------
    numpy.ndarray of shape (n_samples, 2)
        Columns: P(class=0: malignant), P(class=1: benign).
    """
    model = joblib.load(MODEL_PATH)
    return model.predict_proba(X)
