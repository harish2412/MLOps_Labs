from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
from data import load_data, split_data

MODEL_PATH = (Path(__file__).resolve().parent.parent / "model" / "breast_rf.pkl")

def fit_model(X_train, y_train):
    """
    Train a Random Forest Classifier and save the model to a file.
    """
    rf_classifier = RandomForestClassifier(n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None,
        n_jobs=-1,
        random_state=12)
    rf_classifier.fit(X_train, y_train)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_classifier, MODEL_PATH)
    return rf_classifier

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    rf_classifier = fit_model(X_train, y_train)
