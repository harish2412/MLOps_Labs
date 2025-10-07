import pandas as pd
import os
import pickle
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load data from a CSV file
def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"]) if "target" in ds.frame.columns else ds.data
    y = pd.Series(ds.target, name="target")
    return X, y

# Preprocess the data
def data_preprocessing(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# Build and save a logistic regression model
def build_model(data, filename):
    X_train, X_test, y_train, y_test = data

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    clf = GridSearchCV(rf, grid, cv=3, n_jobs=-1, scoring="accuracy")
    clf.fit(X_train, y_train)

    # Save best estimator under dags/model/<filename>
    model_dir = Path(__file__).resolve().parent.parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / filename

    with open(output_path, "wb") as f:
        pickle.dump(clf.best_estimator_, f)

    print(f"Saved best RandomForest model to {output_path}")
    print(f"Best CV score: {clf.best_score_:.4f}")
    return str(output_path)

# Load a saved logistic regression model and evaluate it
def load_model(data, filename):
    X_train, X_test, y_train, y_test = data
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    # Load the saved model from a file
    loaded_model = pickle.load(open(output_path, 'rb'))

    # Make predictions on the test data and print the model's score
    predictions = loaded_model.predict(X_test)
    print(f"Model score on test data: {loaded_model.score(X_test, y_test)}")

    return predictions[0]


if __name__ == '__main__':
    x = load_data()
    x = data_preprocessing(x)
    build_model(x, 'model.sav')