from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Breast Cancer dataset and return features and labels.

    Returns:
        X (numpy.ndarray): Feature matrix of shape.
        y (numpy.ndarray): Binary labels where 0 = malignant and 1 = benign.
    """

    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    return X, y

def split_data(X, y):
    """
    Split the breast cancer data into training and test sets.
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Binary labels.
    Returns:
        X_train, X_test, y_train, y_test: The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)
    return X_train, X_test, y_train, y_test