from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMClassifier:
    def __init__(self, C=1.0, kernel='linear', gamma='scale', random_state=42):
        if kernel not in {'linear', 'rbf'} or C <= 0:
            raise ValueError('SVM needs positive C and a linear or rbf kernel.')
        self.config = {'C': C, 'kernel': kernel, 'gamma': gamma, 'random_state': random_state}
        self.estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**self.config)),
        ])

    def fit(self, features, labels):
        self.estimator.fit(features, labels)
        return self

    def predict(self, features):
        return self.estimator.predict(features)

    def decision_function(self, features):
        return self.estimator.decision_function(features)


class AlexNet:
    def __init__(self) -> None:
        pass
