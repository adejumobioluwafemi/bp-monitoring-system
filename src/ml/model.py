# src/ml/model.py
from sklearn.ensemble import RandomForestClassifier
import joblib


class Model:
    def __init__(self):
        self.pipeline = None
        self.classifier = RandomForestClassifier(
            n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def save(self, filepath):
        joblib.dump(self.pipeline, filepath)

    def load(self, filepath):
        self.pipeline = joblib.load(filepath)
