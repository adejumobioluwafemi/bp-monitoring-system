from sklearn.pipeline import Pipeline
from .data_processor import get_data_preprocessor
from .model import Model


class TrainingPipeline:
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.model = Model()
        self.preprocessor = get_data_preprocessor(
            numerical_cols, categorical_cols)

    def fit(self, X, y):
        self.model.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model.classifier)
        ])
        self.model.fit(X, y)

    def get_model(self):
        return self.model
