import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from lightgbm import LGBMClassifier

_SUPPORTED_MODELS = {
    'multinomial_nb':      MultinomialNB,
    'bernoulli_nb':        BernoulliNB,
    'logistic_regression': LogisticRegression,
    'linear_svc':          LinearSVC,
    'random_forest':       RandomForestClassifier,
    'lightgbm':            LGBMClassifier,
}

class Classifier:

    def __init__(self, config: dict):
        self.config = config
        self.model_type = config['model']
        self.params = config.get('params', {})
        self.model = self._build_model()

    def _build_model(self):
        if self.model_type not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Classificador '{self.model_type}' não suportado. "
                f"Opções: {list(_SUPPORTED_MODELS)}"
            )
        return _SUPPORTED_MODELS[self.model_type](**self.params)

    def fit(self, X, y):
        self.model.fit(np.array(X), y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(np.array(X))

    def evaluate(self, X, y_true) -> dict:
        y_pred = self.predict(X)
        metrics = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'f1':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        report = classification_report(y_true, y_pred, zero_division=0)
        print("\n[classificador] métricas de avaliação:")
        print(f"  Acurácia : {metrics['accuracy']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}")
        print(f"  Precisão : {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print("\n[classificador] relatório por classe:")
        print(report)
        return metrics

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(
                f"'{self.model_type}' não suporta predict_proba. "
                "Use LinearSVC com decision_function ou troque de modelo."
            )
        return self.model.predict_proba(np.array(X))


