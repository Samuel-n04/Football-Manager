import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle
import os

FEATURES = ['FatigueScore', 'MinutesPlayed', 'Accelerations', 'Distance',
            'SprintCount', 'SpeedAvg', 'PerformanceScore']
TARGET = 'InjuryRisk'


class InjuryPredictor:
    """
    Prédit le niveau de risque de blessure (Faible / Moyen / Eleve)
    avec deux modèles : Random Forest et XGBoost.
    """

    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.xgb_model = None  # created dynamically in train() based on class count
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance_rf: dict = {}
        self.results: dict = {}

    def train(self, df: pd.DataFrame) -> dict:
        available = [f for f in FEATURES if f in df.columns]
        X = df[available].fillna(0)
        y = self.label_encoder.fit_transform(df[TARGET])

        n_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_preds = self.rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)

        # XGBoost — set objective and num_class based on actual class count
        if n_classes <= 2:
            xgb_params = dict(n_estimators=100, random_state=42, eval_metric='logloss')
        else:
            xgb_params = dict(n_estimators=100, random_state=42, eval_metric='mlogloss',
                              objective='multi:softmax', num_class=n_classes)
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(X_train, y_train,
                           eval_set=[(X_test, y_test)],
                           verbose=False)
        xgb_preds = self.xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_preds)

        # Feature importance from RF
        self.feature_importance_rf = dict(zip(available, self.rf_model.feature_importances_))

        self.results = {
            'rf_accuracy': round(rf_acc, 4),
            'xgb_accuracy': round(xgb_acc, 4),
            'rf_report': classification_report(
                y_test, rf_preds,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'xgb_report': classification_report(
                y_test, xgb_preds,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'feature_importance': self.feature_importance_rf,
            'best_model': 'RandomForest' if rf_acc >= xgb_acc else 'XGBoost',
        }
        self.is_trained = True
        return self.results

    def predict(self, df: pd.DataFrame, model: str = 'best') -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Train the model first with .train(df).")
        available = [f for f in FEATURES if f in df.columns]
        X = df[available].fillna(0)
        if model == 'rf' or (model == 'best' and self.results.get('best_model') == 'RandomForest'):
            preds = self.rf_model.predict(X)
        else:
            preds = self.xgb_model.predict(X)
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Train the model first.")
        available = [f for f in FEATURES if f in df.columns]
        X = df[available].fillna(0)
        proba = self.rf_model.predict_proba(X)
        return pd.DataFrame(proba, columns=self.label_encoder.classes_)

    def save(self, path: str = 'stubs/injury_predictor.pkl') -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = 'stubs/injury_predictor.pkl') -> 'InjuryPredictor':
        with open(path, 'rb') as f:
            return pickle.load(f)
