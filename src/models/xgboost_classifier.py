import os
import joblib
import numpy as np

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.utils.class_weight import compute_sample_weight


class XGBoostIntrusionClassifier:

    def __init__(
        self,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_balanced_weights=False
    ):

        self.use_balanced_weights = use_balanced_weights
        self.random_state = random_state

        self.label_encoder = LabelEncoder()

        self.model = XGBClassifier(
            objective="multi:softprob",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1
        )

        self.feature_names = None

    def fit(
        self,
        X_train,
        y_train,
        X_validation=None,
        y_validation=None
    ):

        self.feature_names = list(X_train.columns)

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        sample_weights = None

        if self.use_balanced_weights:
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train_encoded
            )

        eval_set = None

        if X_validation is not None and y_validation is not None:
            y_validation_encoded = self.label_encoder.transform(
                y_validation
            )

            eval_set = [
                (X_validation, y_validation_encoded)
            ]

        self.model.fit(
            X_train,
            y_train_encoded,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False
        )

        return self

    def predict(self, X):

        encoded_predictions = self.model.predict(X)

        return self.label_encoder.inverse_transform(
            encoded_predictions.astype(int)
        )

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def predict_with_confidence(self, X):

        probabilities = self.predict_proba(X)

        predicted_indices = np.argmax(
            probabilities,
            axis=1
        )

        confidence = np.max(
            probabilities,
            axis=1
        )

        predicted_classes = self.label_encoder.inverse_transform(
            predicted_indices
        )

        return (
            predicted_classes,
            confidence,
            probabilities
        )

    def evaluate(self, X, y):

        predictions = self.predict(X)

        accuracy = accuracy_score(
            y,
            predictions
        )

        precision = precision_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )

        macro_f1 = f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )

        print("\n========== MODEL PERFORMANCE ==========")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"Weighted F1    : {weighted_f1:.4f}")
        print(f"Macro F1       : {macro_f1:.4f}")

        print(
            "\n========== CLASSIFICATION REPORT ==========\n"
        )

        print(
            classification_report(
                y,
                predictions,
                zero_division=0
            )
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1
        }

    def save(self, path):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        joblib.dump(
            self,
            path
        )

        print(
            f"\nModel saved to: {path}"
        )

    @staticmethod
    def load(path):

        model = joblib.load(path)

        print(
            f"\nModel loaded from: {path}"
        )

        return model