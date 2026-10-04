# src/pipeline/member2_pipeline.py

import os
import sys

import pandas as pd

# Allow imports from src/
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from models.xgboost_classifier import (
    XGBoostIntrusionClassifier
)


# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = "data/processed"

TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"

LABEL_COLUMN = "Label"

MODEL_PATH = "saved-models/xgboost/xgboost_final.joblib"

RESULT_DIR = "outputs/results"


# ==================================================
# LOAD DATA
# ==================================================

def load_data():

    print("Loading datasets...")

    train = pd.read_csv(
        os.path.join(DATA_DIR, TRAIN_FILE)
    )

    validation = pd.read_csv(
        os.path.join(DATA_DIR, VALIDATION_FILE)
    )

    test = pd.read_csv(
        os.path.join(DATA_DIR, TEST_FILE)
    )

    print("\nDataset shapes:")

    print("Train      :", train.shape)
    print("Validation :", validation.shape)
    print("Test       :", test.shape)

    return train, validation, test


# ==================================================
# SPLIT FEATURES AND LABEL
# ==================================================

def prepare_data(train, validation, test):

    X_train = train.drop(
        columns=[LABEL_COLUMN]
    )

    y_train = train[LABEL_COLUMN]

    X_validation = validation.drop(
        columns=[LABEL_COLUMN]
    )

    y_validation = validation[LABEL_COLUMN]

    X_test = test.drop(
        columns=[LABEL_COLUMN]
    )

    y_test = test[LABEL_COLUMN]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    )


# ==================================================
# MAIN
# ==================================================

def main():

    # 1. Load data
    train, validation, test = load_data()

    # 2. Prepare X and y
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = prepare_data(
        train,
        validation,
        test
    )

    print("\nNumber of features:", X_train.shape[1])

    print("\nClasses:")
    print(y_train.value_counts())

    # 3. Create model
    model = XGBoostIntrusionClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,

        # Start with False.
        # We will experiment later.
        use_balanced_weights=False
    )

    # 4. Train
    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    print("\nTraining completed.")

    # 5. Validation evaluation
    print("\n========== VALIDATION ==========")

    validation_results = model.evaluate(
        X_validation,
        y_validation
    )

    # 6. Test evaluation
    print("\n========== TEST ==========")

    test_results = model.evaluate(
        X_test,
        y_test
    )

    # 7. Save model
    model.save(
        MODEL_PATH
    )

    print("\n========== COMPLETE ==========")


if __name__ == "__main__":
    main()