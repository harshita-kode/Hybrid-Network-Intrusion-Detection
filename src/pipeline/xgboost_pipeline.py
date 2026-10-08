import os
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.insert(
    0,
    os.path.join(PROJECT_ROOT, "src")
)


from models.xgboost_classifier import (
    XGBoostIntrusionClassifier
)

from models.confidence import (
    route_by_confidence,
    DEFAULT_CONFIDENCE_THRESHOLD
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "Data_V2"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "results"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved-models",
    "xgboost_final.joblib"
)


# ============================================================
# DATA FILES
# ============================================================

TRAIN_FILE = "known_train.parquet"
VALIDATION_FILE = "known_validation.parquet"
TEST_FILE = "known_test.parquet"

LABEL_COLUMN = "Label"
METADATA_COLUMNS = ["source_file"]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading Data_V2 datasets...")

    train = pd.read_parquet(
        os.path.join(
            DATA_DIR,
            TRAIN_FILE
        )
    )

    validation = pd.read_parquet(
        os.path.join(
            DATA_DIR,
            VALIDATION_FILE
        )
    )

    test = pd.read_parquet(
        os.path.join(
            DATA_DIR,
            TEST_FILE
        )
    )

    print("\nDataset shapes:")
    print("Train      :", train.shape)
    print("Validation :", validation.shape)
    print("Test       :", test.shape)

    return train, validation, test


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    train,
    validation,
    test
):

    columns_to_drop = [
        LABEL_COLUMN,
        *METADATA_COLUMNS
    ]

    X_train = train.drop(
        columns=columns_to_drop
    )

    y_train = train[LABEL_COLUMN]

    X_validation = validation.drop(
        columns=columns_to_drop
    )

    y_validation = validation[LABEL_COLUMN]

    X_test = test.drop(
        columns=columns_to_drop
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


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    train, validation, test = load_data()

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

    print(
        "\nNumber of features:",
        X_train.shape[1]
    )

    print("\nKnown classes:")
    print(
        y_train.value_counts()
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = XGBoostIntrusionClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_balanced_weights=False
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    print("\nTraining completed.")

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\n========== VALIDATION =========="
    )

    validation_results = model.evaluate(
        X_validation,
        y_validation
    )

    # ========================================================
    # TEST
    # ========================================================

    print(
        "\n========== TEST =========="
    )

    test_results = model.evaluate(
        X_test,
        y_test
    )

    # ========================================================
    # CONFIDENCE ROUTING
    # ========================================================

    print(
        "\n========== CONFIDENCE ROUTING =========="
    )

    predictions, confidence, probabilities = (
        model.predict_with_confidence(
            X_test
        )
    )

    confidence, status = route_by_confidence(
        predictions,
        probabilities,
        threshold=DEFAULT_CONFIDENCE_THRESHOLD
    )

    correct = (
        predictions == y_test.to_numpy()
    )

    routed_mask = (
        status == "LOW_CONFIDENCE"
    )

    routed_samples = routed_mask.sum()

    total_errors = (~correct).sum()

    errors_caught = (
        routed_mask & ~correct
    ).sum()

    correct_samples_routed = (
        routed_mask & correct
    ).sum()

    routing_percentage = (
        routed_samples /
        len(y_test) *
        100
    )

    error_capture_rate = (
        errors_caught /
        total_errors *
        100
        if total_errors > 0
        else 0
    )

    print(
        f"Confidence threshold : "
        f"{DEFAULT_CONFIDENCE_THRESHOLD}"
    )

    print(
        f"Low-confidence samples : "
        f"{routed_samples:,}"
    )

    print(
        f"Routing percentage : "
        f"{routing_percentage:.4f}%"
    )

    print(
        f"Total classification errors : "
        f"{total_errors:,}"
    )

    print(
        f"Errors caught : "
        f"{errors_caught:,}"
    )

    print(
        f"Correct samples routed : "
        f"{correct_samples_routed:,}"
    )

    print(
        f"Error capture rate : "
        f"{error_capture_rate:.2f}%"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        MODEL_PATH
    )

    # ========================================================
    # SAVE TEST CLASSIFICATION REPORT
    # ========================================================

    from sklearn.metrics import classification_report

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "xgboost_test_classification_report.csv"
        )
    )

    # ========================================================
    # SAVE CONFIDENCE ROUTING
    # ========================================================

    confidence_results = pd.DataFrame({
        "actual_class": y_test.to_numpy(),
        "predicted_class": predictions,
        "confidence": confidence,
        "status": status
    })

    confidence_results.to_csv(
        os.path.join(
            RESULT_DIR,
            "xgboost_confidence_routing.csv"
        ),
        index=False
    )

    # ========================================================
    # SAVE CONFIDENCE SUMMARY
    # ========================================================

    confidence_summary = pd.DataFrame({
        "metric": [
            "confidence_threshold",
            "total_test_samples",
            "high_confidence_samples",
            "low_confidence_samples",
            "routing_percentage",
            "total_classification_errors",
            "errors_caught",
            "correct_samples_routed",
            "error_capture_rate"
        ],
        "value": [
            DEFAULT_CONFIDENCE_THRESHOLD,
            len(y_test),
            (~routed_mask).sum(),
            routed_samples,
            routing_percentage,
            total_errors,
            errors_caught,
            correct_samples_routed,
            error_capture_rate
        ]
    })

    confidence_summary.to_csv(
        os.path.join(
            RESULT_DIR,
            "xgboost_confidence_summary.csv"
        ),
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n========== PIPELINE COMPLETE =========="
    )

    print(
        "\nFinal test metrics:"
    )

    for metric, value in test_results.items():
        print(
            f"{metric:15s}: {value:.4f}"
        )


if __name__ == "__main__":
    main()