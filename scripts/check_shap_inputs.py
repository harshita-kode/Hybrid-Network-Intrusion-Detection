import os

import joblib
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved-models",
    "xgboost_final.joblib"
)

TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "classifier_test.csv"
)


def main():

    print("Loading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Loaded model type:",
        type(model)
    )

    print("Loading test data...")

    test = pd.read_csv(
        TEST_PATH
    )

    # These columns were explicitly excluded during
    # XGBoost training.
    columns_to_drop = [
        column
        for column in [
            "Label",
            "source_file"
        ]
        if column in test.columns
    ]

    X_test = test.drop(
        columns=columns_to_drop
    )

    y_test = test["Label"]

    print(
        "\n========== INPUT CHECK =========="
    )

    print(
        "Test feature count:",
        X_test.shape[1]
    )

    # XGBoost stores information about the number of
    # features expected by the trained model.
    model_feature_count = (
        model.get_booster().num_features()
    )

    print(
        "Model feature count:",
        model_feature_count
    )

    if X_test.shape[1] != model_feature_count:

        raise ValueError(
            "Feature count mismatch. "
            f"Model expects {model_feature_count}, "
            f"but test data contains "
            f"{X_test.shape[1]}."
        )

    # Check whether XGBoost retained feature names.
    booster_feature_names = (
        model.get_booster().feature_names
    )

    if booster_feature_names is not None:

        print(
            "Model contains feature names: YES"
        )

        exact_match = (
            list(X_test.columns)
            == booster_feature_names
        )

        print(
            "Exact feature match:",
            exact_match
        )

        if not exact_match:

            model_features = set(
                booster_feature_names
            )

            test_features = set(
                X_test.columns
            )

            missing = (
                model_features
                - test_features
            )

            extra = (
                test_features
                - model_features
            )

            print(
                "Missing from test:",
                missing
            )

            print(
                "Extra in test:",
                extra
            )

            if model_features == test_features:

                print(
                    "Features are the same but "
                    "column order differs."
                )

                X_test = X_test[
                    booster_feature_names
                ]

            else:

                raise ValueError(
                    "Test features do not match "
                    "model features."
                )

    else:

        print(
            "Model contains feature names: NO"
        )

        print(
            "Feature count matches. "
            "Using classifier_test.csv column "
            "order from the training pipeline."
        )

    print(
        "\nTest samples:",
        len(X_test)
    )

    print(
        "Label samples:",
        len(y_test)
    )

    print(
        "\nInput verification completed successfully."
    )


if __name__ == "__main__":
    main()