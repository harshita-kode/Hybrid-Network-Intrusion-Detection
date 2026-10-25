
import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.shap_data import load_shap_inputs
from src.analysis.explainability import XGBoostExplainer


# ============================================================
# SELECT CORRECTLY CLASSIFIED ATTACK
# ============================================================

def find_correct_attack_sample(
    model,
    X_test,
    y_test,
    label_classes,
    search_size=5000
):
    """
    Find a correctly classified attack sample
    for the individual SHAP demonstration.
    """

    search_size = min(search_size, len(X_test))

    X_search = X_test.iloc[:search_size]

    probabilities = model.predict_proba(X_search)

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    for index, predicted_index in enumerate(
        predicted_indices
    ):
        actual_class = str(y_test.iloc[index])

        predicted_class = str(
            label_classes[predicted_index]
        )

        if (
            actual_class != "BENIGN"
            and actual_class == predicted_class
        ):
            return index

    raise ValueError(
        "No correctly classified attack found "
        f"within the first {search_size} samples."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        model,
        X_test,
        y_test,
        feature_names,
        label_classes
    ) = load_shap_inputs()

    print("\nFinding a correctly classified attack...")

    sample_index = find_correct_attack_sample(
        model=model,
        X_test=X_test,
        y_test=y_test,
        label_classes=label_classes
    )

    sample = X_test.iloc[[sample_index]]

    actual_class = str(
        y_test.iloc[sample_index]
    )

    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    print("Creating SHAP explainer...")

    explainer = XGBoostExplainer(
        model=model,
        feature_names=feature_names,
        label_classes=label_classes,
        top_n=5
    )

    # --------------------------------------------------------
    # EXPLAIN PREDICTION
    # --------------------------------------------------------

    (
        predicted_class,
        confidence,
        explanation
    ) = explainer.explain(sample)

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n========== INDIVIDUAL SHAP ==========")

    print(f"Sample index    : {sample_index}")
    print(f"Actual class    : {actual_class}")
    print(f"Predicted class : {predicted_class}")
    print(f"Confidence      : {confidence:.2%}")

    print("\nTop contributing features:")

    for rank, contribution in enumerate(
        explanation.contributions,
        start=1
    ):
        direction = (
            "Toward predicted class"
            if contribution.shap_value > 0
            else "Away from predicted class"
            if contribution.shap_value < 0
            else "Neutral"
        )

        print(f"\n{rank}. {contribution.feature}")

        print(
            f"   Feature value : "
            f"{contribution.feature_value:.6f}"
        )

        print(
            f"   SHAP value    : "
            f"{contribution.shap_value:+.6f}"
        )

        print(
            f"   Contribution  : {direction}"
        )

    print("\n=====================================")


if __name__ == "__main__":
    main()
