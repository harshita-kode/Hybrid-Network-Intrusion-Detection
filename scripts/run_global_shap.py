
import sys
from pathlib import Path


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.shap_data import load_shap_inputs
from src.analysis.explainability import XGBoostExplainer


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

RESULT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "shap_global_importance_v3.csv"
)

SAMPLE_SIZE = 1000
RANDOM_STATE = 42
TOP_FEATURES = 15


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

    # --------------------------------------------------------
    # SELECT GLOBAL SHAP SAMPLE
    # --------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(X_test)
    )

    if sample_size == 0:
        raise ValueError(
            "The test dataset contains no samples."
        )

    global_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    print(
        f"\nGlobal SHAP sample size: {sample_size:,}"
    )

    # --------------------------------------------------------
    # CREATE EXPLAINER
    # --------------------------------------------------------

    explainer = XGBoostExplainer(
        model=model,
        feature_names=feature_names,
        label_classes=label_classes,
        top_n=5
    )

    # --------------------------------------------------------
    # CALCULATE GLOBAL IMPORTANCE
    # --------------------------------------------------------

    print("Calculating global SHAP importance...")

    importance = explainer.explain_global(
        data=global_sample,
        top_n=TOP_FEATURES
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n========== GLOBAL SHAP ==========")

    print(
        importance.to_string(index=False)
    )

    print("\n=================================")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    importance.to_csv(
        RESULT_PATH,
        index=False
    )

    print(
        f"\nGlobal SHAP results saved to:\n"
        f"{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
