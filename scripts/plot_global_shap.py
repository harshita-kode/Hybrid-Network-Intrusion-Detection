import os

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "results",
    "shap_global_importance_v3.csv"
)

FIGURE_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "figures",
    "shap_global_importance_v3.png"
)


def main():

    importance = pd.read_csv(
        RESULT_PATH
    )

    # Reverse order so highest importance appears at the top.
    plot_data = importance.sort_values(
        by="mean_abs_shap",
        ascending=True
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_data["feature"],
        plot_data["mean_abs_shap"]
    )

    plt.xlabel(
        "Mean |SHAP Value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Global XGBoost Feature Importance Using SHAP"
    )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(
            FIGURE_PATH
        ),
        exist_ok=True
    )

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved SHAP figure to:\n"
        f"{FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()