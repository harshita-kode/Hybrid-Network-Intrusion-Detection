
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    "XGBoost model":
        "saved-models/xgboost_final.joblib",
    "XGBoost inference":
        "src/pipeline/xgboost_inference.py",
    "SHAP implementation":
        "src/analysis/explainability.py",
    "Autoencoder configuration":
        "saved-models/version20_ssh_features/selection.json",
    "Autoencoder calibration":
        "saved-models/version20_ssh_features/calibration.json",
    "Autoencoder preprocessing":
        "saved-models/version20_ssh_features/preprocessing.joblib",
    "KNN KB implementation":
        "notebooks/kb/kb_flow.py",
    "KNN initial model":
        "saved-models/knn_kb/initial_kb.joblib",
    "Data V3 experiment configuration":
        "data/processed/Data_V3_SSH_Unknown_final/experiment_config.json",
}

def main():
    print("\n=== GC2 COMPONENT CHECK ===")

    for name, relative_path in REQUIRED_FILES.items():
        path = ROOT / relative_path
        status = "FOUND" if path.is_file() else "MISSING"
        print(f"{name}: {status}")

    model_dir = ROOT / "saved-models"
    print("\n=== SAVED MODELS ===")
    if model_dir.exists():
        for path in sorted(model_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".joblib", ".keras", ".json", ".ubj"
            }:
                print(path.relative_to(ROOT))

    data_dir = ROOT / "data" / "processed"
    print("\n=== DATASETS ===")
    if data_dir.exists():
        for path in sorted(data_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".csv", ".parquet"
            }:
                print(path.relative_to(ROOT))

    selection_path = (
        ROOT / "saved-models"
        / "version20_ssh_features"
        / "selection.json"
    )

    if selection_path.exists():
        with selection_path.open(encoding="utf-8") as file:
            selection = json.load(file)

        print("\n=== AUTOENCODER ===")
        print("Version:", selection.get("version"))
        print("Input dimension:", selection.get("input_dim"))
        print("Features:", len(selection.get("features", [])))
        print("Models:", selection.get("model_files"))

    print("\nComponent check completed.")

if __name__ == "__main__":
    main()
