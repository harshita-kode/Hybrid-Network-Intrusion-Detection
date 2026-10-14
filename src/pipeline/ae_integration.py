
"""
Version20 Autoencoder integration for GC2.

Accepts a flow with correctly prepared chronological
behavioural features.

Does not retrain models or update the Knowledge Base.
"""

from pathlib import Path

from src.analysis.anomaly_explanation import (
    explain_autoencoder,
)
from src.analysis.explanation_adapter import (
    adapt_autoencoder_explanation,
)


ROOT = Path(__file__).resolve().parents[2]


class AutoencoderIntegration:

    def __init__(self):

        # Import lazily so TensorFlow is only required
        # when Autoencoder integration is initialized.
        from notebooks.kb.kb_flow import load_version20

        print("Loading Version20 Autoencoder ensemble...")

        self.loaded_ae = load_version20(ROOT)

        print("Version20 Autoencoder loaded.")

    def analyze(self, prepared_flow):
        """
        Analyze one flow with chronological behavioural
        features already calculated.

        Returns
        -------
        result:
            Original reconstruction explanation dictionary.

        explanation:
            Unified ExplanationResult.
        """

        result = explain_autoencoder(
            flow=prepared_flow,
            loaded_ae=self.loaded_ae,
            top_n=5,
        )

        if not result["mse_consistency"]:
            raise ValueError(
                "Autoencoder explanation MSE verification failed"
            )

        explanation = adapt_autoencoder_explanation(
            result
        )

        return result, explanation
