# Hybrid Network Intrusion Detection System

## Overview

The **Hybrid Network Intrusion Detection System (NIDS)** is designed to identify known network attacks and analyse potentially unknown threats using a combination of machine learning techniques. It integrates supervised classification, confidence-based routing, similarity matching, anomaly detection, explainability, and risk assessment.

The system aims to improve intrusion detection by routing uncertain predictions for additional analysis rather than relying entirely on a single classifier.

## System Architecture

The system follows this workflow:

1. **Network Traffic Input** — Receives network-flow records.
2. **Data Preprocessing** — Cleans, encodes, and normalises the data.
3. **Feature Selection** — Retains informative features for model training and inference.
4. **XGBoost Classification** — Predicts known attack classes and produces class probabilities.
5. **Confidence-Based Routing** — Uses a threshold of `0.95` to determine whether further analysis is needed.
6. **Knowledge Base Lookup** — Uses KNN-based similarity matching to compare uncertain flows with stored attack patterns.
7. **Autoencoder Anomaly Detection** — Uses reconstruction error to identify potentially novel traffic patterns.
8. **Historical Attack Comparison** — Compares suspicious traffic with previously observed patterns.
9. **Explainability and Risk Assessment** — Uses SHAP and risk-scoring rules to support investigation.
10. **Security Alerts** — Presents threat classifications, scores, and explanations for review.

## Models and Techniques

| Component | Purpose |
|---|---|
| XGBoost | Classifies known network attacks |
| Supervised KNN | Maps flows to known attack families |
| KNN Knowledge Base | Matches suspicious flows against stored patterns |
| Autoencoder Ensemble | Detects anomalous traffic using reconstruction error |
| SHAP | Explains XGBoost predictions |
| Rule-based Risk Assessment | Combines security-related factors to prioritise threats |

### Key Model Configuration

**XGBoost**
- Number of trees: `300`
- Maximum depth: `8`
- Learning rate: `0.1`
- Input features: `84`
- Classes: `13`

**Supervised KNN**
- Number of neighbours (`k`): `3`
- Selected using validation macro F1-score

**KNN Knowledge Base**
- Features: `73`
- Distance cutoff: `0.945`
- Stored records: `5,203`

**Autoencoder Ensemble**
- Number of models: `3`
- Training data: Benign traffic only
- Inputs: `93`, including 10 behavioural features
- Reconstruction-error threshold: `0.0468`

*These settings reflect the reported configuration and should be checked against the final implementation before publication.*

## Dataset and Preprocessing

The project uses the CIC-IDS2017 dataset and an improved dataset variant for intrusion detection experiments.

Preprocessing includes:
- Combining traffic records from multiple CSV files.
- Standardising column names and attack labels.
- Handling missing and infinite values.
- Removing invalid records and duplicates.
- Excluding metadata from model inputs.
- Selecting numerical features for machine learning.
- Preserving rare attack classes for evaluation.

The dataset is highly imbalanced, so per-class precision, recall, and F1-score are important alongside overall accuracy.

## Evaluation Results

### XGBoost Classification

The reported evaluation on **272,989 test samples** achieved:

| Metric | Result |
|---|---:|
| Accuracy | 99.9451% |
| Weighted Precision | 99.9450% |
| Weighted Recall | 99.9451% |
| Weighted F1-score | 99.9449% |
| Macro F1-score | 96.9050% |

### Confidence-Based Routing

Using a confidence threshold of `0.95`:

| Metric | Result |
|---|---:|
| Test samples | 272,989 |
| Samples routed for further analysis | 517 |
| Routing percentage | 0.1894% |
| Classification errors | 150 |
| Errors captured by routing | 127 |
| Error capture rate | 84.67% |

These routing metrics measure how effectively low-confidence predictions capture classification errors; they do not directly measure unknown-attack detection performance.

The autoencoder was separately reported to achieve a **97.90% F1-score** on its balanced unknown-attack test set. This result should not be interpreted as the end-to-end system F1-score.

## Explainability and Risk Assessment

The system supports investigation through three forms of explanation:

- **SHAP:** Explains which features influenced the XGBoost prediction.
- **KNN Feature-Distance Analysis:** Shows how closely a flow resembles a stored pattern.
- **Autoencoder Feature-Level Error:** Highlights features contributing to reconstruction error.

The risk-assessment component combines security-related factors, including asset value, likelihood, impact, and vulnerability, to help prioritise alerts.

## Future Improvements

- Tune model hyperparameters using validation data.
- Calibrate the confidence threshold to balance error capture and processing overhead.
- Improve evaluation for rare attack classes.
- Test detection performance on genuinely unseen attacks.
- Integrate and validate all components in an end-to-end pipeline.
- Evaluate false-positive rates under realistic network conditions.
- Update the knowledge base as new attack patterns are identified.

## Limitations

- Severe class imbalance can make rare-class metrics unstable.
- High classification accuracy does not guarantee reliable detection of unseen attacks.
- Confidence-based routing can send correctly classified samples for additional analysis.
- The complete hybrid system requires end-to-end evaluation under realistic traffic conditions.

## Conclusion

This project combines XGBoost, KNN-based similarity matching, and autoencoder anomaly detection to support the identification of known and potentially unknown network threats. Confidence-based routing directs uncertain predictions for further analysis, while explainability and risk assessment provide additional context for security analysts.

The reported results demonstrate strong XGBoost classification performance and promising confidence-based error routing. Further end-to-end testing is required to establish the effectiveness of the complete hybrid detection pipeline.

## Technologies

- Python
- XGBoost
- Scikit-learn
- K-Nearest Neighbours (KNN)
- Autoencoders
- SHAP
- Pandas and NumPy
- CIC-IDS2017 network-flow dataset

## Project Status

**Research and development.** Model-level evaluation results are available; the complete integrated system should be validated before production deployment.