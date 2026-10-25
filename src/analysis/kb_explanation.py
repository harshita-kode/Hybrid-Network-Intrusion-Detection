
import numpy as np
import pandas as pd

from notebooks.kb.kb_flow import search, transform


def explain_kb_match(flow, artifact, reference, top_n=5):
    """
    Explain a KNN/KB similarity decision using the same
    fitted preprocessing and Euclidean distance as the KB.
    """

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if isinstance(flow, pd.DataFrame):
        frame = flow.copy()
    elif isinstance(flow, dict):
        frame = pd.DataFrame([flow])
    else:
        raise TypeError("flow must be a DataFrame or dictionary")

    if len(frame) != 1:
        raise ValueError("Exactly one flow is required")

    if len(reference) != artifact["count"]:
        raise ValueError("KB reference and index sizes differ")

    if artifact.get("cutoff") is None:
        raise ValueError("KB matching cutoff is missing")

    # Obtain the actual KB decision.
    lookup = search(
        frame,
        artifact,
        reference,
        k=5,
        trace=False
    )[0]

    # Apply the exact preprocessing used by KNN.
    query_vector = np.asarray(
        transform(frame, artifact)
    )[0]

    distances, indices = artifact["index"].kneighbors(
        query_vector.reshape(1, -1),
        n_neighbors=1
    )

    nearest_index = int(indices[0, 0])
    nearest_distance = float(distances[0, 0])

    # Retrieve the transformed stored neighbour.
    fitted_vectors = artifact["index"]._fit_X

    neighbor_vector = np.asarray(
        fitted_vectors[nearest_index],
        dtype=float
    )

    feature_names = list(artifact["columns"])

    if len(feature_names) != len(query_vector):
        raise ValueError("KNN feature count mismatch")

    squared_differences = (
        query_vector - neighbor_vector
    ) ** 2

    reproduced_distance = float(
        np.sqrt(squared_differences.sum())
    )

   
    # Allow small floating-point differences between
    # sklearn's optimized distance calculation and
    # direct Euclidean reconstruction.

    if not np.isclose(
        reproduced_distance,
        nearest_distance,
        rtol=1e-5,
        atol=1e-6
    ):
        raise ValueError(
            "Feature differences do not reproduce "
            "the KNN distance. Verify its metric."
        )


    total_squared = float(squared_differences.sum())

    ranked_indices = np.argsort(
        squared_differences
    )[::-1][:top_n]

    contributions = []

    for index in ranked_indices:
        contributions.append({
            "feature": feature_names[index],
            "squared_difference": float(
                squared_differences[index]
            ),
            "distance_share": (
                float(squared_differences[index] / total_squared)
                if total_squared > 0 else 0.0
            )
        })

    neighbors = lookup.get("neighbors", [])

    nearest_pattern = (
        neighbors[0].get("pattern_id")
        if neighbors else None
    )

    return {
        "method": "knn_feature_distance",
        "kb_matched": bool(lookup["matched"]),
        "route": (
            "KB_MATCH"
            if lookup["matched"]
            else "AUTOENCODER"
        ),
        "nearest_pattern_id": nearest_pattern,
        "nearest_distance": nearest_distance,
        "match_cutoff": float(artifact["cutoff"]),
        "top_feature_differences": contributions,
    }
