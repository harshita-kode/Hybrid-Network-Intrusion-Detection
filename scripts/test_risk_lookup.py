
"""Test GC2 asset and threat lookup."""

from src.analysis.risk_lookup import RiskLookup


def main():
    lookup = RiskLookup()

    assert len(lookup.assets) == 10
    assert len(lookup.threats) == 12

    # Explicitly assign illustrative asset A001.
    result = lookup.assess(
        detection_status="KNOWN_ATTACK",
        predicted_class="DDoS",
        detection_confidence=0.96,
        asset_id="A001",
        destination_port=443,
    )

    assert result is not None
    assert abs(result.asset_value - 11 / 3) < 1e-10
    assert result.threat_value == 12
    assert result.inherent_risk == 132
    assert result.decision == "NOT_EVALUATED"

    # A benign flow should not receive attack risk.
    assert lookup.assess(
        detection_status="BENIGN",
        predicted_class="BENIGN",
        asset_id="A001",
    ) is None

    # Unknown anomaly without a rated threat.
    assert lookup.assess(
        detection_status="POTENTIAL_NEW_UNKNOWN",
        predicted_class="DDoS",
        asset_id="A001",
    ) is None

    # Missing asset assignment.
    assert lookup.assess(
        detection_status="KNOWN_ATTACK",
        predicted_class="DDoS",
    ) is None

    # A configured acceptable-risk threshold.
    assessed = lookup.assess(
        detection_status="KNOWN_ATTACK",
        predicted_class="DDoS",
        asset_id="A001",
        acceptable_risk=100,
    )

    assert assessed.decision == "REQUIRES_TREATMENT"

    print("Asset inventory: PASSED")
    print("Threat register: PASSED")
    print("Inherent risk calculation: PASSED")
    print("Missing asset handling: PASSED")
    print("Benign and unknown handling: PASSED")
    print("Acceptable risk comparison: PASSED")
    print()
    print("Asset:", result.asset_name)
    print("Threat:", result.threat_name)
    print("Inherent risk:", result.inherent_risk)
    print()
    print("ALL RISK LOOKUP TESTS PASSED")


if __name__ == "__main__":
    main()
