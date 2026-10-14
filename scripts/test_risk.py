
"""Test GC2 asset-based risk assessment."""

from src.analysis.risk import calculate_asset_risk


def main():

    # Example from the team's risk document.
    result = calculate_asset_risk(
        asset_name="Production Web Server",
        threat_name="DDoS",
        confidentiality=4,
        integrity=4,
        availability=4,
        probability=3,
        impact=4,
        vulnerability=3,
        destination_port=443,
        service="HTTPS",
        detection_confidence=0.96,
    )

    assert result.asset_value == 4
    assert result.threat_value == 12
    assert result.inherent_risk == 144
    assert result.decision == "NOT_EVALUATED"

    print("=== ASSET-BASED RISK ASSESSMENT ===")
    print("Asset:", result.asset_name)
    print("Threat:", result.threat_name)
    print("Destination port:", result.destination_port)
    print("Service:", result.service)
    print("Asset value:", result.asset_value)
    print("Threat value:", result.threat_value)
    print("Vulnerability:", result.vulnerability)
    print("Inherent risk:", result.inherent_risk)
    print("Detection confidence:", result.detection_confidence)
    print("Decision:", result.decision)

    # Illustrative threshold only, not a project policy.
    assessed = calculate_asset_risk(
        asset_name="Production Web Server",
        threat_name="DDoS",
        confidentiality=4,
        integrity=4,
        availability=4,
        probability=3,
        impact=4,
        vulnerability=3,
        acceptable_risk=100,
    )

    assert assessed.decision == "REQUIRES_TREATMENT"

    # Confirm invalid rating is rejected.
    try:
        calculate_asset_risk(
            asset_name="Server",
            threat_name="DDoS",
            confidentiality=5,
            integrity=4,
            availability=4,
            probability=3,
            impact=4,
            vulnerability=3,
        )
    except ValueError:
        print("Invalid rating validation: PASSED")
    else:
        raise AssertionError("Invalid rating was accepted")

    print("\nAll risk tests passed.")


if __name__ == "__main__":
    main()
