
from src.analysis.risk import calculate_asset_risk


def calculate(
    c=4, i=4, a=4,
    p=3, impact=4, v=3,
    acceptable=None
):
    return calculate_asset_risk(
        asset_name="Test Server",
        threat_name="Test Threat",
        confidentiality=c,
        integrity=i,
        availability=a,
        probability=p,
        impact=impact,
        vulnerability=v,
        acceptable_risk=acceptable,
    )


def expect_error(error_type, **kwargs):
    try:
        calculate(**kwargs)
    except error_type:
        return
    raise AssertionError(
        f"Expected {error_type.__name__}: {kwargs}"
    )


def main():
    minimum = calculate(
        c=1, i=1, a=1,
        p=1, impact=1, v=1
    )
    assert minimum.inherent_risk == 1

    maximum = calculate(
        c=4, i=4, a=4,
        p=3, impact=4, v=4
    )
    assert maximum.inherent_risk == 192

    example = calculate()
    assert example.inherent_risk == 144
    assert example.decision == "NOT_EVALUATED"

    equal = calculate(acceptable=144)
    assert equal.decision == "ACCEPTABLE"

    above = calculate(acceptable=100)
    assert above.decision == "REQUIRES_TREATMENT"

    below = calculate(acceptable=150)
    assert below.decision == "ACCEPTABLE"

    expect_error(ValueError, c=5)
    expect_error(ValueError, p=0)
    expect_error(ValueError, v=6)
    expect_error(TypeError, c=2.5)
    expect_error(TypeError, p=True)
    expect_error(ValueError, acceptable=-1)
    expect_error(ValueError, acceptable=193)

    print("Minimum risk: PASSED")
    print("Maximum risk: PASSED")
    print("Worked example: PASSED")
    print("Acceptable threshold: PASSED")
    print("Risk treatment threshold: PASSED")
    print("Invalid ratings: PASSED")
    print("\nALL RISK VERIFICATION TESTS PASSED")


if __name__ == "__main__":
    main()
