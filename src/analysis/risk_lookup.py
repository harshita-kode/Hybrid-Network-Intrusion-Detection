
"""
GC2 asset and threat risk lookup.

Uses explicit asset inventory and threat ratings.
Does not derive risk ratings from ML model scores.
"""

import json
import ipaddress
from pathlib import Path

from src.analysis.risk import calculate_asset_risk


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class RiskLookup:

    def __init__(self, config_dir=CONFIG):
        config_dir = Path(config_dir)

        assets_config = read_json(
            config_dir / "assets.json"
        )
        threats_config = read_json(
            config_dir / "threats.json"
        )
        mappings_config = read_json(
            config_dir / "asset_mappings.json"
        )

        self.illustrative = (
            assets_config.get("config_type")
            == "illustrative_test_inventory"
            or threats_config.get("config_type")
            == "illustrative_test_threat_register"
            or mappings_config.get("config_type")
            == "illustrative_test_asset_mappings"
        )

        self.assets = {}
        for asset in assets_config["assets"]:
            asset_id = asset["asset_id"]

            if asset_id in self.assets:
                raise ValueError(
                    f"Duplicate asset ID: {asset_id}"
                )

            for key in (
                "confidentiality",
                "integrity",
                "availability",
                "vulnerability",
            ):
                value = asset[key]
                if type(value) is not int or not 1 <= value <= 4:
                    raise ValueError(
                        f"Invalid {key} for {asset_id}"
                    )

            self.assets[asset_id] = asset

        self.threats = {}
        for threat in threats_config["threats"]:
            name = threat["name"]

            if name in self.threats:
                raise ValueError(
                    f"Duplicate threat: {name}"
                )

            probability = threat["probability"]
            impact = threat["impact"]

            if (
                type(probability) is not int
                or not 1 <= probability <= 3
                or type(impact) is not int
                or not 1 <= impact <= 4
            ):
                raise ValueError(
                    f"Invalid threat ratings: {name}"
                )

            self.threats[name] = threat

        self.ip_to_asset = {}

        for mapping in mappings_config["mappings"]:
            ip = str(
                ipaddress.ip_address(
                    mapping["destination_ip"]
                )
            )
            asset_id = mapping["asset_id"]

            if asset_id not in self.assets:
                raise ValueError(
                    f"Unknown mapped asset: {asset_id}"
                )

            if ip in self.ip_to_asset:
                raise ValueError(
                    f"Duplicate destination IP: {ip}"
                )

            self.ip_to_asset[ip] = asset_id

    def resolve_asset(self, asset_id=None, destination_ip=None):
        if asset_id is not None:
            if asset_id not in self.assets:
                raise ValueError(
                    f"Unknown asset ID: {asset_id}"
                )

            return self.assets[asset_id]

        if destination_ip is None:
            return None

        try:
            ip = str(ipaddress.ip_address(destination_ip))
        except ValueError:
            return None

        matched_id = self.ip_to_asset.get(ip)

        if matched_id is None:
            return None

        return self.assets[matched_id]

    def assess(
        self,
        detection_status,
        predicted_class,
        detection_confidence=None,
        asset_id=None,
        destination_ip=None,
        destination_port=None,
        acceptable_risk=None,
    ):
        """
        Return AssetRiskResult or None.

        Risk is calculated only for an accepted
        high-confidence known attack.
        """

        if detection_status != "KNOWN_ATTACK":
            return None

        if predicted_class not in self.threats:
            return None

        asset = self.resolve_asset(
            asset_id=asset_id,
            destination_ip=destination_ip,
        )

        if asset is None:
            return None

        threat = self.threats[predicted_class]

        if destination_port is not None:
            destination_port = int(destination_port)

        return calculate_asset_risk(
            asset_name=asset["asset_name"],
            threat_name=predicted_class,
            confidentiality=asset["confidentiality"],
            integrity=asset["integrity"],
            availability=asset["availability"],
            probability=threat["probability"],
            impact=threat["impact"],
            vulnerability=asset["vulnerability"],
            acceptable_risk=acceptable_risk,
            destination_port=destination_port,
            detection_confidence=detection_confidence,
        )
