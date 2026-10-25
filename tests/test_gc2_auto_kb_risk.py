"""Run: python tests/test_gc2_auto_kb_risk.py from repository root."""
import tempfile
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from src.pipeline.kb_promotion import WritableKB
from src.analysis.risk_lookup import RiskLookup

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    cols = [f"f{i}" for i in range(73)]
    train = pd.DataFrame(np.zeros((3, 73)), columns=cols)
    train.iloc[1, :] = 1
    train.iloc[2, :] = 2
    imputer = SimpleImputer().fit(train)
    scaler = RobustScaler().fit(imputer.transform(train))
    vectors = scaler.transform(imputer.transform(train))
    index = NearestNeighbors(n_neighbors=1).fit(vectors)
    artifact = root / 'initial_kb.joblib'
    joblib.dump({'columns': cols, 'imputer': imputer, 'scaler': scaler,
                 'vectors': vectors, 'index': index, 'cutoff': 0.5,
                 'count': 3}, artifact)
    writable = WritableKB(frozen_path=artifact,
                          writable_path=root / 'writable.sqlite3',
                          candidate_db=root / 'candidates.sqlite3')
    flow = {name: 5.0 for name in cols}
    evidence = {'detection_status': 'POTENTIAL_NEW_UNKNOWN',
                'confidence_route': 'LOW_CONFIDENCE',
                'kb_matched': False, 'ae_anomalous': True}
    first = writable.store_automatic(flow, evidence)
    second = writable.store_automatic(flow, evidence)
    assert first['created'] is True
    assert second['created'] is False
    assert first['pattern_id'] == second['pattern_id']
    assert writable.lookup(flow)['matched'] is True
    assert len(writable.list_promoted()) == 1
    try:
        writable.store_automatic(flow, {**evidence, 'ae_anomalous': False})
    except ValueError:
        pass
    else:
        raise AssertionError('Non-anomalous insertion should be rejected')
    assert joblib.load(artifact)['count'] == 3
    print('AUTO KB STORAGE / DEDUP / FROZEN PRESERVATION: PASSED')

config = Path(__file__).resolve().parent / 'fixtures'
lookup = RiskLookup(config_dir=config)
known = lookup.assess('KNOWN_ATTACK', 'DDoS', asset_id='A001', acceptable_risk=100)
unknown = lookup.assess('POTENTIAL_NEW_UNKNOWN', 'Portscan', asset_id='A001', acceptable_risk=100)
benign = lookup.assess('BENIGN', 'BENIGN', asset_id='A001', acceptable_risk=100)
assert known and round(known.inherent_risk) == 132
assert known.risk_basis == 'ASSESSED_KNOWN_THREAT'
assert unknown and round(unknown.inherent_risk) == 99
assert unknown.risk_level == 'HIGH'
assert unknown.risk_basis == 'PROVISIONAL_UNKNOWN_THREAT'
assert benign is None
assert lookup.assess('POTENTIAL_NEW_UNKNOWN', 'Portscan') is None
print('KNOWN / PROVISIONAL UNKNOWN / BENIGN RISK: PASSED')
