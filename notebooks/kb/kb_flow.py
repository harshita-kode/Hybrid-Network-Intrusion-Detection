"""CSV-backed, label-free nearest-record retrieval; immutable initial KB and demo copy."""
import json
import logging
import sys
import os
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

LOGGER = logging.getLogger("intrusion.kb")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def trace_flow(stage, flow=None, **details):
    """Log a routing stage, including the full incoming record if supplied."""
    LOGGER.info("[%s] %s", stage, json.dumps(details, default=str, ensure_ascii=False))
    if flow is not None:
        LOGGER.info("[%s] flow=%s", stage, json.dumps(dict(flow), default=str, ensure_ascii=False))


# All names are casefolded before matching; labels/metadata never enter KNN.
EXCLUDE = {
    'label', 'attempted category', 'source_file', 'is_attempted',
    'id', 'flow id', 'src ip', 'dst ip', 'timestamp',
    'pattern_id', 'attack_label', 'anomaly_score',
    'date_added', 'source', 'reviewed', 'ae_verified'
}


def read_partition(path):
    df=pd.read_parquet(path)
    df.columns=df.columns.astype(str).str.strip()
    if df.columns.duplicated().any(): raise ValueError('Duplicate column names')
    if 'is_attempted' in df:
        mask=df['is_attempted'].astype(str).str.strip().str.lower().isin(['true','1','yes','t'])
        if mask.any(): raise ValueError(f'{path}: attempted records in frozen partition')
    return df.reset_index(drop=True)

def numeric(df, columns):
    missing=set(columns)-set(df.columns)
    if missing: raise ValueError(f'Missing features: {sorted(missing)}')
    return df[columns].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)

def signed_log(a):
    return np.sign(a)*np.log1p(np.abs(a))

def build(df, columns=None, cutoff=None):
    if columns is None:
        candidates=[c for c in df if c.casefold() not in EXCLUDE]
        columns=[c for c in candidates if pd.to_numeric(df[c],errors='coerce').notna().mean()>.99
                 and pd.to_numeric(df[c],errors='coerce').nunique(dropna=True)>1]
    forbidden = [c for c in columns if c.strip().casefold() in EXCLUDE]
    if forbidden:
        raise ValueError(f"Excluded metadata cannot enter KNN: {forbidden}")
    if not columns: raise ValueError('No numerical reference features')
    x=numeric(df,columns)
    # Impute BEFORE log; fit all transformations on reference only.
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    a=signed_log(imp.fit_transform(x))
    scaler=RobustScaler().fit(a)
    vectors=np.clip(scaler.transform(a),-20,20)
    index=NearestNeighbors(n_neighbors=min(5,len(df)),metric='euclidean',algorithm='brute').fit(vectors)
    return {'columns':columns,'imputer':imp,'scaler':scaler,'vectors':vectors,
            'index':index,'cutoff':cutoff,'count':len(df)}

def transform(df,artifact):
    forbidden = [c for c in artifact['columns'] if c.strip().casefold() in EXCLUDE]
    if forbidden:
        raise ValueError(f"Saved KNN index contains excluded columns: {forbidden}")
    x=numeric(df,artifact['columns'])
    return np.clip(artifact['scaler'].transform(signed_log(artifact['imputer'].transform(x))),-20,20)

def search(df,artifact,reference,k=5,trace=False):
    z=transform(df,artifact)
    d,ix=artifact['index'].kneighbors(z,n_neighbors=min(k,artifact['count']))
    result=[]
    for row in range(len(df)):
        hits=[{'pattern_id':str(reference.iloc[int(j)]['pattern_id']),
               'distance':float(distance)}
              for distance,j in zip(d[row],ix[row])]
        cutoff=artifact['cutoff']
        matched=bool(cutoff is not None and hits[0]['distance']<=cutoff)
        result.append({'matched':matched,
                       'nearest_distance':hits[0]['distance'],'neighbors':hits})
        if trace:
            trace_flow("KB_LOOKUP", df.iloc[row].to_dict(), row=row, matched=matched,
                       nearest_distance=hits[0]['distance'], cutoff=cutoff,
                       nearest_pattern_id=hits[0]['pattern_id'],
                       next_stage="KB_MATCH" if matched else "AUTOENCODER")
    return result

def atomic_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.csv')
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f: frame.to_csv(f,index=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def update_after_ae(flow,score,threshold,reference_path,model_path):
    """Automatically insert only AE-positive, KB-unmatched flow. No label needed."""
    if not np.isfinite(score) or not np.isfinite(threshold) or score<=threshold:
        raise ValueError('AE score does not exceed global threshold')
    kb=pd.read_csv(reference_path)
    artifact=joblib.load(model_path) # Load only your own trusted artifacts.
    if artifact['cutoff'] is None: raise ValueError('KB cutoff not calibrated')
    query=pd.DataFrame([flow]); hits=search(query,artifact,kb,k=1)[0]
    if hits['matched']: raise ValueError('Pattern already matches KB; no update')
    record={c:flow.get(c,np.nan) for c in artifact['columns']}
    record.update(pattern_id='KB_'+uuid4().hex[:16],attack_label='',
                  anomaly_score=float(score),date_added=datetime.now(timezone.utc).isoformat(),
                  source='ae_auto_update',reviewed=False,ae_verified=True)
    # Preserve original reference fields and append the new record.
    kb=pd.concat([kb,pd.DataFrame([record])],ignore_index=True)
    atomic_csv(kb,reference_path)
    new_artifact=build(kb,artifact['columns'],cutoff=artifact['cutoff'])
    joblib.dump(new_artifact,model_path)
    trace_flow("KB_INSERTED", record, pattern_id=record["pattern_id"],
               anomaly_mse=score, threshold=threshold, kb_size=len(kb))
    return {'pattern_id':record['pattern_id'],'kb_size':len(kb),'before_distance':hits['nearest_distance']}


# ---- Version20 integration: invoked only for KB-unmatched flows ----
"""Loads Version20 ensemble. Caller MUST supply flow row with correctly precomputed behavioural history."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def load_version20(root):
    import tensorflow as tf
    ae_dir=Path(root)/"notebooks"/"autoencoder"
    if str(ae_dir) not in sys.path:
        sys.path.insert(0,str(ae_dir))
    import version20_common as common
    folder=Path(root)/'saved-models/version20_ssh_features'
    selection=json.loads((folder/'selection.json').read_text())
    if selection.get('version')!='version20':raise ValueError('Expected Version20 model')
    if selection.get('data_folder')!='Data_V3_SSH_Unknown_final':raise ValueError('Wrong dataset model')
    prep=joblib.load(folder/'preprocessing.joblib')
    if selection.get('features')!=prep['columns']:raise ValueError('Model feature order mismatch')
    models=[tf.keras.models.load_model(folder/f) for f in selection['model_files']]
    if len(models)!=3:raise ValueError('Expected all three Version20 models')
    calibration=json.loads((folder/selection['calibration_file']).read_text())
    return common,prep,models,float(calibration['global_threshold'])

def score_precomputed_flow(row,loaded):
    common,prep,models,threshold=loaded
    frame=pd.DataFrame([row])
    missing=set(common.BEHAVIOURAL_FEATURES)-set(frame.columns)
    if missing:raise ValueError(f'Behavioural history not supplied: {sorted(missing)}')
    x=common.transform(frame,prep)
    score=float(common.reconstruction_mse(models,x)[0])
    trace_flow("AUTOENCODER_RESULT", row, mse=score, global_threshold=threshold,
               anomalous=bool(score > threshold))
    return score,threshold


def prepare_ae_context(root, flow, source_row_id=None):
    """Calculate V20 features from preceding full-stream traffic, only on a KB miss.

    A unique matching row is required. If duplicate rows exist, supply a stable
    original row index; this function fails closed rather than inventing history.
    """
    import sys
    root=Path(root)
    module_dir=root/"notebooks"/"autoencoder"
    if str(module_dir) not in sys.path:
        sys.path.insert(0,str(module_dir))
    from version20_common import add_behavioural_features
    full_path=root/"data/processed/Data_V3_SSH_Unknown_final/unknown_test.parquet"
    full=pd.read_parquet(full_path)
    full.columns=full.columns.astype(str).str.strip()
    for col in ("Timestamp","Src IP","Dst IP"):
        if col not in full: raise ValueError(f"Full stream missing {col}")
    if not ({"Dst Port","Destination Port"} & set(full.columns)):
        raise ValueError("Full stream missing destination port")
    shared=[c for c in flow if c in full.columns and c!="demo_row"]
    target=pd.DataFrame([flow])[shared].reset_index(drop=True)
    hashes=pd.util.hash_pandas_object(full[shared],index=False).to_numpy()
    target_hash=pd.util.hash_pandas_object(target,index=False).iloc[0]
    matches=[int(i) for i in np.flatnonzero(hashes==target_hash)
             if full.iloc[[int(i)]][shared].reset_index(drop=True).equals(target)]
    if source_row_id is None:
        if len(matches)!=1:
            raise ValueError(f"Expected exactly one full-stream matching row, got {len(matches)}")
        pos=matches[0]
    else:
        pos=int(source_row_id)
        if pos not in matches: raise ValueError("Supplied original row index does not match query")
    ts=pd.to_datetime(full["Timestamp"],errors="coerce")
    if ts.isna().any():raise ValueError("Invalid timestamps")
    earlier=(ts<ts.iloc[pos])|((ts==ts.iloc[pos])&(np.arange(len(full))<pos))
    context=pd.concat([full.loc[earlier],full.iloc[[pos]]],ignore_index=True)
    prepared=add_behavioural_features(context)
    trace_flow("AUTOENCODER_INPUT", prepared.iloc[-1].to_dict(),
               historical_flow_count=int(earlier.sum()))
    return prepared.iloc[-1].to_dict(),int(earlier.sum())


def process_incoming_flow(root, flow, reference_path, model_path,
                          source_row_id=None, loaded_ae=None,
                          trace=True):
    """KNN -> (on miss) Version20 -> automatic AE-positive KB update, with routing logs.

    For demonstrations use the DEMO KB paths, never the frozen initial KB.
    """
    flow=dict(flow)
    if trace:
        trace_flow("INCOMING_FLOW", flow)
    kb=pd.read_csv(reference_path)
    artifact=joblib.load(model_path)  # Trusted project artifacts only.
    if len(kb)!=artifact["count"]:
        raise ValueError("KB CSV/index count mismatch")
    lookup=search(pd.DataFrame([flow]),artifact,kb,k=5,trace=trace)[0]
    result={"kb_matched":lookup["matched"],"lookup":lookup,
            "ae_called":False,"kb_updated":False}
    if lookup["matched"]:
        if trace:
            trace_flow("ROUTED_TO_KB", flow,
                       nearest_pattern_id=lookup["neighbors"][0]["pattern_id"],
                       nearest_distance=lookup["nearest_distance"])
        return result
    if trace:
        trace_flow("ROUTED_TO_AUTOENCODER", flow,
                   nearest_distance=lookup["nearest_distance"],cutoff=artifact["cutoff"])
    prepared,n_history=prepare_ae_context(root,flow,source_row_id=source_row_id)
    ae=loaded_ae if loaded_ae is not None else load_version20(root)
    score,threshold=score_precomputed_flow(prepared,ae)
    result.update(ae_called=True,ae_mse=score,ae_threshold=threshold,
                  ae_anomalous=bool(score>threshold),historical_flows=n_history)
    if score<=threshold:
        if trace:
            trace_flow("NOT_ANOMALOUS_NO_UPDATE",flow,mse=score,threshold=threshold)
        return result
    update=update_after_ae(flow,score,threshold,reference_path,model_path)
    result.update(kb_updated=True,update=update)
    if trace:
        trace_flow("UPDATE_COMPLETE",flow,pattern_id=update["pattern_id"],
                   kb_size=update["kb_size"])
    return result
