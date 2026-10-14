"""Supervised known-class KNN classifier. Label is target, never an input feature."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.neighbors import KNeighborsClassifier

EXCLUDE = {'label','attempted category','source_file','is_attempted','id',
           'flow id','src ip','dst ip','timestamp','pattern_id','attack_label',
           'anomaly_score','date_added','source','reviewed','ae_verified','demo_row'}

def root_from_cwd():
    root=Path.cwd().resolve()
    if root.name.lower()=='knn_known' and root.parent.name.lower()=='notebooks': return root.parent.parent
    if root.name.lower()=='notebooks': return root.parent
    return root

def read_data(path, require_label=True):
    df=pd.read_parquet(path)
    df.columns=df.columns.astype(str).str.strip()
    if df.columns.duplicated().any(): raise ValueError('Duplicate column names')
    if 'is_attempted' in df:
        attempted=df['is_attempted'].astype(str).str.strip().str.lower().isin(['true','1','yes','t'])
        print(f'{Path(path).name}: filtered {int(attempted.sum())} attempted rows')
        df=df.loc[~attempted].copy()
    if require_label and 'Label' not in df: raise KeyError('Missing Label target')
    if df.empty: raise ValueError('No records after filtering')
    return df.reset_index(drop=True)

def choose_columns(df):
    columns=[]
    for col in df.columns:
        if col.casefold() in EXCLUDE: continue
        values=pd.to_numeric(df[col],errors='coerce')
        if values.notna().mean() >= 0.99 and values.nunique(dropna=True)>1:
            columns.append(col)
    if not columns: raise ValueError('No suitable numeric features')
    return columns

def feature_frame(df,columns):
    if any(c.casefold() in EXCLUDE for c in columns):
        raise ValueError('Target or metadata leaked into features')
    missing=sorted(set(columns)-set(df.columns))
    if missing: raise ValueError(f'Missing features: {missing}')
    return df[columns].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)

def signed_log(x):
    a=np.asarray(x,dtype=float)
    return np.sign(a)*np.log1p(np.abs(a))

def build_pipeline(k=5,weights='distance',metric='euclidean'):
    return Pipeline([
        ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('signed_log',FunctionTransformer(signed_log,validate=False)),
        ('scaler',RobustScaler()),
        ('knn',KNeighborsClassifier(n_neighbors=int(k),weights=weights,metric=metric,n_jobs=-1))
    ])

def save_artifact(path,model,columns,config):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'columns':columns,'config':config,'classes':list(model.classes_)},path)

def load_artifact(path):
    return joblib.load(path)  # Load only artifacts you created/trust.

def predict_rows(df,artifact):
    x=feature_frame(df,artifact['columns'])
    model=artifact['model']
    pred=model.predict(x)
    proba=model.predict_proba(x)
    confidence=proba.max(axis=1)
    return pd.DataFrame({'predicted_known_class':pred,'knn_confidence':confidence,
                         'interpretation':['Known-class prediction (not proof of membership)']*len(pred)})


def stratified_sample(df, total, seed=42, preserve_rare=False, rare_limit=100):
    """Exact-size, reproducible stratified sampling without replacement.

    When preserve_rare=True, classes with fewer than rare_limit rows are
    included in full. Other classes receive at least one row, then remaining
    slots are distributed in proportion to available capacity.
    """
    if "Label" not in df.columns:
        raise KeyError("Label column missing")
    if not 0 < total <= len(df):
        raise ValueError(f"Requested {total} samples from {len(df)} rows")
    counts = df["Label"].value_counts().sort_index()
    labels = counts.index.to_numpy()
    available = counts.to_numpy(dtype=np.int64)
    allocation = np.ones(len(labels), dtype=np.int64)
    if preserve_rare:
        rare = available < rare_limit
        allocation[rare] = available[rare]
    if allocation.sum() > total:
        raise ValueError("Sample budget cannot retain all required classes")
    remaining = int(total - allocation.sum())
    capacity = available - allocation
    while remaining > 0:
        eligible = capacity > 0
        if not eligible.any():
            raise ValueError("Insufficient remaining samples")
        weights = capacity.astype(np.float64) / capacity.sum()
        expected = weights * remaining
        extra = np.minimum(np.floor(expected).astype(np.int64), capacity)
        allocated = int(extra.sum())
        allocation += extra
        capacity -= extra
        remaining -= allocated
        if allocated == 0:
            fractions = expected - np.floor(expected)
            for idx in np.argsort(-fractions, kind="stable"):
                if remaining == 0:
                    break
                if capacity[idx] > 0:
                    allocation[idx] += 1
                    capacity[idx] -= 1
                    remaining -= 1
    parts = []
    for label, count in zip(labels, allocation):
        group = df.loc[df["Label"] == label]
        parts.append(group.sample(n=int(count), random_state=seed))
    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    assert len(result) == total
    return result
