"""GEFCom2014 renewable series loader (geographically diverse solar + wind).

Rescues Finding 2 from the single-Alabama-panel PV concern: GEFCom2014 provides 3 solar zones
(Australian PV plants) and 10 wind zones (Australian wind farms), all spatially distinct.

Loaders return (series: np.ndarray, change_points: None, name: str), matching daci.datasets.

Data layout expected (produced by scripts/prep_gefcom.py, or by hand):
    <project_root>/data/gefcom/gefcom_solar_z1.csv ... gefcom_solar_z3.csv
    <project_root>/data/gefcom/gefcom_wind_z1.csv  ... gefcom_wind_z10.csv
Each file: two columns [timestamp, power]; power is the normalized generation in [0,1].
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GEFCOM_DIR = DATA_DIR / "gefcom"


def _load_one(fp, T=None):
    df = pd.read_csv(fp)
    # value = the 'power'/'target'/'value' column if present, else the last numeric column
    lc = {c.lower(): c for c in df.columns}
    col = next((lc[k] for k in ("power", "targetvar", "target", "value", "generation") if k in lc), None)
    if col is None:
        num = df.select_dtypes("number").columns
        if len(num) == 0:
            raise ValueError(f"{fp}: no numeric column found")
        col = num[-1]
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if T:
        x = x[:T]
    return x, None, fp.stem


def load_gefcom_grid(T=None):
    """Return a list of loader lambdas for every gefcom_*.csv in data/gefcom/ (solar first)."""
    if not GEFCOM_DIR.exists():
        raise FileNotFoundError(
            f"{GEFCOM_DIR} not found. See docs/RUNBOOK_GEFCOM.md: download GEFCom2014-S (solar) and "
            f"GEFCom2014-W (wind), then run  python scripts/prep_gefcom.py  to build the per-zone CSVs.")
    files = sorted(GEFCOM_DIR.glob("gefcom_*.csv"),
                   key=lambda p: (0 if "solar" in p.stem else 1, p.stem))
    if not files:
        raise FileNotFoundError(f"No gefcom_*.csv in {GEFCOM_DIR}; run scripts/prep_gefcom.py first.")
    return [(lambda fp=fp: _load_one(fp, T)) for fp in files]
