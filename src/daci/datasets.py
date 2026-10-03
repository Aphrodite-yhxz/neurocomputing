"""
daci.datasets
=============
Loaders return (series: np.ndarray, change_points: list|None, name: str).

ETT CSVs are downloaded ONCE into <project_root>/data/ and reused forever after.
Multiple mirrors are tried in order (jsDelivr CDN first: usually reachable from CN),
each with retries. If every mirror fails, the error tells you how to place the file manually.
"""
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# <project_root>/data   (this file lives at <root>/src/daci/datasets.py)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ETT_MIRRORS = [
    "https://cdn.jsdelivr.net/gh/zhouhaoyi/ETDataset@main/ETT-small/{name}.csv",
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv",
    "https://gh-proxy.com/https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv",
    "https://ghproxy.net/https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv",
]

# Real energy datasets (UCI Electricity Load Diagrams; Alabama solar production)
LSTNET_MIRRORS = [
    "https://cdn.jsdelivr.net/gh/laiguokun/multivariate-time-series-data@master/{path}",
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/{path}",
    "https://gh-proxy.com/https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/{path}",
]
LSTNET_FILES = {
    "electricity": "electricity/electricity.txt.gz",   # 26304 x 321 clients, hourly 2012-2014
    "solar":       "solar-energy/solar_AL.txt.gz",     # 52560 x 137 PV plants, 10-min 2006
}

_UA = {"User-Agent": "Mozilla/5.0 (daci-research)"}


def _download(url, timeout=90, retries=2):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                     # try next attempt / mirror
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def fetch_ett_csv(which="ETTh1", force=False, verbose=True):
    """Return local path of <which>.csv, downloading once into data/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / f"{which}.csv"
    if dest.exists() and not force and dest.stat().st_size > 10_000:
        return dest
    errors = []
    for tpl in ETT_MIRRORS:
        url = tpl.format(name=which)
        try:
            if verbose:
                print(f"  [data] fetching {which}.csv via {url.split('/')[2]} ...", flush=True)
            raw = _download(url)
            if len(raw) < 10_000:
                raise ValueError(f"suspiciously small response ({len(raw)} bytes)")
            dest.write_bytes(raw)
            if verbose:
                print(f"  [data] cached -> {dest}  ({len(raw)/1e6:.1f} MB)", flush=True)
            return dest
        except Exception as e:
            errors.append(f"{url.split('/')[2]}:{type(e).__name__}")
    raise RuntimeError(
        f"Could not download {which}.csv from any mirror ({'; '.join(errors)}).\n"
        f"MANUAL FIX: download it in a browser from\n"
        f"  https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/{which}.csv\n"
        f"(click 'Raw' / download) and save it to:  {dest}")


def make_synthetic(T=6000, seed=1):
    g = np.random.default_rng(seed)
    cps = [int(T * f) for f in (0.25, 0.50, 0.75)]   # == [1500, 3000, 4500] at T=6000
    regimes = [(0.0, 1.0), (0.0, 2.2), (5.0, 0.8), (-3.0, 1.6)]  # (mean, vol)
    bounds = [0] + cps + [T]
    x = np.zeros(T)
    for i in range(len(bounds) - 1):
        a, vol = regimes[i]
        n = bounds[i + 1] - bounds[i]
        e = g.normal(0, vol, n); s = np.zeros(n); s[0] = a
        for j in range(1, n):
            s[j] = 0.7 * s[j - 1] + 0.3 * a + e[j]
        tt = np.arange(bounds[i], bounds[i + 1])
        x[bounds[i]:bounds[i + 1]] = s + 2.0 * np.sin(2 * np.pi * tt / 200)
    # seed=1 keeps the legacy name (backward-compatible with all archived results);
    # other seeds get a distinct name so traces/rank pivots don't collide.
    name = f"synthetic_T{T}" if seed == 1 else f"synthetic_T{T}_s{seed}"
    return x, cps, name


def load_ett(which="ETTh1", col="OT", T=8000, path=None):
    """Load an ETT column. Downloads+caches the CSV once; later calls read from data/."""
    src = Path(path) if path else fetch_ett_csv(which)
    df = pd.read_csv(src)
    return df[col].values[:T].astype(float), None, f"{which}_{col}"


def load_csv(path, col, T=None, change_points=None, name=None):
    df = pd.read_csv(path)
    x = df[col].values.astype(float)
    if T:
        x = x[:T]
    return x, change_points, name or f"{path}:{col}"


def fetch_energy_gz(which="electricity", force=False, verbose=True):
    """Download+cache a real energy dataset (electricity | solar) into data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / f"{which}.txt.gz"
    if dest.exists() and not force and dest.stat().st_size > 100_000:
        return dest
    path = LSTNET_FILES[which]
    errors = []
    for tpl in LSTNET_MIRRORS:
        url = tpl.format(path=path)
        try:
            if verbose:
                print(f"  [data] fetching {which} via {url.split('/')[2]} ...", flush=True)
            raw = _download(url, timeout=180)
            if len(raw) < 100_000:
                raise ValueError(f"too small ({len(raw)} bytes)")
            dest.write_bytes(raw)
            if verbose:
                print(f"  [data] cached -> {dest} ({len(raw)/1e6:.1f} MB)", flush=True)
            return dest
        except Exception as e:
            errors.append(f"{url.split('/')[2]}:{type(e).__name__}")
    raise RuntimeError(f"Could not download {which} ({'; '.join(errors)}). "
                       f"Manual: get {path} from github.com/laiguokun/multivariate-time-series-data "
                       f"and save as {dest}")


def _energy_matrix(which="electricity"):
    """Load the full (T x n_series) matrix, cached in memory."""
    import gzip
    key = f"_mat_{which}"
    if not hasattr(_energy_matrix, key):
        p = fetch_energy_gz(which)
        with gzip.open(p, "rt") as fh:
            mat = np.loadtxt(fh, delimiter=",")
        setattr(_energy_matrix, key, mat)
    return getattr(_energy_matrix, key)


def load_electricity(client=0, T=8000):
    """UCI Electricity Load Diagrams: one client's hourly consumption (real energy demand)."""
    mat = _energy_matrix("electricity")
    return mat[:T, client].astype(float), None, f"electricity_c{client}"


def load_solar(plant=0, T=8000):
    """Alabama solar production: one PV plant (real renewable generation)."""
    mat = _energy_matrix("solar")
    return mat[:T, plant].astype(float), None, f"solar_p{plant}"


def make_spliced_energy(source="electricity", series_ids=(10, 40, 70, 100),
                        seg_len=1500, T_off=1000, name=None):
    """REAL-DATA change-point benchmark: splice consecutive segments taken from DIFFERENT
    real energy series. Dynamics/seasonality/noise are genuinely real; only the switch points
    are imposed -- so post-change-point coverage IS measurable on real data.
    Each segment keeps its own scale (a real regime change alters level, volatility AND shape).
    Returns (series, change_points, name).
    """
    mat = _energy_matrix(source)
    segs, cps = [], []
    for k, sid in enumerate(series_ids):
        seg = mat[T_off:T_off + seg_len, sid].astype(float)
        segs.append(seg)
        if k > 0:
            cps.append(k * seg_len)
    x = np.concatenate(segs)
    return x, cps, name or f"spliced_{source}_{len(series_ids)}seg"


def make_seasonal_energy(client=20, seg_len=1200, T0=4392, name=None):
    """REAL seasonal regime benchmark (cleanest real-shift test): take the SAME electricity
    client across alternating summer/winter windows. The regime changes (level, volatility,
    daily profile) are genuine seasonal shifts of one real consumer -- no cross-series splice
    artefacts -- while the switch points are known, so post-shift coverage is measurable.
    Windows (hourly data starting 2012-01-01): Jul'12, Jan'13, Jul'13, Jan'14.
    """
    mat = _energy_matrix("electricity")
    starts = [T0, T0 + 4392, T0 + 8736, T0 + 13152]      # ~summer, winter, summer, winter
    segs, cps = [], []
    for k, s in enumerate(starts):
        seg = mat[s:s + seg_len, client].astype(float)
        if len(seg) < seg_len:
            break
        segs.append(seg)
        if k > 0:
            cps.append(k * seg_len)
    x = np.concatenate(segs)
    return x, cps, name or f"seasonal_elec_c{client}"


def load_ercot(zone="ERCOT", T=16000, path=None):
    """ERCOT hourly load (the dataset of arXiv:2602.10848 -- direct dialogue with that paper).

    MANUAL DOWNLOAD (no stable public API):
      1. Go to  https://www.ercot.com/gridinfo/load/load_hist
      2. Download 'Native Load' archives for 2022-2024 (xlsx, one per year).
      3. Open each, save as CSV, concatenate rows (keep the header once), save to:
             <project_root>/data/ercot_load.csv
         Expected columns: 'Hour Ending' (or 'HourEnding') + zone columns
         COAST, EAST, FCWEST, NCENT, NORTH, SCENT, SOUTH, WEST, ERCOT.
    Alternative (pip install gridstatus):
         import gridstatus; df = gridstatus.Ercot().get_load("2022-01-01", "2024-12-31")
         df.to_csv("data/ercot_load.csv")   # then set zone to the matching column name
    zone: 'ERCOT' (system total) or a weather-zone column, e.g. 'COAST', 'NORTH'.
    """
    src = Path(path) if path else DATA_DIR / "ercot_load.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. See load_ercot.__doc__ for the 2-minute manual download.")
    df = pd.read_csv(src)
    cols = {c.strip().upper().replace(" ", ""): c for c in df.columns}
    key = zone.strip().upper().replace(" ", "")
    if key not in cols:
        raise KeyError(f"zone '{zone}' not in {list(df.columns)}")
    x = pd.to_numeric(df[cols[key]], errors="coerce").dropna().values.astype(float)
    return x[:T], None, f"ercot_{zone}"


def make_benchmark_grid(n_elec=10, n_solar=8, include_ercot=True, T=8000):
    """The 30+ series camera-ready grid: returns a list of loader lambdas.

    Deterministic client/plant selection (spread across the panel, seed-free), the four ETT
    columns, two spliced + two seasonal real-shift benchmarks, two synthetic references, and
    ERCOT zones when data/ercot_load.csv exists. ~31-34 series total.
    """
    elec_ids = np.linspace(5, 300, n_elec).astype(int).tolist()       # UCI electricity: 321 cols
    solar_ids = np.linspace(3, 130, n_solar).astype(int).tolist()     # solar AL: 137 plants
    grid = []
    grid += [lambda c=c: load_electricity(client=c, T=T) for c in elec_ids]
    grid += [lambda p=p: load_solar(plant=p, T=T) for p in solar_ids]
    grid += [lambda: load_ett("ETTh1", "OT", T=T),
             lambda: load_ett("ETTh2", "OT", T=T),
             lambda: load_ett("ETTm1", "HUFL", T=12000),
             lambda: load_ett("ETTm2", "OT", T=12000)]
    grid += [lambda: make_spliced_energy("electricity", (10, 40, 70, 100)),
             lambda: make_spliced_energy("electricity", (25, 90, 140, 210),
                                         name="spliced_electricity_4seg_b"),
             lambda: make_seasonal_energy(client=20),
             lambda: make_seasonal_energy(client=80)]
    grid += [lambda: make_synthetic(T=6000, seed=1),
             lambda: make_synthetic(T=6000, seed=7)]
    if include_ercot and (DATA_DIR / "ercot_load.csv").exists():
        for z in ["ERCOT", "COAST", "NORTH"]:
            grid.append(lambda z=z: load_ercot(zone=z, T=16000))
    return grid


REGISTRY = {
    "synthetic": lambda: make_synthetic(),
    "ETTh1": lambda: load_ett("ETTh1", "OT"),
    "ETTh2": lambda: load_ett("ETTh2", "OT"),
    "ETTm1": lambda: load_ett("ETTm1", "OT", T=12000),
}
