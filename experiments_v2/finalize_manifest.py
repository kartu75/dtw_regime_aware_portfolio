# -*- coding: utf-8 -*-
"""Write manifest.json for a snapshot folder from the parquet files it contains.

snapshot_data.py stopped before writing its manifest because the Nifty Midcap 100 index
(NIFTY_MIDCAP_100.NS) is no longer served by Yahoo. This records what was saved.
Usage: python finalize_manifest.py data/snapshot_YYYYMMDD
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

from snapshot_data import NIFTY50_TICKERS, BANK_TICKERS, RENAMES

snap = sys.argv[1]
files = sorted(f for f in os.listdir(snap) if f.endswith(".parquet"))
first_mtime = min(os.path.getmtime(os.path.join(snap, f)) for f in files)
manifest = dict(
    download_date=datetime.fromtimestamp(first_mtime).isoformat(timespec="seconds"),
    yfinance_version=yf.__version__, pandas_version=pd.__version__,
    note="auto_adjust=True Close prices; each stocks_* frame repeats one original yf.download call "
         "(same tickers/start/end/retry logic). Index frames are single-ticker downloads.",
    missing_from_snapshot={
        "NIFTY_MIDCAP_100.NS": "Yahoo returned no data for 2015-01-01..2026-08-09 on the download date "
                               "(ticker used by expanding_window_midcap.py in Aug 2026). Alternatives "
                               "^NSEMDCP100, ^CNXMIDCAP, ^NIFMDCP100, NIFTYMIDCAP100.NS, ^CRSMID also empty; "
                               "only ^NSEMDCP50 (Midcap 50) is available."},
    frames={},
)
universe_lists = {"N50": [RENAMES.get(t, t) for t in NIFTY50_TICKERS],
                  "BANK": [RENAMES.get(t, t) for t in BANK_TICKERS]}
for f in files:
    df = pd.read_parquet(os.path.join(snap, f))
    key = f[:-len(".parquet")]
    info = dict(file=f, shape=list(df.shape), first=str(df.index.min().date()),
                last=str(df.index.max().date()), n_missing=int(df.isna().sum().sum()),
                columns=list(map(str, df.columns)))
    if key in ("stocks_N50_A", "stocks_BANK_A"):
        u = key.split("_")[1]
        info["period_a_rule"] = "current constituents with first valid price <= 2009-01-01"
        info["excluded"] = [t for t in universe_lists[u] if t not in df.columns]
    manifest["frames"][key] = info
with open(os.path.join(snap, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
for k, v in manifest["frames"].items():
    print(f"{k:22s} {str(v['shape']):12s} {v['first']}..{v['last']} missing={v['n_missing']}"
          + (f" excluded={v['excluded']}" if 'excluded' in v else ""))
