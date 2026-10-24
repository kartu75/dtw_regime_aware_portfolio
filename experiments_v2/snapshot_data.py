# -*- coding: utf-8 -*-
"""
Freeze the price data used by every existing study into an on-disk snapshot.

Each download below repeats the exact yf.download call (same tickers, start, end,
auto_adjust=True, same per-ticker retry logic) made by the original notebook/scripts,
so each saved frame has the same shape and date index the original pipeline saw.
All later experiments read from this snapshot and never call yfinance.

Run once:  python snapshot_data.py
Output:    data/snapshot_<YYYYMMDD>/*.parquet + manifest.json
"""
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))

NIFTY50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "ITC", "INFY", "INDIGO",
    "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TMPV", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]
NIFTY50_TICKERS = [s.replace("&", "%26") + ".NS" if s == "M&M" else s + ".NS" for s in NIFTY50_SYMBOLS]

BANK_SYMBOLS = ["AUBANK", "AXISBANK", "BANKBARODA", "CANBK", "FEDERALBNK", "HDFCBANK",
                "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "PNB", "SBIN",
                "UNIONBANK", "YESBANK"]
BANK_TICKERS = [s + ".NS" for s in BANK_SYMBOLS]

MIDCAP_SYMBOLS = [
    "360ONE", "APLAPOLLO", "AUBANK", "ATGL", "ABCAPITAL", "ALKEM", "ASHOKLEY", "ASTRAL",
    "AUROPHARMA", "BSE", "BANKINDIA", "BDL", "BHARATFORG", "BHEL", "GROWW", "BIOCON",
    "BLUESTARCO", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "DABUR",
    "DIXON", "EXIDEIND", "NYKAA", "FEDERALBNK", "FORTIS", "GVT&D", "GMRAIRPORT",
    "GLENMARK", "GODFRYPHLP", "GODREJPROP", "HAVELLS", "HEROMOTOCO", "HINDPETRO",
    "POWERINDIA", "HUDCO", "ICICIGI", "ICICIAMC", "IDFCFIRSTB", "INDIANB", "IRCTC",
    "IREDA", "INDUSTOWER", "INDUSINDBK", "NAUKRI", "JSWENERGY", "JUBLFOOD", "KEI",
    "KPITTECH", "KALYANKJIL", "LTF", "LGEINDIA", "LICHSGFIN", "LAURUSLABS", "LENSKART",
    "LUPIN", "MRF", "M&MFIN", "MANKIND", "MARICO", "MFSL", "MOTILALOFS", "MPHASIS",
    "MCX", "NHPC", "NMDC", "NATIONALUM", "OBEROIRLTY", "OIL", "PAYTM", "OFSS",
    "POLICYBZR", "PIIND", "PAGEIND", "PATANJALI", "PERSISTENT", "PHOENIXLTD", "POLYCAB",
    "PREMIERENE", "PRESTIGE", "RADICO", "RVNL", "SBICARD", "SRF", "SAIL", "SUPREMEIND",
    "SUZLON", "SWIGGY", "TATACOMM", "TATAELXSI", "TATAINVEST", "TIINDIA", "UPL", "VMM",
    "IDEA", "VOLTAS", "WAAREEENER", "YESBANK",
]
MIDCAP_SPECIAL = {"M&MFIN": "M%26MFIN", "GVT&D": "GVT%26D"}
MIDCAP_TICKERS = [MIDCAP_SPECIAL.get(s, s) + ".NS" for s in MIDCAP_SYMBOLS]

RENAMES = {"M%26M.NS": "M&M.NS", "M%26MFIN.NS": "M&MFIN.NS", "GVT%26D.NS": "GVT&D.NS"}

END_FULL = "2026-08-09"
END_A = "2020-01-01"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a)
    sys.stdout.flush()


def download_universe(tickers, start, end):
    """Batch download + per-ticker retry of all-NaN columns, as in the original scripts."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    missing = data.isna().sum()
    failed = missing[missing == len(data)].index.tolist()
    retried = {}
    for t in failed:
        for attempt in range(3):
            try:
                df = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)['Close']
                if len(df) > 0:
                    retried[t] = df
                    break
            except Exception as e:
                log(f"  {t} attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    for t, s in retried.items():
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        data[t] = s.reindex(data.index)
    data = data.rename(columns=RENAMES)
    still_empty = [c for c in data.columns if data[c].notna().sum() == 0]
    return data, failed, list(retried), still_empty


def download_index(ticker, start, end):
    for attempt in range(3):
        try:
            s = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)['Close'].squeeze()
            if len(s) > 0:
                s.name = ticker
                return s
        except Exception as e:
            log(f"  {ticker} attempt {attempt + 1} failed: {e}")
        time.sleep(3)
    raise RuntimeError(f"index {ticker} could not be downloaded")


def period_a_qualifiers(tickers, retry_individually):
    """Current constituents with a first valid price on or before 2009-01-01 (probe window
    2008-01-01..2009-06-01), as in the two-period scripts."""
    probe = yf.download(tickers, start="2008-01-01", end="2009-06-01", auto_adjust=True, progress=False)['Close']
    cutoff = pd.Timestamp("2009-01-01")
    qualify, exclude = [], []
    for t in tickers:
        fv = probe[t].first_valid_index() if t in probe.columns else None
        if fv is None and retry_individually:
            for attempt in range(3):
                try:
                    df = yf.Ticker(t).history(start="2008-01-01", end="2009-06-01", auto_adjust=True)['Close']
                    if len(df) > 0:
                        fv = df.index.min()
                        if fv.tz is not None:
                            fv = fv.tz_localize(None)
                        break
                except Exception as e:
                    log(f"  probe {t} attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        if fv is not None and fv <= cutoff:
            qualify.append(t)
        else:
            exclude.append((t, None if fv is None else str(fv.date())))
    return qualify, exclude


def main():
    stamp = datetime.now()
    out_dir = os.path.join(HERE, "data", f"snapshot_{stamp:%Y%m%d}")
    os.makedirs(out_dir, exist_ok=False)
    manifest = dict(created=stamp.isoformat(timespec="seconds"), yfinance_version=yf.__version__,
                    pandas_version=pd.__version__, note="auto_adjust=True Close prices; frames mirror the "
                    "original per-study yf.download calls", frames={})

    def save(key, obj, **meta):
        path = os.path.join(out_dir, f"{key}.parquet")
        df = obj.to_frame() if isinstance(obj, pd.Series) else obj
        df.to_parquet(path)
        info = dict(file=os.path.basename(path), shape=list(df.shape),
                    first=str(df.index.min().date()), last=str(df.index.max().date()),
                    n_missing=int(df.isna().sum().sum()), **meta)
        manifest["frames"][key] = info
        log(f"saved {key}: shape={df.shape} {info['first']}..{info['last']} missing={info['n_missing']}")

    # ---- stock universes (one frame per original download call) ----
    specs = [
        ("stocks_N50_2015", NIFTY50_TICKERS, "2015-01-01", END_FULL),     # single split + expanding N50
        ("stocks_BANK_2015", BANK_TICKERS, "2015-01-01", END_FULL),       # expanding Bank
        ("stocks_MID_2015", MIDCAP_TICKERS, "2015-01-01", END_FULL),      # expanding Midcap
        ("stocks_N50_B", NIFTY50_TICKERS, "2020-01-01", END_FULL),        # two-period B, N50
        ("stocks_BANK_B", BANK_TICKERS, "2020-01-01", END_FULL),          # two-period B, Bank
    ]
    for key, tickers, start, end in specs:
        log(f"downloading {key} ({len(tickers)} tickers, {start}..{end})")
        df, failed, retried, empty = download_universe(tickers, start, end)
        save(key, df, start=start, end=end, retried=retried, still_empty=empty)
        time.sleep(2)

    for key, tickers, retry in [("N50", NIFTY50_TICKERS, False), ("BANK", BANK_TICKERS, True)]:
        log(f"probing Period A coverage for {key}")
        qualify, exclude = period_a_qualifiers(tickers, retry)
        log(f"  {key} Period A qualifying: {len(qualify)}/{len(tickers)}; excluded: {exclude}")
        df, failed, retried, empty = download_universe(qualify, "2008-01-01", END_A)
        save(f"stocks_{key}_A", df, start="2008-01-01", end=END_A, retried=retried, still_empty=empty,
             qualifying=[RENAMES.get(t, t) for t in qualify], excluded=exclude)
        time.sleep(2)

    # ---- indices (regime detection + market series for Treynor) ----
    idx_specs = [
        ("^NSEI", "2015-01-01", END_FULL), ("^NSEI", "2008-01-01", END_A), ("^NSEI", "2020-01-01", END_FULL),
        ("^NSEBANK", "2015-01-01", END_FULL), ("^NSEBANK", "2008-01-01", END_A), ("^NSEBANK", "2020-01-01", END_FULL),
        ("NIFTY_MIDCAP_100.NS", "2015-01-01", END_FULL),
    ]
    for ticker, start, end in idx_specs:
        key = f"index_{ticker.replace('^', '').replace('.NS', '')}_{start[:4]}"
        save(key, download_index(ticker, start, end), ticker=ticker, start=start, end=end)
        time.sleep(1)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"snapshot complete: {out_dir}")


if __name__ == "__main__":
    main()
