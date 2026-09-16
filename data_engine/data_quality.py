"""Data Quality Engine."""
from __future__ import annotations
import pandas as pd
def run_checks(raw,canonical,mapping):
    f=[]
    def add(s,m): f.append({"status":s,"message":m})
    for role,label in [("date","Date"),("revenue","Sales/Revenue"),("customer","Customer"),("product","Product")]:
        add("ok",f"{label} field identified ({mapping[role]}).") if role in mapping else add("warn",f"{label} field not identified.")
    for role in ["revenue","volume","cost"]:
        if role in canonical.columns and canonical[role].notna().any():
            miss=canonical[role].isna().mean()
            if miss>0: add("warn",f"{miss*100:.1f}% of rows missing {role}.")
    if raw.duplicated().sum(): add("warn",f"{raw.duplicated().sum()} duplicate row(s) detected.")
    if "date" in mapping:
        bad=pd.to_datetime(raw[mapping["date"]],errors="coerce").isna().sum()
        if bad: add("warn",f"{bad} row(s) have an unparseable date.")
    add("ok","Budget data present.") if "budget_revenue" in mapping else add("warn","Budget data incomplete.")
    add("ok","Prior-year data present.") if "py_revenue" in mapping else add("warn","Prior-year data missing.")
    if "market_growth" not in mapping: add("warn","Market data missing — market-share analysis limited.")
    return f
def summarise(f): return {"ok":sum(1 for x in f if x["status"]=="ok"),"warn":sum(1 for x in f if x["status"]=="warn"),"error":sum(1 for x in f if x["status"]=="error")}
