"""Standardised (Canonical) FACT_SALES model + derivations."""
from __future__ import annotations
import numpy as np
import pandas as pd
CANONICAL_COLUMNS = ["date","customer","product","region","business_unit","volume","revenue","price","cost","gross_margin","opex","budget_revenue","budget_volume","budget_cost","budget_opex","py_revenue","py_volume","py_cost","market_growth"]
TEXT_ROLES = {"customer","product","region","business_unit"}
def to_canonical(df, mapping):
    out=pd.DataFrame(index=df.index)
    for role in CANONICAL_COLUMNS:
        col=mapping.get(role)
        if col is None or col not in df.columns: out[role]=np.nan; continue
        if role=="date": out[role]=pd.to_datetime(df[col],errors="coerce")
        elif role in TEXT_ROLES: out[role]=df[col].astype(str)
        else: out[role]=pd.to_numeric(df[col],errors="coerce")
    if out["gross_margin"].isna().all() and out["revenue"].notna().any() and out["cost"].notna().any(): out["gross_margin"]=out["revenue"]-out["cost"]
    if out["price"].isna().all() and out["revenue"].notna().any() and out["volume"].notna().any(): out["price"]=np.where(out["volume"]!=0,out["revenue"]/out["volume"],np.nan)
    for role in TEXT_ROLES:
        if out[role].isna().all(): out[role]="All"
    return out.dropna(subset=["revenue"]).reset_index(drop=True)
def canonical_summary(df):
    return {"rows":int(len(df)),"date_min":str(pd.to_datetime(df["date"]).min().date()) if df["date"].notna().any() else None,
        "date_max":str(pd.to_datetime(df["date"]).max().date()) if df["date"].notna().any() else None,
        "customers":int(df["customer"].nunique()) if "customer" in df else 0,"products":int(df["product"].nunique()) if "product" in df else 0,
        "has_budget":bool(df["budget_revenue"].notna().any()),"has_prior_year":bool(df["py_revenue"].notna().any()),"has_market":bool(df["market_growth"].notna().any())}
