"""Data Understanding / Semantic Mapping Engine (deterministic, auditable)."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
SYNONYMS = {
    "date": ["date","month","period","posting date","fiscal","week","quarter","yr","year"],
    "customer": ["customer","account","client","buyer","sold to","customer name"],
    "product": ["product","material","sku","item","article","product line","grade"],
    "region": ["region","geography","country","area","territory","market region"],
    "business_unit": ["business unit","bu","division","segment","business"],
    "volume": ["volume","qty","quantity","units","units sold","kg","tonnes","litres","vol"],
    "revenue": ["revenue","sales","net sales","turnover","net revenue","gross sales","income"],
    "price": ["price","asp","avg price","unit price","selling price","rate"],
    "cost": ["cost","cogs","cost of goods","cost of sales","material cost","std cost"],
    "gross_margin": ["gross margin","gm","margin","contribution"],
    "opex": ["opex","operating expense","overhead","sg&a","sga","operating cost"],
    "budget_revenue": ["budget","plan","budget revenue","budget sales","plan revenue","target"],
    "budget_volume": ["budget volume","plan volume","budget qty","plan qty"],
    "budget_cost": ["budget cost","plan cost","budget cogs"], "budget_opex": ["budget opex","plan opex"],
    "py_revenue": ["prior year","py","last year","ly","py revenue","prior year sales","previous year"],
    "py_volume": ["py volume","prior year volume","ly volume"], "py_cost": ["py cost","prior year cost","ly cogs"],
    "market_growth": ["market growth","industry growth","market","sector growth","market volume","market index"]}
NUMERIC_ROLES = {"volume","revenue","price","cost","gross_margin","opex","budget_revenue","budget_volume","budget_cost","budget_opex","py_revenue","py_volume","py_cost","market_growth"}
def _norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def _score(c, syns):
    n=_norm(c); best=0.0
    for syn in syns:
        if n==syn: return 1.0
        if n.startswith(syn) or n.endswith(syn): best=max(best,0.9)
        elif syn in n.split(): best=max(best,0.8)
        elif syn in n: best=max(best,0.75)
    return best
def infer_mapping(df):
    headers=list(df.columns); numeric=set(df.select_dtypes(include=[np.number]).columns); cands=[]
    for role,syns in SYNONYMS.items():
        for col in headers:
            sc=_score(col,syns)
            if role in NUMERIC_ROLES and col not in numeric: sc*=0.4
            if sc>0: cands.append((sc,role,col))
    cands.sort(reverse=True); mapping,uc,ur={},set(),set()
    for sc,role,col in cands:
        if role in ur or col in uc or sc<0.7: continue
        mapping[role]=col; ur.add(role); uc.add(col)
    return mapping
def derivation_notes(mapping):
    n=[]
    if "gross_margin" not in mapping and "revenue" in mapping and "cost" in mapping: n.append("Gross Margin not supplied — derived as Revenue − COGS.")
    if "price" not in mapping and "revenue" in mapping and "volume" in mapping: n.append("Price/ASP not supplied — derived as Revenue ÷ Volume.")
    if "budget_revenue" not in mapping: n.append("No Budget column — budget comparatives unavailable.")
    if "py_revenue" not in mapping: n.append("No Prior-Year column — YoY comparatives unavailable.")
    if "market_growth" not in mapping: n.append("No Market/Industry growth — market-share analysis limited.")
    return n
def data_dictionary(df, mapping):
    rows=[]; inv={v:k for k,v in mapping.items()}
    for col in df.columns:
        rows.append({"Source column":col,"Mapped to":inv.get(col,"(unmapped)"),"Dtype":str(df[col].dtype),
            "Sample values":", ".join(map(lambda x:str(x)[:18], df[col].dropna().head(3).tolist()))})
    return pd.DataFrame(rows)
