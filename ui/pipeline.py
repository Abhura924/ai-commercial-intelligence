"""Ingestion pipeline: raw -> semantic mapping -> canonical -> quality -> store."""
from __future__ import annotations
import pandas as pd
from data_engine import semantic_mapping as sm
from data_engine import canonical_model as cm
from data_engine import data_quality as dq
from data_engine.storage import store, DatasetRecord

def read_upload(file, sheet=None):
    if file.name.lower().endswith(".csv"): return {"CSV": pd.read_csv(file)}
    xls = pd.ExcelFile(file); return {s: xls.parse(s) for s in xls.sheet_names}

def process(raw, mapping_override=None, filename="upload", company="demo-co", source="excel"):
    mapping = mapping_override or sm.infer_mapping(raw)
    canonical = cm.to_canonical(raw, mapping)
    findings = dq.run_checks(raw, canonical, mapping)
    summary = cm.canonical_summary(canonical)
    ds_id = store.new_id()
    try:
        store.save_raw(ds_id, raw, filename); store.save_processed(ds_id, canonical)
        store.save_metadata(DatasetRecord(dataset_id=ds_id, filename=filename, uploaded_at=ds_id.split("-")[0],
            company=company, source=source, period_min=summary.get("date_min"), period_max=summary.get("date_max"),
            mapping=mapping, quality=dq.summarise(findings), status="processed"))
    except Exception: pass
    return {"dataset_id": ds_id, "mapping": mapping, "canonical": canonical, "findings": findings,
            "summary": summary, "dictionary": sm.data_dictionary(raw, mapping), "derivations": sm.derivation_notes(mapping)}
