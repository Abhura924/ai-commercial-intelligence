"""Storage layer (abstracted for cloud/SaaS). RAW never overwritten; audit kept."""
from __future__ import annotations
import os,json,uuid,datetime as dt
from dataclasses import dataclass,asdict
import pandas as pd
BASE=os.environ.get("ACI_DATA_DIR",os.path.join(os.path.dirname(__file__),"..","data"))
RAW=os.path.join(BASE,"raw"); PROCESSED=os.path.join(BASE,"processed"); METADATA=os.path.join(BASE,"metadata")
for _p in (RAW,PROCESSED,METADATA): os.makedirs(_p,exist_ok=True)
AUDIT=os.path.join(METADATA,"audit_log.jsonl")
@dataclass
class DatasetRecord:
    dataset_id:str; filename:str; uploaded_at:str; company:str; source:str; period_min:str|None; period_max:str|None; mapping:dict; quality:dict; status:str
class Storage:
    def new_id(self): return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:6]
    def save_raw(self,i,raw,fn): p=os.path.join(RAW,f"{i}__{fn}.parquet"); raw.to_parquet(p,index=False); return p
    def save_processed(self,i,c): p=os.path.join(PROCESSED,f"{i}.parquet"); c.to_parquet(p,index=False); return p
    def load_processed(self,i): return pd.read_parquet(os.path.join(PROCESSED,f"{i}.parquet"))
    def save_metadata(self,r):
        with open(os.path.join(METADATA,f"{r.dataset_id}.json"),"w") as f: json.dump(asdict(r),f,indent=2,default=str)
        with open(AUDIT,"a") as f: f.write(json.dumps(asdict(r),default=str)+"\n")
    def audit_log(self):
        if not os.path.exists(AUDIT): return []
        with open(AUDIT) as f: return [json.loads(l) for l in f if l.strip()]
store=Storage()
