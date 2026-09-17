"""Realistic fictional chemical-manufacturing demo dataset (5 cust x 5 prod x 24 months)."""
from __future__ import annotations
import numpy as np
import pandas as pd
CUSTOMERS=["Customer A","Customer B","Customer C","Customer D","Customer E"]
PRODUCTS=["Product A","Product B","Product C","Product D","Product E"]
CUST={"Customer A":(0.14,1.30),"Customer B":(0.05,1.00),"Customer C":(-0.12,0.90),"Customer D":(0.02,0.75),"Customer E":(0.07,0.60)}
PROD={"Product A":(920,0.03,0.34,0.000),"Product B":(610,0.05,0.41,0.004),"Product C":(275,0.02,0.28,-0.002),"Product D":(480,-0.02,0.24,-0.006),"Product E":(1350,0.04,0.46,0.001)}
def build_canonical(seed=4242):
    rng=np.random.default_rng(seed); months=pd.date_range("2023-01-01","2024-12-01",freq="MS"); rows=[]
    for cust in CUSTOMERS:
        cg,cscale=CUST[cust]
        for prod in PRODUCTS:
            bp,pg,bgm,gmd=PROD[prod]; bu=rng.uniform(60,160)*cscale
            for i,m in enumerate(months):
                yrs=i/12.0; seas=1+0.06*np.sin((m.month-1)/12*2*np.pi)
                units=bu*(1+cg)**yrs*seas*(1+rng.normal(0,0.05)); price=bp*(1+pg)**yrs*(1+rng.normal(0,0.02)); rev=units*price
                gmr=float(np.clip(bgm+gmd*yrs+rng.normal(0,0.015),0.08,0.6)); cost=rev*(1-gmr)
                b_units=bu*(1+max(cg,0.03))**yrs*seas; b_price=bp*(1+pg)**yrs; b_rev=b_units*b_price; b_cost=b_rev*(1-bgm)
                mg=0.06+0.02*np.sin(yrs)+rng.normal(0,0.005); opex=rev*(0.16-0.01*yrs); b_opex=b_rev*0.155
                rows.append(dict(date=m,customer=cust,product=prod,region="EMEA",business_unit="Chemicals",volume=units,revenue=rev,price=price,cost=cost,gross_margin=rev-cost,opex=opex,budget_revenue=b_rev,budget_volume=b_units,budget_cost=b_cost,budget_opex=b_opex,market_growth=mg))
    df=pd.DataFrame(rows).sort_values(["customer","product","date"]).reset_index(drop=True)
    for col,pyc in [("revenue","py_revenue"),("volume","py_volume"),("cost","py_cost")]:
        df[pyc]=df.groupby(["customer","product"])[col].shift(12)
    return df[df["date"]>="2024-01-01"].reset_index(drop=True)
def build_raw_excel_like(seed=4242):
    c=build_canonical(seed)
    return pd.DataFrame({"Account":c["customer"],"Material":c["product"],"Period":c["date"],"Qty":c["volume"].round(1),"Turnover":c["revenue"].round(2),"Cost":c["cost"].round(2),"Plan":c["budget_revenue"].round(2),"Industry Growth":c["market_growth"].round(4)})
