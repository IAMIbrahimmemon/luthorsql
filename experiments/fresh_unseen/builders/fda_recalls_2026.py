#!/usr/bin/env python3
"""Build fda_recalls_2026 from openFDA enforcement endpoints (public domain/CC0)."""
import json, os, sqlite3, csv, time, urllib.request
ROOT = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = "/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/fda_recalls_2026"
OUT = f"{ROOT}/data/sys1/fresh/dbs/fda_recalls_2026"
os.makedirs(DL, exist_ok=True); os.makedirs(f"{OUT}/database_description", exist_ok=True)
SEARCH = "report_date:[20260701+TO+20261021]"
def fetch(kind):
    path = f"{DL}/{kind}.json"
    if os.path.exists(path): return json.load(open(path))
    res, skip = [], 0
    while True:
        url = f"https://api.fda.gov/{kind}/enforcement.json?search={SEARCH}&limit=1000&skip={skip}"
        d = json.load(urllib.request.urlopen(url, timeout=120))
        res += d["results"]; skip += 1000
        if skip >= d["meta"]["results"]["total"]: break
        time.sleep(1)
    json.dump(res, open(path, "w")); return res
def dt(s):
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if s and len(s) == 8 else None
def n(s):
    s = (s or "").strip()
    return None if s in ("", "N/A") else s
recs = []
for k in ("drug", "device", "food"):
    for r in fetch(k): r["_kind"] = k; recs.append(r)
dbp = f"{OUT}/fda_recalls_2026.sqlite"
if os.path.exists(dbp): os.remove(dbp)
c = sqlite3.connect(dbp); c.execute("PRAGMA foreign_keys=ON")
c.executescript("""
CREATE TABLE classifications(classification TEXT PRIMARY KEY, severity_rank INTEGER, meaning TEXT);
CREATE TABLE firms(firm_id INTEGER PRIMARY KEY, firm_name TEXT, address_1 TEXT, address_2 TEXT, city TEXT, state TEXT, postal_code TEXT, country TEXT, UNIQUE(firm_name,address_1,city,state,postal_code,country));
CREATE TABLE events(event_id INTEGER PRIMARY KEY, product_category TEXT, recall_count INTEGER, first_report_date TEXT);
CREATE TABLE recalls(recall_number TEXT PRIMARY KEY, event_id INTEGER NOT NULL REFERENCES events(event_id), firm_id INTEGER NOT NULL REFERENCES firms(firm_id), classification TEXT REFERENCES classifications(classification), status TEXT, product_category TEXT, voluntary_mandated TEXT, initial_firm_notification TEXT, distribution_pattern TEXT, reason_for_recall TEXT, recall_initiation_date TEXT, center_classification_date TEXT, report_date TEXT, termination_date TEXT);
CREATE TABLE products(product_id INTEGER PRIMARY KEY, recall_number TEXT NOT NULL REFERENCES recalls(recall_number), product_description TEXT, product_type TEXT, code_info TEXT, product_quantity TEXT);
""")
c.executemany("INSERT INTO classifications VALUES(?,?,?)", [
 ("Class I",1,"Dangerous or defective products that predictably could cause serious health problems or death."),
 ("Class II",2,"Products that might cause a temporary health problem, or pose only a slight threat of a serious nature."),
 ("Class III",3,"Products unlikely to cause any adverse health reaction, but that violate FDA labeling or manufacturing laws."),
 ("Not Yet Classified",None,"Recall reported but FDA has not yet assigned a hazard class.")])
cat = {"drug":"Drugs","device":"Devices","food":"Food"}
firms, ev = {}, {}
for r in recs:
    key = tuple(n(r.get(x)) for x in ("recalling_firm","address_1","city","state","postal_code","country"))
    if key not in firms:
        firms[key] = len(firms)+1
        c.execute("INSERT INTO firms VALUES(?,?,?,?,?,?,?,?)", (firms[key], key[0], key[1], n(r.get("address_2")), key[2], key[3], key[4], key[5]))
    eid = int(r["event_id"]); ev.setdefault(eid, []).append(r)
for eid, rs in ev.items():
    cats = sorted({cat[x["_kind"]] for x in rs})
    c.execute("INSERT INTO events VALUES(?,?,?,?)", (eid, "/".join(cats), len(rs), min(dt(x["report_date"]) for x in rs)))
seen = set(); pid = 0
for r in recs:
    rn = r["recall_number"]
    if rn in seen: continue
    seen.add(rn)
    key = tuple(n(r.get(x)) for x in ("recalling_firm","address_1","city","state","postal_code","country"))
    c.execute("INSERT INTO recalls VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (rn,int(r["event_id"]),firms[key],r.get("classification"),r.get("status"),cat[r["_kind"]],r.get("voluntary_mandated"),r.get("initial_firm_notification"),r.get("distribution_pattern"),r.get("reason_for_recall"),dt(r.get("recall_initiation_date")),dt(r.get("center_classification_date")),dt(r.get("report_date")),dt(r.get("termination_date"))))
    pid += 1
    c.execute("INSERT INTO products VALUES(?,?,?,?,?,?)", (pid,rn,r.get("product_description"),r.get("product_type"),n(r.get("code_info")),r.get("product_quantity")))
c.commit()
D = {
"classifications":[("classification","classification","FDA hazard classification label","text","Class I / Class II / Class III / Not Yet Classified"),("severity_rank","severity_rank","1 = most severe; NULL if unclassified","integer","1,2,3"),("meaning","meaning","Meaning of the class","text","")],
"firms":[("firm_id","firm_id","Surrogate key of the recalling firm location","integer",""),("firm_name","firm_name","Recalling firm name (recalling_firm in openFDA)","text",""),("address_1","address_1","Street address line 1","text",""),("address_2","address_2","Street address line 2","text","NULL if N/A"),("city","city","City","text",""),("state","state","State or province code","text","may be NULL for non-US"),("postal_code","postal_code","Postal code","text",""),("country","country","Country","text","")],
"events":[("event_id","event_id","FDA recall event id grouping related recalls","integer",""),("product_category","product_category","Product center(s) of the event","text","Drugs, Devices, Food (joined by / if mixed)"),("recall_count","recall_count","Number of recalls in this dataset for the event","integer",""),("first_report_date","first_report_date","Earliest report_date among its recalls","date","YYYY-MM-DD")],
"recalls":[("recall_number","recall_number","Unique recall number","text","prefix D-=drug, Z-=device, F-/H-=food"),("event_id","event_id","Event the recall belongs to","integer","FK events"),("firm_id","firm_id","Recalling firm","integer","FK firms"),("classification","classification","Hazard class","text","Class I, II, III, Not Yet Classified; FK classifications"),("status","status","Recall status","text","Ongoing, Completed, Terminated, Pending"),("product_category","product_category","Source endpoint","text","Drugs, Devices, Food"),("voluntary_mandated","voluntary_mandated","Whether recall was voluntary or mandated","text","e.g. Voluntary: Firm initiated, FDA Mandated"),("initial_firm_notification","initial_firm_notification","How the firm first notified consignees","text","e.g. Letter, E-Mail, Press Release"),("distribution_pattern","distribution_pattern","Free-text areas of distribution","text",""),("reason_for_recall","reason_for_recall","Why the product was recalled","text",""),("recall_initiation_date","recall_initiation_date","Date firm began the recall","date","YYYY-MM-DD"),("center_classification_date","center_classification_date","Date FDA classified the recall","date","YYYY-MM-DD"),("report_date","report_date","Date the recall appeared in the enforcement report","date","YYYY-MM-DD; >= 2026-07-01"),("termination_date","termination_date","Date the recall was terminated","date","NULL if not terminated")],
"products":[("product_id","product_id","Surrogate key","integer",""),("recall_number","recall_number","Parent recall","text","FK recalls"),("product_description","product_description","Description of recalled product(s)","text",""),("product_type","product_type","FDA product type","text","Drugs, Devices, Food"),("code_info","code_info","Lot, serial, UDI or batch codes","text","may be NULL"),("product_quantity","product_quantity","Quantity recalled, free text with units","text","")],
}
for t, rows in D.items():
    with open(f"{OUT}/database_description/{t}.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["original_column_name","column_name","column_description","data_format","value_description"]); w.writerows(rows)
cnt = {t: c.execute(f"select count(*) from {t}").fetchone()[0] for t in D}
mn,mx = c.execute("select min(report_date),max(report_date) from recalls").fetchone()
json.dump({"db":"fda_recalls_2026","sources":[{"url":f"https://api.fda.gov/{k}/enforcement.json","licence":"CC0 1.0 / public domain (openFDA)","retrieved":"2026-10-21"} for k in cat],"date_filter":SEARCH,"tables":cnt,"notes":f"report_date range {mn} to {mx}. openFDA gives one product record per recall_number, so products is 1:1 with recalls. Firms deduplicated by name+address."},open(f"{ROOT}/experiments/fresh_unseen/sources/fda_recalls_2026.json","w"),indent=1)
print(cnt, mn, mx)
print(c.execute("PRAGMA foreign_key_check").fetchall(), c.execute("PRAGMA integrity_check").fetchall())
