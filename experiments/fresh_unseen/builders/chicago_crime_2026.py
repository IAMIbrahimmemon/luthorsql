"""Build chicago_crime_2026 sqlite from City of Chicago open data (Socrata). Re-runnable."""
import json, os, sqlite3, urllib.request, urllib.parse, csv
DB = "chicago_crime_2026"
ROOT = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = f"/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/{DB}"
OUT = f"{ROOT}/data/sys1/fresh/dbs/{DB}"
BASE = "https://data.cityofchicago.org/resource/"
os.makedirs(DL, exist_ok=True); os.makedirs(OUT, exist_ok=True)

def get(ds, params):
    url = BASE + ds + ".json?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.load(r)

def page(ds, params, size=50000):
    rows, off = [], 0
    while True:
        p = dict(params, **{"$limit": size, "$offset": off, "$order": ":id"})
        b = get(ds, p); rows += b
        if len(b) < size: return rows
        off += size

crimes = page("ijzp-q8t2", {"$where": "date>='2026-07-01T00:00:00'",
    "$select": "id,case_number,date,iucr,primary_type,description,location_description,arrest,domestic,beat,district,ward,community_area,fbi_code,year,updated_on"})
areas = get("igwz-8jzy", {"$select": "area_numbe,community", "$limit": 200})
dists = get("24zt-jpfn", {"$select": "dist_num,dist_label", "$limit": 200})
iucr = page("c7ck-438e", {"$select": "iucr,primary_description,secondary_description,index_code,active"})
json.dump({"crimes": crimes, "areas": areas, "dists": dists, "iucr": iucr}, open(f"{DL}/raw.json", "w"))

path = f"{OUT}/{DB}.sqlite"
if os.path.exists(path): os.remove(path)
c = sqlite3.connect(path); c.execute("PRAGMA foreign_keys=ON")
c.executescript("""
CREATE TABLE community_areas (community_area_id INTEGER PRIMARY KEY, community_area_name TEXT NOT NULL);
CREATE TABLE police_districts (district_id INTEGER PRIMARY KEY, district_label TEXT);
CREATE TABLE iucr_codes (iucr TEXT PRIMARY KEY, primary_description TEXT, secondary_description TEXT, index_code TEXT, active INTEGER);
CREATE TABLE crimes (
 crime_id INTEGER PRIMARY KEY, case_number TEXT, crime_date TEXT, iucr TEXT, primary_type TEXT, description TEXT,
 location_description TEXT, arrest INTEGER, domestic INTEGER, beat TEXT, district_id INTEGER, ward INTEGER,
 community_area_id INTEGER, fbi_code TEXT, year INTEGER, updated_on TEXT,
 FOREIGN KEY (iucr) REFERENCES iucr_codes(iucr),
 FOREIGN KEY (district_id) REFERENCES police_districts(district_id),
 FOREIGN KEY (community_area_id) REFERENCES community_areas(community_area_id));
""")
c.executemany("INSERT INTO community_areas VALUES (?,?)", [(int(a["area_numbe"]), a["community"]) for a in areas])
c.executemany("INSERT OR IGNORE INTO police_districts VALUES (?,?)", [(int(d["dist_num"]), d["dist_label"]) for d in dists])
c.executemany("INSERT INTO iucr_codes VALUES (?,?,?,?,?)", [(r["iucr"], r.get("primary_description"), r.get("secondary_description"), r.get("index_code"), int(bool(r.get("active")))) for r in iucr])
A = {int(a["area_numbe"]) for a in areas}; Dd = {int(d["dist_num"]) for d in dists}; I = {r["iucr"] for r in iucr}
def n(x, ok=None):
    if x in (None, ""): return None
    v = int(x); return v if ok is None or v in ok else None
rows = []
for r in crimes:
    iu = r.get("iucr"); iu = iu if iu in I else None
    rows.append((int(r["id"]), r.get("case_number"), r["date"].replace("T", " ")[:19], iu, r.get("primary_type"), r.get("description"),
      r.get("location_description"), int(r["arrest"]) if isinstance(r["arrest"], bool) else int(r["arrest"] == "true"),
      int(r["domestic"]) if isinstance(r["domestic"], bool) else int(r["domestic"] == "true"),
      r.get("beat"), n(r.get("district"), Dd), n(r.get("ward")), n(r.get("community_area"), A), r.get("fbi_code"),
      n(r.get("year")), (r.get("updated_on") or "").replace("T", " ")[:19]))
c.executemany("INSERT INTO crimes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
c.commit()
print("fk", c.execute("PRAGMA foreign_key_check").fetchall(), c.execute("PRAGMA integrity_check").fetchall())
cnt = {t: c.execute(f"select count(*) from {t}").fetchone()[0] for t in ["community_areas","police_districts","iucr_codes","crimes"]}
mm = c.execute("select min(crime_date),max(crime_date) from crimes").fetchone()
nulls = c.execute("select sum(iucr is null),sum(district_id is null),sum(community_area_id is null) from crimes").fetchone()
print(cnt, mm, nulls, os.path.getsize(path)/1e6, "MB")
json.dump({"db": DB, "sources": [
 {"url": BASE+"ijzp-q8t2.json", "licence": "City of Chicago Data Portal Terms of Use (open public data, Chicago Police Department)", "retrieved": "2026-10-21"},
 {"url": BASE+"igwz-8jzy.json", "licence": "City of Chicago Data Portal Terms of Use", "retrieved": "2026-10-21"},
 {"url": BASE+"24zt-jpfn.json", "licence": "City of Chicago Data Portal Terms of Use", "retrieved": "2026-10-21"},
 {"url": BASE+"c7ck-438e.json", "licence": "City of Chicago Data Portal Terms of Use", "retrieved": "2026-10-21"}],
 "date_filter": "crimes.date >= 2026-07-01", "tables": cnt,
 "notes": f"crimes date range {mm[0]} to {mm[1]} (full set, no sampling). Dropped block, coordinates. Unmatched FK values set NULL (iucr/district/community_area nulls: {nulls}). Reference tables unfiltered by date."},
 open(f"{ROOT}/experiments/fresh_unseen/sources/{DB}.json", "w"), indent=1)
