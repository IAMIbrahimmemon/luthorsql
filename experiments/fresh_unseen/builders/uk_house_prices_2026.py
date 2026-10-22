"""Build uk_house_prices_2026 from HM Land Registry Price Paid Data (OGL v3). Re-runnable.
Keeps transfers dated >= 2026-07-01 from pp-2026.csv, overlaid with the monthly update file
(monthly rows win on transaction id; they carry record_status C/D changes). No sampling needed (<60MB).
paon/saon dropped; a property is (postcode, street, locality, town_city, district, property_type)."""
import csv, os, sqlite3, json, subprocess, re, collections
ROOT = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = "/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/uk_house_prices_2026"
B = "http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/"
OUT = f"{ROOT}/data/sys1/fresh/dbs/uk_house_prices_2026"
CUT = "2026-07-01"
os.makedirs(DL, exist_ok=True)
for u, f in [("pp-2026.csv", "pp-2026.csv"), ("pp-monthly-update-new-version.csv", "monthly.csv")]:
    if not os.path.exists(f"{DL}/{f}"):
        subprocess.check_call(["curl", "-sSLf", "-o", f"{DL}/{f}", B + u])
rows = {}
for f in ["pp-2026.csv", "monthly.csv"]:
    for r in csv.reader(open(f"{DL}/{f}", encoding="utf-8", errors="replace")):
        if len(r) == 16 and r[2][:10] >= CUT:
            rows[r[0].strip("{}")] = r
PT = [("D","Detached"),("S","Semi-Detached"),("T","Terraced"),("F","Flats/Maisonettes"),("O","Other")]
TEN = [("F","Freehold"),("L","Leasehold")]
PPD = [("A","Standard price paid transaction"),("B","Additional price paid transaction (e.g. buy-to-let, repossession, company purchase)")]
ST = [("A","Addition: new record"),("C","Change: amendment to an existing record"),("D","Delete: record removed")]
NB = [("Y","Newly built"),("N","Not newly built (established residential building)")]
dbp = f"{OUT}/uk_house_prices_2026.sqlite"
if os.path.exists(dbp): os.remove(dbp)
c = sqlite3.connect(dbp); c.execute("PRAGMA foreign_keys=ON")
c.executescript("""
CREATE TABLE property_types(property_type_code TEXT PRIMARY KEY, property_type_name TEXT NOT NULL);
CREATE TABLE tenure_types(tenure_code TEXT PRIMARY KEY, tenure_name TEXT NOT NULL);
CREATE TABLE ppd_categories(ppd_category_code TEXT PRIMARY KEY, ppd_category_name TEXT NOT NULL);
CREATE TABLE record_statuses(record_status_code TEXT PRIMARY KEY, record_status_name TEXT NOT NULL);
CREATE TABLE new_build_flags(new_build_code TEXT PRIMARY KEY, new_build_name TEXT NOT NULL);
CREATE TABLE counties(county_id INTEGER PRIMARY KEY, county_name TEXT NOT NULL UNIQUE);
CREATE TABLE districts(district_id INTEGER PRIMARY KEY, district_name TEXT NOT NULL UNIQUE, county_id INTEGER NOT NULL REFERENCES counties(county_id));
CREATE TABLE postcode_areas(outward_code TEXT PRIMARY KEY, postcode_area TEXT NOT NULL, main_district_id INTEGER REFERENCES districts(district_id), main_town_city TEXT, property_count INTEGER);
CREATE TABLE properties(property_id INTEGER PRIMARY KEY, postcode TEXT, outward_code TEXT REFERENCES postcode_areas(outward_code), street TEXT, locality TEXT, town_city TEXT, district_id INTEGER NOT NULL REFERENCES districts(district_id), property_type_code TEXT NOT NULL REFERENCES property_types(property_type_code));
CREATE TABLE transactions(transaction_id TEXT PRIMARY KEY, property_id INTEGER NOT NULL REFERENCES properties(property_id), price INTEGER NOT NULL, date_of_transfer TEXT NOT NULL, new_build_code TEXT NOT NULL REFERENCES new_build_flags(new_build_code), tenure_code TEXT NOT NULL REFERENCES tenure_types(tenure_code), ppd_category_code TEXT NOT NULL REFERENCES ppd_categories(ppd_category_code), record_status_code TEXT NOT NULL REFERENCES record_statuses(record_status_code));
CREATE INDEX ix_tx_prop ON transactions(property_id); CREATE INDEX ix_tx_date ON transactions(date_of_transfer);
CREATE INDEX ix_prop_dist ON properties(district_id);
""")
for t,col,v in [("property_types","property_type_code, property_type_name",PT),("tenure_types","tenure_code, tenure_name",TEN),("ppd_categories","ppd_category_code, ppd_category_name",PPD),("record_statuses","record_status_code, record_status_name",ST),("new_build_flags","new_build_code, new_build_name",NB)]:
    c.executemany(f"INSERT INTO {t}({col}) VALUES(?,?)", v)
dist2county = {}
for r in rows.values():
    dist2county.setdefault(r[12].strip(), collections.Counter())[r[13].strip()] += 1
counties = sorted({cn.most_common(1)[0][0] for cn in dist2county.values()})
cid = {n:i+1 for i,n in enumerate(counties)}
c.executemany("INSERT INTO counties VALUES(?,?)", [(i,n) for n,i in cid.items()])
did = {}
for i,(d,cn) in enumerate(sorted(dist2county.items()),1):
    did[d]=i; c.execute("INSERT INTO districts VALUES(?,?,?)",(i,d,cid[cn.most_common(1)[0][0]]))
pid = {}; oc_stat = {}; txs = []
for tid, r in sorted(rows.items()):
    pc=r[3].strip(); oc=pc.split(" ")[0] if pc else None
    key=(pc,r[9].strip(),r[10].strip(),r[11].strip(),did[r[12].strip()],r[4])
    if key not in pid:
        pid[key]=len(pid)+1
        if oc:
            s=oc_stat.setdefault(oc,[collections.Counter(),collections.Counter()]); s[0][key[4]]+=1; s[1][key[3]]+=1
    txs.append((tid,pid[key],int(r[1]),r[2][:10],r[5],r[6],r[14],r[15]))
for oc,(dc,tc) in sorted(oc_stat.items()):
    c.execute("INSERT INTO postcode_areas VALUES(?,?,?,?,?)",(oc,re.match(r"[A-Z]+",oc).group(0),dc.most_common(1)[0][0],tc.most_common(1)[0][0],sum(dc.values())))
c.executemany("INSERT INTO properties VALUES(?,?,?,?,?,?,?,?)",[(i,k[0] or None,(k[0].split(" ")[0] if k[0] else None),k[1] or None,k[2] or None,k[3] or None,k[4],k[5]) for k,i in pid.items()])
c.executemany("INSERT INTO transactions VALUES(?,?,?,?,?,?,?,?)",txs)
c.commit()
counts={t:c.execute(f"select count(*) from {t}").fetchone()[0] for t in ["transactions","properties","districts","counties","postcode_areas","property_types","tenure_types","ppd_categories","record_statuses","new_build_flags"]}
mm=c.execute("select min(date_of_transfer),max(date_of_transfer) from transactions").fetchone()
print(counts,mm,c.execute("PRAGMA foreign_key_check").fetchall(),c.execute("PRAGMA integrity_check").fetchall())
D={
"transactions":[("transaction_id","transaction_id","Land Registry unique transaction identifier (GUID without braces)","text","primary key"),("property_id","property_id","Property the transfer relates to","integer","FK to properties"),("price","price","Sale price in GBP stated on the transfer deed","integer",""),("date_of_transfer","date_of_transfer","Date of the transfer (completion)","date YYYY-MM-DD","2026-07-01 onward"),("new_build_code","new_build_code","Newly built flag","text","Y=newly built; N=established"),("tenure_code","tenure_code","Estate type","text","F=Freehold; L=Leasehold"),("ppd_category_code","ppd_category_code","Price Paid Data category","text","A=standard; B=additional (buy-to-let, repossession, company etc.)"),("record_status_code","record_status_code","Monthly-file record status","text","A=addition; C=change; D=delete")],
"properties":[("property_id","property_id","Generated surrogate id","integer","primary key; one per postcode+street+locality+town+district+type (house number/flat dropped)"),("postcode","postcode","Full postcode","text",""),("outward_code","outward_code","First half of postcode","text","FK to postcode_areas"),("street","street","Street name","text","may be null"),("locality","locality","Locality","text","may be null"),("town_city","town_city","Town or city","text",""),("district_id","district_id","Local authority district","integer","FK to districts"),("property_type_code","property_type_code","Property type","text","D,S,T,F,O; FK to property_types")],
"districts":[("district_id","district_id","Generated district id","integer","primary key"),("district_name","district_name","District name (upper case)","text",""),("county_id","county_id","County of the district (most frequent in data)","integer","FK to counties")],
"counties":[("county_id","county_id","Generated county id","integer","primary key"),("county_name","county_name","County name (upper case)","text","")],
"postcode_areas":[("outward_code","outward_code","Postcode outward code e.g. SW1A","text","primary key"),("postcode_area","postcode_area","Leading letters of the outward code (postcode area)","text",""),("main_district_id","main_district_id","Most frequent district among properties in the outward code","integer","FK to districts"),("main_town_city","main_town_city","Most frequent town/city among its properties","text",""),("property_count","property_count","Number of properties in this table with this outward code","integer","")],
"property_types":[("property_type_code","property_type_code","Code","text","primary key"),("property_type_name","property_type_name","Meaning","text","D=Detached; S=Semi-Detached; T=Terraced; F=Flats/Maisonettes; O=Other")],
"tenure_types":[("tenure_code","tenure_code","Code","text","primary key"),("tenure_name","tenure_name","Meaning","text","F=Freehold; L=Leasehold")],
"ppd_categories":[("ppd_category_code","ppd_category_code","Code","text","primary key"),("ppd_category_name","ppd_category_name","Meaning","text","A, B")],
"record_statuses":[("record_status_code","record_status_code","Code","text","primary key"),("record_status_name","record_status_name","Meaning","text","A, C, D")],
"new_build_flags":[("new_build_code","new_build_code","Code","text","primary key"),("new_build_name","new_build_name","Meaning","text","Y, N")]}
for t,cols in D.items():
    with open(f"{OUT}/database_description/{t}.csv","w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["original_column_name","column_name","column_description","data_format","value_description"]); w.writerows(cols)
json.dump({"db":"uk_house_prices_2026","sources":[{"url":B+"pp-2026.csv","licence":"Open Government Licence v3.0 (HM Land Registry Price Paid Data)","retrieved":"2026-10-21"},{"url":B+"pp-monthly-update-new-version.csv","licence":"Open Government Licence v3.0","retrieved":"2026-10-21"}],"date_filter":"date_of_transfer >= 2026-07-01","tables":counts,"notes":f"transactions date range {mm[0]} to {mm[1]}. No sampling: all rows kept. paon/saon dropped. Monthly file overrides yearly on transaction id. County per district = most frequent in data. postcode_areas has no area names (no open lookup used); area = leading letters."},open(f"{ROOT}/experiments/fresh_unseen/sources/uk_house_prices_2026.json","w"),indent=1)
