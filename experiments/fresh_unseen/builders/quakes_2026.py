"""Build quakes_2026 sqlite from USGS FDSN (public domain). Run: python3 -I quakes_2026.py"""
import json, os, re, sqlite3, csv, urllib.request, datetime
ROOT = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = "/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/quakes_2026"
OUT = f"{ROOT}/data/sys1/fresh/dbs/quakes_2026"
os.makedirs(DL, exist_ok=True); os.makedirs(f"{OUT}/database_description", exist_ok=True)
RANGES = [("2026-07-01","2026-08-01"),("2026-08-01","2026-09-01"),("2026-09-01","2026-10-01"),("2026-10-01","2026-10-16")]
feats = {}
for s, e in RANGES:
    fp = f"{DL}/{s}.geojson"
    if not os.path.exists(fp):
        u = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={s}&endtime={e}&minmagnitude=2.5&orderby=time-asc"
        urllib.request.urlretrieve(u, fp)
    d = json.load(open(fp))
    assert d["metadata"]["count"] < 20000
    for f in d["features"]: feats[f["id"]] = f

NETS = {"us":"USGS National Earthquake Information Center","ak":"Alaska Earthquake Center","pr":"Puerto Rico Seismic Network","nc":"USGS Northern California Seismic Network","tx":"Texas Seismological Network","ci":"Southern California Seismic Network","hv":"USGS Hawaiian Volcano Observatory","nn":"Nevada Seismological Laboratory","uu":"University of Utah Seismograph Stations","uw":"Pacific Northwest Seismic Network","av":"Alaska Volcano Observatory","se":"Southeast US Seismic Network","mb":"Montana Bureau of Mines and Geology","ok":"Oklahoma Geological Survey","ld":"Lamont-Doherty Cooperative Seismographic Network","nm":"Central United States Earthquake Consortium (New Madrid)","us_other":"Other"}
MAGT = {"mb":"Short-period body-wave magnitude","ml":"Local (Richter) magnitude","md":"Duration (coda) magnitude","mww":"Moment magnitude from W-phase inversion","mwr":"Moment magnitude from regional waveform inversion","mw":"Moment magnitude","mwb":"Moment magnitude from body waves","mwc":"Moment magnitude from centroid moment tensor","mb_lg":"Lg-wave body magnitude (regional)","ms":"Surface-wave magnitude","mi":"Integrated P-wave moment magnitude","mlg":"Lg-wave local magnitude","mh":"Hand-assigned magnitude","unknown":"Magnitude type not reported"}
STATES = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}
US_NAMES = set(STATES.values())
def region_of(place):
    p = (place or "").strip()
    if not p: return ("Unknown","unknown")
    if "," in p:
        suf = p.rsplit(",",1)[1].strip()
        if suf in STATES: return (STATES[suf],"us_state")
        if suf in US_NAMES: return (suf,"us_state")
        if suf in ("Puerto Rico","Guam","Northern Mariana Islands","U.S. Virgin Islands","American Samoa"): return (suf,"us_territory")
        return (suf,"country_or_area")
    p = re.sub(r"^\d+ km [NSEW]+ of ","",p)
    return (p,"offshore_region")
regions = {}
def rid(key):
    if key not in regions: regions[key] = len(regions)+1
    return regions[key]

db = f"{OUT}/quakes_2026.sqlite"
if os.path.exists(db): os.remove(db)
c = sqlite3.connect(db); c.execute("PRAGMA foreign_keys=ON")
c.executescript("""
CREATE TABLE networks(net_code TEXT PRIMARY KEY, network_name TEXT NOT NULL);
CREATE TABLE magnitude_types(mag_type_code TEXT PRIMARY KEY, description TEXT NOT NULL);
CREATE TABLE regions(region_id INTEGER PRIMARY KEY, region_name TEXT NOT NULL, region_type TEXT NOT NULL);
CREATE TABLE events(
 event_id TEXT PRIMARY KEY, event_time TEXT NOT NULL, event_date TEXT NOT NULL,
 latitude REAL, longitude REAL, depth_km REAL, magnitude REAL NOT NULL,
 mag_type_code TEXT REFERENCES magnitude_types(mag_type_code),
 net_code TEXT NOT NULL REFERENCES networks(net_code),
 region_id INTEGER NOT NULL REFERENCES regions(region_id),
 place TEXT, event_type TEXT, review_status TEXT, significance INTEGER, tsunami_flag INTEGER,
 num_stations INTEGER, azimuthal_gap REAL, min_station_dist_deg REAL, rms_residual REAL, updated_time TEXT);
CREATE TABLE event_impacts(
 event_id TEXT PRIMARY KEY REFERENCES events(event_id),
 felt_reports INTEGER, community_intensity REAL, max_instrumental_intensity REAL, pager_alert TEXT);
""")
iso = lambda ms: datetime.datetime.fromtimestamp(ms/1000, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S") if ms else None
nets=set(); mts=set(); ev=[]; imp=[]; reg_rows={}
for f in feats.values():
    p=f["properties"]; g=f["geometry"]["coordinates"]
    nets.add(p["net"]); mt=p["magType"] or "unknown"; mts.add(mt)
    rn,rt=region_of(p["place"]); r=rid((rn,rt)); reg_rows[r]=(r,rn,rt)
    t=iso(p["time"])
    ev.append((f["id"],t,t[:10],g[1],g[0],g[2],p["mag"],mt,p["net"],r,p["place"],p["type"],p["status"],p["sig"],p["tsunami"],p["nst"],p["gap"],p["dmin"],p["rms"],iso(p["updated"])))
    if any(p[k] is not None for k in ("felt","cdi","mmi","alert")):
        imp.append((f["id"],p["felt"],p["cdi"],p["mmi"],p["alert"]))
c.executemany("INSERT INTO networks VALUES(?,?)",[(n,NETS.get(n,f"Network '{n}' (name not mapped)")) for n in sorted(nets)])
c.executemany("INSERT INTO magnitude_types VALUES(?,?)",[(m,MAGT.get(m,"Other magnitude type")) for m in sorted(mts)])
c.executemany("INSERT INTO regions VALUES(?,?,?)",sorted(reg_rows.values()))
c.executemany("INSERT INTO events VALUES("+",".join("?"*20)+")",ev)
c.executemany("INSERT INTO event_impacts VALUES(?,?,?,?,?)",imp)
c.commit()
print("fk",c.execute("PRAGMA foreign_key_check").fetchall(),c.execute("PRAGMA integrity_check").fetchall())
counts={t:c.execute(f"select count(*) from {t}").fetchone()[0] for t in ("events","networks","magnitude_types","regions","event_impacts")}
print(counts); mm=c.execute("select min(event_time),max(event_time) from events").fetchone(); print(mm)

D=[("original_column_name","column_name","column_description","data_format","value_description")]
desc={
"networks":[("net_code","net_code","USGS network code of the contributing seismic network","text","e.g. us, ak, pr, nc, ci"),("network_name","network_name","Name of the network","text","")],
"magnitude_types":[("mag_type_code","mag_type_code","Magnitude type code","text","mb, ml, md, mww, mwr, mw, mwb, mb_lg"),("description","description","What the magnitude type means","text","")],
"regions":[("region_id","region_id","Surrogate region id","integer",""),("region_name","region_name","Region name derived from the text after the last comma of place (US state, country) or the offshore region name","text",""),("region_type","region_type","Kind of region","text","us_state, us_territory, country_or_area, offshore_region (no country in place text), unknown")],
"events":[("event_id","event_id","USGS event id","text",""),("event_time","event_time","Origin time, UTC","text","YYYY-MM-DD HH:MM:SS"),("event_date","event_date","UTC date of origin","text","YYYY-MM-DD"),("latitude","latitude","Epicentre latitude","real","degrees"),("longitude","longitude","Epicentre longitude","real","degrees"),("depth_km","depth_km","Hypocentre depth","real","km"),("magnitude","magnitude","Event magnitude (>= 2.5)","real",""),("mag_type_code","mag_type_code","Magnitude type; FK to magnitude_types","text",""),("net_code","net_code","Preferred-origin network; FK to networks","text",""),("region_id","region_id","FK to regions","integer",""),("place","place","USGS place description","text","e.g. 94 km SSE of Nikolski, Alaska"),("event_type","event_type","Event type","text","earthquake, quarry blast, explosion, etc."),("review_status","review_status","Review status","text","reviewed or automatic"),("significance","significance","USGS significance score","integer","0-1000+; larger = more significant"),("tsunami_flag","tsunami_flag","Tsunami-related flag","integer","1 = tsunami bulletin applies, 0 = no"),("num_stations","num_stations","Number of stations used","integer","may be null"),("azimuthal_gap","azimuthal_gap","Largest azimuthal gap between stations","real","degrees"),("min_station_dist_deg","min_station_dist_deg","Horizontal distance to nearest station","real","degrees"),("rms_residual","rms_residual","Root-mean-square travel-time residual","real","seconds"),("updated_time","updated_time","Last update time, UTC","text","")],
"event_impacts":[("event_id","event_id","FK to events; only events with felt/intensity/alert data","text",""),("felt","felt_reports","Number of Did You Feel It? reports","integer",""),("cdi","community_intensity","Maximum community decimal intensity","real","1-10"),("mmi","max_instrumental_intensity","Maximum instrumental intensity (ShakeMap)","real","1-10"),("alert","pager_alert","USGS PAGER alert level","text","green, yellow, orange, red")],
}
for t,rows in desc.items():
    with open(f"{OUT}/database_description/{t}.csv","w",newline="") as fh: csv.writer(fh).writerows(D+rows)
src={"db":"quakes_2026","sources":[{"url":"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2026-07-01&endtime=2026-10-16&minmagnitude=2.5 (fetched monthly)","licence":"US Government public domain (USGS)","retrieved":"2026-10-21"}],"date_filter":"origin time 2026-07-01 to 2026-10-15 UTC, magnitude >= 2.5, worldwide","tables":counts,"notes":f"events min/max event_time: {mm[0]} / {mm[1]}. Networks names, magnitude type descriptions hand-written lookups. Regions derived from place text (suffix after last comma; US state abbreviations expanded; no-comma places treated as offshore regions with 'N km DIR of ' stripped). event_impacts holds only events with felt/cdi/mmi/alert values."}
json.dump(src,open(f"{ROOT}/experiments/fresh_unseen/sources/quakes_2026.json","w"),indent=1)
