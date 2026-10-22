"""Build nws_alerts_2026 from the public NWS API (api.weather.gov, US Gov public domain).
Run: python3 -I builders/nws_alerts_2026.py
Note: /alerts only serves alerts issued/active in roughly the last week, so the
snapshot covers whatever the API returns at run time (start=2026-07-01 is passed but
the API retains no older records). Storm Events 2026 (NCEI) ends in June 2026, hence this fallback."""
import json, os, re, sqlite3, time, urllib.request
ROOT = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = "/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/nws_alerts_2026"
DB = f"{ROOT}/data/sys1/fresh/dbs/nws_alerts_2026/nws_alerts_2026.sqlite"
UA = "luthor-research-db-builder (public-data research; no contact)"
os.makedirs(DL, exist_ok=True)

def get(url):
    for i in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/geo+json"}), timeout=60))
        except Exception as e:
            time.sleep(2 * (i + 1)); err = e
    raise err

def cached(name, url):
    p = f"{DL}/{name}"
    if not os.path.exists(p):
        json.dump(get(url), open(p, "w")); time.sleep(0.3)
    return json.load(open(p))

# alerts
if not os.path.exists(f"{DL}/p0.json"):
    u, n = "https://api.weather.gov/alerts?start=2026-07-01T00:00:00Z&limit=500", 0
    while u:
        d = get(u)
        if not d["features"]: break
        json.dump(d, open(f"{DL}/p{n}.json", "w")); n += 1
        u = (d.get("pagination") or {}).get("next"); time.sleep(0.5)
feats = {}
i = 0
while os.path.exists(f"{DL}/p{i}.json"):
    for f in json.load(open(f"{DL}/p{i}.json"))["features"]: feats[f["properties"]["id"]] = f["properties"]
    i += 1
feats = {k: v for k, v in feats.items() if v["status"] == "Actual" and v["event"] != "Test Message"}

zones = {}
for t in ("forecast", "county", "fire"):
    for f in cached(f"z_{t}.json", f"https://api.weather.gov/zones?type={t}&include_geometry=false")["features"]:
        p = f["properties"]; zones[f"{t}/{p['id']}"] = (t, p)
office_ids = {c for _, p in zones.values() for c in (p.get("cwa") or [])}
offices = {}
for o in sorted(office_ids):
    try: offices[o] = cached(f"o_{o}.json", f"https://api.weather.gov/offices/{o}")
    except Exception as e: print("office fail", o, e)

def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
name2off = {}
for o, d in offices.items():
    nm = d["name"]
    name2off[norm("NWS " + nm)] = o
def office_for(p):
    w = (p.get("parameters", {}).get("WMOidentifier") or [""])[0].split()
    if len(w) > 1:
        for cand in (w[1][1:], w[1][-3:]):
            if cand in offices: return cand
    return name2off.get(norm(p.get("senderName") or ""))

if os.path.exists(DB): os.remove(DB)
c = sqlite3.connect(DB); c.execute("PRAGMA foreign_keys=ON")
c.executescript("""
CREATE TABLE offices(office_id TEXT PRIMARY KEY, office_name TEXT, city TEXT, state TEXT, postal_code TEXT, region TEXT, website TEXT);
CREATE TABLE zones(zone_key TEXT PRIMARY KEY, zone_code TEXT, zone_type TEXT, zone_name TEXT, state TEXT, office_id TEXT REFERENCES offices(office_id), time_zone TEXT, radar_station TEXT);
CREATE TABLE alerts(alert_id TEXT PRIMARY KEY, office_id TEXT REFERENCES offices(office_id), event TEXT, headline TEXT, sender_name TEXT, area_desc TEXT,
 sent TEXT, effective TEXT, onset TEXT, expires TEXT, ends TEXT, message_type TEXT, category TEXT, severity TEXT, certainty TEXT, urgency TEXT, response TEXT,
 max_wind_gust TEXT, max_hail_size TEXT, description TEXT, instruction TEXT);
CREATE TABLE alert_zones(alert_id TEXT NOT NULL REFERENCES alerts(alert_id), zone_key TEXT NOT NULL REFERENCES zones(zone_key), PRIMARY KEY(alert_id, zone_key));
""")
for o, d in offices.items():
    a = d.get("address") or {}
    c.execute("INSERT INTO offices VALUES(?,?,?,?,?,?,?)", (o, d["name"], a.get("addressLocality"), a.get("addressRegion"), a.get("postalCode"), d.get("nwsRegion"), d.get("sameAs")))
for k, (t, p) in zones.items():
    cw = (p.get("cwa") or [None])[0]
    tz = (p.get("timeZone") or [None])[0]
    c.execute("INSERT INTO zones VALUES(?,?,?,?,?,?,?,?)", (k, p["id"], t, p["name"], p.get("state") or None, cw if cw in offices else None, tz, p.get("radarStation")))
def one(p, k):
    v = p.get("parameters", {}).get(k); return v[0] if v else None
from datetime import datetime, timezone
def iso(s):
    return datetime.fromisoformat(s).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if s else None
for a, p in feats.items():
    c.execute("INSERT INTO alerts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (a, office_for(p), p["event"], p["headline"], p["senderName"], p["areaDesc"],
        iso(p["sent"]), iso(p["effective"]), iso(p["onset"]), iso(p["expires"]), iso(p["ends"]), p["messageType"], p["category"], p["severity"], p["certainty"], p["urgency"], p["response"],
        one(p, "maxWindGust"), one(p, "maxHailSize"), p["description"], p["instruction"]))
    for u in p["affectedZones"]:
        k = "/".join(u.split("/")[-2:])
        if k in zones: c.execute("INSERT OR IGNORE INTO alert_zones VALUES(?,?)", (a, k))
c.commit()
for t in ("offices", "zones", "alerts", "alert_zones"): print(t, c.execute(f"select count(*) from {t}").fetchone()[0])
print(c.execute("select min(sent),max(sent) from alerts").fetchone())
print("alerts without office", c.execute("select count(*) from alerts where office_id is null").fetchone())
