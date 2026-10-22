#!/usr/bin/env python3
"""Build f1_2026.sqlite from the Jolpica F1 API (Ergast-compatible). Races on/after 2026-07-01 only."""
import json, os, sqlite3, time, urllib.request, urllib.error, csv, datetime
ROOT = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = "/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/f1_2026"
OUT = f"{ROOT}/data/sys1/fresh/dbs/f1_2026"
BASE = "https://api.jolpi.ca/ergast/f1/2026"
CUT = "2026-07-01"; TODAY = "2026-10-21"
os.makedirs(DL, exist_ok=True); os.makedirs(OUT + "/database_description", exist_ok=True)

def get(path):
    f = f"{DL}/" + path.replace("/", "_") + ".json"
    if os.path.exists(f): return json.load(open(f))
    for a in range(6):
        time.sleep(0.3)
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{BASE}/{path}.json?limit=200", headers={"User-Agent": "luthor-research"}), timeout=30) as r:
                d = json.load(r)
            json.dump(d, open(f, "w")); return d
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(2 ** (a + 1)); continue
            raise
    raise RuntimeError(path)

def fnum(x):
    try: return float(x)
    except: return None
def inum(x):
    try: return int(x)
    except: return None
def secs(t):  # "1:23.456" -> seconds
    if not t: return None
    try:
        p = t.split(":"); return round(sum(float(v) * 60 ** i for i, v in enumerate(reversed(p))), 3)
    except: return None

races = [r for r in get("races")["MRData"]["RaceTable"]["Races"] if CUT <= r["date"] <= TODAY]
circuits, drivers, cons, R, Q, S, P, DS, CS, SP = {}, {}, {}, [], [], [], [], [], [], []
def addc(c): circuits[c["circuitId"]] = (c["circuitId"], c["circuitName"], c["Location"]["locality"], c["Location"]["country"], fnum(c["Location"]["lat"]), fnum(c["Location"]["long"]))
def addd(d): drivers[d["driverId"]] = (d["driverId"], d.get("code"), inum(d.get("permanentNumber")), d["givenName"], d["familyName"], d.get("dateOfBirth"), d.get("nationality"))
def addk(k): cons[k["constructorId"]] = (k["constructorId"], k["name"], k.get("nationality"))
def res_rows(rd, rid, lst, sprint):
    out = []
    for x in lst:
        addd(x["Driver"]); addk(x["Constructor"])
        out.append((rid, x["Driver"]["driverId"], x["Constructor"]["constructorId"], inum(x["number"]), inum(x["grid"]), inum(x["position"]), x["positionText"], fnum(x["points"]), inum(x["laps"]), x["status"], inum(x.get("Time", {}).get("millis")) if "Time" in x else None, x.get("Time", {}).get("time"), inum(x.get("FastestLap", {}).get("rank")), inum(x.get("FastestLap", {}).get("lap")), secs(x.get("FastestLap", {}).get("Time", {}).get("time"))))
    return out
RACES = []
for r in races:
    rd = r["round"]; rid = int(rd)
    addc(r["Circuit"])
    RACES.append((rid, r["raceName"], r["Circuit"]["circuitId"], r["date"], r.get("time"), 1 if r.get("Sprint") or r.get("SprintQualifying") else 0))
    rr = get(f"{rd}/results")["MRData"]["RaceTable"]["Races"]
    if rr: R += res_rows(rd, rid, rr[0]["Results"], False)
    sp = get(f"{rd}/sprint")["MRData"]["RaceTable"]["Races"]
    if sp:
        SP += res_rows(rd, rid, sp[0]["SprintResults"], True); RACES[-1] = RACES[-1][:5] + (1,)
    q = get(f"{rd}/qualifying")["MRData"]["RaceTable"]["Races"]
    if q:
        for x in q[0]["QualifyingResults"]:
            addd(x["Driver"]); addk(x["Constructor"])
            Q.append((rid, x["Driver"]["driverId"], x["Constructor"]["constructorId"], inum(x["number"]), inum(x["position"]), x.get("Q1") or None, x.get("Q2") or None, x.get("Q3") or None))
    p = get(f"{rd}/pitstops")["MRData"]["RaceTable"]["Races"]
    if p:
        for x in p[0]["PitStops"]:
            P.append((rid, x["driverId"], inum(x["stop"]), inum(x["lap"]), x["time"], fnum(x["duration"])))
    ds = get(f"{rd}/driverstandings")["MRData"]["StandingsTable"]["StandingsLists"]
    if ds:
        for x in ds[0]["DriverStandings"]:
            addd(x["Driver"])
            for k in x["Constructors"][-1:]: addk(k)
            DS.append((rid, x["Driver"]["driverId"], x["Constructors"][-1]["constructorId"] if x["Constructors"] else None, inum(x["position"]), fnum(x["points"]), inum(x["wins"])))
    cs = get(f"{rd}/constructorstandings")["MRData"]["StandingsTable"]["StandingsLists"]
    if cs:
        for x in cs[0]["ConstructorStandings"]:
            addk(x["Constructor"]); CS.append((rid, x["Constructor"]["constructorId"], inum(x["position"]), fnum(x["points"]), inum(x["wins"])))
# drop pit stops for drivers unseen (safety)
P = [p for p in P if p[1] in drivers]

fn = f"{OUT}/f1_2026.sqlite"
if os.path.exists(fn): os.remove(fn)
db = sqlite3.connect(fn); db.execute("PRAGMA foreign_keys=ON")
db.executescript("""
CREATE TABLE circuits(circuit_id TEXT PRIMARY KEY, circuit_name TEXT NOT NULL, locality TEXT, country TEXT, latitude REAL, longitude REAL);
CREATE TABLE drivers(driver_id TEXT PRIMARY KEY, driver_code TEXT, car_number INTEGER, given_name TEXT, family_name TEXT, date_of_birth TEXT, nationality TEXT);
CREATE TABLE constructors(constructor_id TEXT PRIMARY KEY, constructor_name TEXT NOT NULL, nationality TEXT);
CREATE TABLE races(round INTEGER PRIMARY KEY, race_name TEXT NOT NULL, circuit_id TEXT NOT NULL REFERENCES circuits(circuit_id), race_date TEXT NOT NULL, race_time_utc TEXT, has_sprint INTEGER);
CREATE TABLE results(round INTEGER NOT NULL REFERENCES races(round), driver_id TEXT NOT NULL REFERENCES drivers(driver_id), constructor_id TEXT NOT NULL REFERENCES constructors(constructor_id), car_number INTEGER, grid_position INTEGER, finish_position INTEGER, position_text TEXT, points REAL, laps_completed INTEGER, status TEXT, race_time_ms INTEGER, race_time_text TEXT, fastest_lap_rank INTEGER, fastest_lap_number INTEGER, fastest_lap_seconds REAL, PRIMARY KEY(round, driver_id));
CREATE TABLE sprint_results(round INTEGER NOT NULL REFERENCES races(round), driver_id TEXT NOT NULL REFERENCES drivers(driver_id), constructor_id TEXT NOT NULL REFERENCES constructors(constructor_id), car_number INTEGER, grid_position INTEGER, finish_position INTEGER, position_text TEXT, points REAL, laps_completed INTEGER, status TEXT, race_time_ms INTEGER, race_time_text TEXT, fastest_lap_rank INTEGER, fastest_lap_number INTEGER, fastest_lap_seconds REAL, PRIMARY KEY(round, driver_id));
CREATE TABLE qualifying(round INTEGER NOT NULL REFERENCES races(round), driver_id TEXT NOT NULL REFERENCES drivers(driver_id), constructor_id TEXT NOT NULL REFERENCES constructors(constructor_id), car_number INTEGER, quali_position INTEGER, q1_time TEXT, q2_time TEXT, q3_time TEXT, PRIMARY KEY(round, driver_id));
CREATE TABLE pit_stops(round INTEGER NOT NULL REFERENCES races(round), driver_id TEXT NOT NULL REFERENCES drivers(driver_id), stop_number INTEGER NOT NULL, lap INTEGER, local_time TEXT, duration_seconds REAL, PRIMARY KEY(round, driver_id, stop_number));
CREATE TABLE driver_standings(round INTEGER NOT NULL REFERENCES races(round), driver_id TEXT NOT NULL REFERENCES drivers(driver_id), constructor_id TEXT REFERENCES constructors(constructor_id), standing_position INTEGER, points REAL, wins INTEGER, PRIMARY KEY(round, driver_id));
CREATE TABLE constructor_standings(round INTEGER NOT NULL REFERENCES races(round), constructor_id TEXT NOT NULL REFERENCES constructors(constructor_id), standing_position INTEGER, points REAL, wins INTEGER, PRIMARY KEY(round, constructor_id));
""")
def ins(t, rows):
    if rows: db.executemany(f"INSERT OR REPLACE INTO {t} VALUES({','.join('?'*len(rows[0]))})", rows)
ins("circuits", list(circuits.values())); ins("drivers", list(drivers.values())); ins("constructors", list(cons.values()))
ins("races", RACES); ins("results", R); ins("sprint_results", SP); ins("qualifying", Q); ins("pit_stops", P); ins("driver_standings", DS); ins("constructor_standings", CS)
db.commit()

DESC = {
"circuits": [("circuit_id","slug id of the circuit","text"),("circuit_name","official circuit name","text"),("locality","town or city","text"),("country","country","text"),("latitude","latitude in degrees","real"),("longitude","longitude in degrees","real")],
"drivers": [("driver_id","slug id of the driver","text"),("driver_code","three-letter timing code, e.g. VER","text"),("car_number","permanent car number","integer"),("given_name","first name(s)","text"),("family_name","surname","text"),("date_of_birth","date of birth, YYYY-MM-DD","text"),("nationality","nationality adjective, e.g. Dutch","text")],
"constructors": [("constructor_id","slug id of the team","text"),("constructor_name","team name","text"),("nationality","team nationality adjective","text")],
"races": [("round","championship round number in the 2026 season","integer"),("race_name","grand prix name","text"),("circuit_id","circuit; links to circuits","text"),("race_date","race date, YYYY-MM-DD","text"),("race_time_utc","scheduled start time, UTC","text"),("has_sprint","1 if the weekend had a sprint, else 0","integer")],
"results": [("round","race round; links to races","integer"),("driver_id","driver; links to drivers","text"),("constructor_id","team driven for; links to constructors","text"),("car_number","car number used","integer"),("grid_position","starting grid slot; 0 means pit lane start","integer"),("finish_position","classified finishing position; NULL if not classified","integer"),("position_text","finishing position as text, or R retired, D disqualified, W withdrawn, N not classified","text"),("points","championship points scored","real"),("laps_completed","laps completed","integer"),("status","finish status, e.g. Finished, +1 Lap, or retirement reason","text"),("race_time_ms","total race time in ms; NULL for most non-winners behind the leader lap","integer"),("race_time_text","race time or gap to winner as text","text"),("fastest_lap_rank","rank of driver's fastest lap in the race","integer"),("fastest_lap_number","lap on which fastest lap was set","integer"),("fastest_lap_seconds","fastest lap time in seconds","real")],
"sprint_results": [("round","race weekend round; links to races","integer"),("driver_id","driver; links to drivers","text"),("constructor_id","team; links to constructors","text"),("car_number","car number","integer"),("grid_position","sprint grid slot","integer"),("finish_position","sprint finishing position","integer"),("position_text","position as text or status letter (R, D, W, N)","text"),("points","sprint points scored","real"),("laps_completed","laps completed","integer"),("status","finish status","text"),("race_time_ms","total sprint time in ms","integer"),("race_time_text","sprint time or gap as text","text"),("fastest_lap_rank","rank of fastest lap","integer"),("fastest_lap_number","lap of fastest lap","integer"),("fastest_lap_seconds","fastest lap time in seconds","real")],
"qualifying": [("round","race round; links to races","integer"),("driver_id","driver; links to drivers","text"),("constructor_id","team; links to constructors","text"),("car_number","car number","integer"),("quali_position","final qualifying position","integer"),("q1_time","best Q1 lap, M:SS.mmm; NULL if none","text"),("q2_time","best Q2 lap; NULL if eliminated in Q1","text"),("q3_time","best Q3 lap; NULL if eliminated earlier","text")],
"pit_stops": [("round","race round; links to races","integer"),("driver_id","driver; links to drivers","text"),("stop_number","1-based pit stop count for the driver in the race","integer"),("lap","lap of the stop","integer"),("local_time","local clock time of the stop, HH:MM:SS","text"),("duration_seconds","pit lane time in seconds","real")],
"driver_standings": [("round","standings after this round; links to races","integer"),("driver_id","driver; links to drivers","text"),("constructor_id","team the driver was with at that round; links to constructors","text"),("standing_position","championship position after the round","integer"),("points","cumulative championship points","real"),("wins","cumulative race wins","integer")],
"constructor_standings": [("round","standings after this round; links to races","integer"),("constructor_id","team; links to constructors","text"),("standing_position","constructors' championship position","integer"),("points","cumulative points","real"),("wins","cumulative race wins","integer")],
}
for t, cols in DESC.items():
    with open(f"{OUT}/database_description/{t}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["original_column_name","column_name","column_description","data_format","value_description"])
        for c, d, fmt in cols: w.writerow([c, c, d, fmt, ""])
counts = {t: db.execute(f"select count(*) from {t}").fetchone()[0] for t in DESC}
print(counts)
print("fk", db.execute("PRAGMA foreign_key_check").fetchall(), "integrity", db.execute("PRAGMA integrity_check").fetchall())
rng = db.execute("select min(race_date),max(race_date) from races").fetchone(); print(rng)
json.dump({"db":"f1_2026","sources":[{"url":"https://api.jolpi.ca/ergast/f1/2026/","licence":"Jolpica-F1 API (Ergast-compatible); data derives from Ergast, licensed CC BY-NC-SA 3.0 (non-commercial, share-alike); server code Apache-2.0. Research use only.","retrieved":"2026-10-21"}],"date_filter":f"races with race_date >= {CUT} and already run by {TODAY}","tables":counts,"notes":f"Main tables date range {rng[0]} to {rng[1]} (rounds {db.execute('select min(round),max(round) from races').fetchone()}). Reference tables drivers/constructors/circuits hold only entities appearing in those rounds. Standings are cumulative after each round (season-to-date, so include pre-July points). Round 16 is named 'Bahrain Grand Prix in Malaysia' by source. Rate limit ~4 req/s respected."}, open(f"{ROOT}/experiments/fresh_unseen/sources/f1_2026.json","w"), indent=1)
