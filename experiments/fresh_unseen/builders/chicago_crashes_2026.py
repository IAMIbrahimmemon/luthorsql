"""Build chicago_crashes_2026 (fallback for nyc_crashes_2026: NYC data ends 2026-06-11).
Run: python3 -I builders/chicago_crashes_2026.py"""
import json, os, sqlite3, subprocess, urllib.parse, csv
DB = "chicago_crashes_2026"
ROOT = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = "/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/" + DB
OUT = f"{ROOT}/data/sys1/fresh/dbs/{DB}"
os.makedirs(DL, exist_ok=True); os.makedirs(OUT + "/database_description", exist_ok=True)
for id_, n in [("85ca-t3if", "crashes"), ("u6pd-qa9d", "people"), ("68nd-jvt3", "vehicles")]:
    p = f"{DL}/{n}.json"
    if not os.path.exists(p):
        q = urllib.parse.urlencode({"$where": "crash_date>='2026-07-01T00:00:00'", "$limit": "100000", "$order": ":id"})
        subprocess.run(["curl", "-s", "-m", "300", "-o", p, f"https://data.cityofchicago.org/resource/{id_}.json?{q}"], check=True)
L = lambda n: json.load(open(f"{DL}/{n}.json"))
def num(t):
    def f(v):
        if v in (None, ""): return None
        return int(float(v)) if t == "INTEGER" else float(v)
    return f
# (column, type, description, value_description)
CR = [
 ("crash_record_id","TEXT","Unique crash identifier (hash)",""),
 ("crash_date","TEXT","Date and time of crash, ISO 8601",""),
 ("posted_speed_limit","INTEGER","Posted speed limit in mph",""),
 ("traffic_control_device","TEXT","Traffic control device present","e.g. NO CONTROLS, TRAFFIC SIGNAL, STOP SIGN/FLASHER"),
 ("device_condition","TEXT","Condition of the traffic control device","e.g. FUNCTIONING PROPERLY, NO CONTROLS"),
 ("weather_condition","TEXT","Weather at time of crash","e.g. CLEAR, RAIN, SNOW"),
 ("lighting_condition","TEXT","Lighting at time of crash","e.g. DAYLIGHT, DARKNESS, LIGHTED ROAD"),
 ("first_crash_type","TEXT","Type of first collision","e.g. REAR END, PARKED MOTOR VEHICLE, PEDESTRIAN"),
 ("trafficway_type","TEXT","Trafficway type","e.g. NOT DIVIDED, FOUR WAY"),
 ("alignment","TEXT","Roadway alignment","e.g. STRAIGHT AND LEVEL"),
 ("roadway_surface_cond","TEXT","Road surface condition","e.g. DRY, WET"),
 ("road_defect","TEXT","Road defect","e.g. NO DEFECTS"),
 ("report_type","TEXT","Where the report was taken","ON SCENE, NOT ON SCENE (DESK REPORT), AMENDED"),
 ("crash_type","TEXT","Crash severity class","NO INJURY / DRIVE AWAY; INJURY AND / OR TOW DUE TO CRASH"),
 ("intersection_related_i","TEXT","Intersection-related flag","Y or N; null if not recorded"),
 ("hit_and_run_i","TEXT","Hit-and-run flag","Y or N; null if not recorded"),
 ("damage","TEXT","Estimated damage band","OVER $1,500; $501 - $1,500; $500 OR LESS"),
 ("date_police_notified","TEXT","Date and time police were notified",""),
 ("prim_contributory_cause","TEXT","Primary contributory cause (FK to contributory_causes.cause)",""),
 ("sec_contributory_cause","TEXT","Secondary contributory cause (FK to contributory_causes.cause)",""),
 ("street_no","INTEGER","Street number of crash location",""),
 ("street_direction","TEXT","Street direction","N, S, E, W"),
 ("street_name","TEXT","Street name",""),
 ("beat_of_occurrence","INTEGER","Chicago police beat",""),
 ("num_units","INTEGER","Number of units (vehicles/pedestrians/cyclists) involved",""),
 ("most_severe_injury","TEXT","Most severe injury in the crash","FATAL, INCAPACITATING INJURY, NONINCAPACITATING INJURY, REPORTED, NOT EVIDENT, NO INDICATION OF INJURY"),
 ("injuries_total","INTEGER","Total people injured",""),
 ("injuries_fatal","INTEGER","Fatal injuries",""),
 ("injuries_incapacitating","INTEGER","Incapacitating injuries",""),
 ("injuries_non_incapacitating","INTEGER","Non-incapacitating injuries",""),
 ("injuries_reported_not_evident","INTEGER","Injuries reported but not evident",""),
 ("injuries_no_indication","INTEGER","People with no indication of injury",""),
 ("crash_hour","INTEGER","Hour of day (0-23)",""),
 ("crash_day_of_week","INTEGER","Day of week","1=Sunday ... 7=Saturday"),
 ("crash_month","INTEGER","Month number",""),
 ("latitude","REAL","Crash latitude",""),
 ("longitude","REAL","Crash longitude",""),
]
VE = [
 ("crash_unit_id","INTEGER","Unique unit identifier (a unit may be a vehicle, pedestrian or cyclist)",""),
 ("crash_record_id","TEXT","Crash this unit belongs to (FK to crashes)",""),
 ("unit_no","INTEGER","Unit number within the crash",""),
 ("unit_type","TEXT","Unit type","DRIVER, PARKED, PEDESTRIAN, PEDALCYCLIST, DRIVERLESS, NON-MOTOR VEHICLE"),
 ("vehicle_id","INTEGER","Vehicle identifier; null for non-vehicle units. Referenced by people.vehicle_id",""),
 ("make","TEXT","Vehicle make",""),
 ("model","TEXT","Vehicle model",""),
 ("vehicle_year","INTEGER","Model year",""),
 ("vehicle_defect","TEXT","Vehicle defect","e.g. NONE, BRAKES"),
 ("vehicle_type","TEXT","Vehicle type","e.g. PASSENGER, SPORT UTILITY VEHICLE (SUV), PICKUP"),
 ("vehicle_use","TEXT","Vehicle use","e.g. PERSONAL, TAXI/FOR HIRE"),
 ("travel_direction","TEXT","Direction of travel","N, S, E, W, UNKNOWN"),
 ("maneuver","TEXT","Maneuver just before crash","e.g. STRAIGHT AHEAD, TURNING LEFT"),
 ("occupant_cnt","INTEGER","Number of occupants",""),
 ("first_contact_point","TEXT","First point of contact on vehicle","e.g. FRONT, REAR, SIDE-RIGHT"),
 ("towed_i","TEXT","Vehicle towed flag","Y if towed, otherwise null"),
 ("fire_i","TEXT","Fire flag","Y if fire, otherwise null"),
 ("cmrc_veh_i","TEXT","Commercial vehicle flag","Y if commercial, otherwise null"),
]
PE = [
 ("person_id","TEXT","Unique person record identifier (anonymous code)",""),
 ("person_type","TEXT","Role of person","DRIVER, PASSENGER, PEDESTRIAN, BICYCLE, NON-MOTOR VEHICLE, NON-CONTACT VEHICLE"),
 ("crash_record_id","TEXT","Crash (FK to crashes)",""),
 ("vehicle_id","INTEGER","Vehicle the person was in (FK to vehicles.vehicle_id); null for pedestrians",""),
 ("sex","TEXT","Sex","M, F, X"),
 ("age","INTEGER","Age in years",""),
 ("safety_equipment","TEXT","Safety equipment used","e.g. SAFETY BELT USED, NONE PRESENT"),
 ("airbag_deployed","TEXT","Airbag deployment","e.g. DEPLOYED, FRONT; DID NOT DEPLOY"),
 ("ejection","TEXT","Ejection status","NONE, TOTALLY EJECTED, PARTIALLY EJECTED, TRAPPED/EXTRICATED"),
 ("injury_classification","TEXT","Injury severity","FATAL, INCAPACITATING INJURY, NONINCAPACITATING INJURY, REPORTED, NOT EVIDENT, NO INDICATION OF INJURY"),
 ("driver_action","TEXT","Driver action contributing to crash","e.g. NONE, FAILED TO YIELD, TOO FAST FOR CONDITIONS"),
 ("driver_vision","TEXT","Driver vision obscured by","e.g. NOT OBSCURED, PARKED VEHICLES"),
 ("physical_condition","TEXT","Driver physical condition","e.g. NORMAL, IMPAIRED - ALCOHOL"),
 ("bac_result","TEXT","Blood alcohol test result category","e.g. TEST NOT OFFERED, TEST PERFORMED"),
 ("hospital","TEXT","Hospital the person was taken to",""),
 ("pedpedal_action","TEXT","Pedestrian/cyclist action","e.g. CROSSING - WITH SIGNAL"),
 ("pedpedal_visibility","TEXT","Pedestrian/cyclist visibility","e.g. NO CONTRASTING CLOTHING"),
 ("pedpedal_location","TEXT","Pedestrian/cyclist location","e.g. IN ROADWAY, IN CROSSWALK"),
]
CA = [("cause","TEXT","Contributory cause text as used in crashes.prim/sec_contributory_cause",""),
      ("category","TEXT","Grouped cause category (derived: driver behaviour, impairment, distraction, environment, vehicle/road, other, unknown)","")]
def cat(c):
    u=c.upper()
    if any(k in u for k in("ALCOHOL","DRUGS","PHYSICAL CONDITION")): return "impairment"
    if any(k in u for k in("DISTRACTION","CELL PHONE","TEXTING","RADIO")): return "distraction"
    if any(k in u for k in("WEATHER","VISION OBSCURED","ROAD CONSTRUCTION","ROAD ENGINEERING","ANIMAL","OBSTRUCTED","EVASIVE","RELATED TO BUS")): return "environment"
    if any(k in u for k in("EQUIPMENT","DEFECT")): return "vehicle/road"
    if "UNABLE TO DETERMINE" in u or "NOT APPLICABLE" in u: return "unknown"
    if any(k in u for k in("FOLLOWING","FAILING","DISREGARDING","IMPROPER","SPEED","DRIVING SKILLS","OPERATING","EXCEEDING","TURNING","PASSING","OVERTAKING","DISTRACTION")): return "driver behaviour"
    return "other"
cr, pe, ve = L("crashes"), L("people"), L("vehicles")
cr = [r for r in cr if int(r["crash_record_id"][:6],16) % 10 < 8]  # deterministic 80% whole-crash sample to stay under 60 MB
ids = {r["crash_record_id"] for r in cr}
ve = [r for r in ve if r["crash_record_id"] in ids]
pe = [r for r in pe if r["crash_record_id"] in ids]
vids = {r["vehicle_id"] for r in ve if r.get("vehicle_id")}
for r in pe:
    if r.get("vehicle_id") and r["vehicle_id"] not in vids: r["vehicle_id"] = None
seen=set(); ve2=[]
for r in ve:
    v=r.get("vehicle_id")
    if v:
        if v in seen: r["vehicle_id"]=None
        seen.add(v)
    ve2.append(r)
ve=ve2
causes = sorted({r.get(k) for r in cr for k in ("prim_contributory_cause","sec_contributory_cause") if r.get(k)})
path = f"{OUT}/{DB}.sqlite"
if os.path.exists(path): os.remove(path)
con = sqlite3.connect(path); con.execute("PRAGMA foreign_keys=ON")
def ddl(name, cols, pk, fks=(), extra=()):
    parts=[f'"{c}" {t}' for c,t,_,_ in cols]+[f"PRIMARY KEY ({pk})"]+list(fks)+list(extra)
    con.execute(f'CREATE TABLE {name} ({", ".join(parts)})')
ddl("contributory_causes", CA, "cause")
ddl("crashes", CR, "crash_record_id", ["FOREIGN KEY (prim_contributory_cause) REFERENCES contributory_causes(cause)","FOREIGN KEY (sec_contributory_cause) REFERENCES contributory_causes(cause)"])
ddl("vehicles", VE, "crash_unit_id", ["FOREIGN KEY (crash_record_id) REFERENCES crashes(crash_record_id)"], ["UNIQUE (vehicle_id)"])
ddl("people", PE, "person_id", ["FOREIGN KEY (crash_record_id) REFERENCES crashes(crash_record_id)","FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)"])
def ins(name, cols, rows):
    fs=[num(t) if t in("INTEGER","REAL") else (lambda v: v if v not in (None,"") else None) for _,t,_,_ in cols]
    data=[[f(r.get(c)) for f,(c,_,_,_) in zip(fs,cols)] for r in rows]
    con.executemany(f'INSERT INTO {name} VALUES ({",".join("?"*len(cols))})', data)
ins("contributory_causes", CA, [{"cause":c,"category":cat(c)} for c in causes])
ins("crashes", CR, cr); ins("vehicles", VE, ve); ins("people", PE, pe); con.commit()
for n,cols in [("contributory_causes",CA),("crashes",CR),("vehicles",VE),("people",PE)]:
    with open(f"{OUT}/database_description/{n}.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["original_column_name","column_name","column_description","data_format","value_description"])
        for c,t,d,v in cols: w.writerow([c,c.replace("_i","_flag") if False else c,d,{"TEXT":"text","INTEGER":"integer","REAL":"real"}[t],v])
print("fk", con.execute("PRAGMA foreign_key_check").fetchall()[:5], con.execute("PRAGMA integrity_check").fetchall())
counts={n:con.execute(f"select count(*) from {n}").fetchone()[0] for n in("contributory_causes","crashes","vehicles","people")}
print(counts, con.execute("select min(crash_date),max(crash_date) from crashes").fetchone(), os.path.getsize(path)/1e6,"MB")
json.dump({"db":DB,"sources":[{"url":f"https://data.cityofchicago.org/resource/{i}.json","licence":"City of Chicago Data Portal public data (open data terms of use; Chicago Police Dept / CDOT crash reports)","retrieved":"2026-10-21"} for i in("85ca-t3if","u6pd-qa9d","68nd-jvt3")],
 "date_filter":"crash_date >= 2026-07-01","tables":counts,
 "notes":"Primary NYC db nyc_crashes_2026 abandoned: NYC Open Data crashes/persons/vehicles max crash_date is 2026-06-11 (0 rows from 2026-07-01). Fallback Chicago used. crashes date range "+str(con.execute("select min(crash_date),max(crash_date) from crashes").fetchone())+". Dropped person residence (city/state/zip), licence, EMS run, carrier and commercial-vehicle ID fields. contributory_causes is a derived lookup (category column is a builder heuristic). Whole-crash deterministic ~80% sample (hash of crash_record_id) to fit 60 MB. Units with duplicate vehicle_id had the duplicate set to null."},
 open(f"{ROOT}/experiments/fresh_unseen/sources/{DB}.json","w"),indent=1)
