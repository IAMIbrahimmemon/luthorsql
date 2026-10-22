#!/usr/bin/env python3
"""Build wc2026.sqlite from openfootball/worldcup.json (CC0). Re-runnable."""
import json, os, re, sqlite3, csv, urllib.request, collections
ROOT = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = "/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/wc2026"
BASE = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/"
OUT = f"{ROOT}/data/sys1/fresh/dbs/wc2026"
os.makedirs(DL, exist_ok=True); os.makedirs(OUT + "/database_description", exist_ok=True)
for f in ("worldcup-full.json", "worldcup.groups.json"):
    p = f"{DL}/{f}"
    if not os.path.exists(p):
        urllib.request.urlretrieve(BASE + f, p)
full = json.load(open(f"{DL}/worldcup-full.json"))["matches"]
groups = json.load(open(f"{DL}/worldcup.groups.json"))["groups"]

SCHEMA = """
CREATE TABLE teams (team_id INTEGER PRIMARY KEY, team_name TEXT NOT NULL UNIQUE, group_letter TEXT);
CREATE TABLE stadiums (stadium_id INTEGER PRIMARY KEY, stadium_name TEXT NOT NULL UNIQUE, city TEXT, country TEXT);
CREATE TABLE matches (
 match_id INTEGER PRIMARY KEY, match_date TEXT NOT NULL, kickoff_time TEXT, utc_offset TEXT, stage TEXT NOT NULL, group_letter TEXT,
 home_team_id INTEGER NOT NULL REFERENCES teams(team_id), away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
 stadium_id INTEGER NOT NULL REFERENCES stadiums(stadium_id), attendance INTEGER,
 home_ht INTEGER, away_ht INTEGER, home_ft INTEGER, away_ft INTEGER, home_et INTEGER, away_et INTEGER,
 home_pens INTEGER, away_pens INTEGER, went_to_extra_time INTEGER, went_to_penalties INTEGER,
 winner_team_id INTEGER REFERENCES teams(team_id), referee_name TEXT, referee_country TEXT);
CREATE TABLE goals (
 goal_id INTEGER PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(match_id), scoring_team_id INTEGER NOT NULL REFERENCES teams(team_id),
 player_name TEXT NOT NULL, minute INTEGER NOT NULL, stoppage_minute INTEGER, is_penalty INTEGER NOT NULL, is_own_goal INTEGER NOT NULL);
CREATE TABLE cards (
 card_id INTEGER PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(match_id), team_id INTEGER NOT NULL REFERENCES teams(team_id),
 player_name TEXT NOT NULL, card_type TEXT NOT NULL, minute INTEGER NOT NULL, stoppage_minute INTEGER);
CREATE TABLE lineups (
 lineup_id INTEGER PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(match_id), team_id INTEGER NOT NULL REFERENCES teams(team_id),
 player_name TEXT NOT NULL, is_starter INTEGER NOT NULL, is_captain INTEGER NOT NULL, sub_on_minute INTEGER, sub_off_minute INTEGER);
"""
def mn(s):
    m = re.match(r"(\d+)(?:\+(\d+))?$", str(s))
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)

teams = {}
for g in groups:
    for t in g["teams"]:
        t = {"Bosnia & Herzegovina": "Bosnia and Herzegovina", "Ivory Coast": "Côte d'Ivoire", "Cape Verde": "Cabo Verde", "DR Congo": "Congo DR"}.get(t, t)
        teams[t] = g["name"].replace("Group ", "")
for x in full:
    for t in (x["team1"], x["team2"]):
        teams.setdefault(t, None)
tid = {t: i + 1 for i, t in enumerate(sorted(teams))}
std = {}
for x in full:
    std.setdefault(x["ground"], None)
sid = {s: i + 1 for i, s in enumerate(sorted(std))}
HOST = {"Mexico City": "Mexico", "Guadalajara": "Mexico", "Monterrey": "Mexico", "Toronto": "Canada", "Vancouver": "Canada"}
def split(g):
    name, city = g.split(", ", 1)
    if "Vancouver" in city or "Toronto" in city: c = "Canada"
    elif city in ("Mexico City", "Guadalajara", "Monterrey"): c = "Mexico"
    else: c = "USA"
    return name, city, c

dbp = f"{OUT}/wc2026.sqlite"
if os.path.exists(dbp): os.remove(dbp)
db = sqlite3.connect(dbp); db.executescript(SCHEMA)
db.executemany("INSERT INTO teams VALUES (?,?,?)", [(tid[t], t, teams[t]) for t in tid])
db.executemany("INSERT INTO stadiums VALUES (?,?,?,?)", [(sid[s], *split(s)) for s in sid])
gid = cid = lid = 0
for mi, x in enumerate(sorted(full, key=lambda y: (y["date"], y["time"], y["team1"])), 1):
    sc = x["score"]; g = lambda k, i: sc[k][i] if k in sc else None
    stage = x["round"]; grp = None
    if stage.startswith("First Stage"): grp = stage[-1]; stage = "Group stage"
    final = sc.get("p") or sc.get("et") or sc["ft"]
    a, b = (final[0], final[1])
    if sc.get("p"): a, b = sc["p"]
    win = tid[x["team1"]] if a > b else tid[x["team2"]] if b > a else None
    tm = re.match(r"(\d+:\d+) UTC([+-]\d+)", x["time"])
    ref = (x.get("referees") or [{}])[0]
    db.execute("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
      (mi, x["date"], tm.group(1), "UTC" + tm.group(2), stage, grp, tid[x["team1"]], tid[x["team2"]], sid[x["ground"]], x.get("attendance"),
       g("ht",0), g("ht",1), g("ft",0), g("ft",1), g("et",0), g("et",1), g("p",0), g("p",1),
       int("et" in sc), int("p" in sc), win, ref.get("name"), ref.get("country")))
    for side, key in ((1, "goals1"), (2, "goals2")):
        t = tid[x["team%d" % side]]
        for go in x.get(key, []):
            gid += 1; m, s = mn(go["minute"])
            db.execute("INSERT INTO goals VALUES (?,?,?,?,?,?,?,?)", (gid, mi, t, go["name"], m, s, int(bool(go.get("penalty"))), int(bool(go.get("owngoal")))))
    for side, bl in enumerate(x.get("bookings", [])):
        t = tid[x["team%d" % (side + 1)]]
        for b_ in bl:
            cid += 1; m, s = mn(b_["minute"])
            db.execute("INSERT INTO cards VALUES (?,?,?,?,?,?,?)", (cid, mi, t, b_["name"], b_["type"], m, s))
    for side, lu in enumerate(x.get("lineup", [])):
        t = tid[x["team%d" % (side + 1)]]
        on = {s["on"]: mn(s["minute"])[0] for s in lu.get("subs", [])}
        off = {s["off"]: mn(s["minute"])[0] for s in lu.get("subs", [])}
        for role in ("starter", "bench"):
            for p in lu.get(role, []):
                lid += 1
                db.execute("INSERT INTO lineups VALUES (?,?,?,?,?,?,?,?)", (lid, mi, t, p["name"], int(role == "starter"), int(bool(p.get("captain"))), on.get(p["name"]), off.get(p["name"])))
db.commit()
print("fk", db.execute("PRAGMA foreign_key_check").fetchall(), "integrity", db.execute("PRAGMA integrity_check").fetchall())
counts = {t: db.execute(f"select count(*) from {t}").fetchone()[0] for t in ("teams","stadiums","matches","goals","cards","lineups")}
print(counts)
print(db.execute("select min(match_date),max(match_date),sum(match_date>='2026-07-01'),sum(match_date<'2026-07-01') from matches").fetchone())
json.dump(counts, open(f"{DL}/counts.json", "w"))

DESC = {
"teams": [("team_id","team id","Surrogate key","integer",""),("team_name","team name","National team name (openfootball spelling)","text",""),("group_letter","group letter","Group-stage group, A to L","text","A-L")],
"stadiums": [("stadium_id","stadium id","Surrogate key","integer",""),("stadium_name","stadium name","Neutral FIFA tournament stadium name","text",""),("city","city","Host city or metro area","text",""),("country","country","Host country","text","USA, Canada, Mexico")],
"matches": [("match_id","match id","Match number in chronological order","integer",""),("match_date","match date","Local date of the match","text","YYYY-MM-DD"),("kickoff_time","kickoff time","Local kickoff time","text","HH:MM"),("utc_offset","utc offset","Offset of local time from UTC","text","e.g. UTC-6"),("stage","stage","Tournament stage","text","Group stage, Round of 32, Round of 16, Quarter-final, Semi-final, Bronze final, Final"),("group_letter","group letter","Group for group-stage matches, else NULL","text","A-L"),("home_team_id","home team id","First-listed team (FK teams)","integer",""),("away_team_id","away team id","Second-listed team (FK teams)","integer",""),("stadium_id","stadium id","Venue (FK stadiums)","integer",""),("attendance","attendance","Reported attendance","integer",""),("home_ht","home ht","First-listed team goals at half time","integer",""),("away_ht","away ht","Second-listed team goals at half time","integer",""),("home_ft","home ft","First-listed team goals after 90 minutes","integer",""),("away_ft","away ft","Second-listed team goals after 90 minutes","integer",""),("home_et","home et","First-listed team goals after extra time (cumulative incl. 90 min); NULL if no extra time","integer",""),("away_et","away et","Second-listed team goals after extra time; NULL if no extra time","integer",""),("home_pens","home pens","First-listed team penalty shootout goals; NULL if none","integer",""),("away_pens","away pens","Second-listed team penalty shootout goals; NULL if none","integer",""),("went_to_extra_time","went to extra time","1 if extra time was played","integer","0/1"),("went_to_penalties","went to penalties","1 if decided by shootout","integer","0/1"),("winner_team_id","winner team id","Winning team (FK teams); NULL for a draw","integer",""),("referee_name","referee name","Match referee","text",""),("referee_country","referee country","Referee FIFA country code","text","ISO-like 3 letters")],
"goals": [("goal_id","goal id","Surrogate key","integer",""),("match_id","match id","FK matches","integer",""),("scoring_team_id","scoring team id","Team credited with the goal (FK teams); for own goals the team that benefits","integer",""),("player_name","player name","Scorer","text",""),("minute","minute","Match minute (base minute)","integer",""),("stoppage_minute","stoppage minute","Added-time minutes, e.g. 4 for 90+4; else NULL","integer",""),("is_penalty","is penalty","1 if scored from a penalty kick in play","integer","0/1"),("is_own_goal","is own goal","1 if own goal","integer","0/1")],
"cards": [("card_id","card id","Surrogate key","integer",""),("match_id","match id","FK matches","integer",""),("team_id","team id","Team of the booked player (FK teams)","integer",""),("player_name","player name","Booked player","text",""),("card_type","card type","Card shown","text","Y yellow, R straight red, Y/R second yellow"),("minute","minute","Base minute","integer",""),("stoppage_minute","stoppage minute","Added-time minutes else NULL","integer","")],
"lineups": [("lineup_id","lineup id","Surrogate key","integer",""),("match_id","match id","FK matches","integer",""),("team_id","team id","FK teams","integer",""),("player_name","player name","Player","text",""),("is_starter","is starter","1 if in starting XI, 0 if on bench","integer","0/1"),("is_captain","is captain","1 if captain","integer","0/1"),("sub_on_minute","sub on minute","Minute substituted on, else NULL","integer",""),("sub_off_minute","sub off minute","Minute substituted off, else NULL","integer","")],
}
for t, rows in DESC.items():
    with open(f"{OUT}/database_description/{t}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["original_column_name","column_name","column_description","data_format","value_description"]); w.writerows(rows)
