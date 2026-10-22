"""Build mlb_2026.sqlite from the public MLB Stats API. Re-runnable. Run: python3 mlb_2026.py"""
import json, os, sqlite3, time, urllib.request, csv, datetime
REPO = "/Users/ibrahimmemon/Documents/GitHub/luthor-general"
DL = "/private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/fresh_dl/mlb_2026"
OUT = f"{REPO}/data/sys1/fresh/dbs/mlb_2026"
START, END = datetime.date(2026, 7, 1), datetime.date(2026, 10, 20)
BASE = "https://statsapi.mlb.com/api/v1"
os.makedirs(DL, exist_ok=True); os.makedirs(f"{OUT}/database_description", exist_ok=True)

def get(url, cache):
    p = f"{DL}/{cache}"
    if os.path.exists(p):
        return json.load(open(p))
    time.sleep(0.4)
    d = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-builder"}), timeout=60).read()
    open(p, "wb").write(d)
    return json.loads(d)

games = []
d = START
while d <= END:
    e = min(d + datetime.timedelta(days=6), END)
    j = get(f"{BASE}/schedule?sportId=1&startDate={d}&endDate={e}&hydrate=linescore,team,venue,probablePitcher,decisions", f"sched_{d}.json")
    for dt in j["dates"]:
        games += dt["games"]
    d = e + datetime.timedelta(days=1)
games = [g for g in {g["gamePk"]: g for g in games}.values() if g["status"]["abstractGameState"] == "Final" and g["gameType"] != "E" and g["officialDate"] >= "2026-07-01"]

teams, venues, players = {}, {}, {}
for g in games:
    for s in ("home", "away"):
        t = g["teams"][s]["team"]
        teams[t["id"]] = t
        for k in ("probablePitcher",):
            p = g["teams"][s].get(k)
            if p: players[p["id"]] = p["fullName"]
    venues[g["venue"]["id"]] = g["venue"]["name"]
    for k in ("winner", "loser", "save"):
        p = (g.get("decisions") or {}).get(k)
        if p: players[p["id"]] = p["fullName"]
vl = get(f"{BASE}/venues?venueIds={','.join(map(str, venues))}&hydrate=location", "venues.json")["venues"]
vloc = {v["id"]: v.get("location", {}) for v in vl}

db = f"{OUT}/mlb_2026.sqlite"
if os.path.exists(db): os.remove(db)
c = sqlite3.connect(db); c.execute("PRAGMA foreign_keys=ON")
c.executescript("""
CREATE TABLE teams(team_id INTEGER PRIMARY KEY, name TEXT, team_name TEXT, abbreviation TEXT, location_name TEXT, league_name TEXT, division_name TEXT, first_year_of_play INTEGER, home_venue_id INTEGER);
CREATE TABLE venues(venue_id INTEGER PRIMARY KEY, name TEXT, city TEXT, state TEXT, country TEXT);
CREATE TABLE players(player_id INTEGER PRIMARY KEY, full_name TEXT);
CREATE TABLE games(game_pk INTEGER PRIMARY KEY, game_date TEXT, game_datetime_utc TEXT, game_type TEXT, series_description TEXT, day_night TEXT,
 home_team_id INTEGER REFERENCES teams(team_id), away_team_id INTEGER REFERENCES teams(team_id), venue_id INTEGER REFERENCES venues(venue_id),
 home_score INTEGER, away_score INTEGER, home_hits INTEGER, away_hits INTEGER, home_errors INTEGER, away_errors INTEGER,
 innings_played INTEGER, scheduled_innings INTEGER, double_header TEXT, game_number INTEGER, series_game_number INTEGER, games_in_series INTEGER, winning_team_id INTEGER REFERENCES teams(team_id));
CREATE TABLE linescore_innings(game_pk INTEGER REFERENCES games(game_pk), inning INTEGER, home_runs INTEGER, home_hits INTEGER, home_errors INTEGER, away_runs INTEGER, away_hits INTEGER, away_errors INTEGER, PRIMARY KEY(game_pk, inning));
CREATE TABLE game_pitchers(game_pk INTEGER REFERENCES games(game_pk), player_id INTEGER REFERENCES players(player_id), team_id INTEGER REFERENCES teams(team_id), role TEXT, PRIMARY KEY(game_pk, player_id, role));
""")
for t in teams.values():
    c.execute("INSERT INTO teams VALUES(?,?,?,?,?,?,?,?,?)", (t["id"], t["name"], t.get("teamName"), t.get("abbreviation"), t.get("locationName"),
        t.get("league", {}).get("name"), t.get("division", {}).get("name"), int(t["firstYearOfPlay"]) if t.get("firstYearOfPlay") else None, (t.get("venue") or {}).get("id")))
# home venue may be absent from venues table; add
for vid, nm in venues.items():
    l = vloc.get(vid, {})
    c.execute("INSERT INTO venues VALUES(?,?,?,?,?)", (vid, nm, l.get("city"), l.get("stateAbbrev") or l.get("state"), l.get("country")))
for t in teams.values():
    hv = (t.get("venue") or {}).get("id")
    if hv and hv not in venues:
        c.execute("UPDATE teams SET home_venue_id=NULL WHERE team_id=?", (t["id"],))
for pid, nm in players.items(): c.execute("INSERT INTO players VALUES(?,?)", (pid, nm))
for g in games:
    h, a = g["teams"]["home"], g["teams"]["away"]; ls = g.get("linescore", {}); lt = ls.get("teams", {})
    win = h["team"]["id"] if h.get("isWinner") else a["team"]["id"] if a.get("isWinner") else None
    c.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (g["gamePk"], g["officialDate"], g["gameDate"], g["gameType"], g.get("seriesDescription"), g.get("dayNight"),
        h["team"]["id"], a["team"]["id"], g["venue"]["id"], h.get("score"), a.get("score"),
        lt.get("home", {}).get("hits"), lt.get("away", {}).get("hits"), lt.get("home", {}).get("errors"), lt.get("away", {}).get("errors"),
        len(ls.get("innings", [])), g.get("scheduledInnings"), g.get("doubleHeader"), g.get("gameNumber"), g.get("seriesGameNumber"), g.get("gamesInSeries"), win))
    for i in ls.get("innings", []):
        c.execute("INSERT INTO linescore_innings VALUES(?,?,?,?,?,?,?,?)", (g["gamePk"], i["num"], i.get("home", {}).get("runs"), i.get("home", {}).get("hits"), i.get("home", {}).get("errors"),
            i.get("away", {}).get("runs"), i.get("away", {}).get("hits"), i.get("away", {}).get("errors")))
    roles = set()
    for s, tm in (("home", h), ("away", a)):
        p = tm.get("probablePitcher")
        if p: roles.add((p["id"], tm["team"]["id"], "probable"))
    dec = g.get("decisions") or {}
    for k in ("winner", "loser", "save"):
        p = dec.get(k)
        if not p: continue
        # team: winner on winning team, loser on losing team; save with winner's team
        tid = win if k != "loser" else (a["team"]["id"] if win == h["team"]["id"] else h["team"]["id"])
        if tid: roles.add((p["id"], tid, {"winner": "winning", "loser": "losing", "save": "save"}[k]))
    for pid, tid, r in roles:
        c.execute("INSERT OR IGNORE INTO game_pitchers VALUES(?,?,?,?)", (g["gamePk"], pid, tid, r))
c.commit()

desc = {
"teams": [("team_id","team id","MLB Stats API team id","integer",""),("name","name","full team name","text",""),("team_name","team name","nickname","text",""),("abbreviation","abbreviation","3-letter code","text",""),("location_name","location name","city or region","text",""),("league_name","league name","league","text","American League / National League"),("division_name","division name","division","text","e.g. American League East"),("first_year_of_play","first year of play","first season of franchise","integer",""),("home_venue_id","home venue id","team home ballpark","integer","references venues; NULL if no home game in the period")],
"venues": [("venue_id","venue id","MLB Stats API venue id","integer",""),("name","name","ballpark name","text",""),("city","city","city","text",""),("state","state","state or province abbreviation","text",""),("country","country","country","text","")],
"players": [("player_id","player id","MLB Stats API person id","integer",""),("full_name","full name","public player name (pitchers named in game data)","text","")],
"games": [("game_pk","game pk","unique game id","integer",""),("game_date","game date","official local date","text","YYYY-MM-DD"),("game_datetime_utc","game datetime utc","scheduled start","text","ISO UTC"),("game_type","game type","type of game","text","R regular season, F wild card, D division series, L league championship, W world series, A all-star game"),("series_description","series description","series label","text","e.g. Regular Season"),("day_night","day night","time of day","text","day / night"),("home_team_id","home team id","home team","integer","references teams"),("away_team_id","away team id","away team","integer","references teams"),("venue_id","venue id","ballpark","integer","references venues"),("home_score","home score","home runs","integer",""),("away_score","away score","away runs","integer",""),("home_hits","home hits","home hits","integer",""),("away_hits","away hits","away hits","integer",""),("home_errors","home errors","home errors","integer",""),("away_errors","away errors","away errors","integer",""),("innings_played","innings played","number of innings in linescore","integer",""),("scheduled_innings","scheduled innings","scheduled innings","integer","normally 9"),("double_header","double header","doubleheader flag","text","N no, Y traditional, S split"),("game_number","game number","game number of the day for the pair","integer","1 or 2"),("series_game_number","series game number","game number within series","integer",""),("games_in_series","games in series","scheduled games in series","integer",""),("winning_team_id","winning team id","winner; NULL for a tie","integer","references teams")],
"linescore_innings": [("game_pk","game pk","game","integer","references games"),("inning","inning","inning number","integer","extra innings > 9"),("home_runs","home runs","home runs in inning","integer","NULL if home team did not bat"),("home_hits","home hits","home hits in inning","integer",""),("home_errors","home errors","home errors in inning","integer",""),("away_runs","away runs","away runs in inning","integer",""),("away_hits","away hits","away hits in inning","integer",""),("away_errors","away errors","away errors in inning","integer","")],
"game_pitchers": [("game_pk","game pk","game","integer","references games"),("player_id","player id","pitcher","integer","references players"),("team_id","team id","pitcher's team","integer","references teams"),("role","role","pitcher role in game","text","probable starter / winning / losing / save")],
}
for t, rows in desc.items():
    with open(f"{OUT}/database_description/{t}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["original_column_name","column_name","column_description","data_format","value_description"]); w.writerows(rows)
counts = {t: c.execute(f"select count(*) from {t}").fetchone()[0] for t in desc}
mn, mx = c.execute("select min(game_date),max(game_date) from games").fetchone()
types = c.execute("select game_type,count(*) from games group by 1").fetchall()
json.dump({"db":"mlb_2026","sources":[{"url":"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2026-07-01&endDate=2026-10-20&hydrate=linescore,team,venue,probablePitcher,decisions","licence":"MLB Advanced Media copyright (http://gdx.mlb.com/components/copyright.txt): only individual, non-commercial, non-bulk use permitted; no open licence. Used for research only per owner decision.","retrieved":"2026-10-21"},{"url":"https://statsapi.mlb.com/api/v1/venues?hydrate=location","licence":"same MLBAM terms","retrieved":"2026-10-21"}],
 "date_filter":"officialDate >= 2026-07-01 and <= 2026-10-20; final games only; exhibition excluded","tables":counts,
 "notes":f"games date range {mn}..{mx}; game types {types}. Attendance not in schedule endpoint so omitted. Requests ~2.5/s. Teams includes the AL/NL All-Star squads (game_type A, 1 game). One regular-season game is Final with NULL scores and NULL winning_team_id (source quirk). Rows are regular season + postseason as available."},
 open(f"{REPO}/experiments/fresh_unseen/sources/mlb_2026.json","w"),indent=1)
print(counts, mn, mx, types)
print(c.execute("PRAGMA foreign_key_check").fetchall(), c.execute("PRAGMA integrity_check").fetchall())
