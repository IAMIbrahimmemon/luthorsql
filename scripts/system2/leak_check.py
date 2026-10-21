"""Leak check for System 2 agent transcripts (runs/phase2/system2_dev/PREREG.md): inspects only the tool calls the agent
made (not its system prompt) and flags any access outside the sandbox, any gold/dev file, any web tool, and counts
sql.py executions.

  python3 scripts/system2/leak_check.py <sandbox> <transcript.output>...
"""
import json, re, sys

box = sys.argv[1]
BAD = re.compile(r"dev\.json|dev\.sql|gold|luthor-general|dev_databases|/answers/(?!A[01]/c\d\d\.jsonl)|chunks/A[01]/(?!c\d\d)", re.I)
for path in sys.argv[2:]:
    flags, n_sql, tools = [], 0, {}
    for ln in open(path):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        m = r.get("message") or {}
        if not isinstance(m, dict) or r.get("type") != "assistant":
            continue
        for b in m.get("content") or []:
            if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                continue
            name, inp = b.get("name"), b.get("input") or {}
            tools[name] = tools.get(name, 0) + 1
            text = json.dumps(inp)
            if name in ("WebFetch", "WebSearch"):
                flags.append(f"web tool {name}")
            n_sql += len(re.findall(r"TOOLS/sql\.py\s+\S+\s+(?!--schema)", text))
            for p in re.findall(r"(?<![\w.])(/(?:Users|private|Volumes|tmp|var|etc|home|opt)/[\w./-]+)", text):
                if p.startswith(box) or "/.claude/projects/" in p and "/tool-results/" in p:
                    continue  # the sandbox, or the agent re-reading its own spilled tool output
                flags.append(f"outside path {p}")
            # strip the agent's own chunk/answer paths before searching for gold-like strings
            t2 = re.sub(re.escape(box) + r"/(chunks|answers)/A[01]/c\d\d\.jsonl", "", text)
            for hit in BAD.findall(t2):
                flags.append(f"pattern {hit}")
    print(json.dumps({"transcript": path.rsplit("/", 1)[-1], "tools": tools, "sql_executions": n_sql,
                      "flags": sorted(set(flags))[:10], "clean": not flags}))
