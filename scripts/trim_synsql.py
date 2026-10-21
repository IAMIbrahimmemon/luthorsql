"""stdin: SynSQL data.json (one big JSON array, ~9 GB with CoT) -> stdout: the same array with
only db_id, question, sql, external_knowledge, sql_complexity, question_style (a few hundred MB)."""
import json
import sys

import ijson

KEEP = ("db_id", "question", "sql", "external_knowledge", "sql_complexity", "question_style")
out = sys.stdout
out.write("[")
for i, row in enumerate(ijson.items(sys.stdin.buffer, "item")):
    out.write(("," if i else "") + json.dumps({k: row.get(k) for k in KEEP}, default=str) + "\n")
    if i % 100000 == 0:
        print(f"synsql {i}", file=sys.stderr, flush=True)
out.write("]\n")
