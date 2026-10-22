You are answering BIRD text-to-SQL questions (SQLite) together with a fast text-to-SQL system. For each question that system already proposed candidate queries; they are grouped by identical execution result and listed best first by its own confidence (n_of_32 = how many of its 32 samples gave that result; "result" = a short preview of the rows). Some questions have no candidates.

Input: CHUNK (JSONL: question_id, db_id, question, evidence, candidates[{label, sql, result, n_of_32}]).
Output: append one JSON line per question to OUT as soon as it is done:
  {"question_id": <exactly as given>, "pick": "<label>"}            if one candidate answers the question correctly, or
  {"question_id": <exactly as given>, "pick": "NONE", "sql": "<your own SQLite query>"}   if none is correct or there are none.
Exactly one line per question.

Tool (the ONLY thing you may use besides reading CHUNK and writing OUT):
  python3 TOOLS/sql.py <db_id> --schema        # tables, sample rows, column descriptions
  python3 TOOLS/sql.py <db_id> "<SQL>"         # runs read-only SQL, prints up to 20 rows
Get each DB's schema only if you need it, and reuse it for later questions on the same DB.

Rules (strict — this is a benchmark):
- Do NOT read any other file or directory (no repo files, no gold SQL, no other chunks/answers, no web).
- At most 8 SQL executions per question. Use the candidates to save work: when one is clearly right, pick it without extra exploration; write your own SQL only when you are confident every candidate is wrong.
- Judge correctness as the benchmark does: the result set must equal the intended answer (row order ignored). Use the evidence: it defines terms and formulas as the benchmark intends. Exactly the asked columns, nothing extra.
- If you write any helper script, put it ONLY in OUT's dir + "/../work/<chunk name>/"; never create or edit shared files (other agents run in parallel).
Final message: "done N/N, picked K, own M".
