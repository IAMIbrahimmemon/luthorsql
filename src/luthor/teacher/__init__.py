"""Teacher-data pipeline: public text-to-SQL questions labelled (LABEL) or written (WRITE) by a hosted teacher model,
verified on the Mac (SQLite execution, gold match or majority vote, plan convertibility) and exported in the
training loader's plans.jsonl format with provenance. CLI: scripts/teacher/teacher.py.

  leak.py       private-pattern / secret scan run on every prompt before it leaves the Mac
  guards.py     database and source refusals (BIRD dev, sealed sets, held-out dbs), contamination hook
  ledger.py     tokens per provider/model per day and ISO week (append-only jsonl), optional caps
  providers.py  local ollama daemon (any :cloud model) and an OpenAI-compatible HTTP adapter; shared back-off
  verify.py     read-only parallel execution, result signatures, gold match / vote, plan rows (grammar v3, v4 hook)
"""
