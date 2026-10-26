# LuthorSQL cascade: BIRD test submission (combined GPU + closed LLM)

LuthorSQL (System 1, a 7B open-weight model on one GPU) answers every question. When it is unsure, the question goes
to Claude Opus 5.5 (System 2, Anthropic API) together with System 1's candidate queries. This is the system of our
paper's pre-registered held-out tests, with the threshold fixed in advance (validator margin 3.0).

**Evidence:** used. **`column_meaning.json`:** not used. No gold SQL is read; the `SQL` field of the question file,
if present, is ignored.

## The two stages (GPU code and API code are separate)

1. **Stage 1, GPU, no API** (`run_bird.sh`, `luthor/`, `requirement.txt`, venv `./venv`): the same package as our
   System 1 submission. A grammar-constrained generator writes 32 distinct typed query plans, a compiler turns them
   into SQLite, each is executed read-only, and a validator scores them. Output: `<out>.s1.json` and
   `<out>.s1.json.cands.jsonl` (every candidate with its validator score and a short result preview).
2. **Stage 2, API, no GPU** (`cascade/`, `cascade/requirement_api.txt`, venv `./venv_api`):
   - **Routing.** The margin is the validator's score gap between its top two distinct results. A question goes to
     the agent if the margin is below 3.0 or there is no margin (one distinct result, or nothing ran). On BIRD dev this
     routes 850 of 1,534 questions (55.4%).
   - **Agent.** Routed questions are grouped by database into chunks of at most 25. Each chunk is one Claude Agent
     SDK session (the Claude Code agent loop) on `claude-opus-5-5`. The prompt (`cascade/PROMPT_routed.md`) is the
     one from our paper, verbatim. It lists System 1's top 8 result groups per question and asks the agent to pick a
     clearly right candidate, or write its own query only when every candidate is wrong. The agent is limited to at
     most 8 SQL executions per question by instruction.
   - **Tool.** The agent's database access is `TOOLS/sql.py`: read-only SQLite, the schema with 3 sample rows per
     table and the column descriptions, and at most 20 result rows per query.
   - **Merge.** Kept questions keep System 1's answer. Routed questions take the agent's pick, or its own SQL. A
     routed question with no agent answer falls back to System 1's answer; `merge_report.json` counts these.

## What the agent can and cannot do (compliance)

The agent can read only its sandbox (`<out>.cascade_work/box/`) and write only `answers/` and `work/` there. The
sandbox holds the chunk files, the SQL tool and links to the databases.

- **Tools:** Bash, Read, Write, Edit, Glob and Grep only. There is no web fetch, no web search, no subagents and no
  MCP servers. Skills, plugins, auto-memory, and user and project settings are all off.
- **Guard (`cascade.py: make_guard`):** a check runs before every tool call. It refuses network and install commands
  (curl, wget, ssh, pip, git and so on), `http(s)://` URLs, imports of network modules, and any path outside the
  sandbox. Replaying the 420 tool calls our agents made in the paper's held-out runs, it refused none.
- **OS sandbox:** where the platform supports it, Bash commands also run in Claude Code's sandbox with no network:
  macOS, or Linux with `bubblewrap` installed. Otherwise the guard alone applies; `--os-sandbox on` makes the sandbox
  required.
- **Network:** the only outbound connection is the Anthropic API (`api.anthropic.com`), which receives the questions,
  the evidence, schema excerpts and query results. Telemetry, error reporting and auto-update are off.
- **Authentication:** only `ANTHROPIC_API_KEY`. The run stops if Claude Code would use any other credential.

## Run

```bash
export ANTHROPIC_API_KEY=...        # our key, sent by email; we reset it after the run
bash run_cascade.sh <path>/test.json <path>/test_tables.json <path>/test_databases predict_test.json
```

- **Requirements:** Linux, one NVIDIA GPU with at least 48 GB for stage 1 (see the System 1 notes below), Python 3.12
  through `uv`, and internet access for package installation, the base model and the Anthropic API.
- **Output:** `predict_test.json` in BIRD's format, `{"0": "SQL\t----- bird -----\tdb_id", ...}`, in question-file
  order.
- **Parallel sessions:** 4 by default; set `CASCADE_JOBS` to change it.
- **Logs:**
  - `predict_test.json.cascade_work/logs/<chunk>.jsonl` holds every agent message, tool call and guard decision.
  - `routing.json`, `merge_report.json` and `token_report.json` are in the same folder.
  - `predict_test.json.cascade.log` is the console log.
- **Resume:** rerun the same command. Stage 1 continues from its last saved database. In stage 2, finished chunks are
  skipped and only unanswered questions are retried, up to 2 more times.
- **Pipeline test without API calls:** `CASCADE_DRY=1 bash run_cascade.sh ...` uses fake answers (always candidate A,
  which reproduces System 1's output).

## Tokens and cost (the advance count BIRD asks for)

- **Measured on BIRD dev with this package (2026-10-26):** 1,534 questions, 850 routed in 39 sessions; 9,284,617 prompt
  tokens (610 uncached input, 1,410,848 cache writes, 7,873,159 cache reads), i.e. 6,053 per question and 10,923 per
  routed question. Logged output tokens: 3,312 (the Agent SDK's usage log under-counts output; billed output is
  higher). Dev EX of this package: 83.64% (System 1 alone 59.52%), original dev gold.
- **Estimate from our logged runs under the same harness:** 36k to 41k prompt tokens per routed BIRD question (about
  89% of them cache reads) and about 110 output tokens.
  - Dev (850 routed): about 25M to 35M prompt tokens.
  - Test (about 55% of 1,789 routed, so about 985): about 30M to 40M prompt tokens.
  - At list price ($4 per million input tokens, $5 per million cache writes, $0.20 per million cache reads and $20
    per million output tokens), that is about $25 to $30 for dev and $30 to $35 for test.
- **Why the per-question count is high:** every agent session carries about 46k tokens of Claude Code context
  (system prompt and tool definitions), re-read on each turn.

## Differences from the runs in our paper

1. The agent runs through the Claude Agent SDK with an API key instead of Claude Code subagents on a subscription.
   The session carries no personal configuration (user instructions, connectors), which the paper's sessions did.
   It starts from Claude Code's default system prompt rather than a subagent's.
2. Web tools are removed, and the guard and OS sandbox described above are added.
3. Chunk files name questions by their index in the question file (`q0`, `q1`, ...).

The prompt, the SQL tool, the candidate format, the routing rule, the chunking and the model are unchanged.

## System 1 notes (stage 1)

The stage 1 notes are unchanged from our System 1 submission:
- About 1 hour on one A100 80GB for 1,789 questions; peak GPU memory about 41 GB.
- The weights go in `./weights` (`r1/` and `valhard_rr/`). The base model `seeklhy/OmniSQL-7B` downloads on first use.
- Progress is saved per database and resumes on rerun.
