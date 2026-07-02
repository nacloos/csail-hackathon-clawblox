# EPISODIC_MEMORY.md - What Happened
Memories are stored in `EPISODIC_MEMORY.jsonl`, one JSON object per line with fields `timestamp`, `summary`, `detail`.

Entries are written *after* something happened — not upfront as plans or narration. The goal is to remember **what was done**, not every low-level step. **Not what you learned** — learnings, techniques, and mistakes go in SEMANTIC_MEMORY.

`summary` field: **one short sentence**, past tense, what happened. Easily scannable by extracting just this field across entries.

`detail` field: one short paragraph, at a slightly higher level of detail than `summary`. Skip move-by-move narration and specific coordinates. If you can't say more than `summary`, the event didn't warrant an entry.

`timestamp` field: `YYYY-MM-DDTHH:MM:SSZ` (UTC, second-precision), e.g. `2026-04-22T14:30:00Z`. Generate with `date -u +"%Y-%m-%dT%H:%M:%SZ"`.

Append-only. Add new entries at the end; never rewrite or delete old ones.

---

_Don't modify this file. Read only._
