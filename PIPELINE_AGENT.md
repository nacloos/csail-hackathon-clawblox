# Pipeline Agent — Methodology

You are driving the ClawBlox video-variation pipeline end to end **on your
own**. The operator is not watching for prompts — they explicitly handed this
run to you. **Decide and act.** Do not stall, do not ask for confirmation, do
not narrate plans before doing them. If a gate appears, you choose; if a
prompt is weak, you regenerate it; if you're unsure between two reasonable
options, pick one and write the reason to the decision log.

The pipeline is otherwise scripted; your job is to make the judgment calls at
the three checkpoints where it pauses for input, then let it finish.

## Environment

- The webapp is reachable at the URL in `$WEBAPP_URL` (e.g.
  `http://127.0.0.1:8000`).
- Your run name is in `$RUN_NAME` (e.g. `20260426-153012_clip`).
- The run directory is in `$RUN_DIR` (e.g.
  `runs/20260426-153012_clip`). All artifacts you reason from live there.
- Use `Bash` for HTTP (`curl -fsS …`) and `Read` for inspecting files. You do
  not need network access beyond the webapp.

## Polling loop

Loop forever:

1. `curl -fsS "$WEBAPP_URL/api/runs/$RUN_NAME"` and parse the JSON.
2. Look at `state.gate`. If it is `null` and `state.running` is true, the
   pipeline is mid-stage — sleep ~5 seconds and poll again.
3. If `state.gate` is one of the four named gates (`start`, `pick-segment`,
   `choose-count`, `approve-prompts`), follow the rubric for that gate
   below, POST your decision, and resume polling.
4. If `state.gate` is `null` and `state.running` is false, the run is either
   complete or errored. Read the last ~200 lines of `$RUN_DIR/pipeline.log` to
   confirm. If complete (`state.outputs` populated and `status == "done"`),
   exit cleanly. If errored, summarize the error in one paragraph and exit.

Do not poll faster than once every 4 seconds. Do not exit before the run
finishes unless you encounter an error you cannot resolve.

## Gate 0 — `start`

The very first gate. The clip has been uploaded but the pipeline subprocess
has not been kicked yet — nothing has been sent to TwelveLabs / Pegasus.
The operator has already chosen the render model at upload time; you do
not change it. Your only job here is to start.

**Decision rubric.** Always start immediately.

**Action.**
```bash
curl -fsS -X POST "$WEBAPP_URL/api/runs/$RUN_NAME/start"
```

## Gate 1 — `pick-segment`

**Files to read.**
- `$RUN_DIR/01_twelvelabs/segments.json` — array of segments with
  `start_time`, `end_time`, `title`, `description`.
- `$RUN_DIR/00_input/metadata.json` — original duration / dims for context.

**Decision rubric.** You are selecting training data, not picking a pretty
shot. Apply the SOUL principles 1 and 2 directly:

- **Information density first.** Pick segments where the `description`
  describes manipulation, contact, motion that changes state, or a task
  progression. A robot grasping, placing, pouring, pushing, opening — those
  are signal. A camera pan over a static scene, an idle hover, a transition,
  a blur — those are noise.
- **Reject static/redundant motion.** If the description suggests
  near-static framing, repetitive looping motion, or no environmental change,
  do not pick it.
- **Reject blurs and rapid cuts.** Anything described as a transition,
  motion blur, or rapid camera move makes a bad reskin and is low-signal.
- **Length range.** Segments are already chopped to ≤5s upstream (Aleph's
  per-call cap); among qualifying candidates, length is not the deciding
  factor — content density is.
- **Tie-break.** When two segments are both contact-rich, prefer the one
  with the more concrete object/material vocabulary (object nouns beat
  abstract scene labels) — those reskin more legibly.

**Action.** `POST /api/runs/$RUN_NAME/pick-segment` with body
`{"index": N}` where `N` is 0-based into the array.

```bash
curl -fsS -X POST "$WEBAPP_URL/api/runs/$RUN_NAME/pick-segment" \
  -H "Content-Type: application/json" -d '{"index": 2}'
```

## Gate 2 — `choose-count`

**Files to read.** None required.

**Decision rubric.** **Always pick 3.** Per SOUL principle 4 ("Controlled
Diversity"), every selected segment gets exactly 3 variants — no more, no
less. The webapp accepts `1`, `3`, or `8`; you choose `3`. Do not deviate.

**Action.**
```bash
curl -fsS -X POST "$WEBAPP_URL/api/runs/$RUN_NAME/count" \
  -H "Content-Type: application/json" -d '{"n": 3}'
```

## Gate 3 — `approve-prompts`

This is the longest gate. You can iterate.

**Files to read.**
- `$RUN_DIR/03_analysis/prompts.json` — the original analysis (`prompts`,
  `variation_axes`, `invariants`).
- `$RUN_DIR/04_variations/working_prompts.json` — the live editing buffer
  (created on first regen/edit; before that, treat the first `count` items of
  `analysis.prompts` as the working set).
- `$RUN_DIR/02_segment/first_frame.png` — the image that will be edited.

**Quality bar for each prompt.** Apply SOUL principle 3 directly. A prompt
is acceptable when ALL hold:

1. **Preserves what matters.** Does not contradict the analysis's
   `invariants`. The geometry, the object identities, the trajectory and
   motion of the subject, and the spatial relationships between objects must
   be untouched. The edit only changes surface attributes.
2. **Varies boldly on environment.** Each prompt should anchor in a
   genuinely different plausible environment from its siblings — indoor /
   outdoor / industrial / natural / abstract. Not different shades of the
   same room.
3. **Varies boldly on textures, floor, lighting.** Concrete, named choices
   for material/surface, floor (wood / concrete / grass / metal / sand /
   etc.), and lighting (time-of-day, harsh shadows vs. diffuse, colored vs.
   neutral). No vague mood adjectives.
4. **No subtle changes.** Two prompts that would yield visually similar
   outputs (same family of palette + lighting + setting) are a regen target.
   The 3 variants should look like they came from 3 different worlds.
5. **1–3 sentences, concrete and visual.** No bullet lists inside the
   prompt. No abstract feelings without visual anchors.

**Iteration loop.**
1. Fetch the working prompts:
   ```bash
   curl -fsS -X POST "$WEBAPP_URL/api/runs/$RUN_NAME/working-prompts"
   ```
2. Score each against the bar above. Note which 1-based indices fail.
3. If any indices fail, regenerate them in one batch:
   ```bash
   curl -fsS -X POST "$WEBAPP_URL/api/runs/$RUN_NAME/regen" \
     -H "Content-Type: application/json" -d '{"indices": [2, 5]}'
   ```
   The response includes the new full list. Re-score and repeat. Cap regens
   at **3 rounds total** for this gate; if a slot still fails after that,
   accept it and move on (the human can intervene).
4. If a prompt is *almost* right but needs a small tweak, edit it directly
   instead of regenerating:
   ```bash
   curl -fsS -X POST "$WEBAPP_URL/api/runs/$RUN_NAME/edit-prompt" \
     -H "Content-Type: application/json" \
     -d '{"index": 3, "text": "<full replacement>"}'
   ```
5. When the working set passes the bar (or you've hit the regen cap),
   approve:
   ```bash
   curl -fsS -X POST "$WEBAPP_URL/api/runs/$RUN_NAME/approve"
   ```

After approval the pipeline runs the image edits and Lucy Edit stage with no
further gates. Continue polling until completion.

## Logging and updates

After each decision, write a one-line summary to
`$RUN_DIR/agent_decisions.log` (append-only) so the operator can audit what
you did:

```
2026-04-26T15:30:00Z start model=lucy reason="kept operator's choice"
2026-04-26T15:32:11Z pick-segment idx=2 reason="contact-rich grasp, stable subject (5.4s, 'red mug pickup')"
2026-04-26T15:33:02Z choose-count n=3 reason="SOUL principle 4"
2026-04-26T15:35:14Z regen indices=[2,5] reason="too abstract, no concrete material/palette"
2026-04-26T15:36:48Z approve reason="all 8 distinct on style+palette, invariants respected"
```

Keep reasons short (≤ 80 chars). The operator reads these to spot-check you.

## Failure modes

- **HTTP 4xx from the webapp.** Re-read state — your assumption about the
  current gate may be stale (the human may have advanced manually).
- **`state.running` flips to false with `state.gate == null` mid-run.** The
  pipeline crashed. Read `pipeline.log` tail, summarize, exit.
- **Same gate appears more than ~6 times in a row.** Something is wrong with
  your decisions. Read `pipeline.log` to see what the pipeline is rejecting,
  and adjust.

You do not need to ask the human for confirmation. They chose agent mode
specifically so you would decide.
