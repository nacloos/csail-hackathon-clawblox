/* ClawBlox front-end · node-graph canvas (Tier 1).
   No framework. fetch + DOM + SVG bezier wires.            */

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  runs: [],
  active: null,
  detail: null,
  workingPrompts: null,
  pollTimer: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  drawer: null,
};

// ---------------- helpers
function el(tag, attrs = {}, kids = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v);
  }
  for (const k of [].concat(kids)) {
    if (k == null || k === false) continue;
    node.appendChild(typeof k === "string" ? document.createTextNode(k) : k);
  }
  return node;
}
function fmtSecs(s) { return s == null ? "" : `${Number(s).toFixed(2)}s`; }

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} ${r.status}: ${await r.text()}`);
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}

// ---------------- runs sidebar
async function refreshRuns() {
  state.runs = await api("GET", "/api/runs");
  renderRuns();
}
function dotClass(s) {
  if (s === "running") return "dot-running";
  if (s === "needs-segment" || s === "needs-count" || s === "needs-approval") return "dot-needs";
  if (s === "done") return "dot-done";
  return "dot-idle";
}
function renderRuns() {
  const ul = $("#runsList");
  ul.innerHTML = "";
  if (!state.runs.length) {
    ul.appendChild(el("li", { class: "muted", style: "cursor:default" }, ["no runs yet"]));
    return;
  }
  for (const r of state.runs) {
    const li = el("li", {
      class: state.active === r.name ? "active" : "",
      onclick: () => selectRun(r.name),
    }, [
      el("span", { class: `dot ${dotClass(r.status)}` }),
      el("div", {}, [
        el("div", { class: "name" }, [r.name]),
        el("div", { class: "meta" }, [r.status + (r.duration ? ` · ${fmtSecs(r.duration)}` : "")]),
      ]),
    ]);
    ul.appendChild(li);
  }
}

// ---------------- run detail
async function selectRun(name) {
  state.active = name;
  state.workingPrompts = null;
  $("#empty").classList.add("hidden");
  $("#run").classList.remove("hidden");
  closeDrawer();
  renderRuns();
  await refreshDetail();
  // initial fit on first paint
  setTimeout(fitToScreen, 50);
  startPolling();
}
function showNewRunForm() {
  state.active = null;
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  $("#run").classList.add("hidden");
  $("#empty").classList.remove("hidden");
  closeDrawer();
  renderRuns();
}
function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(refreshDetail, 2500);
}
async function refreshDetail() {
  if (!state.active) return;
  try {
    state.detail = await api("GET", `/api/runs/${state.active}`);
    renderCanvas();
  } catch (e) { console.warn("refresh failed:", e); }
  refreshRuns().catch(() => {});
}

function modelPill(d) {
  const label = d.model === "runway" ? "runway · aleph" : "lucy · decart";
  return el("span", { class: "pill model", title: "render model" }, [label]);
}

function agentButton(d) {
  const running = !!d.agent_running;
  const isAgentMode = d.mode === "agent";
  const btn = el("button", {
    class: `btn small ${running ? "warn" : "ghost"}`,
    title: running
      ? "stop the agent (you take over manual control)"
      : (isAgentMode ? "agent flag is set — relaunch agent" : "let Claude drive this run"),
  }, [running ? "dismiss agent" : (isAgentMode ? "relaunch agent" : "summon agent")]);
  btn.onclick = async (e) => {
    e.preventDefault();
    btn.disabled = true;
    const path = running ? "dismiss-agent" : "summon-agent";
    try {
      await api("POST", `/api/runs/${d.name}/${path}`);
      await refreshDetail();
    } catch (err) {
      alert(err.message);
    } finally {
      btn.disabled = false;
    }
  };
  return btn;
}

function statusPill(d) {
  const k = (text, kind) => el("span", { class: `pill ${kind || ""}` }, [text]);
  const isAgent = d.mode === "agent";
  if (d.running) return k(isAgent ? "agent driving · stage running" : "stage running", "teal");
  if (d.gate === "start")
    return k(isAgent ? "agent starting…" : "ready · confirm & start", isAgent ? "teal" : "gold");
  if (d.gate === "pick-segment")
    return k(isAgent ? "agent picking a scene…" : "pick a scene, human", isAgent ? "teal" : "gold");
  if (d.gate === "choose-count")
    return k(isAgent ? "agent choosing count…" : "how many clones?", isAgent ? "teal" : "gold");
  if (d.gate === "approve-prompts")
    return k(isAgent ? "agent vetting prompts…" : "sign off on the prompts", isAgent ? "teal" : "gold");
  const total = d.outputs?.length || 0;
  const done  = (d.outputs || []).filter(o => o.video).length;
  if (total > 0 && done === total) return k("shipped", "green");
  return k("napping", "");
}

// ---------------- graph layout
function buildGraph(d) {
  const N = d.count || (d.approved_prompts ? d.approved_prompts.length : 0);
  const COL = 320;
  const ROW = 230;            // height per branch row
  const NODE_H = 200;
  const branchH = Math.max(N, 1) * ROW;
  const trunkY = Math.max(0, (branchH - NODE_H) / 2);

  const nodes = {};
  const wires = [];
  const add = (id, col, y, n) => { nodes[id] = { id, x: col * COL, y, w: 280, h: NODE_H, ...n }; };

  add("source",  0, trunkY, sourceNode(d));
  add("pegasus", 1, trunkY, pegasusNode(d));
  add("pick",    2, trunkY, pickNode(d));
  add("analyze", 3, trunkY, analyzeNode(d));
  add("count",   4, trunkY, countNode(d));
  add("approve", 5, trunkY, approveNode(d));

  wires.push(["source","pegasus"], ["pegasus","pick"], ["pick","analyze"],
             ["analyze","count"], ["count","approve"]);

  if (N > 0) {
    for (let i = 0; i < N; i++) {
      const y = i * ROW;
      add(`img_${i}`,  6, y, imageNode(d, i + 1));
      add(`lucy_${i}`, 7, y, lucyNode(d, i + 1));
      wires.push(["approve", `img_${i}`], [`img_${i}`, `lucy_${i}`]);
    }
  }
  return { nodes, wires };
}

// ---- per-node spec ----

function sourceNode(d) {
  const ready = !!d.input;
  return {
    kind: "display",
    title: "01 · Source clip",
    state: ready ? "done" : "locked",
    body: ready
      ? [
          el("div", { class: "node-thumb" }, [
            d.source_video
              ? el("video", { src: d.source_video, muted: "", autoplay: "", loop: "", playsinline: "" })
              : el("span", {}, ["no preview"]),
          ]),
          el("div", { class: "kv" }, [
            el("b", {}, [fmtSecs(d.input.duration)]), " · ",
            `${d.input.width}×${d.input.height}`,
          ]),
        ]
      : [el("div", { class: "muted" }, ["no source clip yet."])],
  };
}

function pegasusNode(d) {
  const ready = !!d.segments;
  if (ready) {
    return {
      kind: "display",
      title: "02 · Pegasus 1.5",
      state: "done",
      body: [
        el("div", { class: "kv" }, [
          el("b", {}, [String(d.segments.length)]), " segments detected",
        ]),
        el("div", { class: "muted", style: "font-size:11.5px" }, [
          "scenes by setting / action / framing",
        ]),
      ],
    };
  }
  // Pre-start gate: clip uploaded but pipeline not yet kicked.
  if (d.gate === "start") {
    const agentMode = d.mode === "agent";
    const modelLabel = (d.model === "runway") ? "Runway · Aleph" : "Lucy · Decart";
    return {
      kind: "choice-inline",
      title: "02 · Pegasus 1.5",
      state: "gate",
      body: [
        el("div", { class: "muted", style: "font-size:11.5px;margin-bottom:8px" },
          [agentMode
            ? "agent will start the pipeline shortly…"
            : `ready · will render with ${modelLabel}`]),
        el("button", {
          class: "btn primary block",
          disabled: agentMode ? "" : null,
          onclick: async (e) => {
            e.stopPropagation();
            if (agentMode) return;
            try {
              await api("POST", `/api/runs/${d.name}/start`);
              await refreshDetail();
            } catch (err) { alert(err.message); }
          },
        }, [agentMode
          ? "agent will send →"
          : "send to TwelveLabs →"]),
      ],
    };
  }
  return {
    kind: "display",
    title: "02 · Pegasus 1.5",
    state: d.input ? "running" : "locked",
    body: [el("div", { class: "muted" }, ["segmenting…"])],
  };
}

function pickNode(d) {
  const has = !!d.segments;
  const chosen = d.chosen;
  let st = "locked";
  if (has && chosen) st = "done";
  else if (has) st = "gate";
  return {
    kind: "choice",
    title: "03 · Pick segment",
    state: st,
    body: [
      el("div", { class: "node-thumb" }, [
        d.first_frame
          ? el("img", { src: d.first_frame })
          : (has
              ? el("div", { class: "muted" }, ["no scene picked yet"])
              : el("div", { class: "muted" }, ["waiting on segmentation"])),
      ]),
      chosen
        ? el("div", { class: "kv" }, [
            el("b", {}, [`${fmtSecs(chosen.start_time)}–${fmtSecs(chosen.end_time)}`]),
            chosen.title ? "  ·  " + chosen.title : "",
          ])
        : el("div", { class: "kv" }, [
            has ? `${d.segments.length} options to choose from` : "—",
          ]),
      el("button", {
        class: "node-cta",
        onclick: (e) => { e.stopPropagation(); openDrawer("segments"); },
      }, [chosen ? "review / change" : "open scene picker →"]),
    ],
  };
}

function analyzeNode(d) {
  if (!d.first_frame)
    return { kind: "display", title: "04 · Frame analysis", state: "locked",
      body: [el("div", { class: "muted" }, ["needs a chosen segment."])] };
  if (!d.analysis)
    return { kind: "display", title: "04 · Frame analysis", state: "running",
      body: [el("div", { class: "kv" }, ["nano is squinting…"])] };
  const axes = (d.analysis.variation_axes || []).slice(0, 4).map(x =>
    el("div", { class: "kv" }, [
      el("b", {}, [(x.name || "?") + ": "]),
      ((x.options || []).slice(0, 3).join(" · ")),
    ])
  );
  return {
    kind: "display",
    title: "04 · GPT-5.4-nano",
    state: "done",
    body: [
      el("div", { class: "muted", style: "font-size:11px;letter-spacing:.4px;text-transform:uppercase" },
        ["variation axes"]),
      ...axes,
    ],
  };
}

function countNode(d) {
  if (!d.analysis)
    return { kind: "display", title: "05 · How many?", state: "locked",
      body: [el("div", { class: "muted" }, ["unlocks after analysis."])] };
  const st = d.count ? "done" : "gate";
  const agentLocked = d.mode === "agent" && !d.count;
  return {
    kind: "choice-inline",
    title: "05 · How many variations?",
    state: st,
    body: [
      el("div", { class: "muted", style: "font-size:11px" },
        [d.count
          ? `chose ${d.count}`
          : (agentLocked ? "agent will choose…" : "click to commit")]),
      el("div", { class: "node-chips" },
        [1, 3, 8].map(n => el("button", {
          class: (d.count === n ? "selected " : "") + (agentLocked ? "locked" : ""),
          disabled: agentLocked ? "" : null,
          onclick: async (e) => {
            e.stopPropagation();
            if (d.count || agentLocked) return;
            try {
              await api("POST", `/api/runs/${d.name}/count`, { n });
              await refreshDetail();
            } catch (err) { alert(err.message); }
          },
        }, [String(n)])),
      ),
    ],
  };
}

function approveNode(d) {
  if (!d.count || !d.analysis)
    return { kind: "display", title: "06 · Approve prompts", state: "locked",
      body: [el("div", { class: "muted" }, ["unlocks once you set a count."])] };
  const approved = !!d.approved_prompts;
  return {
    kind: "choice",
    title: "06 · Approve prompts",
    state: approved ? "done" : "gate",
    body: [
      el("div", { class: "kv" }, [
        approved
          ? el("span", {}, [
              el("b", {}, [String(d.approved_prompts.length)]), " approved · ready to render"
            ])
          : el("span", {}, [el("b", {}, [String(d.count)]), " prompts await your judgment"]),
      ]),
      el("div", { class: "muted", style: "font-size:11.5px" },
        [approved ? "(image stages have been kicked off)" : "edit, regen, or accept as-is"]),
      el("button", {
        class: "node-cta",
        onclick: (e) => { e.stopPropagation(); openDrawer("approve"); },
      }, [approved ? "review prompts" : "open prompt editor →"]),
    ],
  };
}

function imageNode(d, idx) {
  const v = (d.variations || []).find(x => x.index === idx);
  const ready = v && v.image;
  let st;
  if (ready) st = "done";
  else if (!d.approved_prompts && !d.count) st = "locked";
  else st = "running";
  return {
    kind: "display",
    title: `07 · image ${String(idx).padStart(2, "0")}`,
    state: st,
    body: [
      el("div", { class: "node-thumb square" }, [
        ready ? el("img", { src: v.image }) : el("span", { class: "spinner" }),
      ]),
      el("div", { class: "muted", style: "font-size:11.5px" },
        [ready ? "rendered" : "rendering…"]),
    ],
  };
}

function lucyNode(d, idx) {
  const v = (d.outputs || []).find(x => x.index === idx);
  const variation = (d.variations || []).find(x => x.index === idx);
  const ready = v && v.video;
  let st;
  if (ready) st = "done";
  else if (!d.approved_prompts && !d.count) st = "locked";
  else st = "running";
  const thumb = el("div", {
    class: "node-thumb square" + (ready ? " playable" : ""),
    title: ready ? "click to watch" : "",
    onclick: ready ? (e) => { e.stopPropagation(); openDrawer(`output:${idx}`); } : null,
  }, [
    ready
      ? (variation && variation.image
          ? el("img", { src: variation.image, loading: "lazy" })
          : el("div", { class: "video-fallback" }, ["▶ click to watch"]))
      : el("span", { class: "spinner" }),
    ready ? el("div", { class: "play-overlay" }, ["▶"]) : null,
  ]);
  return {
    kind: "display",
    title: `08 · ${d.model === "runway" ? "runway" : "lucy"} ${String(idx).padStart(2, "0")}`,
    state: st,
    body: [
      thumb,
      el("div", { class: "muted", style: "font-size:11.5px" },
        [ready ? "click to watch" : (v?.status || "queued")]),
    ],
  };
}

// ---------------- canvas render
function renderCanvas() {
  const d = state.detail;
  if (!d) return;
  $("#runTitle").textContent = d.name;
  const rs = $("#runStatus");
  rs.innerHTML = "";
  rs.appendChild(modelPill(d));
  rs.appendChild(statusPill(d));
  rs.appendChild(agentButton(d));

  const graph = buildGraph(d);
  const nodesLayer = $("#nodes");
  const svg = $("#wires");
  nodesLayer.innerHTML = "";

  // size canvas content to fit graph
  let maxX = 0, maxY = 0;
  for (const n of Object.values(graph.nodes)) {
    maxX = Math.max(maxX, n.x + n.w + 60);
    maxY = Math.max(maxY, n.y + n.h + 60);
  }
  const content = $("#canvas-content");
  content.style.width  = `${maxX}px`;
  content.style.height = `${maxY}px`;
  svg.setAttribute("width",  String(maxX));
  svg.setAttribute("height", String(maxY));
  svg.setAttribute("viewBox", `0 0 ${maxX} ${maxY}`);
  svg.innerHTML = "";

  // wires first (behind nodes)
  for (const [from, to] of graph.wires) {
    const A = graph.nodes[from], B = graph.nodes[to];
    if (!A || !B) continue;
    const x1 = A.x + A.w, y1 = A.y + A.h / 2;
    const x2 = B.x,       y2 = B.y + B.h / 2;
    const cx = Math.max(40, (x2 - x1) * 0.5);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d",
      `M ${x1} ${y1} C ${x1 + cx} ${y1}, ${x2 - cx} ${y2}, ${x2} ${y2}`);
    let cls = "wire";
    if (A.state === "done" && (B.state === "done" || B.state === "running" || B.state === "gate")) cls += " ";
    else if (A.state !== "done") cls += " dim";
    if ((A.state === "done" && B.state === "running")) cls += " flow";
    path.setAttribute("class", cls);
    svg.appendChild(path);
  }

  // nodes on top
  for (const n of Object.values(graph.nodes)) {
    const cls = `node ${n.kind} ${n.state}`;
    const inner = el("div", { class: cls, style: `left:${n.x}px;top:${n.y}px;width:${n.w}px` }, [
      el("span", { class: "port in" }),
      el("span", { class: "port out" }),
      el("div", { class: "node-head" }, [
        el("div", { class: "stage-num" }, [n.title.split("·")[0].trim()]),
        el("div", { class: "node-title" }, [n.title.split("·").slice(1).join("·").trim()]),
        nodeBadge(n),
      ]),
      el("div", { class: "node-body" }, n.body || []),
    ]);
    nodesLayer.appendChild(inner);
  }

  applyTransform();
  $("#log").textContent = d.log_tail || "(no log yet)";

  // Agent reasoning panel — only visible for agent-driven runs.
  const agentSeen = d.mode === "agent" || (d.agent_decisions && d.agent_decisions.trim());
  const decBox = $("#agentDecisionsBox");
  if (agentSeen) {
    decBox.classList.remove("hidden");
    $("#agentDecisions").textContent = d.agent_decisions && d.agent_decisions.trim()
      ? d.agent_decisions
      : "(no decisions logged yet — agent will append one line per decision here)";
  } else {
    decBox.classList.add("hidden");
  }
}

function nodeBadge(n) {
  if (n.state === "running") return el("span", { class: "pill teal" }, ["running"]);
  if (n.state === "gate")    return el("span", { class: "pill gold" }, ["needs you"]);
  if (n.state === "done")    return el("span", { class: "pill green" }, ["done"]);
  if (n.state === "locked")  return el("span", { class: "pill" }, ["locked"]);
  return el("span", { class: "pill" }, [n.state || ""]);
}

// ---------------- pan / zoom
function applyTransform() {
  const c = $("#canvas-content");
  if (!c) return;
  c.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
}

function fitToScreen() {
  const c = $("#canvas");
  const inner = $("#canvas-content");
  if (!c || !inner) return;
  const cw = c.clientWidth, ch = c.clientHeight;
  const w = inner.offsetWidth || 800, h = inner.offsetHeight || 600;
  const m = 60;
  const z = Math.min((cw - m) / w, (ch - m) / h, 1);
  state.zoom = Math.max(0.3, z);
  state.panX = (cw - w * state.zoom) / 2;
  state.panY = (ch - h * state.zoom) / 2;
  applyTransform();
}

function bindCanvasInteractions() {
  const canvas = $("#canvas");

  // wheel = zoom around mouse
  canvas.addEventListener("wheel", (e) => {
    if (!state.detail) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const oldZ = state.zoom;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    state.zoom = Math.max(0.25, Math.min(2, oldZ * factor));
    state.panX = mx - (mx - state.panX) * (state.zoom / oldZ);
    state.panY = my - (my - state.panY) * (state.zoom / oldZ);
    applyTransform();
  }, { passive: false });

  // pan via empty-space drag
  let panning = false, sx = 0, sy = 0, ox = 0, oy = 0;
  canvas.addEventListener("mousedown", (e) => {
    // skip drags that start inside a node, button, or interactive control
    if (e.target.closest(".node") && !e.target.classList.contains("canvas-content")) return;
    if (e.button !== 0 && e.button !== 1) return;
    panning = true; sx = e.clientX; sy = e.clientY;
    ox = state.panX; oy = state.panY;
    canvas.classList.add("panning");
  });
  window.addEventListener("mousemove", (e) => {
    if (!panning) return;
    state.panX = ox + (e.clientX - sx);
    state.panY = oy + (e.clientY - sy);
    applyTransform();
  });
  window.addEventListener("mouseup", () => {
    panning = false; canvas.classList.remove("panning");
  });

  $("#fitBtn").onclick = fitToScreen;
  $("#resetBtn").onclick = () => { state.zoom = 1; state.panX = 0; state.panY = 0; applyTransform(); };
}

// ---------------- drawer (rich interactions)
function openDrawer(kind) {
  state.drawer = kind;
  const d = state.detail;
  if (!d) return;
  $("#drawer").classList.remove("hidden");
  $("#drawerScrim").classList.remove("hidden");
  if (kind === "segments") drawerSegments(d);
  else if (kind === "approve") drawerApprove(d);
  else if (kind.startsWith("output:")) drawerOutput(d, parseInt(kind.slice(7), 10));
}
function closeDrawer() {
  state.drawer = null;
  $("#drawer").classList.add("hidden");
  $("#drawerScrim").classList.add("hidden");
  state.workingPrompts = null;
}

function drawerSegments(d) {
  $("#drawerTitle").textContent = "Pick a segment";
  const body = $("#drawerBody");
  body.innerHTML = "";
  if (!d.segments) {
    body.appendChild(el("div", { class: "muted" }, ["segments not ready yet"]));
    return;
  }
  const chosenIdx = d.chosen
    ? d.segments.findIndex(s => s.start_time === d.chosen.start_time && s.end_time === d.chosen.end_time)
    : -1;

  if (d.chosen && d.first_frame) {
    body.appendChild(el("div", { class: "card", style: "margin-bottom:12px" }, [
      el("div", { class: "muted", style: "font-size:11.5px;letter-spacing:.4px;text-transform:uppercase;margin-bottom:6px" },
        ["currently chosen"]),
      el("div", { class: "source-row" }, [
        d.segment_video
          ? el("video", { src: d.segment_video, controls: "", muted: "" })
          : el("img", { src: d.first_frame }),
        el("div", { class: "kv" }, [
          el("div", {}, [el("b", {}, ["range: "]),
            `${fmtSecs(d.chosen.start_time)}–${fmtSecs(d.chosen.end_time)}`]),
          d.chosen.title ? el("div", {}, [el("b", {}, ["title: "]), d.chosen.title]) : null,
          d.chosen.description ? el("div", { class: "muted" }, [d.chosen.description]) : null,
        ]),
      ]),
    ]));
  }

  const agentMode = d.mode === "agent";
  const locked = agentMode || !!d.chosen;
  const pickSegment = async (i) => {
    try {
      await api("POST", `/api/runs/${d.name}/pick-segment`, { index: i });
      closeDrawer();
      await refreshDetail();
    } catch (e) { alert(e.message); }
  };
  body.appendChild(el("div", { class: "seg-grid" },
    d.segments.map((s, i) => {
      const dur = s.end_time - s.start_time;
      const previewSrc = d.source_video
        ? `${d.source_video}#t=${s.start_time.toFixed(2)},${s.end_time.toFixed(2)}`
        : null;
      const isChosen = i === chosenIdx;
      const pickBtn = el("button", {
        class: `btn small ${isChosen ? "primary" : "primary"}`,
        disabled: locked ? "" : null,
        onclick: (ev) => {
          ev.stopPropagation();
          if (locked) return;
          pickSegment(i);
        },
      }, [
        isChosen ? "✓ chosen" :
          (agentMode ? "agent will pick" :
            (d.chosen ? "locked" : "use this segment →")),
      ]);
      const tile = el("div", {
        class: "seg-tile" + (isChosen ? " selected" : "")
          + (locked ? " locked" : ""),
      }, [
        previewSrc
          ? el("video", {
              class: "seg-preview",
              src: previewSrc,
              controls: "",
              muted: "",
              preload: "metadata",
              playsinline: "",
            })
          : null,
        el("div", { class: "seg-time" },
          [`${fmtSecs(s.start_time)}–${fmtSecs(s.end_time)}  ·  ${dur.toFixed(2)}s`]),
        el("div", { class: "seg-title" }, [s.title || "(untitled)"]),
        s.description ? el("div", { class: "seg-desc" }, [s.description]) : null,
        el("div", { class: "seg-actions" }, [pickBtn]),
      ]);
      return tile;
    })
  ));
}

function drawerApprove(d) {
  $("#drawerTitle").textContent = "Approve prompts";
  const body = $("#drawerBody");
  body.innerHTML = "";
  if (!d.count || !d.analysis) {
    body.appendChild(el("div", { class: "muted" }, ["set a count first"]));
    return;
  }
  const approved = !!d.approved_prompts;
  const agentLocked = d.mode === "agent" && !approved;
  const readOnly = approved || agentLocked;
  const initial = d.approved_prompts || d.analysis.prompts.slice(0, d.count);

  if (!approved && state.workingPrompts == null) {
    api("POST", `/api/runs/${d.name}/working-prompts`)
      .then(({ prompts }) => {
        if (Array.isArray(prompts) && prompts.length === d.count) {
          state.workingPrompts = prompts.slice();
          if (state.drawer === "approve") drawerApprove(state.detail);
        }
      })
      .catch(() => {});
  }
  const live = approved ? initial : (state.workingPrompts || initial);

  if (agentLocked) {
    body.appendChild(el("div", { class: "muted", style: "margin-bottom:10px;font-size:12px" },
      ["agent is reviewing these prompts. dismiss the agent to take over."]));
  }

  const list = el("div", { class: "prompts" }, live.map((text, i) => {
    const ta = el("textarea", { rows: "3" }, [text || ""]);
    ta.value = text || "";
    ta.disabled = readOnly;
    ta.addEventListener("input", () => {
      if (!state.workingPrompts) state.workingPrompts = live.slice();
      state.workingPrompts[i] = ta.value;
    });
    const acts = readOnly ? el("div") : el("div", { class: "actions" }, [
      el("button", {
        class: "btn small ghost",
        onclick: async () => {
          if (!state.workingPrompts) state.workingPrompts = live.slice();
          state.workingPrompts[i] = ta.value;
          try {
            await api("POST", `/api/runs/${d.name}/edit-prompt`, { index: i + 1, text: ta.value });
          } catch (e) { alert(e.message); }
        },
      }, ["save"]),
      el("button", {
        class: "btn small warn",
        onclick: async () => {
          try {
            const { prompts } = await api("POST", `/api/runs/${d.name}/regen`,
              { indices: [i + 1] });
            state.workingPrompts = prompts;
            drawerApprove(d);
          } catch (e) { alert(e.message); }
        },
      }, ["regen"]),
    ]);
    return el("div", { class: "prompt" }, [
      el("div", { class: "num" }, [String(i + 1).padStart(2, "0")]),
      ta,
      acts,
    ]);
  }));
  body.appendChild(list);

  if (!readOnly) {
    body.appendChild(el("div", { class: "approval-bar", style: "margin-top:14px" }, [
      el("button", { class: "btn warn",
        onclick: async () => {
          const all = Array.from({ length: d.count }, (_, i) => i + 1);
          try {
            const { prompts } = await api("POST", `/api/runs/${d.name}/regen`, { indices: all });
            state.workingPrompts = prompts;
            drawerApprove(d);
          } catch (e) { alert(e.message); }
        }
      }, ["regen all"]),
      el("span", { class: "spacer" }),
      el("span", { class: "hint" }, ["click ‘save’ to persist your edits."]),
      el("button", { class: "btn primary",
        onclick: async () => {
          if (state.workingPrompts) {
            for (let i = 0; i < state.workingPrompts.length; i++) {
              try { await api("POST", `/api/runs/${d.name}/edit-prompt`,
                { index: i + 1, text: state.workingPrompts[i] }); } catch (_) {}
            }
          }
          try {
            await api("POST", `/api/runs/${d.name}/approve`);
            state.workingPrompts = null;
            closeDrawer();
            await refreshDetail();
          } catch (e) { alert(e.message); }
        }
      }, ["approve & generate →"]),
    ]));
  }
}

function drawerOutput(d, idx) {
  $("#drawerTitle").textContent = `Output ${String(idx).padStart(2, "0")}`;
  const body = $("#drawerBody");
  body.innerHTML = "";
  const v = (d.outputs || []).find(x => x.index === idx);
  const variation = (d.variations || []).find(x => x.index === idx);
  if (!v || !v.video) {
    body.appendChild(el("div", { class: "muted" },
      [`output ${idx} not ready (${v?.status || "queued"})`]));
    return;
  }
  body.appendChild(el("video", {
    class: "output-player",
    src: v.video,
    controls: "",
    autoplay: "",
    loop: "",
    playsinline: "",
  }));
  if (variation && variation.prompt) {
    body.appendChild(el("div", { class: "output-prompt" }, [
      el("div", { class: "muted",
        style: "font-size:11.5px;letter-spacing:.4px;text-transform:uppercase;margin-bottom:6px" },
        ["prompt"]),
      el("div", {}, [variation.prompt]),
    ]));
  }
  const links = el("div", { class: "output-links" }, [
    el("a", { href: v.video, download: `${d.name}_${String(idx).padStart(2,"0")}.mp4` },
      ["download mp4"]),
  ]);
  body.appendChild(links);
}

function bindDrawer() {
  $("#drawerClose").onclick = closeDrawer;
  $("#drawerScrim").onclick = closeDrawer;
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
  });
}

// ---------------- upload
function bindUpload() {
  const input = $("#fileInput");
  const drop = $("#dropzone");
  const newBtn = $("#newRunBtn");
  const browse = $("#browseBtn");
  newBtn.onclick = () => showNewRunForm();
  browse.onclick = (e) => { e.preventDefault(); input.click(); };
  input.onchange = () => { if (input.files[0]) uploadFile(input.files[0]); };
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("drag");
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  });
}
async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const modeEl = document.querySelector('input[name="runMode"]:checked');
  fd.append("mode", modeEl ? modeEl.value : "manual");
  const modelEl = document.querySelector('input[name="runModel"]:checked');
  fd.append("model", modelEl ? modelEl.value : "lucy");
  $("#newRunBtn").disabled = true;
  $("#newRunBtn").textContent = "uploading…";
  try {
    const { name } = await api("POST", "/api/runs", fd);
    await refreshRuns();
    selectRun(name);
  } catch (e) { alert(e.message); }
  finally {
    $("#newRunBtn").disabled = false;
    $("#newRunBtn").textContent = "+ new run";
  }
}

// ---------------- init
async function init() {
  bindUpload();
  bindCanvasInteractions();
  bindDrawer();
  await refreshRuns();
  // Default landing: the new-run screen so the model + mode choice is visible.
  // The user can pick an existing run from the sidebar at any time.
  showNewRunForm();
  setInterval(refreshRuns, 4000);
}
init();
