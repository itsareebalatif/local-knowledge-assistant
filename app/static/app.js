"use strict";

let TOKEN = null;
let REFRESH_TOKEN = null;
let ACCOUNT_EMAIL = null;
let AUTH_MODE = "login"; // or "register"

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function authHeaders() {
  return { Authorization: `Bearer ${TOKEN}` };
}

function persistTokens(data) {
  TOKEN = data.access_token;
  REFRESH_TOKEN = data.refresh_token;
  localStorage.setItem("pke_token", TOKEN);
  localStorage.setItem("pke_refresh_token", REFRESH_TOKEN);
}

function clearTokens() {
  localStorage.removeItem("pke_token");
  localStorage.removeItem("pke_refresh_token");
  localStorage.removeItem("pke_email");
  TOKEN = null;
  REFRESH_TOKEN = null;
}

async function tryRefreshToken() {
  if (!REFRESH_TOKEN) return false;
  try {
    const res = await fetch("/api/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: REFRESH_TOKEN }),
    });
    if (!res.ok) return false;
    persistTokens(await res.json());
    return true;
  } catch {
    return false;
  }
}

async function authFetch(url, options = {}, _alreadyRetried = false) {
  const headers = { ...(options.headers || {}), ...authHeaders() };
  const response = await fetch(url, { ...options, headers });
  if (response.status !== 401) return response;

  // Only one retry: if that also fails, the refresh token is gone too.
  if (!_alreadyRetried && (await tryRefreshToken())) {
    return authFetch(url, options, true);
  }

  clearTokens();
  showAuthPanel();
  setAuthMode("login");
  document.getElementById("auth-status").textContent = "Your session expired — please log in again.";
  return response;
}

function showApp(email) {
  ACCOUNT_EMAIL = email;
  document.getElementById("auth-panel").hidden = true;
  document.getElementById("app-main").hidden = false;
  document.getElementById("account-bar").hidden = false;
  document.getElementById("account-email").textContent = email;
}

function showAuthPanel() {
  document.getElementById("auth-panel").hidden = false;
  document.getElementById("app-main").hidden = true;
  document.getElementById("account-bar").hidden = true;
}

function setAuthMode(mode) {
  AUTH_MODE = mode;
  const isRegister = mode === "register";
  document.getElementById("auth-title").textContent = isRegister ? "Register" : "Log in";
  document.getElementById("auth-submit-btn").textContent = isRegister ? "Register" : "Log in";
  document.getElementById("auth-full-name").hidden = !isRegister;
  document.getElementById("auth-toggle-link").textContent = isRegister
    ? "Already have an account? Log in instead."
    : "Need an account? Register instead.";
  document.getElementById("auth-status").textContent = "";
}

async function initAuth() {
  TOKEN = localStorage.getItem("pke_token");
  REFRESH_TOKEN = localStorage.getItem("pke_refresh_token");
  const storedEmail = localStorage.getItem("pke_email");

  if (TOKEN) {
    const check = await authFetch("/api/graph");
    if (check.ok) {
      showApp(storedEmail || "");
      return;
    }
    clearTokens();
  }

  showAuthPanel();
  setAuthMode("login");
}

async function onAuthSubmit(event) {
  event.preventDefault();
  const email = document.getElementById("auth-email").value.trim();
  const password = document.getElementById("auth-password").value;
  const fullName = document.getElementById("auth-full-name").value.trim();
  const statusEl = document.getElementById("auth-status");

  const endpoint = AUTH_MODE === "register" ? "/api/auth/register" : "/api/auth/login";
  const body = AUTH_MODE === "register" ? { email, password, full_name: fullName } : { email, password };

  statusEl.textContent = AUTH_MODE === "register" ? "Registering…" : "Logging in…";
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.detail || "Something went wrong.";
      return;
    }
    persistTokens(data);
    localStorage.setItem("pke_email", data.email);
    showApp(data.email);
    initUpload();
    initQuery();
    initGraph();
    loadDocuments();
  } catch (err) {
    statusEl.textContent = `Error: ${err}`;
  }
}

async function onLogout() {
  try {
    await authFetch("/api/auth/logout", { method: "POST" });
  } catch {
    // Local logout still proceeds even if this request fails.
  }
  clearTokens();
  showAuthPanel();
  setAuthMode("login");
}

function initAuthForm() {
  document.getElementById("auth-form").addEventListener("submit", onAuthSubmit);
  document.getElementById("auth-toggle-link").addEventListener("click", (e) => {
    e.preventDefault();
    setAuthMode(AUTH_MODE === "login" ? "register" : "login");
  });
  document.getElementById("logout-btn").addEventListener("click", onLogout);
}

function initUpload() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  ["dragover", "dragenter"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => handleFiles(e.target.files));
}

async function handleFiles(fileList) {
  // Sequential, not Promise.all: ingests queue behind the same backend anyway.
  for (const file of fileList) {
    await uploadFile(file);
  }
}

async function uploadFile(file) {
  const list = document.getElementById("upload-list");
  const row = document.createElement("li");
  row.innerHTML = `<span>${escapeHtml(file.name)}</span><span class="status">uploading…</span>`;
  list.prepend(row);
  const statusEl = row.querySelector(".status");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await authFetch("/api/ingest", { method: "POST", body: formData });
    // Ingestion is queued, not done synchronously — large files parse/embed
    // in the background now so one slow upload can't tie up the request
    // long enough to trip a hosting platform's proxy timeout. A non-JSON
    // response (e.g. a proxy's own timeout/error page) is handled instead
    // of crashing on res.json().
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      statusEl.textContent = `error: upload failed (${res.status})`;
      statusEl.className = "status status-error";
      console.error("ingest request failed", res.status, text);
      return;
    }
    const accepted = await res.json();
    statusEl.textContent = "processing…";
    statusEl.className = "status status-processing";
    await pollIngestJob(accepted.job_id, statusEl);
  } catch (err) {
    statusEl.textContent = `error: ${err}`;
    statusEl.className = "status status-error";
  }
}

async function pollIngestJob(jobId, statusEl) {
  const maxAttempts = 150; // ~5 minutes at 2s each, generous for a large book-length file
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    await sleep(2000);
    let res;
    try {
      res = await authFetch(`/api/ingest/${jobId}`);
    } catch (err) {
      statusEl.textContent = `error checking status: ${err}`;
      statusEl.className = "status status-error";
      return;
    }
    if (!res.ok) {
      statusEl.textContent = `error: status check failed (${res.status})`;
      statusEl.className = "status status-error";
      return;
    }
    const data = await res.json();
    if (data.status === "processing") continue;

    statusEl.textContent = summarizeIngest(data);
    statusEl.className = `status status-${data.status}`;
    if (data.status === "ingested") {
      loadGraph();
      loadDocuments();
    }
    return;
  }
  statusEl.textContent = "still processing — check back later";
  statusEl.className = "status status-processing";
}

async function loadDocuments() {
  const summaryEl = document.getElementById("documents-summary");
  const listEl = document.getElementById("documents-list");
  try {
    const res = await authFetch("/api/documents");
    if (!res.ok) return;
    const data = await res.json();

    summaryEl.textContent = `${data.total_documents} document(s), ${data.total_chunks} chunk(s) total`;
    listEl.innerHTML = "";
    for (const doc of data.documents) {
      const row = document.createElement("li");
      const uploaded = new Date(doc.created_at).toLocaleString();
      row.innerHTML = `<span>${escapeHtml(doc.file_name)}</span><span class="status">${doc.chunk_count} chunks · ${uploaded}</span>`;
      const deleteBtn = document.createElement("button");
      deleteBtn.textContent = "Delete";
      deleteBtn.className = "secondary delete-doc-btn";
      deleteBtn.addEventListener("click", () => onDeleteDocument(doc.doc_id, doc.file_name));
      row.appendChild(deleteBtn);
      listEl.appendChild(row);
    }
    if (data.documents.length === 0) {
      listEl.innerHTML = `<li class="muted">No documents uploaded yet.</li>`;
    }
  } catch {
    summaryEl.textContent = "Could not load documents.";
  }
}

async function onDeleteDocument(docId, fileName) {
  if (!confirm(`Delete "${fileName}" and all its chunks/embeddings? This can't be undone.`)) return;
  try {
    const res = await authFetch(`/api/documents/${docId}`, { method: "DELETE" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      alert(body.detail || `Delete failed (${res.status}).`);
      return;
    }
    await loadDocuments();
  } catch (err) {
    alert(`Could not reach the API: ${err}`);
  }
}

function summarizeIngest(data) {
  switch (data.status) {
    case "ingested":
      return `${data.total_chunks} chunks (${data.new_chunks} new, ${data.reused_chunks} reused)`;
    case "duplicate_file":
      return "already ingested";
    case "unsupported":
      return "unsupported file type";
    default:
      return data.error || data.status;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function refusalMessage(reason) {
  if (reason === "no_candidates") {
    return "I couldn't find anything in your documents related to that. Try rephrasing, or upload a document that covers this topic.";
  }
  if (reason === "low_keyword_coverage") {
    return "I don't have enough information in your documents to answer that confidently. Try rephrasing using wording closer to what's actually in the document, or ask about something more specific.";
  }
  return "I don't have enough grounded information in your documents to answer that.";
}

async function streamQuery(question, handlers) {
  const res = await authFetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: question }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`request failed: ${res.status}`);
  }

  // EventSource can't send a POST body, so SSE frames are parsed by hand here.
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop(); // may be an incomplete frame

    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      const event = JSON.parse(line.slice(5).trim());
      const handler = handlers[event.type];
      if (!handler) continue;
      if (event.type === "token") {
        // Purely cosmetic pacing; tokens actually arrive much faster.
        await sleep(18);
      }
      handler(event);
    }
  }
}

function groundingSummary(grounding) {
  if (!grounding) return "";
  if (grounding.overall_coverage === null) return "No factual claims to verify.";
  const pct = Math.round(grounding.overall_coverage * 100);
  const warn = grounding.unsupported_sentences.length
    ? ` — ⚠ ${grounding.unsupported_sentences.length} sentence(s) not clearly grounded in sources`
    : "";
  return `Grounding coverage: ${pct}%${warn}`;
}

function renderCitations(citations) {
  const list = document.getElementById("citations");
  list.innerHTML = "";
  citations.forEach((c) => {
    const a = document.createElement("a");
    a.href = "#";
    a.className = "citation-link";
    a.textContent = `[${c.marker}] ${c.file_name}`;
    a.addEventListener("click", (e) => {
      e.preventDefault();
      showCitationDetail(c);
    });
    list.appendChild(a);
  });
}

async function showCitationDetail(citation) {
  const panel = document.getElementById("citation-detail");
  panel.hidden = false;
  panel.innerHTML = "<p class=\"muted\">Loading…</p>";
  const res = await authFetch(`/api/chunks/${citation.chunk_id}`);
  if (!res.ok) {
    panel.innerHTML = "<p class=\"muted\">Could not load this chunk.</p>";
    return;
  }
  const chunk = await res.json();
  panel.innerHTML =
    `<h4>[${citation.marker}] ${escapeHtml(chunk.file_name)}</h4>` + `<p>${escapeHtml(chunk.content)}</p>`;
}

async function onAsk() {
  const input = document.getElementById("question");
  const question = input.value.trim();
  if (!question) return;

  const answerBox = document.getElementById("answer");
  const statusBox = document.getElementById("query-status");
  answerBox.textContent = "";
  statusBox.textContent = "Thinking…";
  document.getElementById("citations").innerHTML = "";
  document.getElementById("citation-detail").hidden = true;

  try {
    await streamQuery(question, {
      token: (event) => {
        answerBox.textContent += event.text;
        statusBox.textContent = "";
      },
      done: (event) => {
        renderCitations(event.citations);
        statusBox.textContent = groundingSummary(event.grounding);
      },
      refused: (event) => {
        answerBox.textContent = refusalMessage(event.reason);
        statusBox.textContent = "";
      },
      error: (event) => {
        statusBox.textContent = `Error: ${event.message}`;
      },
    });
  } catch (err) {
    statusBox.textContent = `Error: ${err}`;
  }
}

function initQuery() {
  document.getElementById("ask-btn").addEventListener("click", onAsk);
  document.getElementById("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter") onAsk();
  });
}

const TYPE_COLOR = {
  Person: "#e07a5f",
  Organization: "#3d5a80",
  Location: "#81b29a",
  Concept: "#f2cc8f",
};

function adjacencyDataToGraph(data) {
  // Undirected edges appear twice in adjacency; deduped below by node-index pair.
  const nodes = data.nodes.map((n) => ({ id: n.id, name: n.name, type: n.type }));
  const indexById = new Map(nodes.map((n, i) => [n.id, i]));
  const links = [];
  const seenPairs = new Set();

  data.adjacency.forEach((neighbors, i) => {
    neighbors.forEach((edge) => {
      const j = indexById.get(edge.id);
      const pairKey = i < j ? `${i}:${j}` : `${j}:${i}`;
      if (seenPairs.has(pairKey)) return;
      seenPairs.add(pairKey);
      links.push({ source: nodes[i].id, target: nodes[j].id, weight: edge.weight || 1 });
    });
  });

  return { nodes, links };
}

function dragBehavior(simulation) {
  return d3
    .drag()
    .on("start", (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (event, d) => {
      d.fx = event.x;
      d.fy = event.y;
    })
    .on("end", (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    });
}

function renderGraph(nodes, links) {
  const svg = d3.select("#graph-svg");
  svg.selectAll("*").remove();
  const width = svg.node().clientWidth || 600;
  const height = svg.node().clientHeight || 420;

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance(80)
    )
    .force("charge", d3.forceManyBody().strength(-220))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(22));

  const g = svg.append("g");
  svg.call(d3.zoom().on("zoom", (event) => g.attr("transform", event.transform)));

  const link = g
    .append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke-width", (d) => Math.max(1, Math.sqrt(d.weight)));

  const node = g
    .append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", 9)
    .attr("fill", (d) => TYPE_COLOR[d.type] || "#888")
    .call(dragBehavior(simulation))
    .on("click", (_event, d) => showGraphNeighbors(d));

  node.append("title").text((d) => `${d.type}: ${d.name}`);

  const label = g
    .append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .text((d) => d.name)
    .attr("font-size", 10)
    .attr("dx", 12)
    .attr("dy", 4);

  simulation.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    label.attr("x", (d) => d.x).attr("y", (d) => d.y);
  });
}

async function showGraphNeighbors(d) {
  const panel = document.getElementById("graph-detail");
  panel.innerHTML = "<p class=\"muted\">Loading…</p>";
  const res = await authFetch(
    `/api/graph/neighbors?name=${encodeURIComponent(d.name)}&type=${encodeURIComponent(d.type)}`
  );
  const data = await res.json();
  const rows = data.neighbors
    .map((n) => `<li>${escapeHtml(n.name)} <span class="badge">${escapeHtml(n.type)}</span> — weight ${n.weight}</li>`)
    .join("");
  panel.innerHTML =
    `<h4>${escapeHtml(d.name)} <span class="badge">${escapeHtml(d.type)}</span></h4>` +
    (rows ? `<ul>${rows}</ul>` : "<p class=\"muted\">No connections yet.</p>");
}

async function loadGraph() {
  const res = await authFetch("/api/graph");
  const data = await res.json();
  const { nodes, links } = adjacencyDataToGraph(data);
  renderGraph(nodes, links);
}

function initGraph() {
  document.getElementById("refresh-graph-btn").addEventListener("click", loadGraph);
  loadGraph();
}

window.addEventListener("DOMContentLoaded", async () => {
  initAuthForm();
  await initAuth();
  if (TOKEN) {
    initUpload();
    initQuery();
    initGraph();
    loadDocuments();
  }
});
