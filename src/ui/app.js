const COMMON_BLOCKING_KEYS = [
  "first_name",
  "last_name",
  "age",
  "zip_code",
  "house_num",
  "street_name",
  "sex",
];

const state = {
  models: [],
  dataset: null,
  selectedModel: null,
  activeTask: "find",
  datasetMode: "path",
  defaults: null,
};

function byId(id) {
  return document.getElementById(id);
}

function showBanner(message, kind = "info") {
  const banner = byId("message-banner");
  banner.textContent = message;
  banner.classList.remove("hidden", "info", "error");
  banner.classList.add(kind);
}

function hideBanner() {
  const banner = byId("message-banner");
  banner.classList.add("hidden");
  banner.textContent = "";
  banner.classList.remove("info", "error");
}

function formatErrorMessage(message) {
  if (message.includes("Candidate pair limit exceeded")) {
    return `${message} Recommended full-database preset: blocking keys first_name, last_name, zip_code with blocking mode set to all.`;
  }
  return message;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "string" ? payload : payload.detail || JSON.stringify(payload);
    throw new Error(detail);
  }
  return payload;
}

function chooseDefaultModel(models, preferredName) {
  if (!models.length) {
    return null;
  }
  return models.find((model) => model.name === preferredName)?.name || models[0].name;
}

function getSelectedModel() {
  return state.models.find((model) => model.name === state.selectedModel) || null;
}

function renderModelOptions() {
  const select = byId("model-select");
  select.innerHTML = "";
  for (const model of state.models) {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = `${model.name} (${model.kind})`;
    select.appendChild(option);
  }
  if (state.selectedModel) {
    select.value = state.selectedModel;
  }
}

function renderModelSummary() {
  const model = getSelectedModel();
  const summary = byId("model-summary");
  if (!model) {
    summary.innerHTML = "No models available.";
    return;
  }

  summary.innerHTML = `
    <table class="summary-table">
      <tr><td>Model</td><td class="mono">${model.name}</td></tr>
      <tr><td>Kind</td><td>${model.kind}</td></tr>
      <tr><td>Default threshold</td><td>${Number(model.threshold).toFixed(3)}</td></tr>
      <tr><td>Training-time blocking</td><td>${(model.blocking_keys || []).join(", ") || "none"}</td></tr>
      <tr><td>Training-time mode</td><td>${model.blocking_mode || "any"}</td></tr>
      <tr><td>Attributes</td><td>${(model.attributes || []).join(", ")}</td></tr>
    </table>
  `;

  syncBlockingControlsFromModel();
}

function renderDatasetSummary() {
  const target = byId("dataset-summary");
  byId("hero-dataset-status").textContent = state.dataset ? `${state.dataset.record_count} records` : "Not loaded";
  if (!state.dataset) {
    target.innerHTML = "No dataset is loaded yet.";
    return;
  }

  target.innerHTML = `
    <table class="summary-table">
      <tr><td>Source</td><td class="mono">${state.dataset.source_name}</td></tr>
      <tr><td>Records</td><td>${state.dataset.record_count}</td></tr>
      <tr><td>Columns</td><td>${state.dataset.columns.join(", ")}</td></tr>
      <tr><td>Saved path</td><td class="mono">${state.dataset.saved_path}</td></tr>
    </table>
  `;
}

function renderBlockingCheckboxes(containerId, selectedKeys) {
  const container = byId(containerId);
  container.innerHTML = "";
  for (const key of COMMON_BLOCKING_KEYS) {
    const label = document.createElement("label");
    label.className = "checkbox-pill";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = key;
    checkbox.checked = selectedKeys.includes(key);

    const span = document.createElement("span");
    span.textContent = key;

    label.appendChild(checkbox);
    label.appendChild(span);
    container.appendChild(label);
  }
}

function getCheckedValues(containerId) {
  return Array.from(byId(containerId).querySelectorAll("input:checked")).map((input) => input.value);
}

function syncBlockingControlsFromModel() {
  const findDefaults = state.defaults?.duplicate_search || {
    blocking_keys: ["first_name", "last_name", "zip_code"],
    blocking_mode: "all",
    top_k: 25,
    max_candidate_pairs: 250000,
  };
  const checkDefaults = state.defaults?.entry_check || {
    blocking_keys: ["first_name", "last_name", "zip_code"],
    blocking_mode: "any",
    top_k: 10,
    max_candidates: 50000,
  };

  renderBlockingCheckboxes("find-blocking-checkboxes", findDefaults.blocking_keys || []);
  renderBlockingCheckboxes("check-blocking-checkboxes", checkDefaults.blocking_keys || []);
  byId("find-blocking-mode").value = findDefaults.blocking_mode || "all";
  byId("check-blocking-mode").value = checkDefaults.blocking_mode || "any";
  byId("find-top-k-input").value = String(findDefaults.top_k || 25);
  byId("check-top-k-input").value = String(checkDefaults.top_k || 10);
  byId("find-max-candidates-input").value = String(findDefaults.max_candidate_pairs || 250000);
  byId("check-max-candidates-input").value = String(checkDefaults.max_candidates || 50000);
}

function setTask(taskName) {
  state.activeTask = taskName;
  for (const button of document.querySelectorAll("#task-toggle .task-card")) {
    button.classList.toggle("active", button.dataset.task === taskName);
  }
  byId("find-task-panel").classList.toggle("hidden", taskName !== "find");
  byId("check-task-panel").classList.toggle("hidden", taskName !== "check");
}

function setDatasetMode(mode) {
  state.datasetMode = mode;
  for (const button of document.querySelectorAll("#dataset-mode-toggle .segmented-option")) {
    button.classList.toggle("active", button.dataset.mode === mode);
  }
  byId("dataset-path-panel").classList.toggle("active", mode === "path");
  byId("dataset-upload-panel").classList.toggle("active", mode === "upload");
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function recordPreview(record) {
  const name = [record.first_name, record.midl_name, record.last_name].filter(Boolean).join(" ").trim();
  const address = [record.house_num, record.street_name].filter(Boolean).join(" ").trim();
  const suffix = [record.zip_code, record.age ? `age ${record.age}` : "", record.sex].filter(Boolean).join(", ");
  return [name ? `<strong>${name}</strong>` : "", address, suffix].filter(Boolean).join(", ");
}

function renderSignalBadges(signals = []) {
  if (!signals.length) {
    return "";
  }
  return `
    <div class="badge-row">
      ${signals.map((signal) => `<span class="badge">${signal}</span>`).join("")}
    </div>
  `;
}

function renderFieldDiffs(pair) {
  const differingFields = pair.differing_fields || [];
  const matchingFields = pair.matching_fields || [];
  if (!differingFields.length && !matchingFields.length) {
    return "";
  }

  return `
    <div class="field-diff-block">
      ${differingFields.length ? `
        <div class="field-diff-group">
          <div class="field-diff-title">Differing fields</div>
          <div class="field-diff-list">
            ${differingFields.map((field) => `
              <div class="field-diff-row">
                <span class="field-diff-label">${field.label}</span>
                <span class="field-diff-values">${field.left || "<blank>"} -> ${field.right || "<blank>"}</span>
              </div>
            `).join("")}
          </div>
        </div>
      ` : ""}
      ${matchingFields.length ? `
        <div class="field-diff-group">
          <div class="field-diff-title">Also matches on</div>
          <div class="badge-row">
            ${matchingFields.slice(0, 6).map((field) => `
              <span class="badge subtle">${field.label}: ${field.value}</span>
            `).join("")}
          </div>
        </div>
      ` : ""}
    </div>
  `;
}

function renderPairSection(title, subtitle, rows = [], options = {}) {
  const reviewNote = options.reviewRecommended
    ? `
      <div class="review-note">
        <strong>Review recommended.</strong>
        These are disagreement cases between models and should be checked manually before acting on them.
      </div>
    `
    : "";
  return `
    <div class="result-block">
      <h3>${title}</h3>
      <p class="section-text">${subtitle}</p>
      ${reviewNote}
      <table class="results-table">
        <thead>
          <tr>
            <th>Pair</th>
            <th>Score</th>
            <th>Record comparison</th>
          </tr>
        </thead>
        <tbody>
          ${rows.length ? rows.map((pair) => `
            <tr>
              <td class="mono">${pair.id1} / ${pair.id2}</td>
              <td>
                ${formatScore(pair.score)}
                ${pair.comparison_score !== undefined ? `<div class="hint-text">${pair.comparison_model_name}: ${formatScore(pair.comparison_score)}</div>` : ""}
              </td>
              <td>
                ${renderSignalBadges(pair.signals)}
                <div>${recordPreview(pair.record1)}</div>
                <div>${recordPreview(pair.record2)}</div>
                ${renderFieldDiffs(pair)}
              </td>
            </tr>
          `).join("") : `
            <tr><td colspan="3">No pairs to show for this section.</td></tr>
          `}
        </tbody>
      </table>
    </div>
  `;
}

function renderScoreSummary(scoreSummary, clusterSummary, payload) {
  const duplicateScores = scoreSummary?.duplicates || {};
  const rejectedScores = scoreSummary?.rejected || {};
  return `
    <div class="result-block">
      <div class="badge-row">
        <span class="badge">Model: ${payload.model_name}</span>
        <span class="badge">Threshold: ${formatScore(payload.threshold)}</span>
        <span class="badge">Candidates: ${payload.candidate_pair_count}</span>
        <span class="badge">Duplicates: ${payload.predicted_duplicate_pair_count}</span>
        <span class="badge">Clusters: ${clusterSummary.total_cluster_count || 0}</span>
      </div>
      <div class="stats-grid">
        <div class="stat-card">
          <span class="mini-label">Acceptance rate</span>
          <strong>${formatScore((scoreSummary?.acceptance_rate || 0) * 100)}%</strong>
        </div>
        <div class="stat-card">
          <span class="mini-label">Duplicate score range</span>
          <strong>${formatScore(duplicateScores.min)} to ${formatScore(duplicateScores.max)}</strong>
          <span class="hint-text">Median ${formatScore(duplicateScores.median)}</span>
        </div>
        <div class="stat-card">
          <span class="mini-label">Rejected score range</span>
          <strong>${formatScore(rejectedScores.min)} to ${formatScore(rejectedScores.max)}</strong>
          <span class="hint-text">Median ${formatScore(rejectedScores.median)}</span>
        </div>
        <div class="stat-card">
          <span class="mini-label">Cluster sizes</span>
          <strong>${clusterSummary.two_record_cluster_count || 0} pairs / ${clusterSummary.three_plus_record_cluster_count || 0} larger clusters</strong>
          <span class="hint-text">Largest cluster ${clusterSummary.largest_cluster_size || 0} records</span>
        </div>
      </div>
    </div>
  `;
}

function renderClusterSummary(clusterSummary) {
  const clusters = clusterSummary.top_clusters || [];
  return `
    <div class="result-block">
      <h3>Duplicate cluster summary</h3>
      <p class="section-text">
        ${clusterSummary.total_cluster_count || 0} predicted clusters, average size
        ${formatScore(clusterSummary.average_cluster_size || 0)}, largest cluster
        ${clusterSummary.largest_cluster_size || 0} records. Showing the top ${clusters.length} cluster profiles.
      </p>
      <div class="cluster-list">
        ${clusters.length ? clusters.map((cluster) => `
          <div class="cluster-chip">
            <strong>${cluster.cluster_id}</strong> (${cluster.size} records)
            <div class="badge-row">
              ${(cluster.shared_characteristics || []).map((item) => `
                <span class="badge">${item.attribute}: ${item.value} (${item.support_count}/${cluster.size})</span>
              `).join("")}
            </div>
            <div class="hint-text">
              ${cluster.member_preview && cluster.member_preview.length
                ? `Example records: ${cluster.member_preview.join(" | ")}`
                : "No record preview available."}
            </div>
          </div>
        `).join("") : "No clusters formed because no duplicate pairs were predicted."}
      </div>
    </div>
  `;
}

function renderPatternSummary(patterns = []) {
  return `
    <div class="result-block">
      <h3>Duplicate pattern summary</h3>
      <p class="section-text">These counts summarize what kinds of predicted duplicates are appearing in this search run.</p>
      <div class="pattern-list">
        ${patterns.length ? patterns.map((pattern) => `
          <div class="pattern-row">
            <span>${pattern.label}</span>
            <strong>${pattern.count}</strong>
            <span class="hint-text">${formatScore(pattern.share * 100)}%</span>
          </div>
        `).join("") : "No duplicate patterns to summarize."}
      </div>
    </div>
  `;
}

function renderRepresentativeCases(cases = {}) {
  const caseRows = [
    {
      label: "Highest-confidence duplicate",
      value: cases.highest_confidence_duplicate,
    },
    {
      label: "Most borderline duplicate",
      value: cases.most_borderline_duplicate,
    },
    {
      label: "Closest rejected pair",
      value: cases.closest_rejected_pair,
    },
  ].filter((item) => item.value);

  const largestCluster = cases.largest_cluster;
  return `
    <div class="result-block">
      <h3>Representative cases</h3>
      <div class="case-grid">
        ${caseRows.map((item) => `
          <div class="case-card">
            <strong>${item.label}</strong>
            <div class="badge-row">
              <span class="badge mono">${item.value.id1} / ${item.value.id2}</span>
              <span class="badge">Score: ${formatScore(item.value.score)}</span>
            </div>
            ${renderSignalBadges(item.value.signals)}
            <div>${recordPreview(item.value.record1)}</div>
            <div>${recordPreview(item.value.record2)}</div>
          </div>
        `).join("")}
        ${largestCluster ? `
          <div class="case-card">
            <strong>Largest predicted cluster</strong>
            <div class="badge-row">
              <span class="badge">${largestCluster.cluster_id}</span>
              <span class="badge">${largestCluster.size} records</span>
            </div>
            <div class="badge-row">
              ${(largestCluster.shared_characteristics || []).map((item) => `
                <span class="badge">${item.attribute}: ${item.value}</span>
              `).join("")}
            </div>
          </div>
        ` : ""}
      </div>
    </div>
  `;
}

function renderDisagreementSection(disagreement) {
  if (!disagreement || !disagreement.available) {
    return `
      <div class="result-block">
        <h3>Model disagreement analysis</h3>
        <p class="section-text">${disagreement?.reason || "No comparison model is available for disagreement analysis."}</p>
      </div>
    `;
  }

  return `
    <div class="result-block">
      <h3>Model disagreement analysis</h3>
      <p class="section-text">
        Compared against <span class="mono">${disagreement.comparison_model_name}</span>.
        Agreement rate across candidate pairs: ${formatScore(disagreement.agreement_rate * 100)}%.
      </p>
      <div class="stats-grid">
        <div class="stat-card">
          <span class="mini-label">Selected model only</span>
          <strong>${disagreement.selected_only_duplicate_count}</strong>
        </div>
        <div class="stat-card">
          <span class="mini-label">Comparison model only</span>
          <strong>${disagreement.comparison_only_duplicate_count}</strong>
        </div>
      </div>
    </div>
    ${renderPairSection(
      "Review recommended: selected model says duplicate",
      "These pairs were flagged as duplicates by the selected model but not by the comparison model.",
      disagreement.selected_only_duplicates || [],
      { reviewRecommended: true }
    )}
    ${renderPairSection(
      "Review recommended: comparison model says duplicate",
      "These pairs were flagged as duplicates by the comparison model but not by the selected model.",
      disagreement.comparison_only_duplicates || [],
      { reviewRecommended: true }
    )}
  `;
}

function renderFindResults(payload) {
  const root = byId("results-root");
  const clusterSummary = payload.duplicate_cluster_summary || {
    total_cluster_count: 0,
    average_cluster_size: 0,
    largest_cluster_size: 0,
    two_record_cluster_count: 0,
    three_plus_record_cluster_count: 0,
    top_clusters: [],
  };

  root.innerHTML = `
    <div class="results-stack">
      ${renderScoreSummary(payload.score_summary, clusterSummary, payload)}
      ${renderPairSection(
        "High-confidence duplicates",
        "The strongest predicted duplicates in this search run.",
        payload.high_confidence_duplicates || []
      )}
      ${renderPairSection(
        "Borderline duplicates",
        "The lowest-scoring pairs that still passed the duplicate threshold.",
        payload.borderline_duplicates || []
      )}
      ${renderPairSection(
        "Near-miss non-duplicates",
        "Pairs that came closest to the threshold but were still rejected.",
        payload.near_miss_non_duplicates || []
      )}
      ${renderRepresentativeCases(payload.representative_cases || {})}
      ${renderPatternSummary(payload.duplicate_pattern_summary || [])}
      ${renderClusterSummary(clusterSummary)}
      ${renderDisagreementSection(payload.disagreement_analysis)}
    </div>
  `;
}

function renderCheckResults(payload) {
  const matches = payload.matches || [];
  const root = byId("results-root");

  root.innerHTML = `
    <div class="results-stack">
      <div class="result-block">
        <div class="badge-row">
          <span class="badge">Model: ${payload.model_name}</span>
          <span class="badge">Threshold: ${Number(payload.threshold).toFixed(3)}</span>
          <span class="badge ${payload.duplicate_exists ? "warning" : ""}">
            Duplicate exists: ${payload.duplicate_exists ? "Yes" : "No"}
          </span>
          <span class="badge">Candidates: ${payload.candidate_count}</span>
        </div>
        <p><strong>Submitted entry:</strong> ${Object.entries(payload.entry)
          .filter(([, value]) => value)
          .map(([key, value]) => `${key}=${value}`)
          .join(", ")}</p>
      </div>
      <div class="result-block">
        <h3>Top matches</h3>
        <div class="match-list">
          ${matches.length ? matches.map((match) => `
            <div class="match-card">
              <div class="badge-row">
                <span class="badge mono">${match.existing_id}</span>
                <span class="badge">Score: ${Number(match.score).toFixed(3)}</span>
                <span class="badge ${match.is_duplicate ? "warning" : ""}">
                  ${match.is_duplicate ? "Predicted duplicate" : "Below threshold"}
                </span>
              </div>
              <div>
                <strong>${match.existing_record.first_name || ""} ${match.existing_record.last_name || ""}</strong>,
                ${match.existing_record.house_num || ""} ${match.existing_record.street_name || ""},
                ${match.existing_record.zip_code || ""}
              </div>
            </div>
          `).join("") : "No candidate matches were produced for the current entry and blocking settings."}
        </div>
      </div>
    </div>
  `;
}

async function refreshAppState(showInfo = false) {
  const payload = await fetchJson("/app/state");
  state.models = payload.models || [];
  state.dataset = payload.dataset;
  state.defaults = payload.defaults || null;
  state.selectedModel = state.selectedModel || chooseDefaultModel(state.models, payload.defaults?.preferred_model);

  byId("hero-model-count").textContent = String(state.models.length);
  renderModelOptions();
  renderModelSummary();
  renderDatasetSummary();

  if (showInfo) {
    showBanner(state.dataset ? "Dataset loaded successfully." : "Application state refreshed.");
  }
}

async function loadDatasetFromPath() {
  const path = byId("dataset-path-input").value.trim();
  if (!path) {
    showBanner("Enter a dataset path before loading.", "error");
    return;
  }
  const payload = await fetchJson("/dataset/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  state.dataset = payload.dataset;
  renderDatasetSummary();
  showBanner(`Dataset loaded from ${payload.dataset.source_name}.`);
}

async function uploadDataset() {
  const fileInput = byId("dataset-file-input");
  if (!fileInput.files || !fileInput.files.length) {
    showBanner("Choose a TSV or CSV file before uploading.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const payload = await fetchJson("/dataset/upload", {
    method: "POST",
    body: formData,
  });
  state.dataset = payload.dataset;
  renderDatasetSummary();
  showBanner(`Dataset uploaded: ${payload.dataset.source_name}.`);
}

async function clearDataset() {
  await fetchJson("/dataset", { method: "DELETE" });
  state.dataset = null;
  renderDatasetSummary();
  byId("results-root").innerHTML = "Dataset cleared. Load another dataset to continue.";
  showBanner("Loaded dataset cleared.");
}

async function runFindDuplicates() {
  if (!state.dataset) {
    showBanner("Load a dataset before running duplicate search.", "error");
    return;
  }

  const thresholdRaw = byId("find-threshold-input").value.trim();
  const selectedBlockingKeys = getCheckedValues("find-blocking-checkboxes");
  if (!selectedBlockingKeys.length) {
    showBanner("Choose at least one blocking key for a full database duplicate search.", "error");
    return;
  }
  const payload = await fetchJson("/duplicates/find", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: state.selectedModel,
      blocking_keys: selectedBlockingKeys,
      blocking_mode: byId("find-blocking-mode").value,
      threshold: thresholdRaw ? Number(thresholdRaw) : null,
      top_k: Number(byId("find-top-k-input").value) || 25,
      max_candidate_pairs: Number(byId("find-max-candidates-input").value) || 250000,
    }),
  });
  renderFindResults(payload);
  showBanner("Duplicate search completed.");
}

async function runCheckEntry() {
  if (!state.dataset) {
    showBanner("Load a dataset before checking a new entry.", "error");
    return;
  }

  const thresholdRaw = byId("check-threshold-input").value.trim();
  const entry = {
    first_name: byId("entry-first-name").value,
    last_name: byId("entry-last-name").value,
    house_num: byId("entry-house-num").value,
    street_name: byId("entry-street-name").value,
    zip_code: byId("entry-zip-code").value,
    age: byId("entry-age").value,
    sex: byId("entry-sex").value,
    race_desc: byId("entry-race").value,
    ethnic_desc: byId("entry-ethnic").value,
  };

  const payload = await fetchJson("/duplicates/check-entry", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: state.selectedModel,
      entry,
      blocking_keys: getCheckedValues("check-blocking-checkboxes"),
      blocking_mode: byId("check-blocking-mode").value,
      threshold: thresholdRaw ? Number(thresholdRaw) : null,
      top_k: Number(byId("check-top-k-input").value) || 10,
      max_candidates: Number(byId("check-max-candidates-input").value) || 50000,
    }),
  });
  renderCheckResults(payload);
  showBanner(payload.duplicate_exists ? "Potential duplicate found." : "No duplicate found above threshold.");
}

function wireEvents() {
  byId("model-select").addEventListener("change", (event) => {
    state.selectedModel = event.target.value;
    renderModelSummary();
    hideBanner();
  });

  for (const button of document.querySelectorAll("#dataset-mode-toggle .segmented-option")) {
    button.addEventListener("click", () => setDatasetMode(button.dataset.mode));
  }

  for (const button of document.querySelectorAll("#task-toggle .task-card")) {
    button.addEventListener("click", () => setTask(button.dataset.task));
  }

  byId("load-path-button").addEventListener("click", async () => {
    hideBanner();
    try {
      await loadDatasetFromPath();
      await refreshAppState();
    } catch (error) {
      showBanner(formatErrorMessage(error.message), "error");
    }
  });

  byId("upload-dataset-button").addEventListener("click", async () => {
    hideBanner();
    try {
      await uploadDataset();
      await refreshAppState();
    } catch (error) {
      showBanner(formatErrorMessage(error.message), "error");
    }
  });

  byId("clear-dataset-button").addEventListener("click", async () => {
    hideBanner();
    try {
      await clearDataset();
    } catch (error) {
      showBanner(formatErrorMessage(error.message), "error");
    }
  });

  byId("run-find-button").addEventListener("click", async () => {
    hideBanner();
    try {
      await runFindDuplicates();
    } catch (error) {
      showBanner(formatErrorMessage(error.message), "error");
    }
  });

  byId("run-check-button").addEventListener("click", async () => {
    hideBanner();
    try {
      await runCheckEntry();
    } catch (error) {
      showBanner(formatErrorMessage(error.message), "error");
    }
  });
}

async function init() {
  wireEvents();
  setTask("find");
  setDatasetMode("path");
  try {
    await refreshAppState();
  } catch (error) {
    showBanner(formatErrorMessage(error.message), "error");
  }
}

window.addEventListener("DOMContentLoaded", init);
