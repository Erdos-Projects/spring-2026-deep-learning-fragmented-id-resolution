const COMMON_BLOCKING_KEYS = [
  "first_name",
  "last_name",
  "age",
  "zip_code",
  "house_num",
  "street_name",
  "sex",
];
const MAX_DISPLAY_ROWS = 10;

const state = {
  models: [],
  dataset: null,
  selectedModel: null,
  activeTask: "find",
  datasetMode: "path",
  defaults: null,
  lastFindPayload: null,
  reviewSummary: null,
  recentReviews: [],
  reviewQueueMode: "pending",
  recentReviewLimit: 10,
  highConfidenceExpanded: false,
  reviewQueueExpanded: false,
  recentReviewsExpanded: false,
  deeperAnalysisExpanded: false,
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

function renderDatasetSummary() {
  const card = byId("dataset-status-card");
  const target = byId("dataset-summary");
  byId("hero-dataset-status").textContent = state.dataset ? `${state.dataset.record_count} records` : "Not loaded";
  if (!state.dataset) {
    card.classList.add("hidden");
    target.innerHTML = "";
    return;
  }

  card.classList.remove("hidden");
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
    top_k: 10,
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
  byId("find-max-candidates-input").value = String(findDefaults.max_candidate_pairs || 250000);
  byId("check-max-candidates-input").value = String(checkDefaults.max_candidates || 50000);
}

function clampRows(rows = [], maxRows = MAX_DISPLAY_ROWS) {
  return rows.slice(0, maxRows);
}

function dedupePairs(rows = []) {
  const seen = new Set();
  const uniqueRows = [];
  for (const row of rows) {
    const key = [row.id1, row.id2].sort().join("::");
    if (!seen.has(key)) {
      seen.add(key);
      uniqueRows.push(row);
    }
  }
  return uniqueRows;
}

function escapeCsv(value) {
  const text = String(value ?? "");
  if (text.includes(",") || text.includes("\"") || text.includes("\n")) {
    return `"${text.replace(/"/g, "\"\"")}"`;
  }
  return text;
}

function pairRowsToCsv(rows = [], sectionName) {
  const headers = [
    "section",
    "id1",
    "id2",
    "score",
    "comparison_model",
    "comparison_score",
    "signals",
    "matching_fields",
    "differing_fields",
    "record1_name",
    "record1_address",
    "record1_zip",
    "record1_age",
    "record1_sex",
    "record2_name",
    "record2_address",
    "record2_zip",
    "record2_age",
    "record2_sex",
  ];
  const lines = [headers.join(",")];
  for (const pair of rows) {
    const record1Name = [pair.record1?.first_name, pair.record1?.midl_name, pair.record1?.last_name].filter(Boolean).join(" ");
    const record2Name = [pair.record2?.first_name, pair.record2?.midl_name, pair.record2?.last_name].filter(Boolean).join(" ");
    const record1Address = [pair.record1?.house_num, pair.record1?.street_name].filter(Boolean).join(" ");
    const record2Address = [pair.record2?.house_num, pair.record2?.street_name].filter(Boolean).join(" ");
    const matchingFields = (pair.matching_fields || []).map((field) => `${field.label}: ${field.value}`).join(" | ");
    const differingFields = (pair.differing_fields || []).map((field) => `${field.label}: ${field.left || "<blank>"} -> ${field.right || "<blank>"}`).join(" | ");
    const row = [
      sectionName,
      pair.id1,
      pair.id2,
      formatScore(pair.score),
      pair.comparison_model_name || "",
      pair.comparison_score !== undefined ? formatScore(pair.comparison_score) : "",
      (pair.signals || []).join(" | "),
      matchingFields,
      differingFields,
      record1Name,
      record1Address,
      pair.record1?.zip_code || "",
      pair.record1?.age || "",
      pair.record1?.sex || "",
      record2Name,
      record2Address,
      pair.record2?.zip_code || "",
      pair.record2?.age || "",
      pair.record2?.sex || "",
    ];
    lines.push(row.map(escapeCsv).join(","));
  }
  return lines.join("\n");
}

function downloadCsv(filename, content) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
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

function formatDecisionLabel(decision) {
  return {
    accept_duplicate: "Accepted duplicate",
    reject_duplicate: "Rejected duplicate",
    uncertain: "Marked uncertain",
  }[decision] || decision || "Unknown";
}

function reviewDecisionBadgeClass(decision) {
  return {
    accept_duplicate: "success",
    reject_duplicate: "danger",
    uncertain: "warning",
  }[decision] || "subtle";
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

function renderReviewState(pair) {
  if (!pair.review_state) {
    return `<div class="hint-text">No saved review decision yet.</div>`;
  }

  return `
    <div class="review-state">
      <div class="badge-row">
        <span class="badge ${reviewDecisionBadgeClass(pair.review_state.decision)}">
          ${pair.review_state.decision_label || formatDecisionLabel(pair.review_state.decision)}
        </span>
        <span class="badge subtle">Saved ${pair.review_state.updated_at}</span>
      </div>
      ${pair.review_state.notes ? `<div class="hint-text">Note: ${pair.review_state.notes}</div>` : ""}
    </div>
  `;
}

function renderReviewNoteEditor(pair) {
  const existingNotes = pair.review_state?.notes || "";
  return `
    <div class="review-note-editor">
      <label class="field-label" for="review-note-${pair.pair_key}">Optional note</label>
      <textarea
        id="review-note-${pair.pair_key}"
        class="review-note-input"
        rows="2"
        data-pair-key="${pair.pair_key}"
        placeholder="Add context for why this was accepted, rejected, or marked uncertain."
      >${existingNotes}</textarea>
    </div>
  `;
}

function renderReviewActions(pair, sourceSection) {
  const currentDecision = pair.review_state?.decision || "";
  const actions = [
    { decision: "accept_duplicate", label: "Accept duplicate" },
    { decision: "reject_duplicate", label: "Reject duplicate" },
    { decision: "uncertain", label: "Mark uncertain" },
  ];
  return `
    <div class="review-actions">
      ${actions.map((action) => `
        <button
          type="button"
          class="ghost-button review-action-button ${currentDecision === action.decision ? "active" : ""}"
          data-review-action="${action.decision}"
          data-id1="${pair.id1}"
          data-id2="${pair.id2}"
          data-score="${pair.score ?? ""}"
          data-comparison-model-name="${pair.comparison_model_name || ""}"
          data-comparison-score="${pair.comparison_score ?? ""}"
          data-cluster-id="${pair.cluster_id || ""}"
          data-source-section="${sourceSection || ""}"
        >
          ${action.label}
        </button>
      `).join("")}
    </div>
  `;
}

function renderThresholdDecision(pair, threshold) {
  if (threshold === null || threshold === undefined) {
    return "";
  }
  const acceptedByModel = Number(pair.score) >= Number(threshold);
  return `
    <div class="badge-row">
      <span class="badge ${acceptedByModel ? "success" : "warning"}">
        ${acceptedByModel
      ? "Model accepted this as a duplicate, but it is close to the threshold"
      : "Model rejected this as a duplicate, but it is close to the threshold"}
      </span>
    </div>
  `;
}

function renderCollapsibleSection({
  title,
  subtitle,
  expanded,
  toggleAttr,
  bodyHtml,
  collapsedHint,
  expandedLabel = "Hide section",
  collapsedLabel = "Show section",
}) {
  return `
    <div class="result-block">
      <div class="section-header-inline">
        <div>
          <h3>${title}</h3>
          <p class="section-text">${subtitle}</p>
        </div>
        <button
          type="button"
          class="ghost-button"
          ${toggleAttr}
        >
          ${expanded ? expandedLabel : collapsedLabel}
        </button>
      </div>
      ${expanded
        ? bodyHtml
        : `<div class="hint-text">${collapsedHint || "This section is hidden. Expand it to inspect the current run."}</div>`}
    </div>
  `;
}

function renderPairSection(title, subtitle, rows = [], options = {}) {
  const displayRows = clampRows(rows);
  const reviewNote = options.reviewRecommended
    ? `
      <div class="review-note">
        <strong>Review recommended.</strong>
        These are disagreement cases between models and should be checked manually before acting on them.
      </div>
    `
    : "";
  const sectionBody = `
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
          ${displayRows.length ? displayRows.map((pair) => `
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
                ${options.threshold !== undefined ? renderThresholdDecision(pair, options.threshold) : ""}
                ${options.reviewActions ? renderReviewState(pair) : ""}
                ${options.reviewActions ? renderReviewNoteEditor(pair) : ""}
                ${options.reviewActions ? renderReviewActions(pair, options.sourceSection) : ""}
              </td>
            </tr>
          `).join("") : `
            <tr><td colspan="3">No pairs to show for this section.</td></tr>
          `}
        </tbody>
      </table>
  `;
  if (options.collapsible) {
    return renderCollapsibleSection({
      title,
      subtitle: `${subtitle} Showing up to ${MAX_DISPLAY_ROWS} rows.`,
      expanded: options.collapsible.expanded,
      toggleAttr: options.collapsible.toggleAttr,
      expandedLabel: options.collapsible.expandedLabel,
      collapsedLabel: options.collapsible.collapsedLabel,
      collapsedHint: options.collapsible.collapsedHint
        || `This section is hidden. Expand it to inspect up to ${MAX_DISPLAY_ROWS} rows from the current run.`,
      bodyHtml: sectionBody,
    });
  }
  return `
    <div class="result-block">
      <h3>${title}</h3>
      <p class="section-text">${subtitle} Showing up to ${MAX_DISPLAY_ROWS} rows.</p>
      ${sectionBody}
    </div>
  `;
}

function renderOverviewSummary(payload, clusterSummary, reviewQueueState, reviewSummary) {
  const scoreSummary = payload.score_summary || {};
  const reviewCandidateCount = reviewQueueState.allRows.length;
  const duplicateSideReviewCount = reviewQueueState.allRows.filter(
    (pair) => Number(pair.score) >= Number(payload.threshold)
  ).length;
  const automaticMergeCount = Math.max(payload.predicted_duplicate_pair_count - duplicateSideReviewCount, 0);

  return `
    <div class="result-block">
      <h3>Summary</h3>
      <p class="section-text">
        Main outcome of the current duplicate-search run, including automatic-merge candidates and near-threshold cases
        that still need human review.
      </p>
      <div class="badge-row">
        <span class="badge">Threshold: ${formatScore(payload.threshold)}</span>
      </div>
      <div class="stats-grid">
        <div class="stat-card">
          <span class="mini-label">Candidate pairs</span>
          <strong>${payload.candidate_pair_count}</strong>
        </div>
        <div class="stat-card">
          <span class="mini-label">Predicted duplicates</span>
          <strong>${payload.predicted_duplicate_pair_count}</strong>
        </div>
        <div class="stat-card">
          <span class="mini-label">Automatic merge candidates</span>
          <strong>${automaticMergeCount}</strong>
          <span class="hint-text">Predicted duplicates outside the near-threshold review queue.</span>
        </div>
        <div class="stat-card">
          <span class="mini-label">Human-review candidates</span>
          <strong>${reviewCandidateCount}</strong>
          <span class="hint-text">Near-threshold pairs on either side of the decision boundary.</span>
        </div>
        <div class="stat-card">
          <span class="mini-label">Reviewed so far</span>
          <strong>${reviewQueueState.reviewedRows.length}</strong>
          <span class="hint-text">${reviewSummary.total || 0} saved decisions for this dataset.</span>
        </div>
        <div class="stat-card">
          <span class="mini-label">Predicted clusters</span>
          <strong>${clusterSummary.total_cluster_count || 0}</strong>
          <span class="hint-text">Largest cluster ${clusterSummary.largest_cluster_size || 0} records.</span>
        </div>
      </div>
    </div>
  `;
}

function renderResultActions(payload) {
  const duplicateRows = dedupePairs(payload.exports?.duplicates || []);
  const reviewRows = dedupePairs(payload.exports?.review || []);

  return `
    <div class="result-block">
      <h3>Export Results</h3>
      <p class="section-text">
        Download the full duplicate queue or the human-review queue built from cases closest to the decision threshold.
      </p>
      <div class="results-toolbar">
        <button
          class="ghost-button"
          type="button"
          data-export-kind="duplicates"
          ${duplicateRows.length ? "" : "disabled"}
        >
          Export duplicates CSV
        </button>
        <button
          class="ghost-button"
          type="button"
          data-export-kind="review"
          ${reviewRows.length ? "" : "disabled"}
        >
          Export human-review CSV
        </button>
      </div>
    </div>
  `;
}

function buildReviewQueue(payload) {
  const allRows = dedupePairs(payload.exports?.review || []);
  const pendingRows = allRows.filter((pair) => !pair.review_state);
  const reviewedRows = allRows.filter((pair) => pair.review_state);
  const activeRows = state.reviewQueueMode === "pending" ? pendingRows : allRows;

  return {
    allRows,
    pendingRows,
    reviewedRows,
    visibleRows: clampRows(activeRows),
  };
}

function renderReviewWorkflow(payload) {
  const summary = payload.review_summary || state.reviewSummary || {
    total: 0,
    accepted_duplicate_count: 0,
    rejected_duplicate_count: 0,
    uncertain_count: 0,
  };
  const reviewQueueState = buildReviewQueue(payload);
  const reviewQueue = reviewQueueState.visibleRows;
  const recentReviews = state.recentReviews || payload.recent_reviews || [];

  return `
    <div class="result-block">
      <h3>Human review</h3>
      <p class="section-text">
        These are the cases closest to the duplicate threshold. Review them before merging records into one person.
      </p>
      <div class="stats-grid">
        <div class="stat-card">
          <span class="mini-label">Pending in queue</span>
          <strong>${reviewQueueState.pendingRows.length}</strong>
        </div>
        <div class="stat-card">
          <span class="mini-label">Already reviewed in queue</span>
          <strong>${reviewQueueState.reviewedRows.length}</strong>
        </div>
        <div class="stat-card">
          <span class="mini-label">Accepted duplicate</span>
          <strong>${summary.accepted_duplicate_count || 0}</strong>
        </div>
        <div class="stat-card">
          <span class="mini-label">Rejected / uncertain</span>
          <strong>${(summary.rejected_duplicate_count || 0) + (summary.uncertain_count || 0)}</strong>
        </div>
      </div>
      <div class="results-toolbar">
        <div class="segmented-control compact-control">
          <button
            type="button"
            class="segmented-option ${state.reviewQueueMode === "pending" ? "active" : ""}"
            data-review-queue-mode="pending"
          >
            Pending only (${reviewQueueState.pendingRows.length})
          </button>
          <button
            type="button"
            class="segmented-option ${state.reviewQueueMode === "all" ? "active" : ""}"
            data-review-queue-mode="all"
          >
            Show all (${reviewQueueState.allRows.length})
          </button>
        </div>
        <button type="button" class="ghost-button" data-reset-reviews>
          Reset review decisions for this dataset
        </button>
        <span class="hint-text">
          ${reviewQueueState.reviewedRows.length} already reviewed in this run.
        </span>
      </div>
    </div>
    ${renderPairSection(
     "Human review queue",
     state.reviewQueueMode === "pending"
       ? `Near-threshold pairs that still need a human decision. The queue auto-fills with the next pending cases from this run.`
       : `Near-threshold pairs from this run, including cases you have already reviewed.`,
     reviewQueue,
     {
       reviewActions: true,
       sourceSection: "review_queue",
       threshold: payload.threshold,
       collapsible: {
         expanded: state.reviewQueueExpanded,
         toggleAttr: "data-toggle-review-queue",
         expandedLabel: "Hide queue",
         collapsedLabel: "Show queue",
         collapsedHint: "The review queue is hidden. Expand it to inspect and label near-threshold cases.",
       },
     }
   )}
    ${renderCollapsibleSection({
      title: "Recent review decisions",
      subtitle: `Showing the most recent ${recentReviews.length} saved review decisions for the currently loaded dataset.`,
      expanded: state.recentReviewsExpanded,
      toggleAttr: "data-toggle-recent-reviews",
      expandedLabel: "Hide recent decisions",
      collapsedLabel: "Show recent decisions",
      collapsedHint: "Recent saved review decisions are hidden. Expand this section to inspect the latest labels and notes.",
      bodyHtml: `
        <div class="match-list">
          ${recentReviews.length ? recentReviews.map((review) => `
            <div class="match-card">
              <div class="badge-row">
                <span class="badge mono">${review.id1} / ${review.id2}</span>
                <span class="badge ${reviewDecisionBadgeClass(review.decision)}">${review.decision_label}</span>
                <span class="badge subtle">${review.updated_at}</span>
              </div>
              <div>${recordPreview(review.record1)}</div>
              <div>${recordPreview(review.record2)}</div>
              ${review.notes ? `<div class="hint-text">Note: ${review.notes}</div>` : ""}
            </div>
          `).join("") : "No review decisions have been saved for the current dataset yet."}
        </div>
        ${summary.total > recentReviews.length ? `
          <div class="results-toolbar">
            <button type="button" class="ghost-button" data-load-more-reviews>
              Load 10 more
            </button>
          </div>
        ` : ""}
      `,
    })}
  `;
}

function renderScoreDiagnostics(scoreSummary = {}) {
  const duplicateScores = scoreSummary.duplicates || {};
  const rejectedScores = scoreSummary.rejected || {};
  return `
    <div class="result-block">
      <h3>Score diagnostics</h3>
      <div class="stats-grid">
        <div class="stat-card">
          <span class="mini-label">Acceptance rate</span>
          <strong>${formatScore((scoreSummary.acceptance_rate || 0) * 100)}%</strong>
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
    { reviewRecommended: true, reviewActions: true, sourceSection: "selected_only_disagreement" }
  )}
    ${renderPairSection(
    "Review recommended: comparison model says duplicate",
    "These pairs were flagged as duplicates by the comparison model but not by the selected model.",
    disagreement.comparison_only_duplicates || [],
    { reviewRecommended: true, reviewActions: true, sourceSection: "comparison_only_disagreement" }
  )}
  `;
}

function renderDeeperAnalysis(payload, clusterSummary) {
  const hasDisagreement = Boolean(payload.disagreement_analysis?.requested);
  return `
    <div class="result-block">
      <div class="section-header-inline">
        <div>
          <h3>Deeper analysis</h3>
          <p class="section-text">
            Optional diagnostics for cluster structure, score behavior, and model disagreements.
          </p>
        </div>
        <button
          type="button"
          class="ghost-button"
          data-toggle-deeper-analysis
        >
          ${state.deeperAnalysisExpanded ? "Hide deeper analysis" : "Show deeper analysis"}
        </button>
      </div>
      ${state.deeperAnalysisExpanded ? `
        ${renderScoreDiagnostics(payload.score_summary || {})}
        ${renderPatternSummary(payload.duplicate_pattern_summary || [])}
        ${renderClusterSummary(clusterSummary)}
        ${hasDisagreement ? renderDisagreementSection(payload.disagreement_analysis) : ""}
      ` : `
        <div class="hint-text">
          Deeper analysis is hidden by default to keep the product view focused on operational decisions.
        </div>
      `}
    </div>
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
  const reviewSummary = payload.review_summary || state.reviewSummary || {
    total: 0,
    accepted_duplicate_count: 0,
    rejected_duplicate_count: 0,
    uncertain_count: 0,
  };
  const reviewQueueState = buildReviewQueue(payload);

  root.innerHTML = `
    <div class="results-stack">
      ${renderOverviewSummary(payload, clusterSummary, reviewQueueState, reviewSummary)}
      ${renderPairSection(
     "High-confidence duplicates",
     "The strongest predicted duplicates in this search run. These are the best candidates for automatic merges.",
     payload.high_confidence_duplicates || [],
     {
       collapsible: {
         expanded: state.highConfidenceExpanded,
         toggleAttr: "data-toggle-high-confidence",
         expandedLabel: "Hide duplicates",
         collapsedLabel: "Show duplicates",
         collapsedHint: "High-confidence duplicates are hidden. Expand this section to inspect the strongest automatic-merge candidates.",
       },
     }
   )}
      ${renderReviewWorkflow(payload)}
      ${renderResultActions(payload)}
      ${renderDeeperAnalysis(payload, clusterSummary)}
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
          <span class="badge">Threshold: ${Number(payload.threshold).toFixed(3)}</span>
          <span class="badge ${payload.duplicate_exists ? "warning" : ""}">
            Duplicate exists: ${payload.duplicate_exists ? "Yes" : "No"}
          </span>
          <span class="badge">Candidates: ${payload.candidate_count}</span>
          <span class="badge">Preview capped at ${MAX_DISPLAY_ROWS}</span>
        </div>
        <p><strong>Submitted entry:</strong> ${Object.entries(payload.entry)
      .filter(([, value]) => value)
      .map(([key, value]) => `${key}=${value}`)
      .join(", ")}</p>
      </div>
      <div class="result-block">
        <h3>Top matches</h3>
        <div class="match-list">
          ${clampRows(matches).length ? clampRows(matches).map((match) => `
            <div class="match-card">
              <div class="badge-row">
                <span class="badge mono">${match.existing_id}</span>
                <span class="badge">Score: ${Number(match.score).toFixed(3)}</span>
                <span class="badge ${match.is_duplicate ? "warning" : ""}">
                  ${match.is_duplicate ? "Predicted duplicate" : "Below threshold"}
                </span>
              </div>
              <div>${recordPreview(match.existing_record)}</div>
            </div>
          `).join("") : "No candidate matches were produced for the current entry and blocking settings."}
        </div>
      </div>
    </div>
  `;
}

function collectFindResultPairs(payload) {
  if (!payload) {
    return [];
  }
  return [
    ...(payload.high_confidence_duplicates || []),
    ...(payload.borderline_duplicates || []),
    ...(payload.near_miss_non_duplicates || []),
    ...(payload.review_queue || []),
    ...(payload.exports?.duplicates || []),
    ...(payload.exports?.review || []),
    ...(payload.disagreement_analysis?.selected_only_duplicates || []),
    ...(payload.disagreement_analysis?.comparison_only_duplicates || []),
  ];
}

function clearReviewStatesInPayload(payload) {
  for (const pair of collectFindResultPairs(payload)) {
    delete pair.review_state;
  }
}

async function loadRecentReviews(limit = state.recentReviewLimit) {
  const payload = await fetchJson(`/reviews?limit=${limit}`);
  state.reviewSummary = payload.review_summary || state.reviewSummary;
  state.recentReviews = payload.reviews || [];
  state.recentReviewLimit = limit;
  if (state.lastFindPayload) {
    state.lastFindPayload.review_summary = state.reviewSummary;
    state.lastFindPayload.recent_reviews = state.recentReviews;
  }
}

async function refreshAppState(showInfo = false) {
  const payload = await fetchJson("/app/state");
  state.models = payload.models || [];
  state.dataset = payload.dataset;
  state.defaults = payload.defaults || null;
  state.reviewSummary = payload.review_summary || null;
  state.recentReviews = payload.recent_reviews || [];
  state.recentReviewLimit = 10;
  state.selectedModel = state.selectedModel || chooseDefaultModel(state.models, payload.defaults?.preferred_model);
  renderDatasetSummary();
  syncBlockingControlsFromModel();

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
  state.lastFindPayload = null;
  renderDatasetSummary();
  byId("results-root").innerHTML = "Dataset loaded. Choose a task to begin review.";
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
  state.lastFindPayload = null;
  renderDatasetSummary();
  byId("results-root").innerHTML = "Dataset loaded. Choose a task to begin review.";
  showBanner(`Dataset uploaded: ${payload.dataset.source_name}.`);
}

async function clearDataset() {
  await fetchJson("/dataset", { method: "DELETE" });
  state.dataset = null;
  state.lastFindPayload = null;
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
      top_k: MAX_DISPLAY_ROWS,
      max_candidate_pairs: Number(byId("find-max-candidates-input").value) || 250000,
      include_disagreement_analysis: byId("find-include-disagreement").checked,
    }),
  });
  state.lastFindPayload = payload;
  state.reviewSummary = payload.review_summary || state.reviewSummary;
  state.recentReviews = payload.recent_reviews || state.recentReviews;
  state.recentReviewLimit = 10;
  state.deeperAnalysisExpanded = false;
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
    midl_name: byId("entry-midl-name").value,
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
      top_k: MAX_DISPLAY_ROWS,
      max_candidates: Number(byId("check-max-candidates-input").value) || 50000,
    }),
  });
  renderCheckResults(payload);
  showBanner(payload.duplicate_exists ? "Potential duplicate found." : "No duplicate found above threshold.");
}

function wireEvents() {
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

  byId("results-root").addEventListener("click", (event) => {
    const exportButton = event.target.closest("[data-export-kind]");
    if (exportButton && state.lastFindPayload) {
      const kind = exportButton.dataset.exportKind;
      const exportsPayload = state.lastFindPayload.exports || {};
      if (kind === "duplicates") {
        const rows = dedupePairs(exportsPayload.duplicates || []);
        if (!rows.length) {
          showBanner("There are no duplicate rows available to export for this run.", "error");
          return;
        }
        downloadCsv("duplicates_found.csv", pairRowsToCsv(rows, "duplicates"));
        return;
      }

      if (kind === "review") {
        const rows = dedupePairs(exportsPayload.review || []);
        if (!rows.length) {
          showBanner("There are no human-review rows available to export for this run.", "error");
          return;
        }
        downloadCsv("human_review_queue.csv", pairRowsToCsv(rows, "human_review"));
        return;
      }
    }

    const toggleDeeperAnalysisButton = event.target.closest("[data-toggle-deeper-analysis]");
    if (toggleDeeperAnalysisButton && state.lastFindPayload) {
      state.deeperAnalysisExpanded = !state.deeperAnalysisExpanded;
      renderFindResults(state.lastFindPayload);
      return;
    }

    const toggleHighConfidenceButton = event.target.closest("[data-toggle-high-confidence]");
    if (toggleHighConfidenceButton && state.lastFindPayload) {
      state.highConfidenceExpanded = !state.highConfidenceExpanded;
      renderFindResults(state.lastFindPayload);
      return;
    }

    const toggleReviewQueueButton = event.target.closest("[data-toggle-review-queue]");
    if (toggleReviewQueueButton && state.lastFindPayload) {
      state.reviewQueueExpanded = !state.reviewQueueExpanded;
      renderFindResults(state.lastFindPayload);
      return;
    }

    const toggleRecentReviewsButton = event.target.closest("[data-toggle-recent-reviews]");
    if (toggleRecentReviewsButton && state.lastFindPayload) {
      state.recentReviewsExpanded = !state.recentReviewsExpanded;
      renderFindResults(state.lastFindPayload);
      return;
    }

    const loadMoreReviewsButton = event.target.closest("[data-load-more-reviews]");
    if (loadMoreReviewsButton) {
      hideBanner();
      loadRecentReviews(state.recentReviewLimit + 10)
        .then(() => {
          if (state.lastFindPayload) {
            renderFindResults(state.lastFindPayload);
          }
        })
        .catch((error) => {
          showBanner(formatErrorMessage(error.message), "error");
        });
      return;
    }

    const resetReviewsButton = event.target.closest("[data-reset-reviews]");
    if (resetReviewsButton) {
      if (!window.confirm("Clear all saved review decisions for the currently loaded dataset?")) {
        return;
      }
      hideBanner();
      fetchJson("/reviews", { method: "DELETE" })
        .then((payload) => {
          state.reviewSummary = payload.review_summary || state.reviewSummary;
          state.recentReviews = payload.recent_reviews || [];
          state.recentReviewLimit = 10;
          if (state.lastFindPayload) {
            clearReviewStatesInPayload(state.lastFindPayload);
            state.lastFindPayload.review_summary = state.reviewSummary;
            state.lastFindPayload.recent_reviews = state.recentReviews;
            renderFindResults(state.lastFindPayload);
          }
          showBanner(payload.message || "Review decisions cleared.");
        })
        .catch((error) => {
          showBanner(formatErrorMessage(error.message), "error");
        });
      return;
    }

    const queueModeButton = event.target.closest("[data-review-queue-mode]");
    if (queueModeButton && state.lastFindPayload) {
      state.reviewQueueMode = queueModeButton.dataset.reviewQueueMode;
      renderFindResults(state.lastFindPayload);
      return;
    }

    const reviewButton = event.target.closest("[data-review-action]");
    if (!reviewButton || !state.lastFindPayload) {
      return;
    }

    const noteInput = reviewButton.closest("td")?.querySelector(".review-note-input");
    hideBanner();
    fetchJson("/reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id1: reviewButton.dataset.id1,
        id2: reviewButton.dataset.id2,
        decision: reviewButton.dataset.reviewAction,
        model_name: state.selectedModel,
        score: reviewButton.dataset.score ? Number(reviewButton.dataset.score) : null,
        comparison_model_name: reviewButton.dataset.comparisonModelName || null,
        comparison_score: reviewButton.dataset.comparisonScore ? Number(reviewButton.dataset.comparisonScore) : null,
        source_section: reviewButton.dataset.sourceSection || null,
        cluster_id: reviewButton.dataset.clusterId || null,
        notes: noteInput ? noteInput.value.trim() : "",
      }),
    })
      .then((payload) => {
        const savedReview = payload.saved_review;
        for (const pair of collectFindResultPairs(state.lastFindPayload)) {
          if (pair.pair_key === savedReview.pair_key) {
            pair.review_state = savedReview;
          }
        }
        state.reviewSummary = payload.review_summary || state.reviewSummary;
        loadRecentReviews(state.recentReviewLimit)
          .catch(() => {
            state.recentReviews = payload.recent_reviews || state.recentReviews;
          })
          .finally(() => {
            state.lastFindPayload.review_summary = state.reviewSummary;
            state.lastFindPayload.recent_reviews = state.recentReviews;
            renderFindResults(state.lastFindPayload);
            showBanner(`Saved review decision: ${savedReview.decision_label}.`);
          });
      })
      .catch((error) => {
        showBanner(formatErrorMessage(error.message), "error");
      });
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
