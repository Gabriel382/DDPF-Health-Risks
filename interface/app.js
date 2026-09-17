(() => {
  "use strict";

  const data = window.DDPF_DEMO_DATA;
  if (!data) {
    throw new Error("DDPF demo data could not be loaded.");
  }

  const state = {
    view: "results",
    disease: "all",
    source: "all",
    granularity: "all",
    layer: "concepts",
    selectedId: data.observations[0].id,
    theme: "dark"
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const formatNumber = new Intl.NumberFormat("en-GB");
  let toastTimer;

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const selectedObservation = () => {
    const current = filteredObservations();
    return current.find((observation) => observation.id === state.selectedId) || current[0] || null;
  };

  const filteredObservations = () =>
    data.observations.filter((observation) => {
      const diseaseMatch = state.disease === "all" || observation.disease === state.disease;
      const sourceMatch = state.source === "all" || observation.source === state.source;
      const granularityMatch =
        state.granularity === "all" || observation.granularity === state.granularity;
      return diseaseMatch && sourceMatch && granularityMatch;
    });

  const observationVolume = (observations) =>
    observations.reduce((total, observation) => total + observation.volume, 0);

  function initialiseSourceFilter() {
    const sources = [...new Set(data.observations.map((item) => item.source))].sort();
    const sourceFilter = $("#sourceFilter");
    sourceFilter.insertAdjacentHTML(
      "beforeend",
      sources.map((source) => `<option value="${escapeHtml(source)}">${escapeHtml(source)}</option>`).join("")
    );
  }

  function renderAll() {
    const observations = filteredObservations();
    if (!observations.some((item) => item.id === state.selectedId) && observations.length) {
      state.selectedId = observations[0].id;
    }
    renderMetrics(observations);
    renderMap(observations);
    renderTimeline(observations);
    renderSignals(observations);
    renderClusters(observations);
    renderDetails(selectedObservation());
    renderSourceBars(observations);
    updateFilterControls(observations);
  }

  function renderMetrics(observations) {
    const volume = observationVolume(observations);
    const sources = new Set(observations.map((item) => item.source)).size;
    const locations = new Set(observations.map((item) => `${item.location}|${item.country}`)).size;
    const clusters = new Set(observations.map((item) => item.cluster)).size;

    $("#metricObservations").textContent = formatNumber.format(volume);
    $("#metricSources").textContent = formatNumber.format(sources);
    $("#metricLocations").textContent = formatNumber.format(locations);
    $("#metricClusters").textContent = formatNumber.format(clusters);
    $("#filterResult").textContent = `${formatNumber.format(volume)} observations`;
  }

  function switchView(view) {
    const page = $(`[data-page="${view}"]`);
    if (!page) return;
    state.view = view;
    $$("[data-page]").forEach((candidate) => {
      const active = candidate.dataset.page === view;
      candidate.hidden = !active;
      candidate.classList.toggle("active", active);
    });
    $$(".nav-item[data-view]").forEach((button) =>
      button.classList.toggle("active", button.dataset.view === view)
    );
    $("#breadcrumbCurrent").textContent = view.charAt(0).toUpperCase() + view.slice(1);
    $$(".results-only").forEach((item) => item.classList.toggle("view-hidden", view !== "results"));
    $("#sidebar").classList.remove("open");
    $("#mobileScrim").classList.remove("open");
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (view === "results") renderAll();
  }

  function renderMap(observations) {
    const markersRoot = $("#mapMarkers");
    const grouped = new Map();

    observations.forEach((observation) => {
      const key = `${observation.location}-${observation.disease}`;
      const current = grouped.get(key) || {
        location: observation.location,
        country: observation.country,
        disease: observation.disease,
        map: [...observation.map],
        volume: 0,
        items: []
      };
      current.volume += observation.volume;
      current.items.push(observation);
      grouped.set(key, current);
    });

    const locationDiseaseCounts = {};
    [...grouped.values()].forEach((group) => {
      locationDiseaseCounts[group.location] = (locationDiseaseCounts[group.location] || 0) + 1;
    });

    markersRoot.innerHTML = [...grouped.values()]
      .map((group) => {
        let [x, y] = group.map;
        if (locationDiseaseCounts[group.location] > 1) {
          x += group.disease === "covid" ? -7 : 7;
          y += group.disease === "covid" ? -4 : 4;
        }
        const radius = Math.max(5, Math.min(11, 4 + Math.sqrt(group.volume) * 0.55));
        const colour = group.disease === "covid" ? "var(--covid)" : "var(--mpox)";
        const selected = group.items.some((item) => item.id === state.selectedId) ? " selected" : "";
        const topItem = [...group.items].sort((a, b) => b.priority - a.priority)[0];
        return `
          <g class="map-marker${selected}" transform="translate(${x} ${y})"
             data-id="${escapeHtml(topItem.id)}" data-location="${escapeHtml(group.location)}"
             data-volume="${group.volume}" data-disease="${group.disease}" tabindex="0" role="button">
            <circle class="glow" r="${radius * 2.1}" fill="${colour}" filter="url(#softGlow)"></circle>
            <circle class="core" r="${radius}" fill="${colour}"></circle>
            <text class="map-label" x="${radius + 6}" y="3">${escapeHtml(group.location)}</text>
          </g>`;
      })
      .join("");

    $$(".map-marker", markersRoot).forEach((marker) => {
      const showTooltip = (event) => {
        const tooltip = $("#mapTooltip");
        const mapStage = $("#mapStage");
        const stageRect = mapStage.getBoundingClientRect();
        const disease = marker.dataset.disease === "covid" ? "COVID-19" : "Mpox";
        tooltip.innerHTML = `<strong>${escapeHtml(marker.dataset.location)}</strong><span>${formatNumber.format(Number(marker.dataset.volume))} ${disease} observations</span>`;
        const x = event.clientX ? event.clientX - stageRect.left + 10 : stageRect.width / 2;
        const y = event.clientY ? event.clientY - stageRect.top + 10 : stageRect.height / 2;
        tooltip.style.left = `${Math.min(x, stageRect.width - 170)}px`;
        tooltip.style.top = `${Math.min(y, stageRect.height - 54)}px`;
        tooltip.classList.add("visible");
      };
      marker.addEventListener("mouseenter", showTooltip);
      marker.addEventListener("focus", showTooltip);
      marker.addEventListener("mouseleave", () => $("#mapTooltip").classList.remove("visible"));
      marker.addEventListener("blur", () => $("#mapTooltip").classList.remove("visible"));
      marker.addEventListener("click", () => selectObservation(marker.dataset.id));
      marker.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectObservation(marker.dataset.id);
        }
      });
    });
  }

  function renderTimeline(observations) {
    const container = $("#timelineChart");
    const width = 410;
    const height = 210;
    const padding = { top: 13, right: 9, bottom: 28, left: 27 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxValue = Math.max(...data.timeline.flatMap((item) => [item.covid, item.mpox]), 1);
    const x = (index) => padding.left + (index / (data.timeline.length - 1)) * plotWidth;
    const y = (value) => padding.top + plotHeight - (value / maxValue) * plotHeight;
    const allCovidVolume = observationVolume(data.observations.filter((item) => item.disease === "covid"));
    const allMpoxVolume = observationVolume(data.observations.filter((item) => item.disease === "mpox"));
    const filteredCovidVolume = observationVolume(observations.filter((item) => item.disease === "covid"));
    const filteredMpoxVolume = observationVolume(observations.filter((item) => item.disease === "mpox"));
    const scale = {
      covid: filteredCovidVolume / allCovidVolume,
      mpox: filteredMpoxVolume / allMpoxVolume
    };
    const linePath = (key) =>
      data.timeline.map((item, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(item[key] * scale[key]).toFixed(1)}`).join(" ");
    const areaPath = (key) =>
      `${linePath(key)} L${x(data.timeline.length - 1)},${padding.top + plotHeight} L${x(0)},${padding.top + plotHeight} Z`;
    const showCovid = state.disease !== "mpox";
    const showMpox = state.disease !== "covid";
    const gridValues = [0, 20, 40, 60];

    container.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="COVID-19 and mpox observation timeline">
        <defs>
          <linearGradient id="covidArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="var(--covid)" stop-opacity=".9"></stop>
            <stop offset="1" stop-color="var(--covid)" stop-opacity="0"></stop>
          </linearGradient>
          <linearGradient id="mpoxArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="var(--mpox)" stop-opacity=".8"></stop>
            <stop offset="1" stop-color="var(--mpox)" stop-opacity="0"></stop>
          </linearGradient>
        </defs>
        ${gridValues.map((value) => `<line class="chart-grid-line" x1="${padding.left}" x2="${width - padding.right}" y1="${y(value)}" y2="${y(value)}"></line><text class="chart-axis-label" x="3" y="${y(value) + 3}">${value}</text>`).join("")}
        ${data.timeline.map((item, index) => index % 3 === 0 || index === data.timeline.length - 1 ? `<text class="chart-axis-label" text-anchor="middle" x="${x(index)}" y="${height - 7}">${escapeHtml(item.label)}</text>` : "").join("")}
        ${showCovid ? `<path class="chart-area" d="${areaPath("covid")}" fill="url(#covidArea)"></path><path class="chart-line" d="${linePath("covid")}" stroke="var(--covid)"></path>` : ""}
        ${showMpox ? `<path class="chart-area" d="${areaPath("mpox")}" fill="url(#mpoxArea)"></path><path class="chart-line" d="${linePath("mpox")}" stroke="var(--mpox)"></path>` : ""}
      </svg>`;

    $("#covidTimelineTotal").textContent = formatNumber.format(filteredCovidVolume);
    $("#mpoxTimelineTotal").textContent = formatNumber.format(filteredMpoxVolume);
    $("#covidTimelineTotal").closest("div").style.opacity = showCovid ? "1" : ".28";
    $("#mpoxTimelineTotal").closest("div").style.opacity = showMpox ? "1" : ".28";
  }

  function renderSignals(observations) {
    const ranked = [...observations].sort((a, b) => b.priority - a.priority).slice(0, 6);
    const signalList = $("#signalList");
    if (!ranked.length) {
      signalList.innerHTML = '<div class="empty-state">No observations match the selected filters.</div>';
      return;
    }
    signalList.innerHTML = ranked
      .map((observation, index) => {
        const active = observation.id === state.selectedId ? " active" : "";
        const colour = observation.disease === "covid" ? "var(--covid)" : "var(--mpox)";
        return `
          <button class="signal-item${active}" data-id="${escapeHtml(observation.id)}">
            <span class="signal-rank">${String(index + 1).padStart(2, "0")}</span>
            <span class="signal-copy">
              <strong>${escapeHtml(observation.title)}</strong>
              <span>${escapeHtml(observation.location)} · ${escapeHtml(observation.source)}</span>
            </span>
            <span class="signal-score"><i style="background:${colour}"></i>${observation.priority}</span>
          </button>`;
      })
      .join("");
    $$(".signal-item", signalList).forEach((button) =>
      button.addEventListener("click", () => selectObservation(button.dataset.id))
    );
  }

  function renderClusters(observations) {
    const container = $("#clusterChart");
    const width = 680;
    const height = 310;
    const offsets = { entity: [-5, 5], concept: [5, -4], concepts: [0, 0] };
    const [layerX, layerY] = offsets[state.layer];
    const pointRows = observations
      .map((observation, index) => {
        const [baseX, baseY] = observation.embedding;
        const hashOffset = ((observation.id.charCodeAt(observation.id.length - 1) + index) % 5) - 2;
        const x = 28 + ((baseX + layerX + hashOffset) / 100) * (width - 56);
        const y = 25 + ((baseY + layerY - hashOffset) / 100) * (height - 50);
        const colour = observation.disease === "covid" ? "var(--covid)" : "var(--mpox)";
        const radius = 4.5 + Math.min(4, observation.volume / 30);
        const selected = observation.id === state.selectedId ? " selected" : "";
        return `<circle class="cluster-point${selected}" data-id="${escapeHtml(observation.id)}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${radius.toFixed(1)}" fill="${colour}" tabindex="0"><title>${escapeHtml(observation.title)}</title></circle>`;
      })
      .join("");

    container.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Qualitative semantic projection">
        <g opacity=".8">
          ${[70, 130, 190, 250].map((y) => `<line class="chart-grid-line" x1="30" x2="650" y1="${y}" y2="${y}"></line>`).join("")}
          ${[120, 240, 360, 480, 600].map((x) => `<line class="chart-grid-line" x1="${x}" x2="${x}" y1="22" y2="282"></line>`).join("")}
        </g>
        <ellipse class="cluster-hull" cx="445" cy="124" rx="152" ry="86" fill="var(--covid)"></ellipse>
        <ellipse class="cluster-hull" cx="233" cy="213" rx="146" ry="78" fill="var(--mpox)"></ellipse>
        <text class="cluster-label" x="520" y="49">Cluster 01</text>
        <text class="cluster-subtitle" x="520" y="61">respiratory / systemic</text>
        <text class="cluster-label" x="86" y="255">Cluster 02</text>
        <text class="cluster-subtitle" x="86" y="267">dermatologic / contact</text>
        ${pointRows}
      </svg>`;

    $$(".cluster-point", container).forEach((point) => {
      point.addEventListener("click", () => selectObservation(point.dataset.id));
      point.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectObservation(point.dataset.id);
        }
      });
    });

    const clusters = new Set(observations.map((item) => item.cluster)).size;
    $("#clusterSelection").textContent = observations.length
      ? `${clusters} cluster${clusters === 1 ? "" : "s"} · ${state.layer} input`
      : "No points visible";
  }

  function renderDetails(observation) {
    if (!observation) {
      $("#detailTitle").textContent = "No matching observation";
      $("#detailConfidence").textContent = "—";
      $("#detailMeta").innerHTML = "";
      $("#detailExcerpt").textContent = "Adjust or reset the filters to inspect an observation.";
      $("#detailEntities").innerHTML = "";
      $("#detailConceptPath").innerHTML = "";
      $("#detailProvenance").textContent = "No record selected";
      return;
    }
    $("#detailTitle").textContent = observation.title;
    $("#detailConfidence").textContent = `${Math.round(observation.confidence * 100)}% match`;
    $("#detailMeta").innerHTML = [
      observation.id,
      new Date(`${observation.date}T00:00:00`).toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric"
      }),
      `${observation.location}, ${observation.country}`,
      observation.source,
      observation.granularity
    ]
      .map((item) => `<span>${escapeHtml(item)}</span>`)
      .join("");
    $("#detailExcerpt").textContent = `“${observation.excerpt}”`;
    $("#detailEntities").innerHTML = observation.entities
      .map((entity) => `<span>${escapeHtml(entity)}</span>`)
      .join("");
    $("#detailConceptPath").innerHTML = observation.concepts
      .map((concept) => `<span>${escapeHtml(concept)}</span>`)
      .join("");
    $("#detailProvenance").textContent = observation.provenance;
  }

  function renderSourceBars(observations) {
    const bySource = observations.reduce((accumulator, observation) => {
      accumulator[observation.source] = (accumulator[observation.source] || 0) + observation.volume;
      return accumulator;
    }, {});
    const rows = Object.entries(bySource).sort((a, b) => b[1] - a[1]);
    const max = Math.max(...rows.map(([, volume]) => volume), 1);
    $("#sourceBars").innerHTML = rows.length
      ? rows
          .map(
            ([source, volume]) => `
              <div class="source-row">
                <label title="${escapeHtml(source)}">${escapeHtml(source)}</label>
                <div class="source-track"><i style="width:${Math.max(4, (volume / max) * 100).toFixed(1)}%"></i></div>
                <strong>${formatNumber.format(volume)}</strong>
              </div>`
          )
          .join("")
      : '<div class="empty-state">No source data for this view.</div>';
  }

  function updateFilterControls(observations) {
    $$("[data-disease]").forEach((button) =>
      button.classList.toggle("active", button.dataset.disease === state.disease)
    );
    $("#sourceFilter").value = state.source;
    $("#granularityFilter").value = state.granularity;
    $$("[data-layer]").forEach((button) =>
      button.classList.toggle("active", button.dataset.layer === state.layer)
    );
    if (!observations.length) {
      $("#filterResult").textContent = "No matching observations";
    }
  }

  function selectObservation(id, options = {}) {
    state.selectedId = id;
    renderMap(filteredObservations());
    renderSignals(filteredObservations());
    renderClusters(filteredObservations());
    renderDetails(selectedObservation());
    if (options.scroll) {
      $(".inspector-panel").scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  function showToast(message) {
    const toast = $("#toast");
    toast.textContent = message;
    toast.classList.add("visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 2600);
  }

  function openBrief() {
    const observations = filteredObservations();
    const volume = observationVolume(observations);
    const top = [...observations].sort((a, b) => b.priority - a.priority)[0];
    const diseases = [...new Set(observations.map((item) => item.disease))]
      .map((name) => (name === "covid" ? "COVID-19" : "mpox"))
      .join(" and ");
    const text = observations.length
      ? `
        <h3>Current selection</h3>
        <p>The current retrospective view contains <strong>${formatNumber.format(volume)} illustrative observations</strong> related to ${escapeHtml(diseases || "the selected events")} across ${new Set(observations.map((item) => item.source)).size} source streams.</p>
        <ul>
          <li>The highest-ranked item is “${escapeHtml(top.title)}” from ${escapeHtml(top.location)}.</li>
          <li>The view uses the <strong>${escapeHtml(state.layer)}</strong> representation layer.</li>
          <li>Cluster separation should be interpreted alongside provenance, time and geography—not as automatic evidence of a new outbreak.</li>
        </ul>`
      : "<h3>Current selection</h3><p>No observations match the selected filters.</p>";
    $("#briefSummary").innerHTML = text;
    const modal = $("#briefModal");
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
    $("#closeBrief").focus();
  }

  function closeBrief() {
    const modal = $("#briefModal");
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
    $("#briefButton").focus();
  }

  function briefPlainText() {
    return $("#briefSummary").innerText.trim();
  }

  function openCommandPalette() {
    if (state.view !== "results") switchView("results");
    const palette = $("#commandPalette");
    palette.classList.add("open");
    palette.setAttribute("aria-hidden", "false");
    $("#commandInput").value = "";
    renderCommandResults("");
    window.setTimeout(() => $("#commandInput").focus(), 10);
  }

  function closeCommandPalette() {
    const palette = $("#commandPalette");
    palette.classList.remove("open");
    palette.setAttribute("aria-hidden", "true");
  }

  function renderCommandResults(query) {
    const normalised = query.trim().toLowerCase();
    const matches = data.observations
      .filter((observation) => {
        const haystack = [
          observation.title,
          observation.location,
          observation.country,
          observation.source,
          ...observation.entities,
          ...observation.concepts
        ]
          .join(" ")
          .toLowerCase();
        return !normalised || haystack.includes(normalised);
      })
      .slice(0, 8);
    $("#commandResults").innerHTML = matches.length
      ? matches
          .map(
            (observation) => `
              <button class="command-result" data-id="${escapeHtml(observation.id)}">
                <i>${observation.disease === "covid" ? "C19" : "MPX"}</i>
                <span><strong>${escapeHtml(observation.title)}</strong><span>${escapeHtml(observation.location)} · ${escapeHtml(observation.source)} · ${escapeHtml(observation.entities.slice(0, 2).join(", "))}</span></span>
                <small>${observation.priority}</small>
              </button>`
          )
          .join("")
      : '<div class="empty-state">No matching observation.</div>';
    $$(".command-result", $("#commandResults")).forEach((button) =>
      button.addEventListener("click", () => {
        state.disease = "all";
        state.source = "all";
        state.granularity = "all";
        state.selectedId = button.dataset.id;
        renderAll();
        closeCommandPalette();
        $(".inspector-panel").scrollIntoView({ behavior: "smooth", block: "center" });
      })
    );
  }

  function exportView() {
    const payload = {
      exportedAt: new Date().toISOString(),
      purpose: "DDPF Studio interface mockup",
      warning: data.meta.note,
      filters: {
        disease: state.disease,
        source: state.source,
        granularity: state.granularity,
        representation: state.layer
      },
      observations: filteredObservations()
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "ddpf-studio-view.json";
    anchor.click();
    URL.revokeObjectURL(url);
    showToast("Current illustrative view exported as JSON.");
  }

  function bindEvents() {
    $$("[data-disease]").forEach((button) =>
      button.addEventListener("click", () => {
        state.disease = button.dataset.disease;
        renderAll();
      })
    );
    $("#sourceFilter").addEventListener("change", (event) => {
      state.source = event.target.value;
      renderAll();
    });
    $("#granularityFilter").addEventListener("change", (event) => {
      state.granularity = event.target.value;
      renderAll();
    });
    $("#resetFilters").addEventListener("click", () => {
      state.disease = "all";
      state.source = "all";
      state.granularity = "all";
      renderAll();
      showToast("Filters reset.");
    });
    $$("[data-layer]").forEach((button) =>
      button.addEventListener("click", () => {
        state.layer = button.dataset.layer;
        renderClusters(filteredObservations());
        updateFilterControls(filteredObservations());
      })
    );

    $("#themeToggle").addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      document.body.dataset.theme = state.theme;
      showToast(`${state.theme === "dark" ? "Dark" : "Light"} theme enabled.`);
    });
    $("#exportButton").addEventListener("click", exportView);
    $("#briefButton").addEventListener("click", openBrief);
    $("#closeBrief").addEventListener("click", closeBrief);
    $("#closeBriefFooter").addEventListener("click", closeBrief);
    $("#briefModal").addEventListener("click", (event) => {
      if (event.target === event.currentTarget) closeBrief();
    });
    $("#copyBrief").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(briefPlainText());
        showToast("Analyst brief copied.");
      } catch {
        showToast("Clipboard access is unavailable in this browser.");
      }
    });

    $("#searchButton").addEventListener("click", openCommandPalette);
    $("#showAllSignals").addEventListener("click", openCommandPalette);
    $("#commandInput").addEventListener("input", (event) => renderCommandResults(event.target.value));
    $("#commandPalette").addEventListener("click", (event) => {
      if (event.target === event.currentTarget) closeCommandPalette();
    });

    $$(".nav-item[data-view]").forEach((button) =>
      button.addEventListener("click", () => switchView(button.dataset.view))
    );
    $$('[data-go-view]').forEach((button) =>
      button.addEventListener("click", () => switchView(button.dataset.goView))
    );

    $$(".mock-action").forEach((button) =>
      button.addEventListener("click", (event) => {
        event.preventDefault();
        showToast("Work in progress — this action is visual only in the current mockup.");
      })
    );

    const uploadZone = $(".upload-zone");
    const datasetUpload = $("#datasetUpload");
    const displayMockFile = (file) => {
      if (!file) return;
      $("#uploadTitle").textContent = file.name;
      $("#uploadSubtitle").textContent = `${Math.max(1, Math.round(file.size / 1024))} KB · mock selection only, not uploaded`;
      showToast("Dataset selected for the interface preview. No file was uploaded.");
    };
    datasetUpload.addEventListener("change", () => displayMockFile(datasetUpload.files[0]));
    ["dragenter", "dragover"].forEach((name) =>
      uploadZone.addEventListener(name, (event) => {
        event.preventDefault();
        uploadZone.classList.add("dragging");
      })
    );
    ["dragleave", "drop"].forEach((name) =>
      uploadZone.addEventListener(name, (event) => {
        event.preventDefault();
        uploadZone.classList.remove("dragging");
      })
    );
    uploadZone.addEventListener("drop", (event) => displayMockFile(event.dataTransfer.files[0]));

    const updatePipelineCount = () => {
      const selected = $$(".representation-card.selected");
      $("#selectedPipelineCount").textContent = `${selected.length} selected`;
      $$(".representation-card").forEach((card) => {
        $(".representation-check", card).textContent = card.classList.contains("selected") ? "✓" : "+";
      });
    };
    $$(".representation-card").forEach((card) =>
      card.addEventListener("click", () => {
        card.classList.toggle("selected");
        updatePipelineCount();
      })
    );

    $$(".choice-card:not(.disabled)").forEach((card) =>
      card.addEventListener("click", () => {
        $$(".choice-card").forEach((item) => item.classList.remove("selected"));
        card.classList.add("selected");
      })
    );

    $$(".algorithm-tabs button").forEach((button) =>
      button.addEventListener("click", () => {
        $$(".algorithm-tabs button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        showToast(`${button.textContent} parameters are represented as a visual mockup.`);
      })
    );

    $("#mobileMenu").addEventListener("click", () => {
      $("#sidebar").classList.add("open");
      $("#mobileScrim").classList.add("open");
    });
    $("#mobileScrim").addEventListener("click", () => {
      $("#sidebar").classList.remove("open");
      $("#mobileScrim").classList.remove("open");
    });

    document.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        openCommandPalette();
      }
      if (event.key === "Escape") {
        closeCommandPalette();
        if ($("#briefModal").classList.contains("open")) closeBrief();
      }
    });
  }

  initialiseSourceFilter();
  bindEvents();
  renderAll();
  switchView("results");
})();
