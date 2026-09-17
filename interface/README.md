# DDPF Studio interface mockup

This folder contains a self-contained, interactive mockup of the complete DDPF
workflow. It is intended to support the journal manuscript discussion and a
future implementation; it is not connected to the research pipeline yet.

The prototype contains four views:

1. **Description** — ontology resources, semantic mappings, composition and
   consistency parameters.
2. **Detection** — a visual `datasets.zip` upload, source streams, extraction
   algorithms, granularities, matching thresholds and entity-linking policy.
3. **Prediction** — representation pipelines, encoder settings, clustering
   algorithms and evaluation metrics.
4. **Results** — the analyst-facing geographic, temporal, cluster and
   provenance dashboard.

## Important data notice

All records displayed by the mockup are **illustrative interface data**. They
are shaped like DDPF observations, but the numbers and excerpts must not be
reported as experimental or epidemiological results.

## Run locally

No third-party front-end package is required.

```bash
cd interface
python serve.py
```

Open <http://127.0.0.1:8080>.

You can choose another port:

Linux/macOS:

```bash
PORT=9000 python serve.py
```

Windows PowerShell:

```powershell
$env:PORT = 9000
python serve.py
```

## Included interactions

- navigate between Description, Detection, Prediction and Results;
- select a local `datasets.zip` in mock mode without transmitting it;
- select representation pipelines and clustering tabs;
- filter by retrospective disease label, source and observation granularity;
- inspect geographic signals and ranked observations;
- switch between Entity, Concept and Concepts representation views;
- select points in the qualitative cluster projection;
- inspect entities, concept paths and provenance;
- search observations with `Ctrl+K` or `Cmd+K`;
- create an analyst brief from the current view;
- export the filtered illustrative records as JSON;
- switch between dark and light themes;
- responsive layout for desktop, tablet and mobile screens.

Buttons that would normally execute a pipeline display a work-in-progress
message. They intentionally do not process data or launch models.

## Files

| File | Purpose |
|---|---|
| `index.html` | Semantic dashboard structure and inline SVG icon set |
| `styles.css` | Responsive visual design and themes |
| `data.js` | Illustrative DDPF-shaped observation records |
| `app.js` | Filtering, charts, selection, search and export behaviour |
| `serve.py` | Small local server using only the Python standard library |
| `assets/favicon.svg` | DDPF Studio browser icon |

## Connecting real pipeline data later

Keep the UI and API/data adapter separate. A real observation payload should
provide at least:

```json
{
  "id": "stable observation identifier",
  "date": "ISO-8601 date",
  "source": "source name",
  "sourceType": "News | Medical | Social | Official | Local",
  "granularity": "Document | Paragraph | Sentence",
  "location": "linked place label",
  "country": "country label",
  "entities": ["linked biomedical mentions"],
  "concepts": ["ontology-grounded concept path"],
  "cluster": 0,
  "confidence": 0.94,
  "provenance": "traceable source record"
}
```

Recommended integration sequence:

1. Export observations, cluster assignments and provenance from a documented
   DDPF run.
2. Add a small backend/API layer that validates the payload and removes any
   sensitive or disallowed source content.
3. Replace `data.js` with an API adapter while keeping a demo-data fallback.
4. Keep data snapshot, model/checkpoint, representation, clustering parameters
   and random seed visible in the reproducibility panel.
5. Add authentication and role-based access before exposing non-public data.

## Scope represented by the mockup

The interface deliberately says **retrospective semantic surveillance**. It
helps an analyst group, inspect and trace previously collected observations.
It does not present DDPF as a validated real-time outbreak-detection, causal or
clinical decision-support system.
