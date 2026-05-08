# TODO_SOC_UPGRADE (SOC monitoring upgrade)

## Step 1 — Upgrade Flask API (`api.py`)
- [ ] Implement `POST /predict` with full traffic feature input schema.
- [ ] Enforce authentication token (Bearer).
- [ ] Return required output fields: `action`, `attack_type`, `risk_score`, `confidence`, `severity`, `explanation`.
- [ ] Implement batch endpoint `POST /predict/batch`.
- [ ] Add API latency stats in-memory + optionally `GET /stats`.
- [ ] Log latency_ms + full input/output fields to `predictions_log.csv` (consistent schema).

## Step 2 — Rewrite Streamlit SOC Dashboard (`dashboard.py`)
- [ ] Implement SOC theme (neon blue/red + dark background).
- [ ] Add animated cyber status indicators (Safe/Suspicious/Attack).
- [ ] Add live RL simulator stream (existing behavior, improved).
- [ ] Add CSV follower that safely reloads `predictions_log.csv` (no crashes on partial writes).
- [ ] Create live panels: traffic stream, AI decisions, attack alerts, blocked alerts, suspicious feed.
- [ ] Add Advanced Analytics KPIs.
- [ ] Add required charts: risk distribution, attack frequency, accuracy over time, live traffic trends, attack timeline heatmap.
- [ ] Implement smooth auto-refresh (short TTL polling + st_autorefresh).

## Step 3 — Integration Consistency
- [ ] Ensure both API and dashboard write consistent CSV columns.
- [ ] Ensure explainable AI outputs match dashboard/API expectations.

## Step 4 — Final-year polish
- [ ] Add architecture diagram + workflow visualization section in Streamlit.
- [ ] Add Traditional IDS vs RL IDS comparison section.
- [ ] Add modular helper functions + production-style comments.

## Done when
- [ ] API endpoints work with auth + batch.
- [ ] Dashboard shows real-time updates + analytics + charts without crashing.
- [ ] Both systems can run simultaneously with concurrent CSV writes.

