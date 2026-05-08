# Enterprise upgrade TODO (RL-Driven Autonomous Network Intrusion Defense System)

## 1) Repo cleanup + structure
- [ ] Create folders: archive/, logs/, models/, data/
- [ ] Move all phase/TODO development artifacts into archive/ (TODO_*.md)
- [ ] Remove duplicates / decide canonical entrypoints (api.py, predictor.py)

## 2) Enterprise prediction pipeline (single source of truth)
- [ ] Implement a unified `enterprise_predict()` pipeline used by API + dashboard
- [ ] Ensure pipeline outputs required decision fields:
  - action, attack_type, risk_score, predicted_risk, confidence, severity_level, human-readable explanation
  - honeypot_status

## 3) Honeypot integration
- [ ] Wire `honeypot.maybe_use_honeypot()` into the pipeline/multi-agent
- [ ] Add honeypot redirect action: REDIRECT_TO_HONEYPOT
- [ ] Log honeypot events into CSV + expose via dashboard

## 4) Predictive threat intelligence
- [ ] Use `predictor.predict_near_future_attack()` inside the pipeline
- [ ] Generate early warnings before attacks fully occur

## 5) Multi-agent compliance for action set
- [ ] Ensure MonitoringAgent / FirewallAgent / IsolationAgent coordinate to produce:
  - ALLOW / MONITOR / THROTTLE / BLOCK / ISOLATE_DEVICE / REDIRECT_TO_HONEYPOT

## 6) Training metrics + persistence
- [ ] Upgrade train.py metrics:
  - precision, recall, false positives, false negatives
  - rewards, accuracy
- [ ] Save plots automatically
- [ ] Save/load Q-table for API/dashboard usage

## 7) API compliance
- [ ] Update POST /predict response schema + include latency
- [ ] Update POST /predict/batch schema
- [ ] Maintain Bearer token auth

## 8) Dashboard compliance
- [ ] SOC-style Streamlit dashboard:
  - neon/dark theme
  - live traffic monitoring
  - attack alerts
  - AI decisions with required fields
  - risk graphs, heatmaps, attack timeline
  - blocked IP counters
  - recovered sessions
  - auto-refresh support
  - honeypot activity

## 9) Final validation
- [ ] Verify training runs
- [ ] Verify API runs and returns schema
- [ ] Verify dashboard runs and updates
- [ ] Verify CSV logging includes honeypot status

## 10) README rewrite
- [ ] Replace README.md with concise professional enterprise GitHub README

---
Execute steps in order. Mark each step complete after verification.

