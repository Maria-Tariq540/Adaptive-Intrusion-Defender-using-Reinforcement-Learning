# TODO — Enterprise AI + RL Cybersecurity Platform Upgrade

## Phase A — Real Dataset Integration
- [ ] Implement real-dataset traffic stream for RL/dashboard (no simulator dependency for training/online ticks)
- [ ] Ensure preprocessing: missing value cleaning, scaling, label mapping, feature selection
- [ ] Add artifact pack for preprocessing so prediction uses identical scaling/columns

## Phase B — High-Accuracy AI Attack Detection
- [ ] Integrate supervised classifier into `enterprise_pipeline.py` (attack_type + confidence)
- [ ] Replace rule-based `predictor.py` usage in core path with classifier outputs (keep fallback)
- [ ] Calibrate risk_score/severity from classifier probabilities (no fake accuracy)

## Phase C — Real Attack Classification Support
- [ ] Map dataset labels to required classes everywhere (dashboard/logs/API)
- [ ] Ensure logs store attack_type + classifier_confidence + RL_action

## Phase D — RL Improvement + Stability
- [ ] Upgrade `environment.py` reward to use dataset ground-truth labels (not heuristic)
- [ ] Enrich RL state mapping from real features (request_rate, packet_rate, flow_duration, dest_port, SYN flags, classifier confidence)
- [ ] Stabilize epsilon/random ALLOW behavior based on classifier confidence and predicted severity

## Phase E — Enterprise AI Analyst
- [ ] Ensure every prediction includes: attack_type, classifier_confidence, RL_action, risk_score, severity, explanation, recommended_response
- [ ] Update dashboard analyst section to reflect real classifier info

## Phase F — Advanced SOC Dashboard Enhancements
- [ ] Fix UI scaling/overlap issues (heatmap, charts, animations)
- [ ] Ensure confidence graphs, severity histogram, blocked vs allowed charts work consistently

## Phase G — Real-Time Early Warning System
- [ ] Implement early-warning triggers from classifier probability/risk deltas and engineered anomalies
- [ ] Feed early_warning into dashboard warnings panel

## Phase H — Logging + Model Storage
- [ ] Persist: classifier model, scaler, encoders, feature_columns, label mapping
- [ ] Verify logs folder writes: timestamp, raw features, predicted attack, classifier confidence, RL action, risk, severity, explanation

## Phase I — API Upgrade
- [ ] `/predict` schema outputs required keys
- [ ] `/predict` loads models automatically on startup (lazy load acceptable)
- [ ] `/predict/batch` returns same schema for each item

## Phase J — Final Cleanup
- [ ] Ensure final structure and move unused experimental files to `archive/`
- [ ] Confirm no existing modules removed

## Phase K — README Generation
- [ ] Generate professional GitHub README.md with architecture + run commands + metrics + API example

## Validation checklist (final)
- [ ] Real dataset training works
- [ ] Achieves realistic 80%+ accuracy (as reported by training metrics)
- [ ] Dashboard runs without crashes and shows real attack types/confidence
- [ ] API `/predict` works and outputs correct schema
- [ ] Logs save correctly
- [ ] Models save/load correctly

