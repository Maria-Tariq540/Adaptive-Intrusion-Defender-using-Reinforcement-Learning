# TODO: RL-Driven Autonomous Network Intrusion Defense System (Enterprise Finalization)

## Step A — Repo cleanup
- [ ] Create `archive/`, `logs/`, `models/`, `data/`
- [ ] Move all `TODO_*.md` and phase artifacts into `archive/`
- [ ] Remove duplicates and keep only: simulator.py environment.py agent.py train.py dashboard.py api.py utils.py config.py

## Step B — Enterprise prediction pipeline
- [ ] Create `enterprise_pipeline.py` (or integrate into existing modules)
- [ ] Produce unified decision dict with required schema

## Step C — Multi-agent action set compliance
- [ ] Add REDIRECT_TO_HONEYPOT action
- [ ] Ensure final action is one of the required values

## Step D — Honeypot integration
- [ ] Use `honeypot.maybe_use_honeypot()` and drive final action
- [ ] Add `honeypot_status` into CSV logging
- [ ] Add honeypot events to dashboard

## Step E — Predictive threat intelligence
- [ ] Use `predictor.predict_near_future_attack()` to generate predicted_risk
- [ ] Add early-warning alerts into dashboard

## Step F — Explainable AI compliance
- [ ] Ensure explainable decision includes: risk_score, predicted_risk, confidence, severity_level, human explanation

## Step G — Training upgrades
- [ ] Add precision/recall + FP/FN metrics
- [ ] Save training artifacts: learning_curve.png, reward_curve.png, precision_recall.png
- [ ] Save/load Q-table under `models/`

## Step H — API compliance
- [ ] Update POST /predict + /predict/batch responses to match required schema
- [ ] Ensure API includes honeypot_status and predicted_risk

## Step I — Dashboard compliance
- [ ] Add risk graphs, heatmap, attack timeline
- [ ] Add blocked IP counters + recovered sessions
- [ ] Auto-refresh

## Step J — Final validation + README
- [ ] Run: python train.py, python api.py, streamlit run dashboard.py
- [ ] Verify CSV logs schema
- [ ] Rewrite README.md

