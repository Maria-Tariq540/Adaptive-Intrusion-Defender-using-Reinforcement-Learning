# TODO - Phase 2 Upgrade (Predictive Detection + Honeypot Defense)

## Information gathered
- `api.py` currently implements `POST /predict` and returns: action, attack_type, risk_score, confidence, severity, explanation.
- `dashboard.py` currently simulates online RL steps and shows ALLOW/BLOCK traffic, plus a rolling risk pill and accuracy graph.
- `environment.py` uses a tabular Q-learning environment with actions 0..3 (ALLOW/MONITOR/THROTTLE/BLOCK) and ground-truth heuristic intrusion detection.
- `utils.py` contains `compute_risk_score()` and `risk_score_to_levels()`.
- `multi_agent.py` exists with actions 0..4 including `ACTION_ISOLATE_DEVICE` but the current dashboard focuses on the simpler RL environment.

## Phase 2 code changes (approved plan pending)
1. Add predictive attack detection
   - Add a lightweight predictor module (rule-based or lightweight model) that examines time-series features:
     - abnormal traffic spikes
     - increasing failed logins
     - suspicious request patterns
     - repeated unknown IP activity
   - Output warning strings + predicted_risk (0..100) and severity.

2. Add risk scoring support for predicted future risk
   - Extend risk scoring output everywhere to include:
     - current_risk
     - predicted_risk

3. Honeypot redirection defense
   - Introduce new defensive action `REDIRECT_TO_HONEYPOT`.
   - When suspicious traffic is detected and prediction indicates imminent attack:
     - redirect attacker to simulated honeypot
     - store honeypot events in logs
     - dashboard must highlight redirected attackers

4. API upgrade
   - Update `/predict` output JSON to include:
     - action, attack_type, current_risk, predicted_risk, severity, explanation, honeypot_status

5. Dashboard enhancements
   - Add live attack prediction panel (animated alerts)
   - Add honeypot activity panel
   - Add future threat indicator using predicted_risk
   - Ensure warnings appear in real time during simulation.

6. Testing after phase 2
   - Run:
     - `python api.py`
     - `streamlit run dashboard.py`
   - Verify:
     - honeypot redirection works
     - predictive attack alerts visible
     - future risk score generated
     - dashboard shows attack predictions
     - API returns enhanced JSON response

