# Cybersecurity Intrusion Detection using Reinforcement Learning (Enterprise SOC Edition)

Enterprise-style intrusion defense project using:
- Tabular Reinforcement Learning (Q-learning) for defensive policy
- Real-time SOC-style monitoring dashboard
- Attack simulation (synthetic traffic)
- Explainable AI narratives
- Risk scoring + confidence + severity
- Honeypot redirection
- Predictive near-future alerts
- Simulated multi-agent security control layer
- API integration (Flask) for SOC consumers
- Logging system (CSV)

---

## Key modules
- `simulator.py`
  - Generates synthetic traffic records (multi-attack scenarios)
- `environment.py`
  - RL environment with ground-truth heuristics and shaped rewards
- `agent.py`
  - Tabular Q-learning agent
- `enterprise_pipeline.py`
  - **Single source of truth** for enterprise predictions:
    - predictive risk (`predictor.py`)
    - honeypot decision (`honeypot.py`)
    - multi-agent coordination (`multi_agent.py`)
    - explainable narrative (`explainable_ai.py`)
- `api.py`
  - Flask API endpoints: `/predict`, `/predict/batch`, `/health`
- `dashboard.py`
  - Streamlit SOC dashboard (live simulation)
- `utils.py`
  - Risk scoring + CSV logging helpers

---

## API usage
### Authentication
All `/predict` endpoints require a bearer token:
- Header: `Authorization: Bearer <API_TOKEN>`

`api_token` is read from `hyperparameters.json` (fallback: `CHANGE_ME`).

### Endpoints
- `POST /health`
  - Returns `{ "status": "ok" }`

- `POST /predict`
  - Input JSON (minimum required):
    - `request_rate`, `failed_logins`, `unknown_ip`, `time_of_day`
  - Optional fields:
    - `traffic_spike`, `session_duration`, `packet_size`, `repeated_requests`, `ip_reputation_score`, `session_id`

  - Output includes (enterprise contract):
    - `action`, `attack_type`, `risk_score`, `predicted_risk`, `predicted_severity`
    - `confidence`, `severity`, `explanation`
    - `honeypot_status`, `honeypot_reason`
    - `early_warning`

---

## Run commands
### Train (synthetic RL bootstrapping)
```bash
python train.py
```

### Start API
```bash
python api.py
```

### Start Dashboard
```bash
streamlit run dashboard.py
```

---

## Enterprise upgrades included (final)
- Honeypot redirection
- Predictive attack alerts
- Simulated multi-agent layer
- Dashboard + API wired to enterprise pipeline output
- CSV logging expanded for honeypot + predictive alert fields

---

## Notes
- This project is a synthetic platform: decisions and attack labels are heuristic-driven for demonstration.
- `streamlit` may need installation on your environment.

