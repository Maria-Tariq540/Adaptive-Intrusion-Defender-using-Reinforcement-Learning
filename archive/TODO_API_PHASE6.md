# TODO_API_PHASE6 - Flask API for RL cybersecurity model

## Step 1
Implement Flask app in `api.py` with `POST /predict`.

## Step 2
On startup, bootstrap a “trained” RL agent by running a short training loop using:
- `IntrusionDetectionEnvironment`
- `RLIntrusionAgent`
- `simulator.generate_traffic`

## Step 3
Implement input parsing/validation for JSON keys:
- request_rate, failed_logins, unknown_ip, time_of_day

## Step 4
Implement output formatting:
- action: ALLOW/BLOCK
- risk_score: 0-100
- explanation: rule-based text

## Step 5
Run the server: `python api.py`

## Step 6
Test endpoint with a sample POST request and ensure prediction JSON returns.

