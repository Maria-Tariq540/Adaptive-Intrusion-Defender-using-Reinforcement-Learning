"""api.py

Flask API for serving predictions from the RL intrusion detection agent.

SOC-grade endpoints:
- POST /predict
  - Auth: Bearer token (Authorization header)
  - Input: traffic features
  - Output: action, attack_type, risk_score, confidence, severity, explanation

- POST /predict/batch
  - Auth: same as /predict
  - Input: {"items": [ ...items... ]}
  - Output: {"results": [ ...predictions... ]}

Notes:
- On startup we bootstrap a "trained" policy via a bounded training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request

from agent import RLIntrusionAgent
from config import get_config
from environment import IntrusionDetectionEnvironment
from simulator import generate_traffic
from enterprise_pipeline import EnterprisePredictInput, enterprise_predict

from utils import append_prediction_to_csv






State = Tuple[int, int, int, int]  # request_rate, failed_logins, unknown_ip, time_of_day


@dataclass(frozen=True)
class PredictInput:
    request_rate: int
    failed_logins: int
    unknown_ip: int
    time_of_day: int


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        iv = int(value)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Expected integer in [{lo}, {hi}], got {value!r}") from e
    return max(lo, min(hi, iv))


def _validate_and_parse_full_payload(payload: Dict[str, Any]) -> Dict[str, int]:
    """Validate and clamp *all* traffic features.

    This matches `simulator.TrafficRecord.to_dict()` keys.
    """

    required = ["request_rate", "failed_logins", "unknown_ip", "time_of_day"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required JSON keys: {missing}")

    return {
        "request_rate": _clamp_int(payload["request_rate"], 0, 100),
        "failed_logins": _clamp_int(payload["failed_logins"], 0, 10),
        "unknown_ip": _clamp_int(payload["unknown_ip"], 0, 1),
        "time_of_day": _clamp_int(payload["time_of_day"], 0, 23),
        # Optional features (safe defaults)
        "traffic_spike": _clamp_int(payload.get("traffic_spike", 0), 0, 1),
        "session_duration": _clamp_int(payload.get("session_duration", 0), 0, 5000),
        "packet_size": _clamp_int(payload.get("packet_size", 0), 0, 50),
        "repeated_requests": _clamp_int(payload.get("repeated_requests", 0), 0, 50),
        "ip_reputation_score": _clamp_int(payload.get("ip_reputation_score", 50), 0, 100),
        "session_id": _clamp_int(payload.get("session_id", -1), -1_000_000, 10_000_000),
    }


def _state_from_full_features(features: Dict[str, int]) -> State:
    return (
        int(features["request_rate"]),
        int(features["failed_logins"]),
        int(features["unknown_ip"]),
        int(features["time_of_day"]),
    )


def _action_to_str(action_id: int) -> str:
    """Map agent action id to SOC action string."""
    # tabular agent: 0=ALLOW, 1=MONITOR, 3=BLOCK in current codebase
    return "BLOCK" if int(action_id) in (1, 3) else "ALLOW"


def bootstrap_trained_agent() -> RLIntrusionAgent:
    """Run a short training loop to bootstrap a reasonable policy at startup."""

    cfg = get_config()

    simulator = type("_Sim", (), {"generate_traffic": staticmethod(generate_traffic)})

    n_samples = int(getattr(cfg, "max_steps_per_episode", 200))
    n_episodes = int(getattr(cfg, "episodes", 100))
    n_episodes = min(n_episodes, 200)
    n_samples = min(n_samples, 400)

    env = IntrusionDetectionEnvironment(
        simulator=simulator,
        n_samples=n_samples,
        seed=int(getattr(cfg, "seed", 42)),
        config=cfg,
    )

    agent = RLIntrusionAgent(config=cfg)

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < n_samples:
            action = agent.choose_action(state)
            next_state, reward, done, _info = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps += 1

    return agent


def _check_bearer_token() -> None:
    cfg = get_config()
    expected = str(getattr(cfg, "api_token", "CHANGE_ME"))

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise PermissionError("Missing/invalid Authorization header")

    token = auth_header.split(" ", 1)[1].strip()
    if token != expected:
        raise PermissionError("Unauthorized")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["agent"] = bootstrap_trained_agent()

    @app.post("/predict")
    def predict():
        import time

        try:
            _check_bearer_token()

            if not request.is_json:
                return jsonify({"error": "Request body must be JSON"}), 400
            payload = request.get_json(silent=True) or {}

            start = time.perf_counter()

            features = _validate_and_parse_full_payload(payload)

            pred = enterprise_predict(
                inp=EnterprisePredictInput(
                    request_rate=features["request_rate"],
                    failed_logins=features["failed_logins"],
                    unknown_ip=features["unknown_ip"],
                    time_of_day=features["time_of_day"],
                    traffic_spike=features["traffic_spike"],
                    session_duration=features["session_duration"],
                    packet_size=features["packet_size"],
                    repeated_requests=features["repeated_requests"],
                    ip_reputation_score=features["ip_reputation_score"],
                    session_id=features["session_id"],
                ),
            )

            latency_ms = int((time.perf_counter() - start) * 1000.0)


            fieldnames = [
                "timestamp",
                "source",
                "latency_ms",
                "session_id",
                "request_rate",
                "failed_logins",
                "unknown_ip",
                "time_of_day",
                "traffic_spike",
                "session_duration",
                "packet_size",
                "repeated_requests",
                "ip_reputation_score",
                "action",
                "attack_type",
                "risk_score",
                "confidence",
                "severity",
                "explanation",
                "honeypot_status",
                "honeypot_reason",
                "early_warning",
                "predicted_risk",
                "predicted_severity",
                "ground_truth",
                "correct",
            ]


            row = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "source": "api",
                "latency_ms": latency_ms,
                "session_id": int(features["session_id"]),
                "request_rate": features["request_rate"],
                "failed_logins": features["failed_logins"],
                "unknown_ip": features["unknown_ip"],
                "time_of_day": features["time_of_day"],
                "traffic_spike": features["traffic_spike"],
                "session_duration": features["session_duration"],
                "packet_size": features["packet_size"],
                "repeated_requests": features["repeated_requests"],
                "ip_reputation_score": features["ip_reputation_score"],
                "action": pred["action"],
                "attack_type": pred["attack_type"],
                "risk_score": int(pred["risk_score"]),
                "confidence": pred["confidence"],
                "severity": pred["severity_level"],
                "explanation": pred["explanation"],
                "predicted_risk": int(pred["predicted_risk"]),
                "predicted_severity": pred.get("predicted_severity", ""),
                "honeypot_status": bool(pred.get("honeypot_status", False)),
                "honeypot_reason": str(pred.get("honeypot_reason", "")),
                "early_warning": bool(pred.get("early_warning", False)),

                "ground_truth": "",
                "correct": "",
            }

            append_prediction_to_csv(
                row,
                csv_path=str(getattr(get_config(), "predictions_csv_path", "predictions_log.csv")),
                fieldnames=fieldnames,
            )

            return jsonify(
                {
                    "action": pred["action"],
                    "attack_type": pred["attack_type"],
                    "risk_score": int(pred["risk_score"]),
                    "predicted_risk": int(pred["predicted_risk"]),
                    "confidence": pred["confidence"],
                    "severity": pred["severity_level"],
                    "explanation": pred["explanation"],
                    "honeypot_status": bool(pred.get("honeypot_status", False)),
                    "latency_ms": latency_ms,
                }
            )


        except PermissionError as e:
            return jsonify({"error": str(e)}), 401
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:  # pragma: no cover
            return jsonify({"error": f"Server error: {e}"}), 500

    @app.post("/predict/batch")
    def predict_batch():
        try:
            _check_bearer_token()

            if not request.is_json:
                return jsonify({"error": "Request body must be JSON"}), 400
            payload = request.get_json(silent=True) or {}
            items = payload.get("items")
            if not isinstance(items, list):
                return jsonify({"error": "Expected 'items' to be a list"}), 400

            results = []
            agent: RLIntrusionAgent = app.config["agent"]

            for item in items:
                if not isinstance(item, dict):
                    results.append({"error": "Item must be an object"})
                    continue

                features = _validate_and_parse_full_payload(item)

                pred = enterprise_predict(
                    inp=EnterprisePredictInput(
                        request_rate=features["request_rate"],
                        failed_logins=features["failed_logins"],
                        unknown_ip=features["unknown_ip"],
                        time_of_day=features["time_of_day"],
                        traffic_spike=features["traffic_spike"],
                        session_duration=features["session_duration"],
                        packet_size=features["packet_size"],
                        repeated_requests=features["repeated_requests"],
                        ip_reputation_score=features["ip_reputation_score"],
                        session_id=features["session_id"],
                    ),
                )

                results.append(
                    {
                        "action": pred["action"],
                        "attack_type": pred["attack_type"],
                        "risk_score": int(pred["risk_score"]),
                        "predicted_risk": int(pred["predicted_risk"]),
                        "confidence": pred["confidence"],
                        "severity": pred["severity_level"],
                        "explanation": pred["explanation"],
                        "honeypot_status": bool(pred.get("honeypot_status", False)),
                        "session_id": int(features["session_id"]),
                    }
                )


            return jsonify({"results": results})

        except PermissionError as e:
            return jsonify({"error": str(e)}), 401
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:  # pragma: no cover
            return jsonify({"error": f"Server error: {e}"}), 500

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

