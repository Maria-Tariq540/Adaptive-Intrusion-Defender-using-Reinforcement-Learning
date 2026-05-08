"""dashboard.py

Streamlit dashboard for the RL-based cybersecurity intrusion detection system.

Run:
    streamlit run dashboard.py

Features:
- Dark cyber-themed UI
- Sidebar: Start simulation / Stop simulation
- Live updates: traffic log (ALLOW/BLOCK), Attack vs Normal counters,
  real-time accuracy graph, and risk indicator (green/yellow/red).

Implementation notes:
- Uses existing simulation logic:
    - simulator.py (synthetic traffic generation)
    - environment.py (RL environment + ground-truth heuristic)
    - agent.py (tabular Q-learning agent)
- The dashboard performs online interaction with the environment while running.
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import streamlit as st


from agent import RLIntrusionAgent
from environment import IntrusionDetectionEnvironment
from simulator import generate_traffic
from multi_agent import MultiAgentSecurityPlatform

from enterprise_pipeline import EnterprisePredictInput, enterprise_predict

from utils import append_prediction_to_csv







# -------------------------
# UI: Cyber dark theme
# -------------------------

def _apply_cyber_theme() -> None:
    st.set_page_config(
        page_title="RL Intrusion Detection - Live Console",
        page_icon="🛡️",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            :root {
                --bg: #070A12;
                --panel: rgba(255, 255, 255, 0.04);
                --panel2: rgba(255, 255, 255, 0.06);
                --text: #E6EAF2;
                --muted: #AAB3C5;
                --cyan: #00E5FF;
                --green: #2DFF8F;
                --yellow: #FFD84D;
                --red: #FF4D4D;
                --border: rgba(255, 255, 255, 0.10);
            }

            html, body, [class*="stApp"] {
                background-color: var(--bg);
                color: var(--text);
            }

            .stSidebar {
                background: rgba(0,0,0,0.25);
                border-right: 1px solid var(--border);
            }

            .panel {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 14px;
            }

            .title {
                font-weight: 800;
                letter-spacing: 0.3px;
                color: var(--cyan);
                text-shadow: 0 0 18px rgba(0, 229, 255, 0.22);
            }

            .kpi {
                font-size: 26px;
                font-weight: 800;
                margin: 0;
                color: var(--text);
            }

            .subkpi {
                color: var(--muted);
                font-size: 12px;
                margin-top: -4px;
            }

            .risk-pill {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 10px 14px;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: rgba(255,255,255,0.03);
                font-weight: 700;
            }

            .dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: var(--green);
                box-shadow: 0 0 14px rgba(45,255,143,0.35);
            }

            .dot.yellow {
                background: var(--yellow);
                box-shadow: 0 0 14px rgba(255,216,77,0.35);
            }

            .dot.red {
                background: var(--red);
                box-shadow: 0 0 14px rgba(255,77,77,0.35);
            }

            .mono {
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--border);
                border-radius: 14px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Simulation loop state
# -------------------------

ActionLogRow = Dict[str, object]


@dataclass
class DashboardState:
    running: bool = False
    step_idx: int = 0

    # Live log (last N)
    log: Deque[ActionLogRow] = None  # type: ignore

    # Counters
    attack_total: int = 0
    normal_total: int = 0
    allow_count: int = 0
    block_count: int = 0

    # Rolling accuracy computation
    correct_total: int = 0
    evaluated_total: int = 0

    # For accuracy graph
    accuracy_points: List[Tuple[int, float]] = None  # type: ignore


def _init_state() -> None:
    if "dash_state" not in st.session_state:
        st.session_state.dash_state = DashboardState(
            running=False,
            step_idx=0,
            log=deque(maxlen=200),
            accuracy_points=[],
        )


def _action_to_str(action: int) -> str:
    # legacy shim (dashboard now uses enterprise_predict)
    return "BLOCK" if action in (1, 3) else "ALLOW"



def _classify_attack(request_rate: int, failed_logins: int) -> str:
    if failed_logins >= 5:
        return "brute force"
    if request_rate >= 75:
        return "DDoS"
    return "unknown behavior"


def _compute_risk_score(request_rate: int, failed_logins: int, unknown_ip: int, action: int) -> int:
    """Compatibility shim.

    Uses the shared single-source-of-truth risk scoring in utils.py.
    """

    from utils import compute_risk_score

    return compute_risk_score(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=None,
        agent_action=action,
    )



def _risk_label_from_score(risk_score: int) -> Tuple[str, str]:
    if risk_score < 35:
        return ("LOW", "")
    if risk_score < 60:
        return ("MEDIUM", "yellow")
    return ("HIGH", "red")


def _compute_risk(level_window: List[float]) -> Tuple[str, str]:

    """Return (risk_label, risk_color_css_class_dot)."""

    if not level_window:
        return ("—", "")

    # Risk is computed from recent intrusion rate among evaluated steps.
    # We'll map: lower intrusion -> green, mid -> yellow, high -> red.
    avg_intrusion_rate = sum(level_window) / len(level_window)  # 0..1

    if avg_intrusion_rate < 0.35:
        return ("LOW", "")  # green (default)
    if avg_intrusion_rate < 0.60:
        return ("MEDIUM", "yellow")
    return ("HIGH", "red")


def _risk_pill(label: str, dot_class: str) -> None:
    dot_class_attr = f"dot {dot_class}" if dot_class else "dot"
    st.markdown(
        f"""
        <div class='risk-pill'>
            <span class='{dot_class_attr}'></span>
            <span class='mono'>Risk: {label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Streamlit app
# -------------------------

_apply_cyber_theme()
_init_state()

st.markdown("<div class='title' style='font-size:22px;'>🛡️ RL Intrusion Detection - Live Console</div>", unsafe_allow_html=True)
st.caption("Online interaction using simulator + environment + tabular Q-learning agent")

with st.sidebar:
    st.markdown("<div class='mono' style='color:#00E5FF;font-weight:800;'>SIMULATION CONTROL</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button("▶ Start simulation", type="primary", use_container_width=True)
    with col2:
        stop_clicked = st.button("■ Stop simulation", use_container_width=True)

    # Optional controls for speed/behavior
    speed_ms = st.slider("Tick delay (ms)", 0, 200, 25, help="Lower is faster updates")
    max_steps_per_run = st.slider(
        "Max steps per run",
        50,
        2000,
        500,
        step=50,
        help="Prevents runaway loops; Stop also stops immediately.",
    )

    st.divider()
    st.markdown("<div class='mono' style='color:#AAB3C5;'>Logs update every tick.</div>", unsafe_allow_html=True)


# Start/stop handlers
if start_clicked:
    # Reset dashboard state but keep Streamlit app responsive.
    st.session_state.dash_state.running = True
    st.session_state.dash_state.step_idx = 0
    st.session_state.dash_state.attack_total = 0
    st.session_state.dash_state.normal_total = 0
    st.session_state.dash_state.allow_count = 0
    st.session_state.dash_state.block_count = 0
    st.session_state.dash_state.correct_total = 0
    st.session_state.dash_state.evaluated_total = 0
    st.session_state.dash_state.log = deque(maxlen=200)
    st.session_state.dash_state.accuracy_points = []

    # Create a fresh environment stream.
    # Using simulator.generate_traffic internally via environment so ground truth aligns.
    simulator = type("_Sim", (), {"generate_traffic": staticmethod(generate_traffic)})

    # We generate enough samples for this run.
    st.session_state._env = IntrusionDetectionEnvironment(
        simulator=simulator,
        n_samples=int(max_steps_per_run),
        seed=42,
        config=None,
    )

    st.session_state._agent = RLIntrusionAgent(config=None)
    st.session_state._state = st.session_state._env.reset()

    # Risk window: store recent risk scores (0..100)
    st.session_state._risk_window: Deque[int] = deque(maxlen=50)


if stop_clicked:
    st.session_state.dash_state.running = False


# Layout
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='mono' style='font-weight:800;color:#00E5FF;'>LIVE TRAFFIC LOG</div>", unsafe_allow_html=True)

    # Create display dataframe from deque
    ds = st.session_state.dash_state
    log_rows = list(ds.log)[-120:]
    if log_rows:
        # Ensure consistent column ordering
        cols = [
            "Step",
            "Action",
            "GroundTruth",
            "Correct",
            "RiskScore",
            "AttackType",
            "Explanation",
        ]

        # Map ground truth bool to label
        for r in log_rows:
            if isinstance(r.get("GroundTruth"), bool):
                r["GroundTruth"] = "ATTACK" if r["GroundTruth"] else "NORMAL"

        # st.dataframe expects list of dicts; keep it simple
        st.dataframe(log_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Click **Start simulation** to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='mono' style='font-weight:800;color:#00E5FF;'>COUNTERS & RISK</div>", unsafe_allow_html=True)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<p class='subkpi'>ALLOW</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='kpi mono'>{ds.allow_count}</p>", unsafe_allow_html=True)
    with c2:
        st.markdown("<p class='subkpi'>BLOCK</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='kpi mono'>{ds.block_count}</p>", unsafe_allow_html=True)
    with c3:
        st.markdown("<p class='subkpi'>ATTACK</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='kpi mono'>{ds.attack_total}</p>", unsafe_allow_html=True)
    with c4:
        st.markdown("<p class='subkpi'>NORMAL</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='kpi mono'>{ds.normal_total}</p>", unsafe_allow_html=True)

    st.divider()

    # Risk indicator
    risk_window = list(st.session_state.get("_risk_window", deque()))
    risk_label, dot_class = _compute_risk(risk_window)
    _risk_pill(risk_label, dot_class)

    st.markdown("</div>", unsafe_allow_html=True)


# Accuracy graph + overall status
st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
colA, colB = st.columns([0.75, 0.25], gap="large")

with colA:
    st.markdown("<div class='mono' style='font-weight:800;color:#00E5FF;'>ACCURACY (REAL-TIME)</div>", unsafe_allow_html=True)
    points = st.session_state.dash_state.accuracy_points
    if points:
        # Convert to two-column series for line_chart
        # line_chart expects y values; x isn't always supported uniformly.
        y_vals = [p[1] for p in points]
        st.line_chart(y_vals, height=260)
    else:
        st.info("Accuracy will appear while simulation runs.")

with colB:
    st.markdown("<div class='mono' style='font-weight:800;color:#00E5FF;'>STATUS</div>", unsafe_allow_html=True)

    running = st.session_state.dash_state.running
    st.markdown(
        f"""
        <div class='mono' style='margin-top:10px; padding:10px; border:1px solid rgba(255,255,255,0.10); border-radius:12px; background:rgba(255,255,255,0.03);'>
            <div><b>Simulation:</b> {'RUNNING' if running else 'STOPPED'}</div>
            <div style='margin-top:6px; color:#AAB3C5;'><b>Steps:</b> {st.session_state.dash_state.step_idx}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Accuracy snapshot
    if ds.evaluated_total > 0:
        acc = ds.correct_total / ds.evaluated_total
        st.metric("Accuracy", f"{acc*100:.2f}%")
    else:
        st.metric("Accuracy", "—")

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Live simulation tick
# -------------------------

def _tick_once() -> None:
    ds = st.session_state.dash_state
    if not ds.running:
        return

    env: IntrusionDetectionEnvironment = st.session_state._env
    agent: RLIntrusionAgent = st.session_state._agent
    state = st.session_state._state

    # One step: take the environment step, then produce enterprise (predictive + honeypot + multi-agent) decision
    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    agent.learn(state, action, reward, next_state)

    request_rate, failed_logins, unknown_ip, time_of_day = state
    session_id = int(env.traffic_data[env._idx - 1].get("session_id", st.session_state.dash_state.step_idx)) if hasattr(env, "traffic_data") else int(st.session_state.dash_state.step_idx)

    # Enterprise prediction drives honeypot redirection + predictive alerts
    pred = enterprise_predict(
        inp=EnterprisePredictInput(
            request_rate=int(request_rate),
            failed_logins=int(failed_logins),
            unknown_ip=int(unknown_ip),
            time_of_day=int(time_of_day),
            traffic_spike=int(env.traffic_data[env._idx - 1].get("traffic_spike", 0)) if hasattr(env, "traffic_data") else 0,
            session_duration=int(env.traffic_data[env._idx - 1].get("session_duration", 0)) if hasattr(env, "traffic_data") else 0,
            packet_size=int(env.traffic_data[env._idx - 1].get("packet_size", 0)) if hasattr(env, "traffic_data") else 0,
            repeated_requests=int(env.traffic_data[env._idx - 1].get("repeated_requests", 0)) if hasattr(env, "traffic_data") else 0,
            ip_reputation_score=int(env.traffic_data[env._idx - 1].get("ip_reputation_score", 50)) if hasattr(env, "traffic_data") else 50,
            session_id=int(session_id),
        )
    )


    ds.step_idx += 1

    is_intrusion = bool(info.get("is_intrusion"))
    correct = bool(info.get("correct"))

    ds.evaluated_total += 1
    if correct:
        ds.correct_total += 1

    if is_intrusion:
        ds.attack_total += 1
    else:
        ds.normal_total += 1

    if action == 1:
        ds.block_count += 1
    else:
        ds.allow_count += 1

    # Update risk window (intrusion occurrence, not correctness)
    st.session_state._risk_window.append(1.0 if is_intrusion else 0.0)

    # Update log using enterprise contract (predictive + honeypot + multi-agent)
    risk_score = int(pred.get("risk_score", 0))
    attack_type = str(pred.get("attack_type", ""))
    explanation = str(pred.get("explanation", ""))

    honeypot_status = bool(pred.get("honeypot_status", False))
    honeypot_reason = str(pred.get("honeypot_reason", ""))
    early_warning = bool(pred.get("early_warning", False))

    action_str = str(pred.get("action", "ALLOW"))

    reasons: List[str] = []
    if early_warning:
        reasons.append("Early warning: predicted near-future risk is high")
    if honeypot_status:
        reasons.append("Honeypot redirection active")
        if honeypot_reason:
            reasons.append(f"Reason: {honeypot_reason}")
    reasons.append(explanation)


    fieldnames = [
        "timestamp",
        "request_rate",
        "failed_logins",
        "unknown_ip",
        "time_of_day",
        "action",
        "risk_score",
        "attack_type",
        "explanation",
        "honeypot_status",
        "honeypot_reason",
        "early_warning",
        "ground_truth",
        "correct",
        "source",
    ]


    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "request_rate": request_rate,
        "failed_logins": failed_logins,
        "unknown_ip": unknown_ip,
        "time_of_day": time_of_day,
        "action": action_str,
        "risk_score": risk_score,
        "attack_type": attack_type,
        "explanation": explanation,
        "honeypot_status": honeypot_status,
        "honeypot_reason": honeypot_reason,
        "early_warning": early_warning,
        "ground_truth": "ATTACK" if is_intrusion else "NORMAL",
        "correct": "YES" if correct else "NO",
        "source": "dashboard",
    }


    append_prediction_to_csv(row, csv_path="predictions_log.csv", fieldnames=fieldnames)

    # Update log shown in UI

    ds.log.append(
        {
            "Step": ds.step_idx,
            "Action": action_str,
            "GroundTruth": "ATTACK" if is_intrusion else "NORMAL",
            "Correct": "YES" if correct else "NO",
            "RiskScore": risk_score,
            "AttackType": attack_type,
            "Explanation": explanation,
            "Honeypot": "ON" if honeypot_status else "OFF",
            "HoneypotReason": honeypot_reason,
            "EarlyWarning": "YES" if early_warning else "NO",
        }
    )



    # Accuracy point
    acc = ds.correct_total / ds.evaluated_total if ds.evaluated_total else 0.0
    ds.accuracy_points.append((ds.step_idx, acc))

    # Persist state
    st.session_state._state = next_state

    # Stop conditions
    if done or ds.step_idx >= int(max_steps_per_run):
        ds.running = False


# Run a small number of ticks per rerun for responsiveness.
# Streamlit reruns top-to-bottom; we use an elapsed loop.
if st.session_state.dash_state.running:
    # Use a bounded number of ticks to avoid blocking the UI too long.
    tick_budget = 10
    for _ in range(tick_budget):
        if not st.session_state.dash_state.running:
            break
        _tick_once()
        if speed_ms > 0:
            time.sleep(speed_ms / 1000.0)
    # Force refresh to update UI
    st.rerun()

