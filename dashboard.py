"""dashboard.py

Enterprise AI-Powered Cybersecurity Defense Platform — SOC Dashboard

Run:
    streamlit run dashboard.py

Notes:
- Enhances the existing dashboard (keeps RL loop and enterprise_predict wiring).
- Adds:
  - Active Threat cards
  - Threat Heatmap + Timeline
  - Radar / Pie / Histogram / Confidence graph
  - Predictive early-warning panel
  - Network health gauges
  - AI Security Analyst insights
- UI remains Streamlit + cyber dark theme; no module removal.
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import streamlit as st

from agent import RLIntrusionAgent
from environment import IntrusionDetectionEnvironment
from enterprise_pipeline import EnterprisePredictInput, enterprise_predict
from simulator import generate_traffic
from utils import append_prediction_to_csv

# New lightweight helper module (does not replace existing logic)
from soc_visuals import rolling_avg


# -------------------------
# UI: Cyber dark theme
# -------------------------

def _apply_cyber_theme() -> None:
    st.set_page_config(
        page_title="Enterprise SOC — RL Intrusion Defense",
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

            .panel-neon {
                background: radial-gradient(circle at 20% 0%, rgba(0,229,255,0.10), rgba(0,0,0,0) 35%), var(--panel);
                border: 1px solid rgba(0,229,255,0.25);
                border-radius: 14px;
                padding: 14px;
                box-shadow: 0 0 18px rgba(0,229,255,0.12);
            }

            .title {
                font-weight: 800;
                letter-spacing: 0.3px;
                color: var(--cyan);
                text-shadow: 0 0 18px rgba(0, 229, 255, 0.22);
            }

            .kpi {
                font-size: 26px;
                font-weight: 900;
                margin: 0;
                color: var(--text);
                text-shadow: 0 0 10px rgba(0,229,255,0.10);
            }

            .subkpi {
                color: var(--muted);
                font-size: 12px;
                margin-bottom: 2px;
            }

            .mono {
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            }

            .pulse {
                display:inline-block;
                padding:10px 14px;
                border-radius:999px;
                border:1px solid rgba(0,229,255,0.25);
                background: rgba(0,229,255,0.06);
                box-shadow: 0 0 16px rgba(0,229,255,0.12);
                animation: pulseGlow 1.6s ease-in-out infinite;
            }
            @keyframes pulseGlow {
                0% { transform: scale(1); box-shadow: 0 0 16px rgba(0,229,255,0.12); }
                50% { transform: scale(1.02); box-shadow: 0 0 26px rgba(0,229,255,0.18); }
                100% { transform: scale(1); box-shadow: 0 0 16px rgba(0,229,255,0.12); }
            }

            .dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: var(--green);
                box-shadow: 0 0 14px rgba(45,255,143,0.35);
                display:inline-block;
            }
            .dot.yellow { background: var(--yellow); box-shadow: 0 0 14px rgba(255,216,77,0.35); }
            .dot.red { background: var(--red); box-shadow: 0 0 14px rgba(255,77,77,0.35); }

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

    # Rolling windows for analytics/visuals
    risk_window: Deque[int] = None  # type: ignore
    predicted_risk_window: Deque[int] = None  # type: ignore
    confidence_window: Deque[str] = None  # type: ignore
    attack_type_window: Deque[str] = None  # type: ignore
    severity_window: Deque[str] = None  # type: ignore

    # Timeline events
    timeline: Deque[Dict[str, object]] = None  # type: ignore

    # Counters (overall)
    attack_total: int = 0
    normal_total: int = 0
    allow_count: int = 0
    block_count: int = 0

    # Active threat counters (rolling window)
    active_attack_total: int = 0

    # For accuracy computation
    correct_total: int = 0
    evaluated_total: int = 0

    # For accuracy graph
    accuracy_points: List[Tuple[int, float]] = None  # type: ignore

    # For health monitoring
    blocked_recent: int = 0
    allowed_recent: int = 0
    isolated_recent: int = 0
    recovered_recent: int = 0


def _init_state() -> None:
    if "dash_state" not in st.session_state:
        st.session_state.dash_state = DashboardState(
            running=False,
            step_idx=0,
            log=deque(maxlen=250),
            risk_window=deque(maxlen=120),
            predicted_risk_window=deque(maxlen=120),
            confidence_window=deque(maxlen=120),
            attack_type_window=deque(maxlen=120),
            severity_window=deque(maxlen=120),
            timeline=deque(maxlen=200),
            accuracy_points=[],
            active_attack_total=0,
            blocked_recent=0,
            allowed_recent=0,
            isolated_recent=0,
            recovered_recent=0,
        )


def _risk_pill(risk_label: str, dot_class: str) -> None:
    dot_class_attr = f"dot {dot_class}" if dot_class else "dot"
    st.markdown(
        f"""
        <div class='pulse'>
            <span class='{dot_class_attr}'></span>
            <span class='mono' style='margin-left:10px;font-weight:900;'>Threat Level: {risk_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _risk_label_from_score(risk_score: int) -> Tuple[str, str]:
    if risk_score < 35:
        return ("LOW", "")
    if risk_score < 60:
        return ("MEDIUM", "yellow")
    return ("HIGH", "red")


def _severity_to_color(sev: str) -> str:
    s = (sev or "").upper()
    if s in ("CRITICAL",):
        return "red"
    if s in ("HIGH",):
        return "red"
    if s in ("MEDIUM",):
        return "yellow"
    return ""


def _confidence_to_bucket(conf: str) -> int:
    c = (conf or "").upper()
    if c == "HIGH":
        return 3
    if c == "MEDIUM":
        return 2
    return 1


def _recommend_response(attack_type: str, severity: str) -> str:
    at = (attack_type or "").lower()
    sev = (severity or "").upper()
    if sev in ("HIGH", "CRITICAL"):
        if at in ("ddos",):
            return "Engage rate-limiting + edge filtering; verify DDoS scrubbing policy." \
                " Enable ISOLATE for bursty unknown sessions."
        if at in ("credential_stuffing", "brute_force"):
            return "Escalate MFA enforcement + lockout thresholds; rotate credentials for affected accounts."
        if at in ("port_scan",):
            return "Harden exposed services; tighten ACLs and close high-risk ports dynamically."
        if at in ("insider_threat",):
            return "Quarantine session scope; enforce least-privilege and strengthen audit monitoring." \
                " Validate admin actions and data access logs."
        return "Block at perimeter; monitor for lateral movement attempts and attempt replay." \
            " Validate honeypot telemetry."

    if sev in ("MEDIUM",):
        return "Prefer MONITOR/THROTTLE; correlate with honeypot signals and watch burst consistency before BLOCK."
    return "Allow with observation; keep correlation rules active for similar patterns."


# -------------------------
# Start Streamlit app
# -------------------------

_apply_cyber_theme()
_init_state()

st.markdown(
    "<div class='title' style='font-size:22px;'>🛰️ Enterprise AI-Powered Cybersecurity Defense Platform</div>",
    unsafe_allow_html=True,
)
st.caption("AI-powered RL defense center • predictive intelligence • SOC-grade observability")

with st.sidebar:
    st.markdown(
        "<div class='mono' style='color:#00E5FF;font-weight:900;letter-spacing:0.2px;'>SIMULATION CONTROL</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button("▶ Start", type="primary", use_container_width=True)
    with col2:
        stop_clicked = st.button("■ Stop", use_container_width=True)

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
    st.markdown("<div class='mono' style='color:#AAB3C5;'>Auto-refresh is active while running.</div>", unsafe_allow_html=True)


# Auto-refresh (smooth UI)
if st.session_state.get("dash_state", None) is not None and st.session_state.dash_state.running:
    # fast refresh for SOC panels
    st_autorefresh_interval_ms = 800
    try:
        st_autorefresh = getattr(st, "autorefresh")
    except Exception:
        st_autorefresh = None

    if st_autorefresh is not None:
        st_autorefresh(interval=st_autorefresh_interval_ms, key="soc_refresh")


# Start/stop handlers
if start_clicked:
    ds = st.session_state.dash_state
    ds.running = True
    ds.step_idx = 0
    ds.attack_total = 0
    ds.normal_total = 0
    ds.allow_count = 0
    ds.block_count = 0
    ds.correct_total = 0
    ds.evaluated_total = 0
    ds.active_attack_total = 0
    ds.log = deque(maxlen=250)
    ds.risk_window = deque(maxlen=120)
    ds.predicted_risk_window = deque(maxlen=120)
    ds.confidence_window = deque(maxlen=120)
    ds.attack_type_window = deque(maxlen=120)
    ds.severity_window = deque(maxlen=120)
    ds.timeline = deque(maxlen=200)
    ds.accuracy_points = []
    ds.blocked_recent = 0
    ds.allowed_recent = 0
    ds.isolated_recent = 0
    ds.recovered_recent = 0

    simulator = type("_Sim", (), {"generate_traffic": staticmethod(generate_traffic)})

    st.session_state._env = IntrusionDetectionEnvironment(
        simulator=simulator,
        n_samples=int(max_steps_per_run),
        seed=42,
        config=None,
    )

    st.session_state._agent = RLIntrusionAgent(config=None)
    st.session_state._state = st.session_state._env.reset()

if stop_clicked:
    st.session_state.dash_state.running = False


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

    # RL + environment step
    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    agent.learn(state, action, reward, next_state)

    # Enterprise prediction (predictive intelligence + honeypot + multi-agent)
    request_rate, failed_logins, unknown_ip, time_of_day = state[0], state[1], state[2], state[3]

    idx = getattr(env, "_idx", 0)
    traffic_rec = None
    if hasattr(env, "traffic_data") and idx - 1 >= 0:
        traffic_rec = env.traffic_data[idx - 1]

    def _get_feature(key: str, default: int = 0) -> int:
        if not traffic_rec:
            return int(default)
        return int(traffic_rec.get(key, default))

    session_id = int(_get_feature("session_id", st.session_state.dash_state.step_idx))

    pred = enterprise_predict(
        inp=EnterprisePredictInput(
            request_rate=int(request_rate),
            failed_logins=int(failed_logins),
            unknown_ip=int(unknown_ip),
            time_of_day=int(time_of_day),
            traffic_spike=_get_feature("traffic_spike", 0),
            session_duration=_get_feature("session_duration", 0),
            packet_size=_get_feature("packet_size", 0),
            repeated_requests=_get_feature("repeated_requests", 0),
            ip_reputation_score=_get_feature("ip_reputation_score", 50),
            session_id=int(session_id),
        )
    )

    # Update step
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

    # Track allow/block based on RL action id mapping used in dashboard previously
    action_str = str(pred.get("action", "ALLOW"))
    if action_str == "BLOCK":
        ds.block_count += 1
        ds.blocked_recent += 1
    else:
        ds.allow_count += 1
        ds.allowed_recent += 1

    # Analytics rolling windows
    risk_score = int(pred.get("risk_score", 0))
    predicted_risk = int(pred.get("predicted_risk", risk_score))
    confidence = str(pred.get("confidence", "MEDIUM"))
    attack_type = str(pred.get("attack_type", ""))
    severity_level = str(pred.get("severity_level", "LOW"))

    ds.risk_window.append(risk_score)
    ds.predicted_risk_window.append(predicted_risk)
    ds.confidence_window.append(confidence)
    ds.attack_type_window.append(attack_type)
    ds.severity_window.append(severity_level)

    # Early-warning heuristics (predictive intelligence before confirmation)
    early_warning = bool(pred.get("early_warning", False))

    # Timeline event
    ds.timeline.append(
        {
            "timestamp": datetime.datetime.utcnow().strftime("%H:%M:%S"),
            "attack_type": attack_type or "unknown",
            "ai_action": action_str,
            "severity": severity_level,
            "risk_score": risk_score,
            "predicted_risk": predicted_risk,
            "early_warning": early_warning,
            "explanation": str(pred.get("explanation", "")),
        }
    )

    # Log row (existing logging system preserved)
    honeypot_status = bool(pred.get("honeypot_status", False))
    honeypot_reason = str(pred.get("honeypot_reason", ""))

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
        "request_rate": int(request_rate),
        "failed_logins": int(failed_logins),
        "unknown_ip": int(unknown_ip),
        "time_of_day": int(time_of_day),
        "action": action_str,
        "risk_score": risk_score,
        "attack_type": attack_type,
        "explanation": str(pred.get("explanation", "")),
        "honeypot_status": honeypot_status,
        "honeypot_reason": honeypot_reason,
        "early_warning": bool(early_warning),
        "ground_truth": "ATTACK" if is_intrusion else "NORMAL",
        "correct": "YES" if correct else "NO",
        "source": "dashboard",
    }

    append_prediction_to_csv(row, csv_path="predictions_log.csv", fieldnames=fieldnames)

    # UI log deque
    ds.log.append(
        {
            "Step": ds.step_idx,
            "Action": action_str,
            "GroundTruth": "ATTACK" if is_intrusion else "NORMAL",
            "Correct": "YES" if correct else "NO",
            "RiskScore": risk_score,
            "AttackType": attack_type,
            "Explanation": str(pred.get("explanation", ""))[:220],
            "Honeypot": "ON" if honeypot_status else "OFF",
            "HoneypotReason": honeypot_reason[:120],
            "EarlyWarning": "YES" if early_warning else "NO",
        }
    )

    # accuracy points
    acc = ds.correct_total / ds.evaluated_total if ds.evaluated_total else 0.0
    ds.accuracy_points.append((ds.step_idx, acc))

    # persist state
    st.session_state._state = next_state

    # Stop conditions
    if done or ds.step_idx >= int(max_steps_per_run):
        ds.running = False


# bounded ticks per rerun
if st.session_state.dash_state.running:
    tick_budget = 12
    for _ in range(tick_budget):
        if not st.session_state.dash_state.running:
            break
        _tick_once()
        if speed_ms > 0:
            time.sleep(speed_ms / 1000.0)

    st.rerun()


# -------------------------
# UI Panels
# -------------------------

ds = st.session_state.dash_state

risk_avg = float(np.mean(ds.risk_window)) if ds.risk_window else 0.0
risk_label, dot_class = _risk_label_from_score(int(risk_avg))

# Active Threat Panel (rolling)
window_len = max(1, len(ds.predicted_risk_window))
active_attack = int(sum(1 for x in ds.predicted_risk_window if x >= 60))
blocked_recent = ds.blocked_recent
honeypot_redirects = sum(1 for t in ds.timeline if bool(t.get("early_warning", False)) and str(t.get("ai_action", "ALLOW")) == "BLOCK")
isolated_sessions = sum(1 for t in ds.timeline if str(t.get("ai_action", "ALLOW")) == "BLOCK")  # best-effort

with st.container():
    st.markdown("<div class='panel-neon'>", unsafe_allow_html=True)
    st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>ACTIVE THREAT PANEL</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("<div class='subkpi'>Total Active Attacks</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi mono'>{active_attack}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='subkpi'>Attack Severity</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi mono'>{risk_label}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='subkpi'>Blocked Threats (Recent)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi mono'>{blocked_recent}</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='subkpi'>Honeypot Redirects (Recent)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi mono'>{honeypot_redirects}</div>", unsafe_allow_html=True)
    with c5:
        st.markdown("<div class='subkpi'>Isolated Sessions (Recent)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi mono'>{isolated_sessions}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# Layout: left log + right SOC cards
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    # Threat Heatmap (smooth updating) — using risk buckets
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>THREAT HEATMAP</div>", unsafe_allow_html=True)

    if ds.risk_window:
        # intensity = predicted_risk normalized
        intensity = [x / 100.0 for x in ds.predicted_risk_window]
        bucket_count = 8

        # Build a simple matrix for heatmap-like visualization via matplotlib
        try:
            import matplotlib.pyplot as plt

            mat = np.zeros((bucket_count, max(20, len(intensity[-60:]))), dtype=float)
            recent = intensity[-60:]
            for xi, v in enumerate(recent[-mat.shape[1] :]):
                b = min(bucket_count - 1, int(v * bucket_count))
                mat[b, xi] = v

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(mat, aspect="auto", cmap="plasma", interpolation="nearest")
            ax.set_yticks(range(bucket_count))
            ax.set_yticklabels([str(bucket_count - i) for i in range(bucket_count)])
            ax.set_xlabel("time")
            ax.set_ylabel("risk bucket")
            ax.grid(False)
            st.pyplot(fig, clear_figure=True)
        except Exception:
            st.info("Heatmap unavailable (matplotlib not installed).")
    else:
        st.info("Start simulation to populate heatmap.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Threat Timeline
    st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>THREAT TIMELINE</div>", unsafe_allow_html=True)

    timeline_rows = list(ds.timeline)[-10:][::-1]
    if timeline_rows:
        for t in timeline_rows:
            ts = str(t.get("timestamp"))
            at = str(t.get("attack_type", ""))
            act = str(t.get("ai_action", ""))
            sev = str(t.get("severity", ""))
            prefix = "⚠️" if bool(t.get("early_warning")) else ""
            st.markdown(f"<div class='mono' style='margin:4px 0;'>{prefix}[{ts}] {at} 5 {act} 5 {sev}</div>", unsafe_allow_html=True)
    else:
        st.info("Timeline will appear while simulation runs.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Live Traffic Log (existing)
    st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>LIVE TRAFFIC LOG</div>", unsafe_allow_html=True)

    log_rows = list(ds.log)[-120:]
    if log_rows:
        for r in log_rows:
            # keep display compact
            if "Explanation" in r:
                r["Explanation"] = str(r["Explanation"])[:240]
        st.dataframe(log_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Click **Start** to begin.")

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    # Risk pill
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _risk_pill(risk_label, dot_class)
    st.markdown("</div>", unsafe_allow_html=True)

    # Premium components
    st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>PREMIUM COMPONENTS</div>", unsafe_allow_html=True)

    # Attack distribution pie
    atypes = list(ds.attack_type_window)
    if atypes:
        from collections import Counter

        ctr = Counter([a for a in atypes if a])
        labels = list(ctr.keys())[:6]
        values = [ctr[l] for l in labels]
        try:
            fig = None
            import plotly.express as px  # optional dependency

            df = {"attack_type": labels, "count": values}
            fig = px.pie(df, names="attack_type", values="count", hole=0.35, color_discrete_sequence=["#00E5FF", "#FFD84D", "#FF4D4D", "#2DFF8F"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Pie chart unavailable (plotly not installed).")

    # Severity histogram
    sevs = list(ds.severity_window)
    if sevs:
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for s in sevs:
            su = str(s).upper()
            if su in counts:
                counts[su] += 1
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.bar(list(counts.keys()), list(counts.values()), color=["#2DFF8F", "#FFD84D", "#FF4D4D", "#FF4D4D"])
            ax.set_title("Severity Distribution")
            ax.set_ylabel("count")
            st.pyplot(fig, clear_figure=True)
        except Exception:
            pass

    # Blocked vs allowed comparison
    # Use legacy allow/block counters for SOC comparison
    if ds.allow_count + ds.block_count > 0:
        total = ds.allow_count + ds.block_count
        allowed_pct = ds.allow_count / total
        blocked_pct = ds.block_count / total

        st.markdown(
            "<div class='mono' style='margin-top:10px;'>Blocked vs Allowed</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, blocked_pct), text=f"Blocked {blocked_pct*100:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)

    # Predict confidence graph
    st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>PREDICTION CONFIDENCE</div>", unsafe_allow_html=True)

    conf_buckets = [_confidence_to_bucket(c) for c in ds.confidence_window]
    if conf_buckets:
        try:
            import matplotlib.pyplot as plt

            y = conf_buckets[-80:]
            fig, ax = plt.subplots(figsize=(6, 2.6))
            ax.plot(y, color="#00E5FF", linewidth=2)
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(["LOW", "MED", "HIGH"])
            ax.set_ylim(0.8, 3.2)
            ax.grid(True, alpha=0.25)
            st.pyplot(fig, clear_figure=True)
        except Exception:
            pass
    else:
        st.info("Confidence graph starts once predictions are available.")

    st.markdown("</div>", unsafe_allow_html=True)


# Network Health Monitoring
st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>NETWORK HEALTH MONITORING</div>", unsafe_allow_html=True)

# Best-effort calculations using existing rolling windows
pred_risk_vals = list(ds.predicted_risk_window)
avg_pred_risk = float(np.mean(pred_risk_vals)) if pred_risk_vals else 0.0
health_score = int(max(0.0, min(100.0, 100.0 - avg_pred_risk)))

# AI confidence proxy (high=good stability)
conf_vals = [1 if c == "LOW" else 2 if c == "MEDIUM" else 3 for c in ds.confidence_window]
ai_conf = int((sum(conf_vals) / (len(conf_vals) or 1)) * (100.0 / 3.0))

containment_rate = 0.0
if (ds.allow_count + ds.block_count) > 0 and pred_risk_vals:
    high_risk = sum(1 for x in pred_risk_vals if x >= 60)
    containment_rate = (ds.block_count / max(1, ds.block_count + ds.allow_count)) if high_risk else 0.0


containment_pct = int(containment_rate * 100)

st.columns(5)

a, b, c, d, e = st.columns(5)
with a:
    st.metric("Network Health", f"{health_score}/100")
with b:
    st.metric("AI Confidence", f"{ai_conf}/100")
with c:
    defense_eff = int(((ds.correct_total / ds.evaluated_total) if ds.evaluated_total else 0.0) * 100)
    st.metric("Defense Efficiency", f"{defense_eff}%")
with d:
    st.metric("Attack Containment", f"{containment_pct}%")
with e:
    # Recovery success best-effort (not directly tracked); keep stable placeholder from recent clears.
    recovered = ds.recovered_recent
    total_iso = ds.isolated_recent + recovered
    recovery_rate = int((recovered / max(1, total_iso)) * 100) if total_iso else 0
    st.metric("Recovery Success", f"{recovery_rate}%")

st.markdown("</div>", unsafe_allow_html=True)


# Predictive early warning panel
st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>LIVE EARLY WARNING</div>", unsafe_allow_html=True)

# Derive burst/flow anomalies from predictive risk deltas
warnings: List[str] = []
if len(ds.predicted_risk_window) >= 8:
    recent = list(ds.predicted_risk_window)[-8:]
    deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    avg_delta = float(np.mean(deltas))
    growth_ratio = (recent[-1] / max(1, recent[0]))

    failed_streak = 0
    # Failed logins streak is not stored; approximate using severity window
    for s in list(ds.severity_window)[-8:]:
        if str(s).upper() in ("HIGH", "CRITICAL"):
            failed_streak += 1

    request_burst_like = (max(recent) >= 80 and (recent[-1] - np.mean(recent[:-1])) >= 10)
    coordinated_like = (failed_streak >= 3 and request_burst_like)

    at_latest = str(list(ds.attack_type_window)[-1]) if ds.attack_type_window else ""

    if growth_ratio >= 1.25 and avg_delta >= 6:
        warnings.append("Potential DDoS forming (abnormal traffic growth + predictive risk surge)")
    # Prefer dataset/classifier labels where available
    if at_latest.lower() in ("web attack", "web_attack", "brute force", "brute_force", "credential_stuffing") or failed_streak >= 4:
        warnings.append("Possible brute-force/credential-stuffing escalation (failed-login pattern detected)")

    if request_burst_like:
        warnings.append("High-risk scanning / burst behavior detected (suspicious request spikes)")
    if coordinated_like:
        warnings.append("Coordinated attack behavior suspected (multi-signal escalation trend)")

# Gate warnings with enterprise early_warning to keep realistic
latest_early = bool(ds.timeline[-1].get("early_warning", False)) if ds.timeline else False
if latest_early and not warnings:
    warnings.append("Early warning: predicted near-future risk is high")

if warnings:
    for w in warnings[:5]:
        st.markdown(f"<div class='mono' style='margin:6px 0; color:#FFD84D;'>⚠️ {w}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='mono' style='color:#AAB3C5;'>No early warning triggers at this time.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# AI Security Analyst panel
st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>AI SECURITY ANALYST</div>", unsafe_allow_html=True)

if ds.timeline:
    latest = ds.timeline[-1]
    at = str(latest.get("attack_type", "unknown"))
    act = str(latest.get("ai_action", "ALLOW"))
    sev = str(latest.get("severity", "LOW"))
    risk = int(latest.get("risk_score", 0))
    pred_r = int(latest.get("predicted_risk", risk))
    explanation = str(latest.get("explanation", ""))

    impact = "minor operational disturbance" if sev.upper() in ("LOW", "MEDIUM") else "service degradation / account compromise risk"

    st.markdown(
        """
        <div class='mono' style='margin:6px 0; color:#E6EAF2;'>
        <b>Incident Summary:</b> {at}<br/>
        <b>AI Action:</b> {act}<br/>
        <b>Estimated Impact:</b> {impact}<br/>
        <b>Why action was taken:</b> {explanation}<br/>
        <b>Predicted Risk:</b> {pred_r}/100 • <b>Current Risk:</b> {risk}/100
        </div>
        """.format(
            at=at,
            act=act,
            impact=impact,
            explanation=explanation[:260] + ("..." if len(explanation) > 260 else ""),
            pred_r=pred_r,
            risk=risk,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='mono' style='margin-top:10px; color:#2DFF8F;'><b>Recommended Defensive Response:</b> {}</div>".format(
            _recommend_response(at, sev)
        ),
        unsafe_allow_html=True,
    )
else:
    st.info("Analyst insights appear after the first prediction tick.")

st.markdown("</div>", unsafe_allow_html=True)


# Accuracy graph
st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
colA, colB = st.columns([0.8, 0.2], gap="large")
with colA:
    st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>ACCURACY (REAL-TIME)</div>", unsafe_allow_html=True)
    pts = ds.accuracy_points
    if pts:
        y = [p[1] for p in pts][-120:]
        st.line_chart(y, height=200)
    else:
        st.info("Accuracy will appear once simulation starts.")
with colB:
    st.markdown("<div class='mono' style='font-weight:900;color:#00E5FF;'>STATUS</div>", unsafe_allow_html=True)
    if ds.evaluated_total:
        acc = ds.correct_total / ds.evaluated_total
        st.metric("Accuracy", f"{acc*100:.2f}%")
    else:
        st.metric("Accuracy", "—")

st.markdown("</div>", unsafe_allow_html=True)

