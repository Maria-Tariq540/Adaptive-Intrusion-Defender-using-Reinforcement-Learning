"""train.py

Purpose:
    Training entrypoint for the reinforcement learning intrusion detection system.

Notes:
    - This script wires together config, environment, and agent.
    - No training logic is implemented yet.
    - Designed to be runnable step-by-step later.

Typical flow (later):
    1) Load config
    2) Create simulator + environment
    3) Create agent
    4) Run training loop for N episodes
"""

from __future__ import annotations


def main() -> None:
    """Training main entrypoint."""
    import csv
    from pathlib import Path


    import matplotlib.pyplot as plt

    from agent import RLIntrusionAgent
    from config import get_config
    from environment import IntrusionDetectionEnvironment
    from simulator import generate_traffic

    cfg = get_config()

    # --- Reproducibility (best-effort) ---
    try:
        import random

        random.seed(cfg.seed)
    except Exception:
        pass

    # Create shared simulator + environment.
    # Use a fixed traffic stream per episode to keep the task well-defined.
    # You can regenerate per episode later if needed.
    # NOTE: Phase-1 supervised ML is trained on REAL dataset.
    # RL environment remains backward-compatible; next phase will switch it to real stream.
    simulator = type("_Sim", (), {"generate_traffic": staticmethod(generate_traffic)})

    env = IntrusionDetectionEnvironment(
        simulator=simulator,
        n_samples=cfg.max_steps_per_episode,
        seed=cfg.seed,
        config=cfg,
    )


    agent = RLIntrusionAgent(config=cfg)

    episodes = int(cfg.episodes)
    max_steps = int(cfg.max_steps_per_episode)

    episodes_axis: list[int] = []
    accuracies: list[float] = []
    rewards: list[float] = []

    correct_window = []  # track correct decisions within an episode

    out_dir = Path(__file__).resolve().parent
    learning_curve_path = out_dir / "learning_curve.csv"

    # Prepare CSV (append-friendly). Overwrite for a clean run.
    with learning_curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "accuracy", "total_reward"])

    # Explainable tracing + CSV logging (training-safe)
    from explainable_ai import explain_decision_from_state
    from utils import safe_append_prediction_to_csv

    # --- Phase 1: supervised ML classifier on real traffic datasets ---
    from dataset_loader import load_cic_ids_2017_csv, load_nsl_kdd_csv
    from feature_engineering import normalize_attack_class_order
    from ml_attack_detector import train_detector, save_detector, evaluate_and_save_artifacts

    ml_enabled = bool(getattr(cfg, "ml_enable", True))

    class_order = normalize_attack_class_order()

    ml_metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }

    if ml_enabled:
        try:
            cic_path = str(getattr(cfg, "cic_ids_2017_csv_path", "data/CIC-IDS2017.csv"))
            ds_bundle = load_cic_ids_2017_csv(
                cic_path,
                test_size=float(getattr(cfg, "dataset_test_size", 0.2)),
                seed=int(getattr(cfg, "seed", 42)),
                min_rows_required=int(getattr(cfg, "min_rows_required", 2000)),
            )
            # Persist preprocessing artifacts needed for prediction-time compatibility
            # (best-effort: feature columns + label mapping are embedded in MLDetector)


            # If dataset loader produces a different label set, keep only intersecting classes.
            # Also ensure class_order used for confusion matrix matches dataset classes ordering.
            import numpy as np
            dataset_classes = [c for c in np.unique(ds_bundle.train_y) if c in class_order]
            if not dataset_classes:
                dataset_classes = class_order


            detector, metrics, cm = train_detector(
                train_X=ds_bundle.train_X,
                train_y=ds_bundle.train_y,
                test_X=ds_bundle.test_X,
                test_y=ds_bundle.test_y,
                feature_columns=ds_bundle.feature_columns,
                class_order=dataset_classes,
                model_type=str(getattr(cfg, "ml_model_type", "rf")),
                seed=int(getattr(cfg, "seed", 42)),
            )

            ml_model_path = str(getattr(cfg, "ml_detector_model_path", "models/ml_detector.joblib"))
            save_detector(detector, ml_model_path)

            art_dir = out_dir / "models" / "ml_artifacts"
            ml_metrics = evaluate_and_save_artifacts(
                detector,
                test_X=ds_bundle.test_X,
                test_y=ds_bundle.test_y,
                class_order=dataset_classes,
                cm=cm,
                out_dir=art_dir,
            )

            print(
                "[ML] Classifier training completed "
                f"(accuracy={ml_metrics.get('accuracy', 0.0):.4f}, "
                f"f1={ml_metrics.get('f1', 0.0):.4f})"
            )
        except Exception as e:
            print(f"[ML] Classifier training skipped due to error: {e}")


    predictions_fieldnames = [
        "timestamp",
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
        "predicted_action",
        "attack_type",
        "risk_score",
        "confidence",
        "explanation",
        "severity_level",
        "ground_truth",
        "correct",
        "source",
    ]


    # Buffer to avoid per-step disk writes
    prediction_buffer = []
    log_flush_every = 25

    import datetime

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        total_reward = 0.0
        correct = 0
        steps_taken = 0

        while not done and steps_taken < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            # Learn first (keeps the decision trace aligned with the chosen action)
            agent.learn(state, action, reward, next_state)

            total_reward += float(reward)
            steps_taken += 1
            if bool(info.get("correct")):
                correct += 1

            # ---- Explainable artifacts (O(1) rule-based) ----
            # state is (request_rate, failed_logins, unknown_ip, time_of_day)
            # state is now an 8-tuple discretized vector.
            # For explainable_ai we still pass the original 4 raw features.
            rr_bin, fl_bin, ui, tod_bin, traffic_spike, rep_bin, ip_rep_bin, sess_bin = state

            # Recover raw-ish values for explainability (best-effort midpoints).
            # Explainable AI uses these features as signals only; exact bin->value mapping isn't critical.
            rr = int(rr_bin * 20)  # coarse midpoint-ish
            fl = int(fl_bin * 2)
            tod = int(tod_bin * 6)

            explain = explain_decision_from_state(
                (rr, fl, int(ui), tod),
                action_id=action,
                traffic_spike=int(traffic_spike),
                repeated_requests=int(rep_bin * 5),
                rl_confidence=0.5,
            )


            action_str = {
                0: "ALLOW",
                1: "MONITOR",
                2: "THROTTLE",
                3: "BLOCK",
            }.get(int(action), "ALLOW")


            row = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "session_id": int(env.traffic_data[env._idx - 1].get("session_id", -1))
                if hasattr(env, "traffic_data") and (env._idx - 1) >= 0
                else -1,
                "request_rate": int(rr),
                "failed_logins": int(fl),
                "unknown_ip": int(ui),
                "time_of_day": int(tod),
                "traffic_spike": int(env.traffic_data[env._idx - 1].get("traffic_spike", 0))
                if hasattr(env, "traffic_data") and (env._idx - 1) >= 0
                else 0,
                "session_duration": int(env.traffic_data[env._idx - 1].get("session_duration", 0))
                if hasattr(env, "traffic_data") and (env._idx - 1) >= 0
                else 0,
                "packet_size": int(env.traffic_data[env._idx - 1].get("packet_size", 0))
                if hasattr(env, "traffic_data") and (env._idx - 1) >= 0
                else 0,
                "repeated_requests": int(env.traffic_data[env._idx - 1].get("repeated_requests", 0))
                if hasattr(env, "traffic_data") and (env._idx - 1) >= 0
                else 0,
                "ip_reputation_score": int(env.traffic_data[env._idx - 1].get("ip_reputation_score", 50))
                if hasattr(env, "traffic_data") and (env._idx - 1) >= 0
                else 50,
                "predicted_action": action_str,
                "attack_type": explain.attack_type,
                "risk_score": int(explain.risk_score),
                "confidence": getattr(explain, "confidence_level", "MEDIUM"),
                "explanation": explain.explanation,
                "severity_level": getattr(explain, "severity_level", "MEDIUM"),
                "ground_truth": "ATTACK" if bool(info.get("is_intrusion")) else "NORMAL",
                "correct": "YES" if bool(info.get("correct")) else "NO",
                "source": "train",
            }


            # Training-safe buffered logging (never crash training)
            safe_append_prediction_to_csv(
                row,
                csv_path="predictions_log.csv",
                fieldnames=predictions_fieldnames,
                buffer=prediction_buffer,
                flush_every=log_flush_every,
            )

            state = next_state

        # Flush remaining buffered logs per episode (still best-effort)
        from utils import flush_prediction_buffer
        flush_prediction_buffer(
            prediction_buffer,
            csv_path="predictions_log.csv",
            fieldnames=predictions_fieldnames,
        )

        accuracy = correct / steps_taken if steps_taken > 0 else 0.0


        episodes_axis.append(ep)
        accuracies.append(accuracy)
        rewards.append(total_reward)

        with learning_curve_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ep, accuracy, total_reward])

        if ep % 50 == 0 or ep == 1 or ep == episodes:
            avg_reward_last = sum(rewards[max(0, ep - 50):ep]) / max(1, min(50, ep))
            print(
                f"Episode {ep}/{episodes} | accuracy={accuracy:.4f} | total_reward={total_reward:.2f} | "
                f"avg_reward_last50={avg_reward_last:.2f} | epsilon={agent.epsilon:.4f}",
                flush=True,
            )

    # Plot learning curve
    plt.figure(figsize=(9, 5))
    plt.plot(episodes_axis, accuracies, label="Accuracy")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Episode vs Accuracy (RL Intrusion Detection)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / "accuracy_curve.png"
    plt.savefig(plot_path)
    plt.show()

    print(f"Training complete. Learning curve saved to: {learning_curve_path}")
    print(f"Accuracy plot saved to: {plot_path}")

    # Persist learned policy so future runs can reuse it.
    agent.save_q_table()
    print("Saved trained Q-table to: q_table.json")




if __name__ == "__main__":
    main()


