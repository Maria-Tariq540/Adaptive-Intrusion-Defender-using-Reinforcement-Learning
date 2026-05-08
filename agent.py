"""agent.py

Q-learning agent for intrusion detection (no deep learning).

Requirements satisfied:
- Q-learning table (dict)
- actions: 0 = ALLOW, 1 = BLOCK
- choose_action(state) with epsilon-greedy exploration
- learn(state, action, reward, next_state)

State is expected to be a 4-tuple:
    [request_rate, failed_logins, unknown_ip, time_of_day]
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple


State = Tuple[int, int, int, int]


@dataclass(frozen=True)
class QLearningParams:
    learning_rate: float = 0.1  # alpha
    discount_factor: float = 0.99  # gamma
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42

    # persistence
    q_table_path: str = "q_table.json"


class RLIntrusionAgent:
    """Tabular Q-learning agent (no deep learning)."""

    # Actions:
    # 0 = ALLOW
    # 1 = MONITOR
    # 2 = THROTTLE
    # 3 = BLOCK
    ACTION_ALLOW = 0
    ACTION_MONITOR = 1
    ACTION_THROTTLE = 2
    ACTION_BLOCK = 3

    ACTIONS = (ACTION_ALLOW, ACTION_MONITOR, ACTION_THROTTLE, ACTION_BLOCK)


    def __init__(self, config: object | None = None, **overrides) -> None:
        # Try to read values from config dataclass if provided; otherwise use defaults.
        p = QLearningParams()

        if config is not None:
            # Best-effort extraction of expected attribute names
            lr = getattr(config, "learning_rate", p.learning_rate)
            gamma = getattr(config, "discount_factor", p.discount_factor)
            eps_start = getattr(config, "epsilon_start", p.epsilon_start)
            eps_end = getattr(config, "epsilon_end", p.epsilon_end)
            eps_decay = getattr(config, "epsilon_decay", p.epsilon_decay)
            seed = getattr(config, "seed", p.seed)
            p = QLearningParams(
                learning_rate=lr,
                discount_factor=gamma,
                epsilon_start=eps_start,
                epsilon_end=eps_end,
                epsilon_decay=eps_decay,
                seed=seed,
            )

        # Apply overrides explicitly provided by caller
        p = QLearningParams(
            learning_rate=overrides.get("learning_rate", p.learning_rate),
            discount_factor=overrides.get("discount_factor", p.discount_factor),
            epsilon_start=overrides.get("epsilon_start", p.epsilon_start),
            epsilon_end=overrides.get("epsilon_end", p.epsilon_end),
            epsilon_decay=overrides.get("epsilon_decay", p.epsilon_decay),
            seed=overrides.get("seed", p.seed),
        )

        self.params = p
        self.rng = random.Random(p.seed)

        # Q-table: state -> [Q(0..3)] in order (ALLOW, MONITOR, THROTTLE, BLOCK)
        self.q_table: Dict[State, list[float]] = {}


        self.epsilon = p.epsilon_start

        # auto-load trained policy if available
        self.load_q_table()



    def _ensure_state(self, state: State) -> None:
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]


    def choose_action(self, state: State) -> int:
        """Epsilon-greedy action selection over 4 actions."""
        self._ensure_state(state)

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(list(self.ACTIONS)))

        q_values = self.q_table[state]
        best_action = int(max(range(len(q_values)), key=lambda i: q_values[i]))
        return best_action

    def learn(self, state: State, action: int, reward: float, next_state: State) -> None:
        """Run one Q-learning update over 4 actions."""
        if action not in self.ACTIONS:
            raise ValueError("action must be 0..3")

        self._ensure_state(state)
        self._ensure_state(next_state)

        q_values = self.q_table[state]
        next_q_values = self.q_table[next_state]

        current_q = q_values[int(action)]
        max_next_q = max(next_q_values)

        alpha = float(self.params.learning_rate)
        gamma = float(self.params.discount_factor)

        td_target = float(reward) + gamma * float(max_next_q)
        td_error = td_target - float(current_q)
        q_values[int(action)] = float(current_q) + alpha * td_error

        self.epsilon = max(self.params.epsilon_end, self.epsilon * self.params.epsilon_decay)

    def save_q_table(self, path: str | None = None) -> None:
        """Persist Q-table to disk (best-effort)."""
        try:
            import json
            from pathlib import Path

            p = path or self.params.q_table_path
            out = Path(p)
            out.parent.mkdir(parents=True, exist_ok=True)

            # Convert tuple keys to strings for JSON.
            serial = {"|".join(map(str, k)): v for k, v in self.q_table.items()}
            out.write_text(json.dumps(serial), encoding="utf-8")
        except Exception:
            return

    def load_q_table(self, path: str | None = None) -> None:
        """Load Q-table if present (best-effort)."""
        try:
            import json
            from pathlib import Path

            p = path or self.params.q_table_path
            fp = Path(p)
            if not fp.exists():
                return

            data = json.loads(fp.read_text(encoding="utf-8"))
            q_table: Dict[State, list[float]] = {}
            for k, v in data.items():
                parts = k.split("|")
                if len(parts) != 4:
                    continue
                s = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
                q_table[s] = list(map(float, v))
            self.q_table = q_table
        except Exception:
            return


    # Backwards-compatible aliases (in case other project code calls these)
    def select_action(self, observation: State) -> int:
        return self.choose_action(observation)

    def observe(self, transition: dict) -> None:
        # transition expected keys: state, action, reward, next_state
        self.learn(
            transition["state"],
            transition["action"],
            transition["reward"],
            transition["next_state"],
        )

    def train_step(self) -> None:
        # No-op for tabular Q-learning; learning happens in learn().
        return


