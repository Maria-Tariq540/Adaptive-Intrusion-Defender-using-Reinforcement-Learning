"""agent.py

Q-learning agent for intrusion detection (no deep learning).

Requirements satisfied:
- Q-learning table (dict)
- actions: 0 = ALLOW, 1 = BLOCK
- choose_action(state) with epsilon-greedy exploration
- learn(state, action, reward, next_state)

State is expected to be a discretized tuple (see environment.py).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

# RL state is a discretized tuple; length can vary as the environment evolves.
State = Tuple[int, ...]


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
    ACTION_ISOLATE_DEVICE = 4
    ACTION_REDIRECT_TO_HONEYPOT = 5

    ACTIONS = (
        ACTION_ALLOW,
        ACTION_MONITOR,
        ACTION_THROTTLE,
        ACTION_BLOCK,
        ACTION_ISOLATE_DEVICE,
        ACTION_REDIRECT_TO_HONEYPOT,
    )


    def __init__(self, config: object | None = None, **overrides) -> None:
        # Try to read values from config dataclass if provided; otherwise use defaults.
        p = QLearningParams()

        if config is not None:
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

        # Q-table: state -> Q-values list ordered by ACTIONS
        self.q_table: Dict[State, list[float]] = {}


        self.epsilon = float(p.epsilon_start)

        # auto-load trained policy if available
        self.load_q_table()

    def _ensure_state(self, state: State) -> None:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.ACTIONS]


    def choose_action(self, state: State) -> int:
        """Epsilon-greedy action selection over 6 actions.


        Stability improvement:
        - If Q-values are clearly separated, we reduce the effective exploration
          even when epsilon is not yet fully decayed.
        """

        self._ensure_state(state)
        q_values = self.q_table[state]

        # Determine best and runner-up to gauge confidence/stability.
        best_i = int(max(range(len(q_values)), key=lambda i: q_values[i]))
        sorted_q = sorted(q_values, reverse=True)
        best_q = float(sorted_q[0])
        second_q = float(sorted_q[1]) if len(sorted_q) > 1 else best_q

        # If the best action is ahead by margin, keep exploitation.
        # This prevents oscillations caused by random exploration.
        margin = best_q - second_q
        effective_epsilon = float(self.epsilon)
        if margin >= 0.75:
            effective_epsilon *= 0.15

        if self.rng.random() < effective_epsilon:
            return int(self.rng.choice(list(self.ACTIONS)))

        return best_i

    def learn(self, state: State, action: int, reward: float, next_state: State) -> None:
        """Run one Q-learning update.

        NOTE: The environment rewards only support legacy actions (0..3), while the agent
        contains extra enterprise maneuvers (4..5). To keep RL stable and backward
        compatible with the current environment contract, we clamp any action outside
        the environment’s supported set (0..3) to ACTION_BLOCK.
        """

        if int(action) not in self.ACTIONS:
            raise ValueError("action must be one of RL actions 0..5")

        # Environment contract supports only actions 0..3.
        action = int(action)
        if action not in (self.ACTION_ALLOW, self.ACTION_MONITOR, self.ACTION_THROTTLE, self.ACTION_BLOCK):
            action = self.ACTION_BLOCK



        self._ensure_state(state)
        self._ensure_state(next_state)

        q_values = self.q_table[state]
        next_q_values = self.q_table[next_state]

        current_q = float(q_values[int(action)])
        max_next_q = float(max(next_q_values))

        alpha = float(self.params.learning_rate)
        gamma = float(self.params.discount_factor)

        td_target = float(reward) + gamma * max_next_q
        td_error = td_target - current_q
        q_values[int(action)] = current_q + alpha * td_error

        # Epsilon schedule: reduce exploration smoothly but avoid unstable rapid decay.
        new_eps = float(self.epsilon) * float(self.params.epsilon_decay)

        # When near the floor, decay even more slowly to keep learning stable.
        floor = float(self.params.epsilon_end)
        if new_eps < (floor * 1.6):
            new_eps = float(self.epsilon) - (float(self.epsilon) - floor) * 0.25

        self.epsilon = max(floor, float(new_eps))

    def save_q_table(self, path: str | None = None) -> None:
        """Persist Q-table to disk (best-effort)."""
        try:
            import json
            from pathlib import Path

            p = path or self.params.q_table_path
            out = Path(p)
            out.parent.mkdir(parents=True, exist_ok=True)

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
                parts = str(k).split("|")
                if len(parts) < 1:
                    continue
                s = tuple(map(int, parts))
                q_table[s] = list(map(float, v))

            self.q_table = q_table
        except Exception:
            return

    # Backwards-compatible aliases (in case other project code calls these)
    def select_action(self, observation: State) -> int:
        return self.choose_action(observation)

    def observe(self, transition: dict) -> None:
        self.learn(
            transition["state"],
            transition["action"],
            transition["reward"],
            transition["next_state"],
        )

    def train_step(self) -> None:
        # No-op for tabular Q-learning; learning happens in learn().
        return

