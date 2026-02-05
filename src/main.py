from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.agent import AgentFactory
from src.environment import CraftaxEnvironment
from src.runner import GameRunner


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    config_dir = Path(__file__).resolve().parent.parent / "configs"

    agent_cfg = load_config(config_dir / "agent.yaml")
    env_cfg = load_config(config_dir / "env.yaml")

    # Build environment
    environment = CraftaxEnvironment(
        env_name=env_cfg.get("env_name", "Craftax-Symbolic-v1"),
        seed=env_cfg.get("seed", 42),
        auto_reset=env_cfg.get("auto_reset", False),
        record=env_cfg.get("record", False),
    )

    # Build agent via factory
    agent_type = agent_cfg.get("type", "random")
    agent_kwargs = {
        "history_length": agent_cfg.get("history_length", 0),
    }
    if agent_type == "openai":
        agent_kwargs["model"] = agent_cfg.get("model", "gpt-4o-mini")
        agent_kwargs["temperature"] = agent_cfg.get("temperature", 0.7)

    print(f"Creating agent: {agent_type}")
    print(f"Available agents: {AgentFactory.list_agents()}")
    agent = AgentFactory.create(agent_type, **agent_kwargs)

    # Build runner
    runner = GameRunner(
        agent=agent,
        environment=environment,
        max_steps=env_cfg.get("max_steps", 1000),
        verbose=True,
        replay_fps=env_cfg.get("replay_fps", 8),
    )

    # Run
    n_episodes = agent_cfg.get("n_episodes", 1)
    runner.run(n_episodes=n_episodes)


if __name__ == "__main__":
    main()
