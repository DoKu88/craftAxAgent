from __future__ import annotations

import logging
from datetime import datetime

import coolname

from src.agent import BaseAgent
from src.environment import CraftaxEnvironment
from src.models import ACTION_NAMES

logger = logging.getLogger("craftax")


class GameRunner:
    """Runs Craftax episodes with an agent."""

    def __init__(
        self,
        agent: BaseAgent,
        environment: CraftaxEnvironment,
        max_steps: int = 1000,
        verbose: bool = True,
        replay_fps: int = 8,
        log_interval: int = 1,
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.max_steps = max_steps
        self.verbose = verbose
        self.replay_fps = replay_fps
        self.log_interval = log_interval

    def run_episode(self) -> dict:
        """Run a single episode. Returns stats dict."""
        self.agent.reset()
        agent_input = self.environment.reset()

        total_reward = 0.0
        steps = 0

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Starting episode with agent: {self.agent.name}")
            print(f"{'='*50}")

        while steps < self.max_steps:
            # Log agent input before acting
            if self.log_interval and steps % self.log_interval == 0:
                obs_text = self.agent._format_observation(agent_input)
                logger.info("AGENT INPUT (step %d):\n%s", steps, obs_text)

            agent_output = self.agent(agent_input)
            action_name = ACTION_NAMES.get(agent_output.action, "UNKNOWN")

            # Log agent output after acting
            if self.log_interval and steps % self.log_interval == 0:
                logger.info(
                    "AGENT OUTPUT (step %d): action=%s (%d) | reasoning=%s | confidence=%s",
                    steps,
                    action_name,
                    agent_output.action,
                    agent_output.reasoning or "-",
                    f"{agent_output.confidence:.2f}" if agent_output.confidence is not None else "-",
                )

            agent_input, reward, done, info = self.environment.step(agent_output.action)
            total_reward += reward
            steps += 1

            if self.verbose and steps % 100 == 0:
                print(
                    f"  Step {steps}: action={action_name}, "
                    f"reward={reward:.2f}, total={total_reward:.2f}, "
                    f"health={agent_input.intrinsics.get('health', '?')}, "
                    f"floor={agent_input.dungeon_floor}"
                )

            if done:
                break

        if self.verbose:
            print(f"\nEpisode finished: steps={steps}, total_reward={total_reward:.2f}")
            print(f"  Final floor: {agent_input.dungeon_floor}")
            print(f"  Final health: {agent_input.intrinsics.get('health', '?')}")
            n_achieved = sum(1 for v in agent_input.achievements.values() if v)
            print(f"  Achievements: {n_achieved}")

        result = {
            "steps": steps,
            "total_reward": total_reward,
            "dungeon_floor": agent_input.dungeon_floor,
            "done": done,
            "achievements": sum(1 for v in agent_input.achievements.values() if v),
        }

        if self.environment.record:
            slug = coolname.generate_slug(2)  # e.g. "brave-tiger"
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            replay_path = self.environment.save_replay(
                path=f"replays/{slug}_{timestamp}.mp4",
                fps=self.replay_fps,
            )
            result["replay_path"] = str(replay_path)

        return result

    def run(self, n_episodes: int = 1) -> list[dict]:
        """Run multiple episodes and print summary."""
        results = []
        self._episode_num = 0
        for ep in range(n_episodes):
            self._episode_num = ep + 1
            if self.verbose:
                print(f"\n--- Episode {ep + 1}/{n_episodes} ---")
            result = self.run_episode()
            results.append(result)

        if n_episodes > 1 and self.verbose:
            avg_reward = sum(r["total_reward"] for r in results) / n_episodes
            avg_steps = sum(r["steps"] for r in results) / n_episodes
            print(f"\n{'='*50}")
            print(f"Summary over {n_episodes} episodes:")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Avg steps:  {avg_steps:.1f}")
            print(f"{'='*50}")

        return results
