from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from src.models import AgentInput, NUM_ACTIONS


class CraftaxEnvironment:
    """Stateful wrapper around the JAX-based Craftax gymnax environment.

    Manages JAX PRNGKey and environment state internally so callers
    interact via simple reset()/step(action) methods.
    """

    def __init__(
        self,
        env_name: str = "Craftax-Symbolic-v1",
        seed: int = 42,
        auto_reset: bool = False,
    ) -> None:
        self.env_name = env_name
        self.seed = seed
        self.auto_reset = auto_reset

        self._rng = jax.random.PRNGKey(seed)
        self._env = make_craftax_env_from_name(env_name, auto_reset=auto_reset)
        self._params = self._env.default_params
        self._state = None
        self._step_count = 0

    @property
    def action_space_size(self) -> int:
        return NUM_ACTIONS

    def reset(self) -> AgentInput:
        """Reset the environment and return initial observation."""
        self._rng, rng_reset = jax.random.split(self._rng)
        obs, self._state = self._env.reset(rng_reset, self._params)
        self._step_count = 0

        obs_np = np.asarray(obs)
        return AgentInput.from_craftax(obs=obs_np, step=0)

    def step(self, action: int) -> tuple[AgentInput, float, bool, dict]:
        """Take one step in the environment.

        Returns:
            (agent_input, reward, done, info)
        """
        self._rng, rng_step = jax.random.split(self._rng)
        action_jnp = jnp.int32(action)

        obs, self._state, reward, done, info = self._env.step(
            rng_step, self._state, action_jnp, self._params
        )

        self._step_count += 1
        obs_np = np.asarray(obs)
        reward_float = float(reward)
        done_bool = bool(done)

        # Convert JAX info arrays to numpy
        info_dict = {}
        for k, v in info.items():
            if hasattr(v, "tolist"):
                info_dict[k] = np.asarray(v)
            else:
                info_dict[k] = v

        agent_input = AgentInput.from_craftax(
            obs=obs_np,
            reward=reward_float,
            done=done_bool,
            info=info_dict,
            step=self._step_count,
        )

        return agent_input, reward_float, done_bool, info_dict
