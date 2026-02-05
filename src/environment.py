from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import imageio
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG
from craftax.craftax.renderer import render_craftax_pixels
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
        record: bool = False,
    ) -> None:
        self.env_name = env_name
        self.seed = seed
        self.auto_reset = auto_reset
        self.record = record

        self._rng = jax.random.PRNGKey(seed)
        self._env = make_craftax_env_from_name(env_name, auto_reset=auto_reset)
        self._params = self._env.default_params
        self._state = None
        self._step_count = 0
        self._recorded_states: list = []

        if record:
            self._render_fn = jax.jit(render_craftax_pixels, static_argnums=(1,))

    @property
    def action_space_size(self) -> int:
        return NUM_ACTIONS

    def reset(self) -> AgentInput:
        """Reset the environment and return initial observation."""
        self._rng, rng_reset = jax.random.split(self._rng)
        obs, self._state = self._env.reset(rng_reset, self._params)
        self._step_count = 0
        self._recorded_states.clear()

        if self.record:
            self._recorded_states.append(self._state)

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

        if self.record:
            self._recorded_states.append(self._state)

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

    def save_replay(
        self,
        path: str | Path = "replay.mp4",
        fps: int = 8,
        block_pixel_size: int | None = None,
    ) -> Path:
        """Render recorded states to an MP4 file.

        Args:
            path: Output file path (.mp4)
            fps: Frames per second
            block_pixel_size: Pixel size per tile (default: BLOCK_PIXEL_SIZE_IMG=16)

        Requires: imageio-ffmpeg (pip install imageio[ffmpeg])
        """
        if not self._recorded_states:
            raise RuntimeError("No recorded states. Set record=True and run an episode first.")

        if block_pixel_size is None:
            block_pixel_size = BLOCK_PIXEL_SIZE_IMG

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n_frames = len(self._recorded_states)
        print(f"Rendering {n_frames} frames...")

        writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
        for i, state in enumerate(self._recorded_states):
            pixels = self._render_fn(state, block_pixel_size)
            frame = np.asarray(pixels, dtype=np.uint8)
            writer.append_data(frame)
            if (i + 1) % 100 == 0:
                print(f"  Rendered {i + 1}/{n_frames} frames")
        writer.close()

        print(f"Replay saved to {path} ({n_frames} frames)")
        return path
