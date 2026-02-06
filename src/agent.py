from __future__ import annotations

import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from openai import OpenAI

logger = logging.getLogger("craftax")

from src.models import (
    ACTION_NAMES,
    NUM_ACTIONS,
    Action,
    AgentInput,
    AgentOutput,
)


class BaseAgent(ABC):
    """Abstract base agent for Craftax. Subclasses implement act()."""

    name: str = "base"

    def __init__(self, history_length: int = 0, **kwargs) -> None:
        self.history_length = history_length
        self._history: List[tuple[AgentInput, AgentOutput]] = []

    @abstractmethod
    def act(self, agent_input: AgentInput) -> AgentOutput:
        """Choose an action given structured input. Must be implemented by subclasses."""
        ...

    def reset(self) -> None:
        """Clear history for a new episode."""
        self._history.clear()

    def __call__(self, agent_input: AgentInput) -> AgentOutput:
        """Public API: calls act() and records history."""
        output = self.act(agent_input)
        self._record_history(agent_input, output)
        return output

    def _record_history(self, agent_input: AgentInput, agent_output: AgentOutput) -> None:
        if self.history_length <= 0:
            return
        # Store deep copies so later mutations can't rewrite history
        input_copy = (
            agent_input.model_copy(deep=True)
            if hasattr(agent_input, "model_copy")
            else agent_input
        )
        output_copy = (
            agent_output.model_copy(deep=True)
            if hasattr(agent_output, "model_copy")
            else agent_output
        )
        self._history.append((input_copy, output_copy))
        if len(self._history) > self.history_length:
            self._history = self._history[-self.history_length :]

    def _format_observation(self, agent_input: AgentInput) -> str:
        """Convert structured input into human-readable text for LLM prompts."""
        lines = []
        lines.append(f"=== Step {agent_input.step} | Floor {agent_input.dungeon_floor} ===")
        lines.append(f"Reward this step: {agent_input.reward}")

        # Intrinsics
        lines.append("\n--- Player Status ---")
        for k, v in agent_input.intrinsics.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"  direction: {agent_input.direction}")
        if agent_input.is_sleeping:
            lines.append("  [SLEEPING]")
        if agent_input.is_resting:
            lines.append("  [RESTING]")

        # Inventory
        lines.append("\n--- Inventory ---")
        for k, v in agent_input.inventory.items():
            if isinstance(v, bool):
                if v:
                    lines.append(f"  {k}: yes")
            elif isinstance(v, str):
                if v != "none":
                    lines.append(f"  {k}: {v}")
            elif v > 0:
                lines.append(f"  {k}: {v}")

        # Potions
        non_zero_potions = {k: v for k, v in agent_input.potions.items() if v > 0}
        if non_zero_potions:
            lines.append("\n--- Potions ---")
            for k, v in non_zero_potions.items():
                lines.append(f"  {k}: {v}")

        # Armor
        equipped = {k: v for k, v in agent_input.armor.items() if v.get("material") != "none"}
        if equipped:
            lines.append("\n--- Armor ---")
            for piece, info in equipped.items():
                ench = f" ({info['enchantment']})" if info.get("enchantment", "none") != "none" else ""
                lines.append(f"  {piece}: {info['material']}{ench}")

        # Spells
        spells = []
        if agent_input.learned_fireball:
            spells.append("fireball")
        if agent_input.learned_iceball:
            spells.append("iceball")
        if spells:
            lines.append(f"\n--- Spells: {', '.join(spells)} ---")

        # Environment
        lines.append(f"\n--- Environment ---")
        lines.append(f"  light: {'day' if agent_input.light_level > 0.5 else 'night'}")
        lines.append(f"  ladder_cleared: {agent_input.ladder_cleared}")
        lines.append(f"  boss_vulnerable: {agent_input.boss_vulnerable}")

        # Nearby
        if agent_input.nearby_mobs:
            lines.append("\n--- Nearby Mobs ---")
            for m in agent_input.nearby_mobs[:15]:
                lines.append(f"  {m}")

        if agent_input.nearby_items:
            lines.append("\n--- Nearby Items ---")
            for it in agent_input.nearby_items:
                lines.append(f"  {it}")

        # Summarize nearby blocks (just notable ones)
        notable_blocks = [
            b for b in agent_input.nearby_blocks
            if not any(skip in b for skip in ("GRASS", "PATH", "STONE", "WALL", "GRAVEL", "SAND"))
        ]
        if notable_blocks:
            lines.append("\n--- Notable Nearby Blocks ---")
            for b in notable_blocks[:20]:
                lines.append(f"  {b}")

        return "\n".join(lines)


class RandomAgent(BaseAgent):
    """Agent that picks a random action each step."""

    name: str = "random"

    def act(self, agent_input: AgentInput) -> AgentOutput:
        action_id = random.randint(0, NUM_ACTIONS - 1)
        return AgentOutput(
            action=action_id,
            reasoning="Random selection",
            confidence=1.0 / NUM_ACTIONS,
        )


SYSTEM_PROMPT = """You are an AI agent playing Craftax, a roguelike survival game. Your goal is to survive as long as possible, gather resources, craft tools and weapons, and descend through dungeon floors.

Game mechanics:
- You explore a 2D grid world across multiple dungeon floors
- Manage health, food, drink, energy, and mana
- Gather resources (wood, stone, coal, iron, diamond, sapphire, ruby)
- Craft pickaxes and swords (wood < stone < iron < diamond)
- Fight monsters in melee, with arrows, or with spells
- Kill 8 monsters per floor to unlock the descending ladder
- Place crafting tables and furnaces to craft advanced items
- Rest to recover energy, sleep to pass night

Strategy tips:
- Prioritize gathering wood and stone early
- Craft a pickaxe first, then a sword
- Keep health, food, and drink above critical levels
- Fight weaker mobs first, avoid strong ones without good gear
- Use DO action to interact with the tile you're facing (mine, chop, attack, pick up)
- Move towards resources and away from danger when under-equipped

Available actions (by ID):
"""

# Append the action list to the system prompt
_action_list = "\n".join(f"  {i}: {name}" for i, name in ACTION_NAMES.items())
SYSTEM_PROMPT += _action_list + """

You MUST respond with a JSON object: {"action": <id>, "reasoning": "<brief explanation>"}
Only output the JSON. No other text."""


class OpenAIAgent(BaseAgent):
    """Agent that uses OpenAI API to decide actions."""

    name: str = "openai"

    def __init__(
        self,
        history_length: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(history_length=history_length, **kwargs)
        self.model = model
        self.temperature = temperature
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self.client = OpenAI(api_key=resolved_key)

    def act(self, agent_input: AgentInput) -> AgentOutput:
        user_prompt = self._build_user_prompt(agent_input)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add history context
        for i, (past_input, past_output) in enumerate(self._history, 1):
            label = f"HISTORY {i}/{len(self._history)}"
            messages.append({
                "role": "user",
                "content": f"[{label}]\n{self._format_observation(past_input)}",
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "action": past_output.action,
                    "reasoning": past_output.reasoning or "",
                }),
            })

        messages.append({"role": "user", "content": f"[CURRENT]\n{user_prompt}"})

        # Log full message array sent to LLM
        logger.info(
            "STEP %d — LLM REQUEST (%d messages, %d history turns)\n%s",
            agent_input.step,
            len(messages),
            len(self._history),
            "\n".join(
                f"--- [{m['role'].upper()}] ---\n{m['content']}"
                for m in messages
            ),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=150,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""

        # Log raw LLM response
        logger.info("STEP %d — LLM RAW RESPONSE\n%s", agent_input.step, content)

        output = self._parse_response(content)

        action_name = ACTION_NAMES.get(output.action, "UNKNOWN")
        logger.info(
            "STEP %d — PARSED OUTPUT: action=%s (%d) | reasoning=%s",
            agent_input.step,
            action_name,
            output.action,
            output.reasoning or "-",
        )

        return output

    def _build_user_prompt(self, agent_input: AgentInput) -> str:
        parts = []

        # Current observation
        parts.append("=== Current Observation ===")
        parts.append(self._format_observation(agent_input))
        parts.append("")
        parts.append('Choose your next action. Respond with JSON: {"action": <id>, "reasoning": "<why>"}')

        return "\n".join(parts)

    def _parse_response(self, content: str) -> AgentOutput:
        """Extract action ID from LLM response."""
        # Try to find JSON-like pattern
        json_match = re.search(r'\{\s*"action"\s*:\s*(\d+)', content)
        if json_match:
            action_id = int(json_match.group(1))
            if 0 <= action_id < NUM_ACTIONS:
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
                reasoning = reasoning_match.group(1) if reasoning_match else None
                return AgentOutput(action=action_id, reasoning=reasoning)

        # Fallback: find any number in valid range
        numbers = re.findall(r"\d+", content)
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num < NUM_ACTIONS:
                return AgentOutput(
                    action=num,
                    reasoning=f"Parsed from response: {content[:100]}",
                )

        # Last resort: NOOP
        return AgentOutput(
            action=Action.NOOP,
            reasoning=f"Failed to parse response, defaulting to NOOP: {content[:100]}",
        )


class AgentFactory:
    """Factory for creating agents by name."""

    _registry: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        cls._registry[name] = agent_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseAgent:
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown agent: '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_agents(cls) -> List[str]:
        return list(cls._registry.keys())


# Register built-in agents
AgentFactory.register("random", RandomAgent)
AgentFactory.register("openai", OpenAIAgent)
