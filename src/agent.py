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


_DIRECTION_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def _parse_offset(entry: str) -> tuple[str, int, int] | None:
    """Parse 'TYPE (+dr,+dc)' into (type_name, dr, dc)."""
    match = re.match(r"^(.+?)\s+\(([+-]?\d+),([+-]?\d+)\)$", entry)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _direction_to(dr: int, dc: int) -> str:
    """Primary direction to reach (dr, dc) from origin."""
    if abs(dr) >= abs(dc):
        return "DOWN" if dr > 0 else "UP"
    return "RIGHT" if dc > 0 else "LEFT"


def _navigation_hints(agent_input: AgentInput) -> list[str]:
    """Compute navigation hints: nearest resources, mobs, items with directions."""
    hints = []
    facing = agent_input.direction
    face_dr, face_dc = _DIRECTION_DELTA.get(facing, (0, 0))

    # Group nearby blocks by type, find nearest of each
    resource_types = {
        "TREE", "COAL", "IRON", "DIAMOND", "SAPPHIRE", "RUBY",
        "WATER", "CRAFTING_TABLE", "FURNACE", "RIPE_PLANT",
        "ENCHANTMENT_TABLE_FIRE", "ENCHANTMENT_TABLE_ICE",
    }
    nearest: dict[str, tuple[int, int, int]] = {}  # type -> (dist, dr, dc)
    for entry in agent_input.nearby_blocks:
        parsed = _parse_offset(entry)
        if not parsed:
            continue
        name, dr, dc = parsed
        if name not in resource_types:
            continue
        dist = abs(dr) + abs(dc)
        if dist == 0:
            continue
        if name not in nearest or dist < nearest[name][0]:
            nearest[name] = (dist, dr, dc)

    for name, (dist, dr, dc) in sorted(nearest.items(), key=lambda x: x[1][0]):
        direction = _direction_to(dr, dc)
        if dist == 1 and dr == face_dr and dc == face_dc:
            hints.append(f"nearest {name}: 1 tile {direction}, facing it → use DO")
        elif dist == 1:
            hints.append(f"nearest {name}: 1 tile {direction} → move {direction} to face it, then DO")
        else:
            hints.append(f"nearest {name}: {dist} tiles, move {direction}")

    # Nearest mob (threat awareness)
    for entry in agent_input.nearby_mobs:
        parsed = _parse_offset(entry)
        if not parsed:
            continue
        name, dr, dc = parsed
        dist = abs(dr) + abs(dc)
        direction = _direction_to(dr, dc)
        flee = _direction_to(-dr, -dc) if (dr, dc) != (0, 0) else "UP"
        hints.append(f"{name}: {dist} tiles {direction} (flee: {flee})")

    # Nearby items (ladders, torches)
    for entry in agent_input.nearby_items:
        parsed = _parse_offset(entry)
        if not parsed:
            continue
        name, dr, dc = parsed
        dist = abs(dr) + abs(dc)
        if dist == 0:
            hints.append(f"{name}: on your tile")
        else:
            direction = _direction_to(dr, dc)
            hints.append(f"{name}: {dist} tiles, move {direction}")

    return hints


class BaseAgent(ABC):
    """Abstract base agent for Craftax. Subclasses implement act()."""

    name: str = "base"

    def __init__(self, history_length: int = 0, **kwargs) -> None:
        self.history_length = history_length
        self._history: List[tuple[AgentInput, AgentOutput]] = []
        self._recent_actions: List[int] = []

    @abstractmethod
    def act(self, agent_input: AgentInput) -> AgentOutput:
        """Choose an action given structured input. Must be implemented by subclasses."""
        ...

    def reset(self) -> None:
        """Clear history for a new episode."""
        self._history.clear()
        self._recent_actions.clear()

    def __call__(self, agent_input: AgentInput) -> AgentOutput:
        """Public API: calls act() and records history."""
        output = self.act(agent_input)
        self._record_history(agent_input, output)
        self._recent_actions.append(output.action)
        if len(self._recent_actions) > 10:
            self._recent_actions = self._recent_actions[-10:]
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

        # Navigation hints (pre-computed directions)
        hints = _navigation_hints(agent_input)
        if hints:
            lines.append("\n--- Navigation Hints ---")
            for h in hints:
                lines.append(f"  {h}")

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

## Movement & Interaction

You are on a 2D grid. Nearby objects are shown as (row, col) offsets from your position:
- Negative row = UP from you, positive row = DOWN
- Negative col = LEFT from you, positive col = RIGHT
- (0, 0) is your current tile

CRITICAL — Movement and facing:
- Moving LEFT/RIGHT/UP/DOWN moves you 1 tile AND sets your facing direction
- DO interacts with the tile you are FACING (chop trees, mine stone, attack mobs, pick up items)
- You MUST face a target before using DO. Example: a tree at (0, +1) is to your RIGHT. Move RIGHT to face it, then DO to chop it.
- If DO has no effect, you are probably not facing your target. Check the Navigation Hints.

## Game Mechanics
- Manage health, food, drink, energy, and mana — if any reaches 0 you weaken or die
- Gather resources: wood (from trees), stone, coal, iron, diamond, sapphire, ruby
- Craft pickaxes and swords at a placed crafting table (wood < stone < iron < diamond)
- Smelt iron at a placed furnace
- Kill 8 monsters per floor to clear the ladder, then DESCEND
- Rest to recover energy, sleep to pass night safely
- Eat ripe plants for food, drink at water tiles

## Strategy
- Gather wood first (chop trees), then stone (mine stone blocks with a pickaxe)
- Craft a wood pickaxe ASAP, then a wood sword for defense
- Keep health/food/drink above 3 — prioritize survival over progression
- Fight passive mobs for easy XP; avoid melee/ranged mobs without a sword
- If a Navigation Hint says "facing it, use DO" — just use DO
- If nothing useful is nearby, explore by moving in a consistent direction

## Available Actions
"""

# Append the action list to the system prompt
_action_list = "\n".join(f"  {i}: {name}" for i, name in ACTION_NAMES.items())
SYSTEM_PROMPT += _action_list + """

## Response Format
Think step-by-step, then respond with JSON:
1. What is my situation? (health, resources, threats)
2. What should I prioritize right now?
3. What is my target and which direction is it?

{"action": <id>, "reasoning": "<your step-by-step reasoning>"}"""


class OpenAIAgent(BaseAgent):
    """Agent that uses OpenAI API to decide actions."""

    name: str = "openai"

    def __init__(
        self,
        history_length: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: str | None = None,
        log_mute_system: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(history_length=history_length, **kwargs)
        self.model = model
        self.temperature = temperature
        self.log_mute_system = log_mute_system
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
        log_messages = messages
        if self.log_mute_system:
            log_messages = [m for m in messages if m["role"] != "system"]
        logger.info(
            "STEP %d — LLM REQUEST (%d messages, %d history turns%s)\n%s",
            agent_input.step,
            len(messages),
            len(self._history),
            ", system prompt muted" if self.log_mute_system else "",
            "\n".join(
                f"--- [{m['role'].upper()}] ---\n{m['content']}"
                for m in log_messages
            ),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=300,
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

        # Stuck detection warning
        if len(self._recent_actions) >= 3:
            last_3 = self._recent_actions[-3:]
            if len(set(last_3)) == 1:
                action_name = ACTION_NAMES.get(last_3[0], "UNKNOWN")
                parts.append(
                    f"WARNING: You have repeated {action_name} for the last {len(last_3)} steps "
                    f"with no change in state. Your action is having no effect. "
                    f"Try something different — you likely need to MOVE to reposition first."
                )
                parts.append("")

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
