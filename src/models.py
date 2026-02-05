from __future__ import annotations

import math
from enum import IntEnum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    DO = 5
    SLEEP = 6
    PLACE_STONE = 7
    PLACE_TABLE = 8
    PLACE_FURNACE = 9
    PLACE_PLANT = 10
    MAKE_WOOD_PICKAXE = 11
    MAKE_STONE_PICKAXE = 12
    MAKE_IRON_PICKAXE = 13
    MAKE_WOOD_SWORD = 14
    MAKE_STONE_SWORD = 15
    MAKE_IRON_SWORD = 16
    REST = 17
    DESCEND = 18
    ASCEND = 19
    MAKE_DIAMOND_PICKAXE = 20
    MAKE_DIAMOND_SWORD = 21
    MAKE_IRON_ARMOUR = 22
    MAKE_DIAMOND_ARMOUR = 23
    SHOOT_ARROW = 24
    MAKE_ARROW = 25
    CAST_FIREBALL = 26
    CAST_ICEBALL = 27
    PLACE_TORCH = 28
    DRINK_POTION_RED = 29
    DRINK_POTION_GREEN = 30
    DRINK_POTION_BLUE = 31
    DRINK_POTION_PINK = 32
    DRINK_POTION_CYAN = 33
    DRINK_POTION_YELLOW = 34
    READ_BOOK = 35
    ENCHANT_SWORD = 36
    ENCHANT_ARMOUR = 37
    MAKE_TORCH = 38
    LEVEL_UP_DEXTERITY = 39
    LEVEL_UP_STRENGTH = 40
    LEVEL_UP_INTELLIGENCE = 41
    ENCHANT_BOW = 42


NUM_ACTIONS = 43

ACTION_NAMES: dict[int, str] = {a.value: a.name for a in Action}

# Block types for the map observation
BLOCK_TYPES = [
    "INVALID", "OUT_OF_BOUNDS", "GRASS", "WATER", "STONE", "TREE", "WOOD",
    "PATH", "COAL", "IRON", "DIAMOND", "CRAFTING_TABLE", "FURNACE", "SAND",
    "LAVA", "PLANT", "RIPE_PLANT", "WALL", "DARKNESS", "WALL_MOSS",
    "STALAGMITE", "SAPPHIRE", "RUBY", "CHEST", "FOUNTAIN", "FIRE_GRASS",
    "ICE_GRASS", "GRAVEL", "FIRE_TREE", "ICE_SHRUB",
    "ENCHANTMENT_TABLE_FIRE", "ENCHANTMENT_TABLE_ICE", "NECROMANCER",
    "GRAVE", "GRAVE2", "GRAVE3", "NECROMANCER_VULNERABLE",
]

ITEM_TYPES = ["NONE", "TORCH", "LADDER_DOWN", "LADDER_UP", "LADDER_DOWN_BLOCKED"]

DIRECTION_NAMES = ["LEFT", "RIGHT", "UP", "DOWN"]

ENCHANTMENT_NAMES = {0: "none", 1: "fire", 2: "ice"}

# Observation layout constants
MAP_OBS_ROWS = 9
MAP_OBS_COLS = 11
CHANNELS_PER_CELL = 83
MAP_OBS_SIZE = MAP_OBS_ROWS * MAP_OBS_COLS * CHANNELS_PER_CELL  # 8217
PLAYER_STATE_SIZE = 51
TOTAL_OBS_SIZE = MAP_OBS_SIZE + PLAYER_STATE_SIZE  # 8268

# Inventory field names (indices 0-15 relative to player state start)
INVENTORY_FIELDS = [
    "wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby",
    "sapling", "torches", "arrows", "books", "pickaxe", "sword",
    "sword_enchantment", "bow_enchantment", "bow",
]

POTION_NAMES = ["red", "green", "blue", "pink", "cyan", "yellow"]

INTRINSIC_FIELDS = [
    "health", "food", "drink", "energy", "mana", "xp",
    "dexterity", "strength", "intelligence",
]

ARMOUR_PIECES = ["helmet", "chestplate", "pants", "boots"]


def _denorm_sqrt(v: float) -> int:
    """Reverse sqrt(x)/10 normalization."""
    return round((v * 10.0) ** 2)


def _denorm_div10(v: float) -> float:
    """Reverse x/10 normalization."""
    return round(v * 10.0, 1)


class AgentInput(BaseModel):
    """Structured input to the agent, parsed from Craftax symbolic observation."""

    observation_raw: list[float] = Field(description="Raw 8268-dim observation vector")
    inventory: dict[str, int | float | str | bool] = Field(default_factory=dict)
    potions: dict[str, int] = Field(default_factory=dict)
    intrinsics: dict[str, float] = Field(default_factory=dict)
    direction: str = "UP"
    armor: dict[str, dict[str, str | float]] = Field(default_factory=dict)
    light_level: float = 1.0
    is_sleeping: bool = False
    is_resting: bool = False
    learned_fireball: bool = False
    learned_iceball: bool = False
    dungeon_floor: int = 0
    ladder_cleared: bool = False
    boss_vulnerable: bool = False
    nearby_blocks: list[str] = Field(default_factory=list)
    nearby_mobs: list[str] = Field(default_factory=list)
    nearby_items: list[str] = Field(default_factory=list)
    step: int = 0
    reward: float = 0.0
    done: bool = False
    achievements: dict[str, bool] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_craftax(
        cls,
        obs: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        info: dict | None = None,
        step: int = 0,
    ) -> AgentInput:
        """Parse a flat Craftax symbolic observation into structured fields."""
        obs = np.asarray(obs, dtype=np.float32).flatten()
        raw = obs.tolist()

        # --- Parse map (9x11 grid, 83 channels per cell) ---
        map_obs = obs[:MAP_OBS_SIZE].reshape(MAP_OBS_ROWS, MAP_OBS_COLS, CHANNELS_PER_CELL)
        nearby_blocks = []
        nearby_mobs = []
        nearby_items = []

        center_r, center_c = MAP_OBS_ROWS // 2, MAP_OBS_COLS // 2
        for r in range(MAP_OBS_ROWS):
            for c in range(MAP_OBS_COLS):
                cell = map_obs[r, c]
                light = cell[82]
                if light < 0.5:
                    continue

                dr, dc = r - center_r, c - center_c
                pos_str = f"({dr:+d},{dc:+d})"

                # Block type (channels 0-36)
                block_idx = int(np.argmax(cell[:37]))
                if block_idx > 1:  # skip INVALID/OUT_OF_BOUNDS
                    nearby_blocks.append(f"{BLOCK_TYPES[block_idx]} {pos_str}")

                # Item type (channels 37-41)
                item_idx = int(np.argmax(cell[37:42]))
                if item_idx > 0:  # skip NONE
                    nearby_items.append(f"{ITEM_TYPES[item_idx]} {pos_str}")

                # Mobs (channels 42-81)
                mob_channels = cell[42:82]
                mob_class_names = [
                    "melee_mob", "passive_mob", "ranged_mob",
                    "mob_projectile", "player_projectile",
                ]
                for mi, mob_ch in enumerate(mob_channels):
                    if mob_ch > 0.5:
                        cls_idx = mi // 8
                        type_idx = mi % 8
                        nearby_mobs.append(
                            f"{mob_class_names[cls_idx]}(type={type_idx}) {pos_str}"
                        )

        # --- Parse player state (indices 8217-8267) ---
        ps = obs[MAP_OBS_SIZE:]

        inventory = {
            "wood": _denorm_sqrt(ps[0]),
            "stone": _denorm_sqrt(ps[1]),
            "coal": _denorm_sqrt(ps[2]),
            "iron": _denorm_sqrt(ps[3]),
            "diamond": _denorm_sqrt(ps[4]),
            "sapphire": _denorm_sqrt(ps[5]),
            "ruby": _denorm_sqrt(ps[6]),
            "sapling": _denorm_sqrt(ps[7]),
            "torches": _denorm_sqrt(ps[8]),
            "arrows": _denorm_sqrt(ps[9]),
            "books": round(ps[10] * 2.0),
            "pickaxe_level": round(ps[11] * 4.0),
            "sword_level": round(ps[12] * 4.0),
            "sword_enchantment": ENCHANTMENT_NAMES.get(int(ps[13]), "none"),
            "bow_enchantment": ENCHANTMENT_NAMES.get(int(ps[14]), "none"),
            "has_bow": bool(ps[15] > 0.5),
        }

        potions = {
            name: _denorm_sqrt(ps[16 + i]) for i, name in enumerate(POTION_NAMES)
        }

        intrinsics = {
            name: _denorm_div10(ps[22 + i]) for i, name in enumerate(INTRINSIC_FIELDS)
        }

        # Direction (one-hot at ps[31:35])
        dir_idx = int(np.argmax(ps[31:35]))
        direction = DIRECTION_NAMES[dir_idx] if np.max(ps[31:35]) > 0.5 else "NONE"

        # Armor (ps[35:39] = levels, ps[39:43] = enchantments)
        armor = {}
        armour_level_names = {0: "none", 1: "iron", 2: "diamond"}
        for i, piece in enumerate(ARMOUR_PIECES):
            level = round(ps[35 + i] * 2.0)
            ench = int(ps[39 + i])
            armor[piece] = {
                "material": armour_level_names.get(level, "none"),
                "enchantment": ENCHANTMENT_NAMES.get(ench, "none"),
            }

        achievements = {}
        if info and "achievements" in info:
            ach = info["achievements"]
            if hasattr(ach, "tolist"):
                ach = ach.tolist()
            if isinstance(ach, (list, np.ndarray)):
                achievements = {f"achievement_{i}": bool(v) for i, v in enumerate(ach)}
            elif isinstance(ach, dict):
                achievements = {k: bool(v) for k, v in ach.items()}

        return cls(
            observation_raw=raw,
            inventory=inventory,
            potions=potions,
            intrinsics=intrinsics,
            direction=direction,
            armor=armor,
            light_level=float(ps[43]),
            is_sleeping=bool(ps[44] > 0.5),
            is_resting=bool(ps[45] > 0.5),
            learned_fireball=bool(ps[46] > 0.5),
            learned_iceball=bool(ps[47] > 0.5),
            dungeon_floor=round(ps[48] * 10.0),
            ladder_cleared=bool(ps[49] > 0.5),
            boss_vulnerable=bool(ps[50] > 0.5),
            nearby_blocks=nearby_blocks,
            nearby_mobs=nearby_mobs,
            nearby_items=nearby_items,
            step=step,
            reward=float(reward),
            done=done,
            achievements=achievements,
        )


class AgentOutput(BaseModel):
    """Structured output from the agent."""

    action: int = Field(ge=0, lt=NUM_ACTIONS, description="Action ID (0-42)")
    action_name: str = Field(default="", description="Human-readable action name")
    reasoning: Optional[str] = Field(default=None, description="Agent's reasoning")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score"
    )

    @field_validator("action_name", mode="before")
    @classmethod
    def _set_action_name(cls, v: str, info) -> str:
        if not v and "action" in info.data:
            return ACTION_NAMES.get(info.data["action"], f"UNKNOWN_{info.data['action']}")
        return v
