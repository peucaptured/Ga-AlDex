# move_interpreter.py
from typing import Dict, List
import re

def interpret_effects_to_build(
    *,
    rank: int,
    is_special: bool,
    effects: Dict[str, bool],
    area: str | None = None,
    perception: bool = False,
    ranged: bool = False,
    resist_override: str | None = None,
) -> str:
    """
    Converte efeitos marcados pelo jogador em uma build M&M completa.
    """

    parts: List[str] = []

    # ------------------
    # DAMAGE
    # ------------------
    if effects.get("damage"):
        dmg = f"Damage {rank}"
        if is_special:
            dmg += " [Custom 0/r: Intelect Based]"
        if area:
            dmg += f" [Area: {area}]"
        if perception:
            dmg += " [Perception]"
        if ranged:
            dmg += " [Ranged]"
        parts.append(dmg)

    # ------------------
    # AFFLICTION
    # ------------------
    if effects.get("affliction"):
        parts.append(
            f"Linked Affliction {rank} "
            "(Fatigued, Dazed, Stunned; Resisted by Fortitude)"
        )

    # ------------------
    # WEAKEN
    # ------------------
    if effects.get("weaken"):
        targets = []
        if effects.get("weaken_stgr"):
            targets.append("Strength")
        if effects.get("weaken_int"):
            targets.append("Intellect")
        if effects.get("weaken_dodge"):
            targets.append("Dodge")
        if effects.get("weaken_will"):
            targets.append("Will")

        if not targets:
            targets = ["Toughness"]

        tgt = " & ".join(targets)
        parts.append(
            f"Linked Weaken {tgt} {rank} (Resisted by Fortitude)"
        )

    # ------------------
    # HEALING
    # ------------------
    if effects.get("healing"):
        parts.append(f"Healing {rank}")

    # ------------------
    # CREATE / ENVIRONMENT
    # ------------------
    if effects.get("create"):
        parts.append(f"Create {rank}")

    if effects.get("environment"):
        parts.append(f"Environment {rank}")

    # ------------------
    # JOIN
    # ------------------
    return "; ".join(parts)
