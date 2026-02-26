# golpe_builder.py
"""
Engine do Golpe Builder: modelo de dados, cálculo de PP,
geração de build string, validação e conversão para formatos de saída.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from powers_data import (
    EFFECTS,
    EXTRAS,
    FLAWS,
    AFFLICTION_CONDITIONS,
    STAT_TARGETS,
    COMBAT_ADVANTAGES,
    ADAPTATION_RULES,
    IMMUNITY_OPTIONS,
    SENSES_OPTIONS,
    MOVEMENT_OPTIONS,
)


# ─────────────────────────────────────────────
# Modelo de Dados
# ─────────────────────────────────────────────

@dataclass
class PowerComponent:
    """Um componente (efeito) individual de um golpe."""
    effect_key: str                          # chave em EFFECTS
    rank: int = 1
    extras: List[Dict[str, Any]] = field(default_factory=list)
    # cada extra: {"key": str, "ranks": int, "description": str}
    flaws: List[Dict[str, Any]] = field(default_factory=list)
    # cada flaw: {"key": str, "ranks": int, "description": str}
    conditions: Optional[Dict[str, str]] = None
    # para affliction: {"degree_1": "Dazed", "degree_2": "Stunned", "degree_3": "Incapacitated"}
    stat_targets: List[str] = field(default_factory=list)
    # para weaken/enhanced_trait: ["Stgr", "Dodge"]
    resist_override: Optional[str] = None
    is_linked: bool = False

    # ── Campos específicos para Immunity ──────────────────────────────────────
    # Lista de itens de imunidade selecionados.
    # Cada item: {"key": str, "label_pt": str, "cost": int,
    #             "custom_label": str (só para key=="custom")}
    # rank é IGNORADO para immunity quando immunity_items estiver preenchida;
    # o rank efetivo = soma dos custos dos itens.
    immunity_items: List[Dict[str, Any]] = field(default_factory=list)

    # ── Campos específicos para Senses ────────────────────────────────────────
    # Lista de sentidos selecionados.
    # Cada item: {"key": str, "label_pt": str, "cost": int,
    #             "ranks": int (para sentidos per_rank),
    #             "custom_label": str (só para key=="custom"),
    #             "custom_desc": str (descrição opcional)}
    # rank é IGNORADO para senses quando sense_items estiver preenchida;
    # o rank efetivo = soma de (cost * ranks) de cada item.
    sense_items: List[Dict[str, Any]] = field(default_factory=list)

    # ── Campos específicos para Communication ─────────────────────────────────
    communication_type: str = ""            # "mental", "auditory", "radio", etc.
    communication_custom_sense: str = ""    # tipo de sentido customizado (ex: "Pesadelos")

    # ── Campos específicos para Movement ──────────────────────────────────────
    # Lista de chaves de MOVEMENT_OPTIONS selecionadas.
    # rank é IGNORADO para movement quando movement_types estiver preenchida;
    # o rank efetivo = len(movement_types).
    movement_types: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PowerComponent":
        # Suporta dados antigos sem os novos campos (compatibilidade)
        known = {
            "effect_key", "rank", "extras", "flaws", "conditions",
            "stat_targets", "resist_override", "is_linked",
            "immunity_items", "sense_items",
            "communication_type", "communication_custom_sense",
            "movement_types",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        return PowerComponent(**filtered)


@dataclass
class GolpeDraft:
    """Rascunho completo de um golpe customizado."""
    name: str = ""
    category: str = "fisico"                 # "fisico" | "especial" | "status"
    components: List[PowerComponent] = field(default_factory=list)
    advantages: List[Dict[str, Any]] = field(default_factory=list)
    # cada advantage: {"key": str, "ranks": int}
    description_pt: str = ""
    pokemon_type: str = ""                   # tipo do golpe (fire, water, etc.)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "components": [c.to_dict() for c in self.components],
            "advantages": list(self.advantages),
            "description_pt": self.description_pt,
            "pokemon_type": self.pokemon_type,
        }

    @staticmethod
    def from_dict(d: dict) -> "GolpeDraft":
        return GolpeDraft(
            name=d.get("name", ""),
            category=d.get("category", "fisico"),
            components=[PowerComponent.from_dict(c) for c in d.get("components", [])],
            advantages=d.get("advantages", []),
            description_pt=d.get("description_pt", ""),
            pokemon_type=d.get("pokemon_type", ""),
        )


# ─────────────────────────────────────────────
# Cálculo de PP
# ─────────────────────────────────────────────

def _effective_rank(comp: "PowerComponent") -> int:
    """
    Retorna o rank efetivo do componente, considerando:
    - Immunity: soma dos custos dos immunity_items (quando há itens)
    - Senses: soma de (cost * ranks) dos sense_items (quando há itens)
    - Movement: número de movement_types selecionados (quando há tipos)
    - Outros: comp.rank normal
    """
    if comp.effect_key == "immunity" and comp.immunity_items:
        return max(1, sum(item.get("cost", 1) for item in comp.immunity_items))
    if comp.effect_key == "senses" and comp.sense_items:
        total = 0
        for item in comp.sense_items:
            cost = item.get("cost", 1)
            ranks = item.get("ranks", 1) if item.get("per_rank", False) else 1
            total += cost * ranks
        return max(1, total)
    if comp.effect_key == "movement" and comp.movement_types:
        return max(1, len(comp.movement_types))
    return comp.rank


def calculate_component_cost(comp: PowerComponent) -> Tuple[float, str]:
    """
    Calcula o custo em PP de um componente individual.
    Retorna (custo_total, explicação_pt).

    Fórmula M&M:
      custo_por_rank = max(1, base + extras_per_rank - falhas_per_rank)
      total = (custo_por_rank × rank) + extras_flat + falhas_flat

    Para Immunity, Senses e Movement o rank é calculado
    automaticamente a partir dos itens/tipos selecionados.
    """
    eff = EFFECTS.get(comp.effect_key)
    if not eff:
        return 0, f"Efeito desconhecido: {comp.effect_key}"

    base = eff["base_cost"]
    per_rank_mod = 0
    flat_mod = 0
    parts = [f"Base {base}/rank"]

    # Rank efetivo (pode diferir de comp.rank para immunity/senses/movement)
    effective_rank = _effective_rank(comp)

    # Extras
    for ex in comp.extras:
        edata = EXTRAS.get(ex["key"])
        if not edata:
            continue
        ranks = max(1, int(ex.get("ranks", 1)))
        ct = edata["cost_type"]
        cost = edata["cost"]

        if ct == "per_rank":
            per_rank_mod += cost
            parts.append(f"+{cost}/rank ({edata['label_en']})")
        elif ct == "flat_per_rank":
            flat_mod += cost * ranks
            parts.append(f"+{cost * ranks} fixo ({edata['label_en']} {ranks})")
        elif ct == "flat":
            flat_mod += cost * ranks
            parts.append(f"+{cost * ranks} fixo ({edata['label_en']})")

    # Falhas
    for fl in comp.flaws:
        fdata = FLAWS.get(fl["key"])
        if not fdata:
            continue
        ranks = max(1, int(fl.get("ranks", 1)))
        ct = fdata["cost_type"]
        cost = fdata["cost"]  # negativo

        if ct == "per_rank":
            per_rank_mod += cost
            parts.append(f"{cost}/rank ({fdata['label_en']})")
        elif ct in ("flat", "flat_per_rank"):
            flat_mod += cost * ranks
            parts.append(f"{cost * ranks} fixo ({fdata['label_en']})")

    cost_per_rank = max(ADAPTATION_RULES["min_cost_per_rank"], base + per_rank_mod)
    total = (cost_per_rank * effective_rank) + flat_mod
    total = max(1, total)

    explanation = (
        f"{eff['label_en']} {effective_rank}: ({' | '.join(parts)}) "
        f"× {effective_rank} rank + {flat_mod} fixo = {total} PP"
    )
    return total, explanation


def calculate_total_pp(draft: GolpeDraft) -> Tuple[float, List[str]]:
    """
    Calcula o PP total do golpe.
    Retorna (pp_total, lista_de_explicações).
    """
    total = 0.0
    explanations: List[str] = []

    for i, comp in enumerate(draft.components):
        cost, exp = calculate_component_cost(comp)
        prefix = "Linked " if comp.is_linked else ""
        explanations.append(f"  {prefix}{exp}")
        total += cost

    return total, explanations


# ─────────────────────────────────────────────
# Geração de Build String
# ─────────────────────────────────────────────

def _build_component_string(comp: PowerComponent, category: str) -> str:
    """Gera a string de build para um único componente."""
    eff = EFFECTS.get(comp.effect_key)
    if not eff:
        return ""

    parts: List[str] = []

    # Prefixo Linked
    prefix = "Linked " if comp.is_linked else ""

    # Nome do efeito + rank
    eff_name = eff["label_en"]
    effective_rank = _effective_rank(comp)

    # Weaken e Enhanced Trait incluem o alvo no nome
    if comp.effect_key in ("weaken", "enhanced_trait") and comp.stat_targets:
        target_str = " & ".join(comp.stat_targets)
        effect_header = f"{prefix}{eff_name} {target_str} {effective_rank}"
    elif comp.effect_key == "communication" and comp.communication_type:
        # Communication: tipo aparece antes do nome
        comm_label = comp.communication_type.capitalize()
        if comp.communication_type == "custom" and comp.communication_custom_sense:
            comm_label = comp.communication_custom_sense
        effect_header = f"{prefix}{comm_label} Communication {effective_rank}"
    else:
        effect_header = f"{prefix}{eff_name} {effective_rank}"

    # Condições de Affliction (entre parênteses)
    condition_str = ""
    if comp.effect_key == "affliction" and comp.conditions:
        conds = []
        for degree_key in ("degree_1", "degree_2", "degree_3"):
            c = comp.conditions.get(degree_key)
            if c:
                conds.append(c)
        if conds:
            condition_str = ", ".join(conds)

    # Itens de Immunity entre parênteses
    immunity_str = ""
    if comp.effect_key == "immunity" and comp.immunity_items:
        labels = []
        for item in comp.immunity_items:
            if item.get("key") == "custom":
                lbl = item.get("custom_label") or "Custom"
                cost = item.get("cost", 1)
                labels.append(f"{lbl} {cost}")
            else:
                labels.append(item.get("label_pt", item["key"]))
        immunity_str = ", ".join(labels)

    # Sentidos de Senses entre parênteses
    senses_str = ""
    if comp.effect_key == "senses" and comp.sense_items:
        labels = []
        for item in comp.sense_items:
            if item.get("key") == "custom":
                lbl = item.get("custom_label") or "Custom"
                cost = item.get("cost", 1)
                labels.append(f"{lbl} {cost}")
            else:
                lbl = item.get("label_pt", item["key"])
                r = item.get("ranks", 1)
                labels.append(f"{lbl}{f' {r}' if r > 1 else ''}")
        senses_str = ", ".join(labels)

    # Tipos de Movement entre parênteses
    movement_str = ""
    if comp.effect_key == "movement" and comp.movement_types:
        _mv_map = {m["key"]: m["label_pt"] for m in MOVEMENT_OPTIONS}
        movement_str = ", ".join(_mv_map.get(k, k) for k in comp.movement_types)

    # Sense type de Communication entre parênteses
    comm_sense_str = ""
    if comp.effect_key == "communication" and comp.communication_custom_sense:
        comm_sense_str = f"Sense Type: {comp.communication_custom_sense}"

    # Resistência
    resist = comp.resist_override or eff.get("default_resist")

    # Monta o bloco de parênteses
    paren_parts = []
    if condition_str:
        paren_parts.append(condition_str)
    if immunity_str:
        paren_parts.append(immunity_str)
    if senses_str:
        paren_parts.append(senses_str)
    if movement_str:
        paren_parts.append(movement_str)
    if comm_sense_str:
        paren_parts.append(comm_sense_str)
    if resist:
        paren_parts.append(f"Resisted by {resist}")

    if paren_parts:
        effect_header += f" ({'; '.join(paren_parts)})"

    parts.append(effect_header)

    # Custom de categoria (Stgr/Intelect Based) -- so para Damage
    # Pula se o jogador ja adicionou manualmente custom_stgr_based ou custom_int_based
    has_manual_custom_cat = any(
        e["key"] in ("custom_stgr_based", "custom_int_based") for e in comp.extras
    )
    if comp.effect_key == "damage" and not has_manual_custom_cat:
        cat_info = ADAPTATION_RULES["category_map"].get(category, {})
        custom = cat_info.get("custom", "")
        if custom:
            parts.append(custom)

    # Range padrao (Close se nao tem Ranged extra)
    has_ranged = any(e["key"] == "ranged" for e in comp.extras)
    has_area = any(e["key"].startswith("area_") for e in comp.extras)
    attack_effects = ("damage", "affliction", "weaken", "nullify", "move_object",
                      "mind_reading", "transform", "deflect")
    if comp.effect_key in attack_effects:
        if not has_ranged and not has_area:
            if eff.get("default_range") == "close":
                parts.append("[Close]")

    # Extras como modifiers
    for ex in comp.extras:
        edata = EXTRAS.get(ex["key"])
        if not edata:
            continue

        ranks = int(ex.get("ranks", 1))
        key = ex["key"]
        desc = ex.get("description", "")

        # Custom extras com texto livre: [Custom X/r: descricao]
        if edata.get("group") == "custom":
            cost_val = edata["cost"]
            sign = f"+{cost_val}" if cost_val >= 0 else str(cost_val)
            if desc:
                parts.append(f"[Custom {sign}/r: {desc}]")
            elif "build_fragment" in edata:
                parts.append(edata["build_fragment"])
            else:
                parts.append(f"[Custom {sign}/r]")
            continue

        # Area extras usam build_fragment
        if "build_fragment" in edata:
            parts.append(edata["build_fragment"])
            continue

        # Ranged
        if key == "ranged":
            parts.append("[Ranged]")
            continue

        # Demais extras
        label = edata["label_en"]
        if desc:
            parts.append(f"[Extra: {label} ({desc})]")
        elif ranks > 1:
            parts.append(f"[Extra: {label} {ranks}]")
        else:
            parts.append(f"[Extra: {label}]")

    # Falhas como modifiers
    for fl in comp.flaws:
        fdata = FLAWS.get(fl["key"])
        if not fdata:
            continue

        key = fl["key"]
        label = fdata["label_en"]
        desc = fl.get("description", "")

        # Custom flaws com texto livre
        if fdata.get("group") == "custom":
            cost_val = fdata["cost"]
            sign = f"+{cost_val}" if cost_val >= 0 else str(cost_val)
            if desc:
                parts.append(f"[Flaw Custom {sign}/r: {desc}]")
            else:
                parts.append(f"[Flaw Custom {sign}/r]")
            continue

        if desc:
            parts.append(f"[Flaw: {label} ({desc})]")
        else:
            parts.append(f"[Flaw: {label}]")

    return " ".join(parts)


def parse_build_string(build: str, default_rank: int = 10) -> List[PowerComponent]:
    """
    Parse de uma build string M&M → lista de PowerComponent.
    Usado para calcular o custo em PP de golpes vindos da base de dados
    com a mesma engine dos golpes criados manualmente.

    Suporta:
    - Prefixo "Linked "
    - Rank como número após o nome do efeito
    - [Ranged], [Close], [Extra: X], [Flaw: X], [Area: Burst/Cone/Line/Cloud]
    - Parênteses de condição para Affliction: (Dazed, Stunned, Incapacitated)
    - Stat targets para Weaken/Enhanced Trait: "Weaken Stgr 14"
    """
    import re

    # ── Mapa nome_en → chave ─────────────────────────────────────────────
    _name_map: Dict[str, str] = {}
    for k, eff in EFFECTS.items():
        _name_map[eff["label_en"].lower()] = k

    _aliases: List[tuple] = sorted([
        ("damage", "damage"),
        ("affliction", "affliction"),
        ("weaken", "weaken"),
        ("healing", "healing"),
        ("regeneration", "regeneration"),
        ("enhanced trait", "enhanced_trait"),
        ("enhanced", "enhanced_trait"),
        ("flight", "flight"),
        ("teleport", "teleport"),
        ("immunity", "immunity"),
        ("senses", "senses"),
        ("movement", "movement"),
        ("communication", "communication"),
        ("environment", "environment"),
        ("nullify", "nullify"),
        ("protection", "protection"),
        ("swimming", "swimming"),
        ("leaping", "leaping"),
        ("mind reading", "mind_reading"),
        ("deflect", "deflect"),
        ("create", "create"),
        ("morph", "morph"),
        ("growth", "growth"),
        ("shrinking", "shrinking"),
        ("concealment", "concealment"),
        ("insubstantial", "insubstantial"),
        ("summon", "summon"),
        ("transform", "transform"),
        ("variable", "variable"),
        ("quickness", "quickness"),
        ("speed", "speed"),
        ("remote sensing", "remote_sensing"),
        ("linked", "damage"),   # fallback: "Linked X" sem prefixo
    ], key=lambda x: -len(x[0]))

    _area_fragments = {
        "area: burst": "area_burst",
        "area: cone": "area_cone",
        "area: cloud": "area_cloud",
        "area: line": "area_line",
        "perception area": "area_perception",
        "cone area": "area_cone",
        "cloud area": "area_cloud",
        "line area": "area_line",
    }

    _stat_keys = {s["label_en"].lower(): s["label_en"] for s in STAT_TARGETS}

    def _split_components(s: str) -> List[str]:
        """Divide por ';' apenas fora de colchetes [ ]."""
        parts, current, depth = [], [], 0
        for ch in s:
            if ch == '[':
                depth += 1
                current.append(ch)
            elif ch == ']':
                depth = max(depth - 1, 0)
                current.append(ch)
            elif ch == ';' and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return [p for p in parts if p]

    components: List[PowerComponent] = []
    raw_parts = _split_components(build)

    for raw_part in raw_parts:
        if not raw_part:
            continue

        part = raw_part.strip()
        is_linked = part.lower().startswith("linked ")
        if is_linked:
            part = part[7:].strip()

        # Extrai conteúdo de colchetes [...]
        bracket_contents = re.findall(r'\[([^\]]+)\]', part)
        part_no_brackets = re.sub(r'\[[^\]]+\]', '', part).strip()

        # Extrai parênteses (condições, itens, etc.)
        paren_match = re.search(r'\(([^)]+)\)', part_no_brackets)
        paren_content = paren_match.group(1) if paren_match else ""
        part_clean = re.sub(r'\([^)]+\)', '', part_no_brackets).strip()

        # Encontra rank (último token numérico) e tokens do nome
        tokens = part_clean.split()
        rank = default_rank
        effect_tokens: List[str] = list(tokens)

        for i in range(len(tokens) - 1, -1, -1):
            try:
                rank = int(tokens[i])
                effect_tokens = tokens[:i]
                break
            except ValueError:
                continue

        effect_text = " ".join(effect_tokens).lower().strip()

        # Remove "strength-based", "int-based", etc. (não afetam custo)
        effect_text = re.sub(r'strength.based|str.based|int.based|special.based', '', effect_text).strip()

        # Resolve efeito
        effect_key: Optional[str] = _name_map.get(effect_text)
        if not effect_key:
            for alias, key in _aliases:
                if alias in effect_text:
                    effect_key = key
                    break

        if not effect_key:
            continue

        # ── Extras e Falhas dos colchetes ─────────────────────────────
        extras: List[Dict] = []
        flaws: List[Dict] = []

        # Cada bracket pode conter múltiplos extras separados por ';'
        # ex: [Ranged; Area: Burst] → processa "Ranged" e "Area: Burst" separadamente
        bracket_tokens: List[str] = []
        for bracket in bracket_contents:
            for frag in bracket.split(";"):
                frag = frag.strip()
                if frag:
                    bracket_tokens.append(frag)

        for b in bracket_tokens:
            b_lower = b.lower()

            if b_lower in ("close",) or b_lower.startswith("resisted by") or b_lower.startswith("stgr based") or b_lower.startswith("strength"):
                continue

            if b_lower == "ranged":
                extras.append({"key": "ranged", "ranks": 1, "description": ""})
                continue

            # Area fragments
            matched_area = False
            for frag, area_key in _area_fragments.items():
                if frag in b_lower:
                    extras.append({"key": area_key, "ranks": 1, "description": ""})
                    matched_area = True
                    break
            if matched_area:
                continue

            # Extra: Label
            if b_lower.startswith("extra:"):
                extra_text = b[6:].strip()
                for ex_key, ex_data in EXTRAS.items():
                    if ex_data["label_en"].lower() in extra_text.lower():
                        extras.append({"key": ex_key, "ranks": 1, "description": extra_text})
                        break
                continue

            # Flaw: Label
            if b_lower.startswith("flaw:"):
                flaw_text = b[5:].strip()
                for fl_key, fl_data in FLAWS.items():
                    if fl_data["label_en"].lower() in flaw_text.lower():
                        flaws.append({"key": fl_key, "ranks": 1, "description": flaw_text})
                        break
                continue

            # Tenta casar diretamente com extras conhecidos (build_fragment ou label_en)
            for ex_key, ex_data in EXTRAS.items():
                label_en = ex_data.get("label_en", "").lower()
                bf = ex_data.get("build_fragment", "").lower()
                if (label_en and label_en in b_lower) or (bf and bf in b_lower):
                    extras.append({"key": ex_key, "ranks": 1, "description": ""})
                    break

        # ── Monta componente ──────────────────────────────────────────
        comp = PowerComponent(
            effect_key=effect_key,
            rank=rank,
            is_linked=is_linked,
            extras=extras,
            flaws=flaws,
        )

        # Condições para Affliction
        if effect_key == "affliction" and paren_content:
            cond_labels = re.split(r'[,;]\s*', paren_content)
            skip_words = {"resisted by toughness", "resisted by will", "resisted by fortitude", "resisted by dodge"}
            conditions: Dict[str, str] = {}
            deg_keys = ["degree_1", "degree_2", "degree_3"]
            idx = 0
            for lbl in cond_labels:
                lbl = lbl.strip()
                if lbl.lower() in skip_words or not lbl:
                    continue
                if idx < 3:
                    conditions[deg_keys[idx]] = lbl
                    idx += 1
            if conditions:
                comp.conditions = conditions

        # Stat targets para Weaken / Enhanced Trait
        if effect_key in ("weaken", "enhanced_trait") and effect_tokens:
            noise = {"weaken", "enhanced", "trait", "linked", "stgr", "based"}
            stat_tokens = [t for t in effect_tokens if t.lower() not in noise]
            matched_stats: List[str] = []
            for tok in stat_tokens:
                st_label = _stat_keys.get(tok.lower())
                if st_label:
                    matched_stats.append(st_label)
                elif tok and tok[0].isupper():
                    matched_stats.append(tok)
            if matched_stats:
                comp.stat_targets = matched_stats

        components.append(comp)

    return components


def generate_build_string(draft: GolpeDraft) -> str:
    """
    Converte um GolpeDraft em uma string de build M&M completa.
    Formato: "Efeito Rank (condições) [modifiers]; Linked Efeito Rank [modifiers]"
    """
    component_strings = []
    for comp in draft.components:
        s = _build_component_string(comp, draft.category)
        if s:
            component_strings.append(s)

    return "; ".join(component_strings)


# ─────────────────────────────────────────────
# Explicação em PT
# ─────────────────────────────────────────────

def generate_explanation(draft: GolpeDraft) -> List[Dict[str, str]]:
    """
    Gera explicação em português de cada componente do golpe.
    Retorna lista de {"component": str, "explanation_pt": str, "cost": float}.
    """
    result = []
    for comp in draft.components:
        eff = EFFECTS.get(comp.effect_key)
        if not eff:
            continue

        cost, _ = calculate_component_cost(comp)
        prefix = "Vinculado (Linked): " if comp.is_linked else ""

        effective_rank = _effective_rank(comp)

        # Explicação base
        exp_parts = [f"{prefix}{eff['desc_pt']}"]
        exp_parts.append(f"Rank {effective_rank} — custo: {cost} PP.")

        # Condições (Affliction)
        if comp.effect_key == "affliction" and comp.conditions:
            for deg_key, label in [("degree_1", "Grau 1"), ("degree_2", "Grau 2"), ("degree_3", "Grau 3")]:
                c = comp.conditions.get(deg_key)
                if c:
                    exp_parts.append(f"  {label}: {c}")

        # Itens de Immunity
        if comp.effect_key == "immunity" and comp.immunity_items:
            for item in comp.immunity_items:
                if item.get("key") == "custom":
                    lbl = item.get("custom_label", "Custom")
                    exp_parts.append(f"  • {lbl} — {item.get('cost', 1)} PP (personalizado)")
                else:
                    exp_parts.append(f"  • {item.get('label_pt', item['key'])} — {item.get('cost', 1)} PP")

        # Itens de Senses
        if comp.effect_key == "senses" and comp.sense_items:
            for item in comp.sense_items:
                if item.get("key") == "custom":
                    lbl = item.get("custom_label", "Custom")
                    exp_parts.append(f"  • {lbl} — {item.get('cost', 1)} PP (personalizado)")
                else:
                    r = item.get("ranks", 1)
                    cost_item = item.get("cost", 1) * r
                    ranks_info = f" (rank {r})" if r > 1 else ""
                    exp_parts.append(f"  • {item.get('label_pt', item['key'])}{ranks_info} — {cost_item} PP")

        # Tipos de Movimento
        if comp.effect_key == "movement" and comp.movement_types:
            _mv_map = {m["key"]: m["label_pt"] for m in MOVEMENT_OPTIONS}
            for k in comp.movement_types:
                exp_parts.append(f"  • {_mv_map.get(k, k)}")

        # Tipo de Comunicação
        if comp.effect_key == "communication":
            if comp.communication_type:
                type_label = comp.communication_type.capitalize()
                if comp.communication_type == "custom" and comp.communication_custom_sense:
                    type_label = f"Personalizado ({comp.communication_custom_sense})"
                exp_parts.append(f"  Tipo: {type_label}")
            if comp.communication_custom_sense and comp.communication_type != "custom":
                exp_parts.append(f"  Sentido personalizado: {comp.communication_custom_sense}")

        # Alvos (Weaken / Enhanced Trait)
        if comp.stat_targets:
            exp_parts.append(f"  Alvos: {', '.join(comp.stat_targets)}")

        # Resistência
        resist = comp.resist_override or eff.get("default_resist")
        if resist:
            exp_parts.append(f"  Resistido por: {resist}")

        # Extras
        for ex in comp.extras:
            edata = EXTRAS.get(ex["key"])
            if edata:
                exp_parts.append(f"  + {edata['label_pt']}: {edata['desc_pt']}")

        # Falhas
        for fl in comp.flaws:
            fdata = FLAWS.get(fl["key"])
            if fdata:
                desc = fl.get("description", "")
                detail = f" — {desc}" if desc else ""
                exp_parts.append(f"  - {fdata['label_pt']}: {fdata['desc_pt']}{detail}")

        component_label = f"{'Linked ' if comp.is_linked else ''}{eff['label_en']} {effective_rank}"
        result.append({
            "component": component_label,
            "explanation_pt": "\n".join(exp_parts),
            "cost": cost,
        })

    # Vantagens
    if draft.advantages:
        adv_lines = []
        for adv in draft.advantages:
            adv_data = next((a for a in COMBAT_ADVANTAGES if a["key"] == adv["key"]), None)
            if adv_data:
                ranks_str = f" {adv.get('ranks', 1)}" if adv_data.get("has_ranks") else ""
                adv_lines.append(f"  {adv_data['label_pt']}{ranks_str}: {adv_data['desc_pt']}")
        if adv_lines:
            result.append({
                "component": "Vantagens de Combate",
                "explanation_pt": "\n".join(adv_lines),
                "cost": 0,
            })

    return result


def generate_description_pt(draft: GolpeDraft) -> str:
    """
    Gera automaticamente uma descrição em linguagem natural (PT)
    a partir dos componentes do golpe.
    """
    parts: List[str] = []
    cat_info = ADAPTATION_RULES["category_map"].get(draft.category, {})
    cat_label = cat_info.get("label_pt", "")

    if cat_label:
        parts.append(f"Golpe {cat_label}.")

    for comp in draft.components:
        eff = EFFECTS.get(comp.effect_key)
        if not eff:
            continue

        linked = "Adicionalmente, " if comp.is_linked else ""

        if comp.effect_key == "damage":
            has_ranged = any(e["key"] == "ranged" for e in comp.extras)
            has_area = any(e["key"].startswith("area_") for e in comp.extras)
            range_txt = "a distância" if has_ranged else "corpo-a-corpo"
            area_txt = ""
            if has_area:
                area_extra = next((e for e in comp.extras if e["key"].startswith("area_")), None)
                if area_extra:
                    area_data = EXTRAS.get(area_extra["key"], {})
                    area_txt = f" em área ({area_data.get('label_en', 'Area')})"
            parts.append(f"{linked}Causa dano rank {comp.rank} {range_txt}{area_txt}.")

        elif comp.effect_key == "affliction":
            conds = []
            if comp.conditions:
                for dk in ("degree_1", "degree_2", "degree_3"):
                    c = comp.conditions.get(dk)
                    if c:
                        conds.append(c)
            cond_txt = f" ({', '.join(conds)})" if conds else ""
            resist = comp.resist_override or eff["default_resist"] or ""
            parts.append(f"{linked}Impõe condições{cond_txt}, resistido por {resist}.")

        elif comp.effect_key == "weaken":
            targets = ", ".join(comp.stat_targets) if comp.stat_targets else "atributo"
            parts.append(f"{linked}Reduz {targets} do alvo em {comp.rank} ranks.")

        elif comp.effect_key == "enhanced_trait":
            targets = ", ".join(comp.stat_targets) if comp.stat_targets else "atributo"
            parts.append(f"{linked}Aumenta {targets} do Pokémon em {comp.rank} ranks.")

        elif comp.effect_key == "healing":
            parts.append(f"{linked}Cura {comp.rank} ranks de dano.")

        elif comp.effect_key == "environment":
            parts.append(f"{linked}Altera as condições ambientais/climáticas na área.")

        elif comp.effect_key == "create":
            parts.append(f"{linked}Cria objetos ou terreno sólido.")

        elif comp.effect_key == "teleport":
            parts.append(f"{linked}Teletransporta o Pokémon.")

        elif comp.effect_key == "nullify":
            parts.append(f"{linked}Anula poderes ativos do alvo.")

        elif comp.effect_key == "immunity":
            parts.append(f"{linked}Confere imunidade a efeitos específicos.")

        elif comp.effect_key == "protection":
            parts.append(f"{linked}Aumenta a resistência física (Toughness) em {comp.rank}.")

        elif comp.effect_key == "regeneration":
            parts.append(f"{linked}Regenera dano continuamente ao longo dos turnos.")

        elif comp.effect_key == "concealment":
            parts.append(f"{linked}Torna o Pokémon oculto/invisível.")

        elif comp.effect_key == "insubstantial":
            parts.append(f"{linked}Torna o Pokémon intangível.")

        else:
            parts.append(f"{linked}{eff['desc_pt']}")

    return " ".join(parts)


# ─────────────────────────────────────────────
# Validação
# ─────────────────────────────────────────────

def validate_draft(draft: GolpeDraft) -> List[str]:
    """
    Valida o rascunho contra as regras de adaptação Pokémon.
    Retorna lista de avisos (vazia = válido).
    """
    warnings: List[str] = []

    if not draft.components:
        warnings.append("O golpe precisa ter pelo menos um componente/efeito.")
        return warnings

    if not draft.name.strip():
        warnings.append("O golpe precisa ter um nome.")

    # Verificar efeitos duplicados em Linked
    if ADAPTATION_RULES["linked_no_repeat"]:
        effect_keys = [c.effect_key for c in draft.components]
        seen = set()
        for ek in effect_keys:
            if ek in seen:
                eff = EFFECTS.get(ek, {})
                warnings.append(
                    f"Efeito '{eff.get('label_pt', ek)}' aparece mais de uma vez. "
                    "Linked effects não podem repetir o mesmo tipo de efeito."
                )
            seen.add(ek)

    for comp in draft.components:
        eff = EFFECTS.get(comp.effect_key)
        if not eff:
            warnings.append(f"Efeito desconhecido: {comp.effect_key}")
            continue

        # Area max rank
        for ex in comp.extras:
            if ex["key"].startswith("area_"):
                if comp.rank > ADAPTATION_RULES["area_max_rank"]:
                    warnings.append(
                        f"Efeito de Área ({eff['label_pt']}) com rank {comp.rank} "
                        f"excede o máximo permitido ({ADAPTATION_RULES['area_max_rank']})."
                    )

        # Enhanced Trait cap
        if comp.effect_key == "enhanced_trait":
            if comp.rank > ADAPTATION_RULES["stat_boost_cap"]:
                warnings.append(
                    f"Enhanced Trait com rank {comp.rank} excede o máximo "
                    f"permitido (+{ADAPTATION_RULES['stat_boost_cap']})."
                )

        # Affliction precisa de condições
        if comp.effect_key == "affliction":
            if not comp.conditions or not any(comp.conditions.values()):
                warnings.append("Affliction precisa de pelo menos uma condição definida.")

        # Weaken/Enhanced precisa de alvo
        if eff.get("has_targets") and not comp.stat_targets:
            warnings.append(f"{eff['label_pt']} precisa de pelo menos um atributo alvo.")

        # Custo mínimo
        cost, _ = calculate_component_cost(comp)
        if cost < 1:
            warnings.append(f"{eff['label_pt']}: custo calculado ({cost}) é menor que 1 PP.")

    return warnings


# ─────────────────────────────────────────────
# Conversão para formatos de saída
# ─────────────────────────────────────────────

def draft_to_move_json(draft: GolpeDraft, accuracy: int = 0) -> dict:
    """
    Converte um GolpeDraft para o formato JSON existente (cg_moves).
    Compatível com o formato de sheets_dump.json / _confirm_move().
    """
    build = generate_build_string(draft)
    pp, _ = calculate_total_pp(draft)

    # Determinar ranged e perception_area
    is_ranged = False
    is_perception = False
    for comp in draft.components:
        for ex in comp.extras:
            if ex["key"] == "ranged":
                is_ranged = True
            if ex["key"] == "area_perception":
                is_perception = True

    # Rank principal = rank do primeiro componente
    main_rank = draft.components[0].rank if draft.components else 1

    cat_info = ADAPTATION_RULES["category_map"].get(draft.category, {})
    category_label = cat_info.get("label_pt", "Status")

    return {
        "name": draft.name,
        "rank": int(main_rank),
        "build": build,
        "pp_cost": int(pp),
        "accuracy": int(accuracy),
        "meta": {
            "ranged": is_ranged,
            "perception_area": is_perception,
            "category": category_label,
        },
        "_ui_id": uuid.uuid4().hex,
    }


def draft_to_firestore_doc(draft: GolpeDraft, trainer_name: str) -> dict:
    """
    Converte um GolpeDraft para o documento Firestore em custom_moves.
    """
    build = generate_build_string(draft)
    pp, _ = calculate_total_pp(draft)
    now = datetime.now(timezone.utc).isoformat()

    cat_info = ADAPTATION_RULES["category_map"].get(draft.category, {})

    return {
        "name": draft.name,
        "build": build,
        "pp_cost": int(pp),
        "description_pt": draft.description_pt,
        "category": cat_info.get("label_pt", "Status"),
        "pokemon_type": draft.pokemon_type,
        "components": [c.to_dict() for c in draft.components],
        "advantages": draft.advantages,
        "trainer_name": trainer_name,
        "created_at": now,
        "updated_at": now,
    }
