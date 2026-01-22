# advantages_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
import streamlit as st


# =========================
# Fontes automáticas (PokeAPI)
# =========================
POKEAPI = "https://pokeapi.co/api/v2"


@st.cache_data(show_spinner=False)
def _pokeapi_move(name: str) -> dict:
    url = f"{POKEAPI}/move/{name.strip().lower()}"
    try:
        r = requests.get(url, timeout=20)

        # se não for sucesso HTTP
        if r.status_code != 200:
            return {}

        # tenta converter pra JSON
        try:
            return r.json()
        except Exception:
            return {}

    except Exception:
        return {}



def pokemon_moveset_from_pjson(pjson: dict) -> Set[str]:
    """Extrai nomes de moves do JSON do pokemon (sem requests extras)."""
    out = set()
    for m in pjson.get("moves", []):
        mn = (m.get("move") or {}).get("name")
        if mn:
            out.add(mn.lower())
    return out


def pokemon_speed_from_stats(base_stats: dict) -> int:
    # base_stats vem do seu pokeapi_parse_stats(pjson) -> dict tipo {"speed": 78, ...}
    return int(base_stats.get("speed", 0) or 0)


# =========================
# Modelo de output
# =========================
@dataclass
class AdvantageSuggestion:
    name: str
    rank: Optional[int] = None
    note: Optional[str] = None

    def label(self) -> str:
        if self.rank is not None:
            return f"{self.name} (Rank {self.rank})"
        return self.name


# =========================
# Regras do seu documento
# =========================
# 1) Accuracy / anti-redução
ACCURACY_IMMUNE_ABILITIES = {
    "keen-eye",          # impede redução de accuracy (stages)
    "mind-s-eye",        # compartilha efeito do keen-eye
}
ACCURACY_BOOST_ABILITIES = {
    "compound-eyes",     # aumenta accuracy
    "victory-star",      # aumenta accuracy do usuário e aliados
    "no-guard",          # golpes não erram (equivalente prático)
}

# 2) Low HP boost (all-out)
LOW_HP_TYPE_BOOST_ABILITIES = {
    "blaze", "torrent", "overgrow", "swarm",
}

# 3) Evasion / ocultação
EVASION_ABILITIES = {
    "snow-cloak",
    "sand-veil",
    # você pode ampliar depois:
    # "tangled-feet",
}

# 4) Uncanny Dodge (não dormir / sempre alerta)
ALWAYS_AWAKE_ABILITIES = {
    "insomnia",
    "vital-spirit",
    "early-bird",
}

# 5) Power Attack
POWER_ATTACK_ABILITIES = {
    "sheer-force",
}

# 6) Sorte
LUCK_ABILITIES = {
    "super-luck",
}

# 7) Move-by Action
MOVE_BY_ABILITIES = {
    "run-away",
}

# 8) Attractive
ATTRACTIVE_ABILITIES = {
    "cute-charm",
}

# 9) Weather / Favored Environment
WEATHER_RAIN_ABILITIES = {"drizzle", "rain-dish", "swift-swim", "hydration"}
WEATHER_SUN_ABILITIES  = {"drought", "chlorophyll", "solar-power"}
WEATHER_SAND_ABILITIES = {"sand-stream", "sand-veil", "sand-rush"}
WEATHER_SNOW_ABILITIES = {"snow-warning", "snow-cloak", "slush-rush"}

# 10) Moves gatilho (detecção “completa” via moveset)
DISARM_MOVES = {"knock-off", "trick"}  # você pediu esses dois
TAUNT_MOVE = "taunt"

# “grabs” / “holds” (aproximação automática)
GRAB_STYLE_MOVES = {
    "bind", "wrap", "clamp", "whirlpool", "infestation", "sand-tomb",
    "seismic-toss", "vital-throw", "submission", "circle-throw", "storm-throw",
}

# “sufocar” é muito interpretativo -> deixo automático só se tiver moves de prender/enrolar
CHOKEHOLD_MOVES = {"bind", "wrap", "clamp", "whirlpool", "infestation", "sand-tomb"}

# “All-out-Attack” por golpe base power > 100 (automático via PokeAPI move power)
@st.cache_data(show_spinner=False)
def _move_power(name: str) -> Optional[int]:
    mj = _pokeapi_move(name)
    return mj.get("power", None)

@st.cache_data(show_spinner=False)
def _move_priority(name: str) -> int:
    mj = _pokeapi_move(name)
    return int(mj.get("priority", 0) or 0)

@st.cache_data(show_spinner=False)
def _move_has_high_crit(name: str) -> bool:
    mj = _pokeapi_move(name)
    meta = mj.get("meta") or {}
    # PokeAPI usa "crit_rate" (0 padrão, >0 = high crit)
    return int(meta.get("crit_rate", 0) or 0) > 0


def suggest_advantages(
    *,
    pjson: dict,
    base_stats: dict,
    types: List[str],
    abilities: List[str],
) -> List[AdvantageSuggestion]:
    """
    Regras automáticas:
    - usa: abilities + tipos + speed + moveset
    - para coisas subjetivas ("bonito", "predador natural", "líder") -> NÃO sugere automaticamente
      (você pode criar overrides depois)
    """
    abilities_set = {a.strip().lower() for a in abilities if a}
    types_set = {t.strip().lower() for t in types if t}
    moves = pokemon_moveset_from_pjson(pjson)
    speed = pokemon_speed_from_stats(base_stats)

    out: List[AdvantageSuggestion] = []

    # ------------------------
    # Accurate Attack
    # ------------------------
    if abilities_set & (ACCURACY_IMMUNE_ABILITIES | ACCURACY_BOOST_ABILITIES):
        out.append(AdvantageSuggestion(
            name="Accurate Attack",
            note="Habilidade impede redução ou aumenta acerto."
        ))

    # ------------------------
    # Agile Feint (objetivo: speed alto)
    # shows “visivelmente ágeis” -> aproximamos por speed
    # ------------------------
    if speed >= 100:
        out.append(AdvantageSuggestion(
            name="Agile Feint",
            note="Sugestão automática por Speed >= 100."
        ))

    # ------------------------
    # All-out-Attack
    # - low HP boosters OU golpe >100 power (detectado via moveset)
    # ------------------------
    all_out = False
    if abilities_set & LOW_HP_TYPE_BOOST_ABILITIES:
        all_out = True

    if not all_out:
        # procura UM move com power > 100 (evita loop pesado)
        for mn in moves:
            pw = _move_power(mn)
            if pw is not None and int(pw) > 100:
                all_out = True
                break

    if all_out:
        out.append(AdvantageSuggestion(
            name="All-out-Attack",
            note="Low HP boost ou possui golpe com power > 100."
        ))

    # ------------------------
    # Attractive
    # ------------------------
    if abilities_set & ATTRACTIVE_ABILITIES:
        out.append(AdvantageSuggestion(
            name="Attractive",
            note="Garantido por Cute Charm."
        ))

    # ------------------------
    # Chokehold (semi-automático)
    # ------------------------
    if moves & CHOKEHOLD_MOVES:
        out.append(AdvantageSuggestion(
            name="Chokehold",
            note="Sugestão por moves de prender/enrolar. Mestre valida."
        ))

    # ------------------------
    # Evasion (regra sua por speed)
    # >100 = rank 1, >120 = rank 2
    # ------------------------
    if speed > 120:
        out.append(AdvantageSuggestion(name="Evasion", rank=2, note="Speed > 120."))
    elif speed > 100:
        out.append(AdvantageSuggestion(name="Evasion", rank=1, note="Speed > 100."))

    # ------------------------
    # Fast Grab / Improved Grab / Improved Hold (aproximação)
    # ------------------------
    if ("fighting" in types_set) or (moves & GRAB_STYLE_MOVES):
        out.append(AdvantageSuggestion(name="Fast Grab", note="Fighting ou moves de agarrar/arremessar."))
        out.append(AdvantageSuggestion(name="Improved Grab", note="Fighting ou moves de agarrar/arremessar."))
        out.append(AdvantageSuggestion(name="Improved Hold", note="Fighting ou moves de agarrar/arremessar."))

    # ------------------------
    # Favored Environment (weather)
    # Somente chuva/sol/areia/neve e só pokémons do clima
    # ------------------------
    if abilities_set & WEATHER_RAIN_ABILITIES:
        out.append(AdvantageSuggestion(name="Favored Environment", note="Chuva (Rain)."))
    if abilities_set & WEATHER_SUN_ABILITIES:
        out.append(AdvantageSuggestion(name="Favored Environment", note="Sol (Sun)."))
    if abilities_set & WEATHER_SAND_ABILITIES:
        out.append(AdvantageSuggestion(name="Favored Environment", note="Tempestade de Areia (Sand)."))
    if abilities_set & WEATHER_SNOW_ABILITIES:
        out.append(AdvantageSuggestion(name="Favored Environment", note="Nevasca/Neve (Snow)."))

    # ------------------------
    # Hide in Plain Sight
    # Abilities evasivas OU água (manual-like)
    # ------------------------
    if abilities_set & EVASION_ABILITIES:
        out.append(AdvantageSuggestion(
            name="Hide in a Plain Sight",
            note="Snow Cloak / Sand Veil. (ou Dex específica)."
        ))
    else:
        # seu exemplo do Vaporeon na água -> aproximamos por Water type, mas marcando mestre
        if "water" in types_set:
            out.append(AdvantageSuggestion(
                name="Hide in a Plain Sight",
                note="Apenas em ambiente aquático (mestre valida / Dex)."
            ))

    # ------------------------
    # Holding Back (ligado a mega/forma) -> não dá pra inferir bem pela PokeAPI sem reforçar forms
    # ------------------------
    # out.append(...)  # deixo fora do automático

    # ------------------------
    # Improved Aim
    # ------------------------
    if ("sniper" in abilities_set) or (abilities_set & (ACCURACY_IMMUNE_ABILITIES | ACCURACY_BOOST_ABILITIES)):
        out.append(AdvantageSuggestion(name="Improved Aim", note="Sniper ou accuracy/anti-redução."))

    # ------------------------
    # Improved Critical
    # Sniper OU super luck OU move high-crit
    # ------------------------
    high_crit = False
    if ("sniper" in abilities_set) or (abilities_set & LUCK_ABILITIES):
        high_crit = True
    if not high_crit:
        for mn in moves:
            if _move_has_high_crit(mn):
                high_crit = True
                break
    if high_crit:
        out.append(AdvantageSuggestion(name="Improved Critical", note="Sniper/Super Luck ou move high-crit."))

    # ------------------------
    # Improved Disarm (Knock Off / Trick)
    # ------------------------
    if moves & DISARM_MOVES:
        out.append(AdvantageSuggestion(name="Improved Disarm", note="Aprende Knock Off/Trick."))

    # ------------------------
    # Inspire / Leadership (somente líder do time) -> NÃO automático
    # ------------------------

    # ------------------------
    # Instant Up (Voadores e Lutadores)
    # ------------------------
    if ("flying" in types_set) or ("fighting" in types_set):
        out.append(AdvantageSuggestion(name="Instant Up", note="Tipo Flying ou Fighting."))

    # ------------------------
    # Move-by Action (Speed > 110 ou Run Away)
    # ------------------------
    if speed > 110 or (abilities_set & MOVE_BY_ABILITIES):
        out.append(AdvantageSuggestion(name="Move-by Action", note="Speed > 110 ou Run Away."))

    # ------------------------
    # Power Attack (Sheer Force)
    # ------------------------
    if abilities_set & POWER_ATTACK_ABILITIES:
        out.append(AdvantageSuggestion(name="Power Attack", note="Sheer Force."))

    # ------------------------
    # Redirect (precisa aprovação do mestre) -> NÃO automático
    # ------------------------

    # ------------------------
    # Seize Initiative (Prioridade)
    # - se aprende algum move com prioridade > 0
    # ------------------------
    has_priority = False
    for mn in moves:
        if _move_priority(mn) > 0:
            has_priority = True
            break
    if has_priority:
        out.append(AdvantageSuggestion(name="Seize Initiative", note="Aprende golpe com prioridade."))

    # ------------------------
    # Taunt (é o golpe)
    # ------------------------
    if TAUNT_MOVE in moves:
        out.append(AdvantageSuggestion(name="Taunt", note="Aprende o golpe Taunt."))

    # ------------------------
    # Uncanny Dodge (não dormir / alerta)
    # ------------------------
    if abilities_set & ALWAYS_AWAKE_ABILITIES:
        out.append(AdvantageSuggestion(name="Uncanny Dodge", note="Insomnia/Vital Spirit/Early Bird."))

    # ------------------------
    # Toda Habilidade que envolva Sorte
    # - habilidade Super Luck OU golpe com “luck” no nome
    # ------------------------
    if (abilities_set & LUCK_ABILITIES) or any("luck" in mn for mn in moves):
        out.append(AdvantageSuggestion(
            name="Toda Habilidade que envolva Sorte",
            note="Super Luck ou move com 'Luck' no nome. Mestre valida."
        ))

    # ------------------------
    # Fearless (Proibido) -> nunca sugerir
    # ------------------------

    # Remove duplicatas por (name, rank)
    seen = set()
    uniq: List[AdvantageSuggestion] = []
    for a in out:
        key = (a.name, a.rank)
        if key not in seen:
            seen.add(key)
            uniq.append(a)

    return uniq
