# golpe_builder_ui.py
"""
UI Streamlit do Golpe Builder — fluxo de 5 passos para criar golpes customizados.
Funciona standalone (streamlit run golpe_builder_ui.py) e como módulo importável.
"""
from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st

from powers_data import (
    EFFECTS,
    EXTRAS,
    FLAWS,
    AFFLICTION_CONDITIONS,
    STAT_TARGETS,
    COMBAT_ADVANTAGES,
    ADAPTATION_RULES,
    TYPE_DESCRIPTORS,
    PRESET_KITS,
    IMMUNITY_OPTIONS,
    SENSES_OPTIONS,
    COMMUNICATION_TYPES,
    MOVEMENT_OPTIONS,
)
from golpe_builder import (
    PowerComponent,
    GolpeDraft,
    calculate_component_cost,
    calculate_total_pp,
    generate_build_string,
    generate_explanation,
    generate_description_pt,
    draft_to_move_json,
    draft_to_firestore_doc,
    validate_draft,
    parse_build_string,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _k(prefix: str, *parts: str) -> str:
    """Gera chave única para widgets Streamlit."""
    return f"{prefix}_{'_'.join(parts)}"


def _get_draft(prefix: str) -> GolpeDraft:
    """Recupera ou cria o rascunho do golpe no session_state."""
    key = _k(prefix, "draft")
    if key not in st.session_state:
        st.session_state[key] = GolpeDraft().to_dict()
    d = st.session_state[key]
    if isinstance(d, dict):
        return GolpeDraft.from_dict(d)
    return d


def _set_draft(prefix: str, draft: GolpeDraft):
    """Salva o rascunho no session_state."""
    st.session_state[_k(prefix, "draft")] = draft.to_dict()


def _get_step(prefix: str) -> int:
    return st.session_state.get(_k(prefix, "step"), 1)


def _set_step(prefix: str, step: int):
    st.session_state[_k(prefix, "step")] = step


# ─────────────────────────────────────────────
# Histórico & Templates
# ─────────────────────────────────────────────
_HISTORY_KEY  = "gb_global_history"
_TEMPLATE_KEY = "gb_global_templates"
_MAX_HISTORY  = 10


def _get_history() -> List[dict]:
    return st.session_state.get(_HISTORY_KEY, [])


def _save_to_history(draft: GolpeDraft):
    if not draft.name or not draft.components:
        return
    h = st.session_state.get(_HISTORY_KEY, [])
    h = [x for x in h if x.get("name") != draft.name]  # dedup por nome
    h.append(draft.to_dict())
    st.session_state[_HISTORY_KEY] = h[-_MAX_HISTORY:]


def _get_templates() -> List[dict]:
    return st.session_state.get(_TEMPLATE_KEY, [])


def _save_template(draft: GolpeDraft):
    t = st.session_state.get(_TEMPLATE_KEY, [])
    t = [x for x in t if x.get("name") != draft.name]
    t.append(draft.to_dict())
    st.session_state[_TEMPLATE_KEY] = t


def _delete_template(idx: int):
    t = st.session_state.get(_TEMPLATE_KEY, [])
    if 0 <= idx < len(t):
        t.pop(idx)
    st.session_state[_TEMPLATE_KEY] = t


# ─────────────────────────────────────────────
# Preview em tempo real
# ─────────────────────────────────────────────

def _render_live_preview(draft: Optional[GolpeDraft], compact: bool = False):
    """Renderiza um card de preview com build + PP + complexidade."""
    if not draft or not draft.components:
        st.info("Configure o golpe ao lado para ver o preview.")
        return

    build = generate_build_string(draft)
    pp, _ = calculate_total_pp(draft)
    n_comps = len(draft.components)
    n_extras = sum(len(c.extras) for c in draft.components)

    complexity_score = n_comps + n_extras
    if complexity_score <= 2:
        complexity = "🟢 Simples"
    elif complexity_score <= 5:
        complexity = "🟡 Médio"
    else:
        complexity = "🔴 Complexo"

    st.markdown("#### 👁️ Preview em Tempo Real")
    st.code(build, language="text")

    col_pp, col_cx = st.columns(2)
    with col_pp:
        st.metric("PP Total", int(pp))
    with col_cx:
        st.markdown(f"**{complexity}**")

    if not compact:
        for item in generate_explanation(draft):
            st.caption(f"**{item['component']}** — {item['cost']} PP: {item['explanation_pt'].split(chr(10))[0]}")


# ─────────────────────────────────────────────
# Modo Simples — mapeamentos e lógica
# ─────────────────────────────────────────────

# O que o golpe faz → efeitos
_WHAT_OPTIONS: List[Dict] = [
    {"label": "⚔️ Causa Dano",                       "effects": ["damage"],         "show_conditions": False, "show_targets": False},
    {"label": "🌀 Impõe Condição / Status",            "effects": ["affliction"],      "show_conditions": True,  "show_targets": False},
    {"label": "⚔️🌀 Dano + Condição",                 "effects": ["damage","affliction"], "show_conditions": True, "show_targets": False},
    {"label": "💊 Cura / Recupera HP",                "effects": ["healing"],         "show_conditions": False, "show_targets": False},
    {"label": "⚔️💊 Drena (Dano + Cura)",             "effects": ["damage","healing"],"show_conditions": False, "show_targets": False},
    {"label": "⬇️ Enfraquece Atributo",               "effects": ["weaken"],          "show_conditions": False, "show_targets": True },
    {"label": "⬆️ Melhora Atributo (próprio/aliado)", "effects": ["enhanced_trait"],  "show_conditions": False, "show_targets": True },
    {"label": "🌿 Altera o Campo (Clima/Terreno)",    "effects": ["environment"],     "show_conditions": False, "show_targets": False},
    {"label": "✈️ Voo",                               "effects": ["flight"],          "show_conditions": False, "show_targets": False},
    {"label": "🔀 Teleporte",                          "effects": ["teleport"],        "show_conditions": False, "show_targets": False},
    {"label": "🧠 Sentidos Especiais",                 "effects": ["senses"],          "show_conditions": False, "show_targets": False},
    {"label": "🛡️ Imunidade",                         "effects": ["immunity"],        "show_conditions": False, "show_targets": False},
    {"label": "🚶 Movimento Especial",                 "effects": ["movement"],        "show_conditions": False, "show_targets": False},
    {"label": "🔕 Nulifica Poder",                     "effects": ["nullify"],         "show_conditions": False, "show_targets": False},
    {"label": "🔋 Regeneração",                        "effects": ["regeneration"],    "show_conditions": False, "show_targets": False},
]

# Alvo/alcance → extras
_TARGET_OPTIONS: List[Dict] = [
    {"label": "🎯 1 alvo — Corpo a corpo",          "extras": []},
    {"label": "🎯 1 alvo — À distância (Ranged)",   "extras": ["ranged"]},
    {"label": "💥 Explosão em área (Burst)",         "extras": ["area_burst"]},
    {"label": "📐 Cone de efeito",                   "extras": ["area_cone"]},
    {"label": "➖ Linha reta",                       "extras": ["area_line"]},
    {"label": "🌫️ Nuvem / Campo (Cloud)",           "extras": ["area_cloud"]},
    {"label": "👁️ Percepção — todos que percebem", "extras": ["area_perception"]},
    {"label": "✋ Si mesmo / Aliado próximo",        "extras": []},
]

# Extras simplificados
_SIMPLE_EXTRAS: List[Dict] = [
    {"label": "⚡ Efeito secundário (Secondary Effect)",  "key": "secondary_effect"},
    {"label": "📈 Progressivo (piora a cada turno)",      "key": "progressive"},
    {"label": "🔁 Cumulativo",                            "key": "cumulative"},
    {"label": "🎯 Preciso (+2 no ataque)",                "key": "accurate"},
    {"label": "💀 Penetrante (ignora parte da defesa)",   "key": "penetrating"},
    {"label": "⚡ Reação (Triggered/Reaction)",           "key": "reaction"},
    {"label": "🔗 Contagioso (Contagious)",               "key": "contagious"},
    {"label": "🔮 Resistência Alternativa",               "key": "alternate_resistance"},
]

# Falhas simplificadas
_SIMPLE_FLAWS: List[Dict] = [
    {"label": "💤 Cansativo (gasta energia extra)",   "key": "tiring"},
    {"label": "✊ Só corpo a corpo (Grab-based)",     "key": "grab_based"},
    {"label": "🎲 Pouco confiável (Unreliable)",      "key": "unreliable"},
    {"label": "📌 Limitado (Limited)",                "key": "limited"},
    {"label": "🧘 Requer concentração (Concentration)","key": "concentration"},
    {"label": "💥 Efeito colateral em si (Side Effect)","key": "side_effect"},
]


def _simple_mode_to_draft(what_label: str, target_label: str,
                           extra_keys: List[str], flaw_keys: List[str],
                           conditions: Optional[Dict[str, str]],
                           stat_targets: List[str],
                           rank: int, category: str,
                           pokemon_type: str) -> GolpeDraft:
    """Converte respostas do modo simples em GolpeDraft."""
    what_data = next((w for w in _WHAT_OPTIONS if w["label"] == what_label), _WHAT_OPTIONS[0])
    target_data = next((t for t in _TARGET_OPTIONS if t["label"] == target_label), _TARGET_OPTIONS[0])

    effects = what_data["effects"]
    base_extras = target_data["extras"]

    components: List[PowerComponent] = []
    for i, eff_key in enumerate(effects):
        comp = PowerComponent(
            effect_key=eff_key,
            rank=rank,
            is_linked=(i > 0),
        )
        # Extras de área/alcance (só no primeiro componente de ataque)
        if i == 0:
            for ex_key in base_extras:
                comp.extras.append({"key": ex_key, "ranks": 1, "description": ""})
            for ex_key in extra_keys:
                if ex_key not in {e["key"] for e in comp.extras}:
                    comp.extras.append({"key": ex_key, "ranks": 1, "description": ""})
            for fl_key in flaw_keys:
                comp.flaws.append({"key": fl_key, "ranks": 1, "description": ""})

        if eff_key == "affliction" and conditions:
            comp.conditions = {k: v for k, v in conditions.items() if v}
        if eff_key in ("weaken", "enhanced_trait") and stat_targets:
            comp.stat_targets = list(stat_targets)

        components.append(comp)

    return GolpeDraft(
        name="",
        category=category,
        components=components,
        pokemon_type=pokemon_type,
    )


# ─────────────────────────────────────────────
# Botão Mágico — keyword → draft
# ─────────────────────────────────────────────

# (keyword_list, {effects, extras, conditions, pokemon_type})
_KW_MAP: List[tuple] = [
    (["dano", "ataque", "golpe", "pancada", "impacto", "bate"],           {"effects": ["damage"]}),
    (["paralisa", "paralisia"],                                            {"effects": ["affliction"], "conditions": {"degree_1": "Dazed", "degree_2": "Immobile", "degree_3": "Paralyzed"}}),
    (["dorme", "sono", "adormece", "sleep"],                              {"effects": ["affliction"], "conditions": {"degree_3": "Asleep"}}),
    (["confunde", "confusão"],                                             {"effects": ["affliction"], "conditions": {"degree_1": "Dazed", "degree_2": "Stunned"}}),
    (["queima", "queimadura", "burn"],                                     {"effects": ["affliction"], "conditions": {"degree_1": "Impaired"}, "pokemon_type": "fire"}),
    (["drena", "drenar", "roubar vida", "absorve vida"],                  {"effects": ["damage", "healing"]}),
    (["cura", "curar", "recupera", "hp"],                                 {"effects": ["healing"]}),
    (["enfraquece", "reduz atributo", "diminui"],                         {"effects": ["weaken"]}),
    (["fortalece", "melhora atributo", "aumenta"],                        {"effects": ["enhanced_trait"]}),
    (["clima", "chuva", "granizo", "nevasca", "tempestade"],              {"effects": ["environment"]}),
    (["voa", "voar", "voo"],                                              {"effects": ["flight"]}),
    (["teleporta", "teleporte", "teletransporte"],                        {"effects": ["teleport"]}),
    (["nulifica", "cancela poder", "nullify"],                            {"effects": ["nullify"]}),
    (["regenera", "regeneração", "cura contínua"],                        {"effects": ["regeneration"]}),
    (["área", "explosão", "todos próximos", "em volta"],                  {"extras": ["area_burst"]}),
    (["cone", "sopra", "bafo"],                                           {"extras": ["area_cone"]}),
    (["linha", "raio direto"],                                            {"extras": ["area_line"]}),
    (["percepção", "visual", "todos que veem"],                           {"extras": ["area_perception"]}),
    (["distância", "longe", "arremessa", "lança", "projétil"],            {"extras": ["ranged"]}),
    (["progressivo", "piora a cada turno"],                               {"extras": ["progressive"]}),
    (["secundário", "efeito adicional"],                                  {"extras": ["secondary_effect"]}),
    (["fogo", "chama", "flamejante"],                                     {"pokemon_type": "fire"}),
    (["água", "aqua", "hidro"],                                           {"pokemon_type": "water"}),
    (["elétrico", "trovão", "relâmpago"],                                 {"pokemon_type": "electric"}),
    (["grama", "planta", "folha"],                                        {"pokemon_type": "grass"}),
    (["gelo", "cryo", "glacial"],                                         {"pokemon_type": "ice"}),
    (["psíquico", "psico", "mental", "telepatia"],                        {"pokemon_type": "psychic"}),
    (["dragão"],                                                           {"pokemon_type": "dragon"}),
    (["noite", "sombra", "escuro", "dark"],                               {"pokemon_type": "dark"}),
    (["fantasma", "espectro"],                                            {"pokemon_type": "ghost"}),
]


def _keyword_to_draft(text: str, rank: int) -> GolpeDraft:
    """Converte texto livre em GolpeDraft usando mapeamento de keywords."""
    text_lower = text.lower()
    effects: List[str] = []
    extras: List[str] = []
    conditions: Dict[str, str] = {}
    pokemon_type = ""

    for keywords, mapping in _KW_MAP:
        if any(kw in text_lower for kw in keywords):
            for eff in mapping.get("effects", []):
                if eff not in effects:
                    effects.append(eff)
            for ex in mapping.get("extras", []):
                if ex not in extras:
                    extras.append(ex)
            if mapping.get("conditions"):
                conditions.update(mapping["conditions"])
            if mapping.get("pokemon_type") and not pokemon_type:
                pokemon_type = mapping["pokemon_type"]

    if not effects:
        effects = ["damage"]

    # Detecta categoria
    category = "status"
    if "damage" in effects:
        category = "fisico"
    elif any(e in effects for e in ("healing", "enhanced_trait", "regeneration", "immunity", "senses", "flight", "teleport", "movement")):
        category = "status"

    components: List[PowerComponent] = []
    for i, eff_key in enumerate(effects):
        comp = PowerComponent(effect_key=eff_key, rank=rank, is_linked=(i > 0))
        if i == 0:
            for ex_key in extras:
                comp.extras.append({"key": ex_key, "ranks": 1, "description": ""})
        if eff_key == "affliction" and conditions:
            comp.conditions = conditions
        components.append(comp)

    return GolpeDraft(
        name="",
        category=category,
        components=components,
        pokemon_type=pokemon_type,
    )


# ─────────────────────────────────────────────
# Passo 1: Busca por Nome
# ─────────────────────────────────────────────

def _render_move_card(mv, rank: int, prefix: str, card_key: str):
    """Renderiza um card de Move com ações de usar/personalizar + PP via engine."""
    build = mv.render_build(rank)

    # ── PP via engine (parse_build_string) ────────────────────────
    try:
        comps = parse_build_string(build, default_rank=rank)
        if comps:
            from golpe_builder import GolpeDraft as _GD
            _d = _GD(components=comps)
            from golpe_builder import calculate_total_pp as _ctp
            pp_engine, _ = _ctp(_d)
            pp_display = int(pp_engine)
            pp_label = f"**{pp_display} PP** (calculado pela engine)"
        else:
            pp_raw, pp_why = mv.pp_cost(rank)
            pp_display = int(pp_raw) if pp_raw else 0
            pp_label = f"**{pp_display} PP** (estimativa: {pp_why})"
    except Exception:
        pp_raw, pp_why = mv.pp_cost(rank)
        pp_display = int(pp_raw) if pp_raw else 0
        pp_label = f"**{pp_display} PP** (estimativa)"

    st.code(build, language="text")
    col_meta, col_pp = st.columns([3, 1])
    with col_meta:
        st.caption(f"**Tipo:** {mv.tipo or '—'}  |  **Categoria:** {mv.categoria or '—'}")
        if mv.descricao:
            st.caption(mv.descricao[:200])
    with col_pp:
        st.metric("PP", pp_display)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Usar como está", key=_k(prefix, card_key, "use")):
            move_json = {
                "name": mv.name,
                "rank": int(rank),
                "build": build,
                "pp_cost": pp_display,
                "accuracy": 0,
                "meta": {
                    "ranged": bool(mv.ranged),
                    "perception_area": bool(mv.perception_area),
                    "category": str(mv.categoria or ""),
                },
                "_ui_id": uuid.uuid4().hex,
            }
            st.session_state[_k(prefix, "final_move")] = move_json
            _set_step(prefix, 5)
            st.rerun()
    with c2:
        if st.button("✏️ Personalizar", key=_k(prefix, card_key, "edit")):
            draft = _existing_move_to_draft(mv, rank)
            _set_draft(prefix, draft)
            _set_step(prefix, 4)
            st.rerun()


def _render_step1(prefix: str, move_db):
    """Busca o golpe pelo nome, mostra sugestões fuzzy, histórico, templates e botão mágico."""
    st.subheader("Passo 1: Buscar ou Criar golpe")

    # ══ Layout principal ═════════════════════════════════════════════
    col_search, col_magic = st.columns([3, 2])

    with col_search:
        st.markdown("##### 🔍 Buscar golpe existente")
        name = st.text_input("Nome do golpe", key=_k(prefix, "s1_name"),
                             placeholder="Ex: Surf, Flamethrower, Parting Shot…")
        rank = st.slider("Rank base", 1, 20, 10, key=_k(prefix, "s1_rank"))

        c1, c2 = st.columns(2)
        with c1:
            search_clicked = st.button("🔍 Buscar", key=_k(prefix, "s1_search"), type="primary")
        with c2:
            skip_clicked = st.button("✨ Criar novo", key=_k(prefix, "s1_skip"))

        if skip_clicked:
            draft = GolpeDraft(name=name)
            _set_draft(prefix, draft)
            _set_step(prefix, 2)
            st.rerun()
            return

    with col_magic:
        st.markdown("##### 🪄 Botão Mágico — descreva em 1 frase")
        magic_text = st.text_input(
            "Ex: uma rajada de fogo em cone que pode queimar",
            key=_k(prefix, "s1_magic_text"),
            label_visibility="collapsed",
            placeholder="Ex: golpe d'água que drena HP e paralisa…",
        )
        magic_rank = st.number_input("Rank", 1, 20, 10, key=_k(prefix, "s1_magic_rank"))
        magic_clicked = st.button("🪄 Gerar Rascunho", key=_k(prefix, "s1_magic_btn"))

        if magic_clicked and magic_text.strip():
            magic_draft = _keyword_to_draft(magic_text.strip(), int(magic_rank))
            magic_draft.name = magic_text.strip()[:40]
            _set_draft(prefix, magic_draft)
            _set_step(prefix, 2)
            st.rerun()
            return

    st.divider()

    # ══ Busca por nome ═══════════════════════════════════════════════
    result_key = _k(prefix, "s1_result")
    if search_clicked and move_db is not None:
        mv = move_db.get_by_name(name)
        if mv:
            st.session_state[result_key] = mv.name
            st.session_state[_k(prefix, "s1_suggestions")] = []
        else:
            st.session_state[result_key] = None
            # Busca fuzzy automática
            suggestions = move_db.suggest_by_description(name, top_k=5) if name.strip() else []
            st.session_state[_k(prefix, "s1_suggestions")] = [(m.name, sc) for m, sc in suggestions]
            if not suggestions:
                st.warning("Golpe não encontrado na base. Crie um novo ou use o Botão Mágico.")
    elif search_clicked and move_db is None:
        st.session_state[result_key] = None
        st.warning("Base de golpes não carregada.")

    # ── Resultado exato ───────────────────────────────────────────────
    found_name = st.session_state.get(result_key)
    if found_name and move_db is not None:
        mv = move_db.get_by_name(found_name)
        if mv:
            st.success(f"✅ **{mv.name}**")
            _render_move_card(mv, rank, prefix, "exact")

    # ── Sugestões fuzzy ───────────────────────────────────────────────
    suggestions = st.session_state.get(_k(prefix, "s1_suggestions"), [])
    if suggestions:
        st.markdown("##### 🔎 Você quis dizer…")
        for i, (mv_name, score) in enumerate(suggestions):
            mv = move_db.get_by_name(mv_name)
            if mv:
                score_pct = int(score * 100)
                with st.expander(f"**{mv.name}** — {mv.tipo} / {mv.categoria}  _(similaridade {score_pct}%)_"):
                    _render_move_card(mv, rank, prefix, f"sug_{i}")

    st.divider()

    # ══ Histórico & Templates ═════════════════════════════════════════
    col_hist, col_tpl = st.columns(2)

    with col_hist:
        history = _get_history()
        if history:
            with st.expander(f"⏱️ Últimos {len(history)} golpes criados"):
                for j, d in enumerate(reversed(history)):
                    nm = d.get("name", "?")
                    cat = d.get("category", "")
                    st.markdown(f"**{nm}** _{cat}_")
                    if st.button("↩️ Clonar", key=_k(prefix, "s1_hist_clone", str(j))):
                        cloned = GolpeDraft.from_dict(d)
                        cloned.name = f"Cópia de {cloned.name}"
                        _set_draft(prefix, cloned)
                        _set_step(prefix, 4)
                        st.rerun()
                    st.caption("")
        else:
            st.caption("_Sem histórico ainda._")

    with col_tpl:
        templates = _get_templates()
        if templates:
            with st.expander(f"📋 Meus Templates ({len(templates)})"):
                for j, d in enumerate(templates):
                    nm = d.get("name", "?")
                    st.markdown(f"**{nm}**")
                    c_use, c_del = st.columns(2)
                    with c_use:
                        if st.button("▶️ Usar", key=_k(prefix, "s1_tpl_use", str(j))):
                            tpl = GolpeDraft.from_dict(d)
                            _set_draft(prefix, tpl)
                            _set_step(prefix, 4)
                            st.rerun()
                    with c_del:
                        if st.button("🗑️", key=_k(prefix, "s1_tpl_del", str(j))):
                            _delete_template(j)
                            st.rerun()
        else:
            st.caption("_Sem templates salvos._")


def _existing_move_to_draft(mv, rank: int) -> GolpeDraft:
    """Converte um Move existente em GolpeDraft para edição."""
    # Parse simples: detectar efeitos principais do build
    build_lower = (mv.build or "").lower()
    components = []

    cat = (mv.categoria or "").lower()
    if "fisico" in cat or "físico" in cat:
        category = "fisico"
    elif "especial" in cat:
        category = "especial"
    else:
        category = "status"

    # Detectar componentes pelo texto da build
    if "damage" in build_lower:
        comp = PowerComponent(effect_key="damage", rank=rank)
        if "ranged" in build_lower:
            comp.extras.append({"key": "ranged", "ranks": 1, "description": ""})
        components.append(comp)

    if "affliction" in build_lower:
        comp = PowerComponent(effect_key="affliction", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "weaken" in build_lower:
        comp = PowerComponent(effect_key="weaken", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "healing" in build_lower:
        comp = PowerComponent(effect_key="healing", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "enhanced" in build_lower:
        comp = PowerComponent(effect_key="enhanced_trait", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "nullify" in build_lower:
        comp = PowerComponent(effect_key="nullify", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "environment" in build_lower:
        comp = PowerComponent(effect_key="environment", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "create " in build_lower:
        comp = PowerComponent(effect_key="create", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "teleport" in build_lower:
        comp = PowerComponent(effect_key="teleport", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "immunity" in build_lower:
        comp = PowerComponent(effect_key="immunity", rank=rank, is_linked=bool(components))
        components.append(comp)

    if "protection" in build_lower:
        comp = PowerComponent(effect_key="protection", rank=rank, is_linked=bool(components))
        components.append(comp)

    # Se nenhum componente detectado, criar um Damage genérico
    if not components:
        components.append(PowerComponent(effect_key="damage", rank=rank))

    return GolpeDraft(
        name=mv.name,
        category=category,
        components=components,
        pokemon_type=(mv.tipo or "").lower(),
    )


# ─────────────────────────────────────────────
# Passo 2: Assistente / Editor de Efeitos
# ─────────────────────────────────────────────

def _render_step2(prefix: str):
    """Modo Simples (assistente guiado) ou Avançado (editor completo), com preview em tempo real."""
    st.subheader("Passo 2: Definir Efeitos")

    draft = _get_draft(prefix)

    # ── Toggle de modo ────────────────────────────────────────────
    mode_key = _k(prefix, "s2_mode")
    mode = st.session_state.get(mode_key, "simple")
    col_tog1, col_tog2, col_spacer = st.columns([1, 1, 4])
    with col_tog1:
        if st.button(
            "🧭 Simples" + (" ◀" if mode == "simple" else ""),
            key=_k(prefix, "s2_btn_simple"),
            type="primary" if mode == "simple" else "secondary",
        ):
            st.session_state[mode_key] = "simple"
            st.rerun()
    with col_tog2:
        if st.button(
            "⚙️ Avançado" + (" ◀" if mode == "advanced" else ""),
            key=_k(prefix, "s2_btn_advanced"),
            type="primary" if mode == "advanced" else "secondary",
        ):
            st.session_state[mode_key] = "advanced"
            st.rerun()

    st.divider()

    # Layout: esquerda = formulário / direita = preview
    col_form, col_prev = st.columns([3, 2])

    # ══════════════════════════════════════════════════════════════
    # MODO SIMPLES
    # ══════════════════════════════════════════════════════════════
    if mode == "simple":
        with col_form:
            st.markdown("#### 🧭 Assistente de Criação")

            # Nome e rank
            col_n, col_r = st.columns([3, 1])
            with col_n:
                simple_name = st.text_input("Nome do golpe", value=draft.name, key=_k(prefix, "s2s_name"))
            with col_r:
                simple_rank = st.number_input("Rank", 1, 20, max(1, draft.components[0].rank if draft.components else 10), key=_k(prefix, "s2s_rank"))

            # Q1: O que o golpe faz?
            what_labels = [w["label"] for w in _WHAT_OPTIONS]
            what_sel = st.selectbox("1️⃣  O que o golpe faz?", what_labels, key=_k(prefix, "s2s_what"))
            what_data = next(w for w in _WHAT_OPTIONS if w["label"] == what_sel)

            # Q2: Alvo / Alcance
            target_labels = [t["label"] for t in _TARGET_OPTIONS]
            target_sel = st.selectbox("2️⃣  Como o golpe alcança o alvo?", target_labels, key=_k(prefix, "s2s_target"))

            # Q3: Categoria (só se "Causa Dano")
            cat_options = list(ADAPTATION_RULES["category_map"].keys())
            cat_labels_map = [ADAPTATION_RULES["category_map"][c]["label_pt"] for c in cat_options]
            if "damage" in what_data["effects"]:
                cat_sel = st.selectbox("3️⃣  Categoria do dano?", cat_labels_map, key=_k(prefix, "s2s_cat"))
                simple_category = cat_options[cat_labels_map.index(cat_sel)]
            else:
                simple_category = "status"

            # Q4: Condições (se Affliction selecionado)
            simple_conditions: Dict[str, str] = {}
            if what_data["show_conditions"]:
                st.markdown("4️⃣  **Graus da condição de status:**")
                for deg_key, deg_data in AFFLICTION_CONDITIONS.items():
                    opts = ["(nenhum)"] + [o["label_en"] for o in deg_data["options"]]
                    sel_c = st.selectbox(
                        deg_data["label_pt"],
                        opts,
                        key=_k(prefix, "s2s_cond", deg_key),
                    )
                    if sel_c != "(nenhum)":
                        simple_conditions[deg_key] = sel_c

            # Q5: Atributos (se Weaken/Enhanced)
            simple_stat_targets: List[str] = []
            if what_data["show_targets"]:
                target_opts = [t["label_en"] for t in STAT_TARGETS]
                simple_stat_targets = st.multiselect(
                    "4️⃣  Atributo(s) afetado(s)?",
                    target_opts,
                    key=_k(prefix, "s2s_targets"),
                )

            # Q6: Extras rápidos
            st.markdown("5️⃣  **Extras adicionais?** _(opcional)_")
            sel_extras = st.multiselect(
                "Extras",
                [e["label"] for e in _SIMPLE_EXTRAS],
                key=_k(prefix, "s2s_extras"),
                label_visibility="collapsed",
            )
            extra_keys = [e["key"] for e in _SIMPLE_EXTRAS if e["label"] in sel_extras]

            # Q7: Falhas
            st.markdown("6️⃣  **Falhas que reduzem custo?** _(opcional)_")
            sel_flaws = st.multiselect(
                "Falhas",
                [f["label"] for f in _SIMPLE_FLAWS],
                key=_k(prefix, "s2s_flaws"),
                label_visibility="collapsed",
            )
            flaw_keys = [f["key"] for f in _SIMPLE_FLAWS if f["label"] in sel_flaws]

            # Q8: Tipo pokémon
            type_options = ["(nenhum)"] + [t["key"] for t in TYPE_DESCRIPTORS]
            type_sel = st.selectbox("7️⃣  Tipo do Pokémon?", type_options, key=_k(prefix, "s2s_type"))
            pokemon_type = "" if type_sel == "(nenhum)" else type_sel

            # Build preview draft em tempo real
            preview_draft = _simple_mode_to_draft(
                what_sel, target_sel, extra_keys, flaw_keys,
                simple_conditions, simple_stat_targets,
                int(simple_rank), simple_category, pokemon_type,
            )
            preview_draft.name = simple_name or "Novo Golpe"

            # Botão gerar
            st.divider()
            if st.button("🚀 Criar Rascunho e Ir para Editor", key=_k(prefix, "s2s_generate"), type="primary"):
                preview_draft.name = simple_name or "Novo Golpe"
                _set_draft(prefix, preview_draft)
                _set_step(prefix, 4)
                st.rerun()

            if st.button("👁️ Ver Preview Completo", key=_k(prefix, "s2s_preview")):
                _set_draft(prefix, preview_draft)
                _set_step(prefix, 3)
                st.rerun()

        # Preview ao vivo (coluna direita)
        with col_prev:
            try:
                _render_live_preview(preview_draft)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════
    # MODO AVANÇADO
    # ══════════════════════════════════════════════════════════════
    else:
        with col_form:
            st.markdown("#### ⚙️ Seletor de Efeitos")

            # Nome e configurações básicas
            col_n, col_c, col_r = st.columns([3, 2, 1])
            with col_n:
                draft.name = st.text_input("Nome do golpe", value=draft.name, key=_k(prefix, "s2_name"))
            with col_c:
                cat_options = list(ADAPTATION_RULES["category_map"].keys())
                cat_labels = [ADAPTATION_RULES["category_map"][c]["label_pt"] for c in cat_options]
                cat_idx = cat_options.index(draft.category) if draft.category in cat_options else 0
                sel_cat = st.selectbox("Categoria", cat_labels, index=cat_idx, key=_k(prefix, "s2_cat"))
                draft.category = cat_options[cat_labels.index(sel_cat)]
            with col_r:
                base_rank = st.number_input("Rank base", 1, 20, 10, key=_k(prefix, "s2_rank"))

            st.divider()

            # ── Efeitos Principais ──
            st.markdown("### Efeitos Principais")
            st.caption("O que o golpe FAZ? Selecione um ou mais efeitos.")

            effects_by_type: Dict[str, List] = {"attack": [], "support": [], "utility": [], "movement": [], "sensory": []}
            for key, eff in EFFECTS.items():
                effects_by_type.setdefault(eff["type"], []).append((key, eff))

            selected_effects: List[str] = []

            col_atk, col_sup, col_util = st.columns(3)
            with col_atk:
                st.markdown("**Ataque**")
                for key, eff in effects_by_type.get("attack", []):
                    if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=f"**{eff['label_en']}** — {eff['desc_pt']}"):
                        selected_effects.append(key)
            with col_sup:
                st.markdown("**Suporte**")
                for key, eff in effects_by_type.get("support", []):
                    if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=f"**{eff['label_en']}** — {eff['desc_pt']}"):
                        selected_effects.append(key)
            with col_util:
                st.markdown("**Utilidade**")
                for key, eff in effects_by_type.get("utility", []):
                    if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=f"**{eff['label_en']}** — {eff['desc_pt']}"):
                        selected_effects.append(key)
            col_mov, col_sens = st.columns(2)
            with col_mov:
                st.markdown("**Movimento**")
                for key, eff in effects_by_type.get("movement", []):
                    if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=f"**{eff['label_en']}** — {eff['desc_pt']}"):
                        selected_effects.append(key)
            with col_sens:
                st.markdown("**Sensorial**")
                for key, eff in effects_by_type.get("sensory", []):
                    if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=f"**{eff['label_en']}** — {eff['desc_pt']}"):
                        selected_effects.append(key)

            st.divider()

    # ── Configurações por efeito ──
    effect_configs: Dict[str, Dict] = {}
    for eff_key in selected_effects:
        eff = EFFECTS[eff_key]
        with st.expander(f"Configurar: {eff['label_pt']}", expanded=True):
            config: Dict[str, Any] = {}

            # ════════════════════════════════════════════════
            # IMMUNITY — seleção de itens com custo dinâmico
            # ════════════════════════════════════════════════
            if eff_key == "immunity":
                st.markdown(
                    "Escolha os itens de imunidade. "
                    "O **rank total** é calculado automaticamente como a soma dos custos."
                )
                immunity_items: List[Dict] = []

                # Multiselect de opções padrão (exceto "custom")
                std_options = [o for o in IMMUNITY_OPTIONS if o["key"] != "custom"]
                std_labels  = [f"{o['label_pt']} ({o['cost']} PP)" for o in std_options]
                selected_labels = st.multiselect(
                    "Imunidades padrão",
                    std_labels,
                    key=_k(prefix, "s2_imm_sel", eff_key),
                )
                label_to_option = {f"{o['label_pt']} ({o['cost']} PP)": o for o in std_options}
                for lbl in selected_labels:
                    opt = label_to_option[lbl]
                    immunity_items.append({
                        "key": opt["key"],
                        "label_pt": opt["label_pt"],
                        "cost": opt["cost"],
                    })

                # Seção de imunidades personalizadas
                st.markdown("**Imunidades personalizadas** (status ou descritores únicos do seu jogo):")
                n_custom = st.number_input(
                    "Quantidade de itens personalizados",
                    min_value=0, max_value=10, value=0,
                    key=_k(prefix, "s2_imm_ncustom", eff_key),
                )
                for i in range(int(n_custom)):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        custom_lbl = st.text_input(
                            f"Nome #{i+1}", key=_k(prefix, "s2_imm_clbl", eff_key, str(i))
                        )
                    with c2:
                        custom_cost = st.number_input(
                            f"PP #{i+1}", min_value=1, max_value=30, value=1,
                            key=_k(prefix, "s2_imm_ccost", eff_key, str(i))
                        )
                    if custom_lbl:
                        immunity_items.append({
                            "key": "custom",
                            "label_pt": custom_lbl,
                            "custom_label": custom_lbl,
                            "cost": int(custom_cost),
                        })

                total_rank = sum(it["cost"] for it in immunity_items)
                st.info(f"**Rank calculado: {total_rank} PP** ({len(immunity_items)} item(s))")
                config["immunity_items"] = immunity_items
                config["rank"] = max(1, total_rank)

            # ════════════════════════════════════════════════
            # SENSES — seleção de sentidos com custo dinâmico
            # ════════════════════════════════════════════════
            elif eff_key == "senses":
                st.markdown(
                    "Escolha os sentidos. "
                    "O **rank total** é calculado automaticamente."
                )
                sense_items: List[Dict] = []

                # Divide em fixos vs per_rank para melhor apresentação
                fixed_opts   = [o for o in SENSES_OPTIONS if not o["per_rank"] and o["key"] != "custom"]
                perrank_opts = [o for o in SENSES_OPTIONS if o["per_rank"]]

                st.markdown("**Sentidos de custo fixo:**")
                fixed_labels = [f"{o['label_pt']} ({o['cost']} PP)" for o in fixed_opts]
                sel_fixed = st.multiselect(
                    "Sentidos fixos",
                    fixed_labels,
                    key=_k(prefix, "s2_sens_fixed", eff_key),
                )
                fl_map = {f"{o['label_pt']} ({o['cost']} PP)": o for o in fixed_opts}
                for lbl in sel_fixed:
                    opt = fl_map[lbl]
                    sense_items.append({
                        "key": opt["key"],
                        "label_pt": opt["label_pt"],
                        "cost": opt["cost"],
                        "per_rank": False,
                        "ranks": 1,
                    })

                st.markdown("**Sentidos por rank** (escolha e defina quantos ranks):")
                pr_labels = [o["label_pt"] for o in perrank_opts]
                sel_pr = st.multiselect(
                    "Sentidos por rank",
                    pr_labels,
                    key=_k(prefix, "s2_sens_pr", eff_key),
                )
                pr_map = {o["label_pt"]: o for o in perrank_opts}
                for lbl in sel_pr:
                    opt = pr_map[lbl]
                    r = st.number_input(
                        f"Ranks — {lbl}",
                        min_value=1, max_value=20, value=1,
                        key=_k(prefix, "s2_sens_prrank", eff_key, opt["key"]),
                    )
                    sense_items.append({
                        "key": opt["key"],
                        "label_pt": opt["label_pt"],
                        "cost": opt["cost"],
                        "per_rank": True,
                        "ranks": int(r),
                    })

                # Sentidos personalizados
                st.markdown("**Sentidos personalizados:**")
                n_custom_s = st.number_input(
                    "Quantidade de sentidos personalizados",
                    min_value=0, max_value=10, value=0,
                    key=_k(prefix, "s2_sens_ncustom", eff_key),
                )
                for i in range(int(n_custom_s)):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        slbl = st.text_input(
                            f"Nome #{i+1}", key=_k(prefix, "s2_sens_clbl", eff_key, str(i))
                        )
                    with c2:
                        scost = st.number_input(
                            f"PP #{i+1}", min_value=1, max_value=10, value=1,
                            key=_k(prefix, "s2_sens_ccost", eff_key, str(i))
                        )
                    with c3:
                        sranks = st.number_input(
                            f"Ranks #{i+1}", min_value=1, max_value=10, value=1,
                            key=_k(prefix, "s2_sens_cranks", eff_key, str(i))
                        )
                    if slbl:
                        sense_items.append({
                            "key": "custom",
                            "label_pt": slbl,
                            "custom_label": slbl,
                            "cost": int(scost),
                            "per_rank": int(sranks) > 1,
                            "ranks": int(sranks),
                        })

                total_sense_rank = sum(
                    it["cost"] * (it.get("ranks", 1) if it.get("per_rank") else 1)
                    for it in sense_items
                )
                st.info(f"**Rank calculado: {total_sense_rank} PP** ({len(sense_items)} sentido(s))")
                config["sense_items"] = sense_items
                config["rank"] = max(1, total_sense_rank)

            # ════════════════════════════════════════════════
            # COMMUNICATION — tipo + sentido customizado
            # ════════════════════════════════════════════════
            elif eff_key == "communication":
                comm_type_labels = [ct["label_pt"] for ct in COMMUNICATION_TYPES]
                sel_ct = st.selectbox(
                    "Tipo de Comunicação",
                    comm_type_labels,
                    key=_k(prefix, "s2_comm_type", eff_key),
                )
                comm_type_map = {ct["label_pt"]: ct["key"] for ct in COMMUNICATION_TYPES}
                config["communication_type"] = comm_type_map[sel_ct]

                if comm_type_map[sel_ct] == "custom":
                    config["communication_custom_sense"] = st.text_input(
                        "Nome do tipo de sentido personalizado (ex: Pesadelos, Sonhos)",
                        key=_k(prefix, "s2_comm_custom", eff_key),
                    )
                else:
                    # Campo opcional para sentido customizado mesmo em tipos padrão
                    extra_sense = st.text_input(
                        "Sense Type específico (opcional, ex: nome de sentido temático)",
                        key=_k(prefix, "s2_comm_extra", eff_key),
                    )
                    config["communication_custom_sense"] = extra_sense

                config["rank"] = st.number_input(
                    "Rank de Communication (determina o alcance)",
                    1, 20, int(base_rank),
                    key=_k(prefix, "s2_comm_rank", eff_key),
                )

            # ════════════════════════════════════════════════
            # MOVEMENT — seleção de tipos de movimento
            # ════════════════════════════════════════════════
            elif eff_key == "movement":
                st.markdown(
                    "Cada tipo de movimento = **1 rank = 2 PP**. "
                    "O rank total é calculado automaticamente."
                )
                mv_labels = [m["label_pt"] for m in MOVEMENT_OPTIONS]
                sel_mv = st.multiselect(
                    "Tipos de movimento",
                    mv_labels,
                    key=_k(prefix, "s2_mov_sel", eff_key),
                )
                mv_map_rev = {m["label_pt"]: m["key"] for m in MOVEMENT_OPTIONS}
                movement_types = [mv_map_rev[lbl] for lbl in sel_mv]
                total_mv_rank = len(movement_types)
                if total_mv_rank:
                    st.info(f"**Rank calculado: {total_mv_rank}** → {total_mv_rank * 2} PP")
                config["movement_types"] = movement_types
                config["rank"] = max(1, total_mv_rank)

            # ════════════════════════════════════════════════
            # TODOS OS OUTROS EFEITOS — configuração genérica
            # ════════════════════════════════════════════════
            else:
                # Rank individual
                config["rank"] = st.number_input(
                    f"Rank de {eff['label_en']}", 1, 20, int(base_rank),
                    key=_k(prefix, "s2_eff_rank", eff_key)
                )

                # Condições para Affliction
                if eff.get("has_conditions"):
                    st.markdown("**Condições (escolha uma por grau):**")
                    for deg_key, deg_data in AFFLICTION_CONDITIONS.items():
                        options = ["(nenhuma)"] + [o["label_en"] for o in deg_data["options"]]
                        sel = st.selectbox(
                            deg_data["label_pt"],
                            options,
                            key=_k(prefix, "s2_cond", eff_key, deg_key),
                            help=None,
                        )
                        config.setdefault("conditions", {})[deg_key] = sel if sel != "(nenhuma)" else ""

                # Alvos para Weaken / Enhanced Trait
                if eff.get("has_targets"):
                    target_options = [t["label_en"] for t in STAT_TARGETS]
                    selected_targets = st.multiselect(
                        "Atributos alvo",
                        target_options,
                        key=_k(prefix, "s2_targets", eff_key),
                    )
                    config["stat_targets"] = selected_targets

                # Resistência override
                resist_default = eff.get("default_resist") or "Toughness"
                from powers_data import RESIST_STATS
                resist_options = RESIST_STATS
                resist_idx = resist_options.index(resist_default) if resist_default in resist_options else 0
                if eff.get("default_resist"):
                    config["resist"] = st.selectbox(
                        "Resistido por",
                        resist_options,
                        index=resist_idx,
                        key=_k(prefix, "s2_resist", eff_key),
                    )

            effect_configs[eff_key] = config

            # ── Extras Rápidos (condicionais) ─────────────────────────────
            st.divider()
            st.markdown("### Modificadores Rápidos (Extras)")
            st.caption("Mostra apenas extras relevantes para os efeitos selecionados.")

            quick_extras: List[str] = []
            has_attack = any(EFFECTS.get(k, {}).get("type") == "attack" for k in selected_effects)
            has_area = False  # será True se algum area extra for marcado

            extras_range = [(k, v) for k, v in EXTRAS.items() if v.get("group") == "range"]
            extras_area = [(k, v) for k, v in EXTRAS.items() if v.get("group") == "area"]
            extras_other = [(k, v) for k, v in EXTRAS.items() if "group" not in v]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Alcance**")
                for key, ex in extras_range:
                    if st.checkbox(ex["label_pt"], key=_k(prefix, "s2_qex", eff_key, key), help=f"**{ex['label_en']}** — {ex['desc_pt']}"):
                        quick_extras.append(key)
            with c2:
                st.markdown("**Área**")
                for key, ex in extras_area:
                    checked = st.checkbox(ex["label_pt"], key=_k(prefix, "s2_qex", eff_key, key), help=f"**{ex['label_en']}** — {ex['desc_pt']}")
                    if checked:
                        quick_extras.append(key)
                        has_area = True
            with c3:
                st.markdown("**Outros**")
                # Esconde extras irrelevantes: Selective só se há área; Accurate só se há ataque
                common_extras = ["selective", "accurate", "counter", "reaction", "cumulative", "improved_critical"]
                for key in common_extras:
                    if key not in EXTRAS:
                        continue
                    # Escolhas condicionais
                    if key == "selective" and not has_area:
                        continue
                    if key in ("accurate", "improved_critical", "counter") and not has_attack:
                        continue
                    ex = EXTRAS[key]
                    if st.checkbox(ex["label_pt"], key=_k(prefix, "s2_qex", eff_key, key), help=f"**{ex['label_en']}** — {ex['desc_pt']}"):
                        quick_extras.append(key)

            # Extras avançados (compatíveis com ao menos um efeito selecionado)
            relevant_remaining = []
            for k, v in extras_other:
                if k in common_extras:
                    continue
                compatible = v.get("compatible_effects")
                if compatible is None or any(e in compatible for e in selected_effects):
                    relevant_remaining.append((k, v))
            if relevant_remaining:
                with st.expander("Mais extras…"):
                    for key, ex in relevant_remaining:
                        if st.checkbox(ex["label_pt"], key=_k(prefix, "s2_qex", eff_key, key), help=f"**{ex['label_en']}** — {ex['desc_pt']}"):
                            quick_extras.append(key)

            st.divider()

            # ── Falhas Rápidas ──────────────────────────────────────────────
            st.markdown("### Falhas (reduzem custo)")
            quick_flaws: List[Dict] = []
            common_flaws = ["limited", "tiring", "unreliable", "grab_based", "concentration", "side_effect"]
            c1, c2 = st.columns(2)
            half = len(common_flaws) // 2

            def _render_flaw_check(key: str, col):
                if key not in FLAWS:
                    return
                fl = FLAWS[key]
                with col:
                    if st.checkbox(fl["label_pt"], key=_k(prefix, "s2_qfl", eff_key, key), help=f"**{fl['label_en']}** — {fl['desc_pt']}"):
                        desc = ""
                        if fl.get("has_description"):
                            desc = st.text_input(f"Detalhe ({fl['label_en']})", key=_k(prefix, "s2_qfl_desc", key))
                        quick_flaws.append({"key": key, "ranks": 1, "description": desc})

            for key in common_flaws[:half + 1]:
                _render_flaw_check(key, c1)
            for key in common_flaws[half + 1:]:
                _render_flaw_check(key, c2)

            remaining_flaws = [k for k in FLAWS if k not in common_flaws]
            if remaining_flaws:
                with st.expander("Mais falhas…"):
                    for key in remaining_flaws:
                        fl = FLAWS[key]
                        if st.checkbox(fl["label_pt"], key=_k(prefix, "s2_qfl", eff_key, key), help=f"**{fl['label_en']}** — {fl['desc_pt']}"):
                            desc = ""
                            if fl.get("has_description"):
                                desc = st.text_input(f"Detalhe ({fl['label_en']})", key=_k(prefix, "s2_qfl_desc", key))
                            quick_flaws.append({"key": key, "ranks": 1, "description": desc})

            st.divider()

            # ── Gerar Build (modo avançado) ─────────────────────────────────
            if st.button("🚀 Gerar Build", key=_k(prefix, "s2_generate"), type="primary", disabled=not selected_effects):
                components_adv: List[PowerComponent] = []
                for i, eff_key in enumerate(selected_effects):
                    cfg = effect_configs.get(eff_key, {})
                    comp = PowerComponent(
                        effect_key=eff_key,
                        rank=int(cfg.get("rank", base_rank)),
                        is_linked=(i > 0),
                        conditions=cfg.get("conditions"),
                        stat_targets=cfg.get("stat_targets", []),
                        resist_override=cfg.get("resist"),
                        immunity_items=cfg.get("immunity_items", []),
                        sense_items=cfg.get("sense_items", []),
                        communication_type=cfg.get("communication_type", ""),
                        communication_custom_sense=cfg.get("communication_custom_sense", ""),
                        movement_types=cfg.get("movement_types", []),
                    )
                    if i == 0 or eff_key in ("damage", "affliction", "weaken"):
                        for ex_key in quick_extras:
                            comp.extras.append({"key": ex_key, "ranks": 1, "description": ""})
                    if i == 0:
                        for fl in quick_flaws:
                            comp.flaws.append(dict(fl))
                    components_adv.append(comp)

                adv_draft = GolpeDraft(
                    name=draft.name,
                    category=draft.category,
                    components=components_adv,
                    pokemon_type=draft.pokemon_type,
                )
                _set_draft(prefix, adv_draft)
                _set_step(prefix, 3)
                st.rerun()

            if not selected_effects:
                st.info("Selecione pelo menos um efeito para gerar a build.")

        # ── Preview ao vivo — modo avançado ─────────────────────────────
        with col_prev:
            if selected_effects:
                try:
                    preview_comps: List[PowerComponent] = []
                    for i, eff_key in enumerate(selected_effects):
                        cfg = effect_configs.get(eff_key, {})
                        preview_comps.append(PowerComponent(
                            effect_key=eff_key,
                            rank=int(cfg.get("rank", base_rank)),
                            is_linked=(i > 0),
                            conditions=cfg.get("conditions"),
                            stat_targets=cfg.get("stat_targets", []),
                            immunity_items=cfg.get("immunity_items", []),
                            sense_items=cfg.get("sense_items", []),
                            movement_types=cfg.get("movement_types", []),
                        ))
                    preview_d = GolpeDraft(
                        name=draft.name or "Novo Golpe",
                        category=draft.category,
                        components=preview_comps,
                    )
                    _render_live_preview(preview_d)
                except Exception:
                    pass
            else:
                st.info("Selecione efeitos para ver o preview.")

    # ── Botão Voltar (fora do bloco de modo) ─────────────────────────
    st.divider()
    if st.button("← Voltar ao Passo 1", key=_k(prefix, "s2_back")):
        _set_step(prefix, 1)
        st.rerun()


# ─────────────────────────────────────────────
# Passo 3: Build Gerada com Explicação
# ─────────────────────────────────────────────

def _render_step3(prefix: str):
    """Mostra a build gerada, explicação e custo."""
    st.subheader("Passo 3: Build Gerada")

    draft = _get_draft(prefix)
    if not draft.components:
        st.error("Nenhum componente definido. Volte ao Passo 2.")
        if st.button("Voltar", key=_k(prefix, "s3_back_err")):
            _set_step(prefix, 2)
            st.rerun()
        return

    # Build string
    build = generate_build_string(draft)
    st.markdown("### Build M&M")
    st.code(build, language="text")

    # PP
    pp, explanations = calculate_total_pp(draft)
    st.metric("PP Total", f"{int(pp)}")
    with st.expander("Detalhes do cálculo de PP"):
        for exp in explanations:
            st.write(exp)

    # Explicação
    st.markdown("### O que cada parte faz")
    for item in generate_explanation(draft):
        st.markdown(f"**{item['component']}** — {item['cost']} PP")
        st.caption(item["explanation_pt"])

    # Validação
    warnings = validate_draft(draft)
    for w in warnings:
        st.warning(w)

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Personalizar (Editor)", key=_k(prefix, "s3_customize"), type="primary"):
            _set_step(prefix, 4)
            st.rerun()
    with c2:
        if st.button("Confirmar e Salvar", key=_k(prefix, "s3_confirm")):
            draft.description_pt = generate_description_pt(draft)
            _set_draft(prefix, draft)
            st.session_state[_k(prefix, "final_move")] = draft_to_move_json(draft)
            _set_step(prefix, 5)
            st.rerun()
    with c3:
        if st.button("Voltar ao Passo 2", key=_k(prefix, "s3_back")):
            _set_step(prefix, 2)
            st.rerun()


# ─────────────────────────────────────────────
# Passo 4: Editor Interativo
# ─────────────────────────────────────────────

def _get_compatible_extras(effect_key: str) -> List[str]:
    """Retorna extras compativeis com um efeito especifico."""
    result = []
    for k, v in EXTRAS.items():
        compatible = v.get("compatible_effects")
        if compatible is None or effect_key in compatible:
            result.append(k)
    return result


def _get_compatible_flaws(effect_key: str) -> List[str]:
    """Retorna falhas compativeis com um efeito especifico."""
    result = []
    for k, v in FLAWS.items():
        compatible = v.get("compatible_effects")
        if compatible is None or effect_key in compatible:
            result.append(k)
    return result


def _render_step4(prefix: str):
    """Editor completo: ajustar rank, extras, falhas, condicoes, vantagens, kits."""
    st.subheader("Passo 4: Editor Interativo")

    draft = _get_draft(prefix)

    # Nome e categoria
    col_n, col_c = st.columns([3, 1])
    with col_n:
        draft.name = st.text_input("Nome do golpe", value=draft.name, key=_k(prefix, "s4_name"))
    with col_c:
        cat_options = list(ADAPTATION_RULES["category_map"].keys())
        cat_labels = [ADAPTATION_RULES["category_map"][c]["label_pt"] for c in cat_options]
        cat_idx = cat_options.index(draft.category) if draft.category in cat_options else 0
        sel_cat = st.selectbox("Categoria", cat_labels, index=cat_idx, key=_k(prefix, "s4_cat"))
        draft.category = cat_options[cat_labels.index(sel_cat)]

    # ── Kits Pre-construidos ──
    st.markdown("### Kits Pre-construidos (Status / Clima)")
    st.caption("Adicione um pacote completo de efeitos com um clique. Os componentes serao adicionados como Linked.")
    kit_keys = list(PRESET_KITS.keys())
    kit_labels = [f"{PRESET_KITS[k]['label_pt']}" for k in kit_keys]

    c_kit1, c_kit2 = st.columns([4, 1])
    with c_kit1:
        sel_kit = st.selectbox(
            "Escolher kit",
            ["---"] + kit_keys,
            format_func=lambda k: PRESET_KITS[k]["label_pt"] if k in PRESET_KITS else "---",
            key=_k(prefix, "s4_kit_sel"),
        )
        if sel_kit != "---":
            kit = PRESET_KITS[sel_kit]
            st.caption(kit["desc_pt"])
    with c_kit2:
        if st.button("Inserir Kit", key=_k(prefix, "s4_kit_add")) and sel_kit != "---":
            kit = PRESET_KITS[sel_kit]
            for comp_data in kit["components"]:
                new_comp = PowerComponent(
                    effect_key=comp_data["effect_key"],
                    rank=comp_data.get("rank", 1),
                    is_linked=comp_data.get("is_linked", True),
                    extras=[dict(e) for e in comp_data.get("extras", [])],
                    flaws=[dict(f) for f in comp_data.get("flaws", [])],
                    conditions=dict(comp_data["conditions"]) if comp_data.get("conditions") else None,
                    stat_targets=list(comp_data.get("stat_targets", [])),
                    resist_override=comp_data.get("resist_override"),
                )
                draft.components.append(new_comp)
            _set_draft(prefix, draft)
            st.rerun()

    st.divider()

    # ── Componentes ──
    components_to_remove: List[int] = []
    for i, comp in enumerate(draft.components):
        eff = EFFECTS.get(comp.effect_key, {})
        linked_label = " (Linked)" if comp.is_linked else ""
        cost_i, _ = calculate_component_cost(comp)

        with st.expander(f"Componente {i+1}: {eff.get('label_pt', comp.effect_key)}{linked_label} — {cost_i} PP", expanded=True):

            # Rank e custo
            c_r, c_cost, c_del = st.columns([2, 1, 1])
            with c_r:
                comp.rank = st.number_input(
                    "Rank", 1, 20, int(comp.rank),
                    key=_k(prefix, "s4_rank", str(i))
                )
            with c_cost:
                st.metric("Custo", f"{cost_i} PP")
            with c_del:
                if len(draft.components) > 1:
                    if st.button("Remover", key=_k(prefix, "s4_del", str(i))):
                        components_to_remove.append(i)

            # Condições (Affliction)
            if comp.effect_key == "affliction":
                st.markdown("**Condições:**")
                if comp.conditions is None:
                    comp.conditions = {}
                for deg_key, deg_data in AFFLICTION_CONDITIONS.items():
                    options = ["(nenhuma)"] + [o["label_en"] for o in deg_data["options"]]
                    current = comp.conditions.get(deg_key, "")
                    idx = options.index(current) if current in options else 0
                    sel = st.selectbox(
                        deg_data["label_pt"],
                        options, index=idx,
                        key=_k(prefix, "s4_cond", str(i), deg_key),
                    )
                    comp.conditions[deg_key] = sel if sel != "(nenhuma)" else ""

            # Alvos (Weaken / Enhanced Trait)
            if eff.get("has_targets"):
                target_options = [t["label_en"] for t in STAT_TARGETS]
                comp.stat_targets = st.multiselect(
                    "Atributos alvo",
                    target_options,
                    default=comp.stat_targets,
                    key=_k(prefix, "s4_targets", str(i)),
                )

            # Resistência
            if eff.get("default_resist"):
                from powers_data import RESIST_STATS
                current_resist = comp.resist_override or eff["default_resist"]
                r_idx = RESIST_STATS.index(current_resist) if current_resist in RESIST_STATS else 0
                comp.resist_override = st.selectbox(
                    "Resistido por",
                    RESIST_STATS, index=r_idx,
                    key=_k(prefix, "s4_resist", str(i)),
                )

            st.markdown("---")

            # ── Extras deste componente ──
            st.markdown("**Extras:**")
            extras_to_remove: List[int] = []
            for j, ex in enumerate(comp.extras):
                edata = EXTRAS.get(ex["key"], {})
                c_e1, c_e2, c_e3 = st.columns([4, 1, 1])
                with c_e1:
                    st.write(f"{edata.get('label_pt', ex['key'])}")
                    st.caption(edata.get("desc_pt", ""))
                    # Campo de texto para extras com has_description (Custom, etc.)
                    if edata.get("has_description"):
                        ex["description"] = st.text_input(
                            "Descricao do extra",
                            value=ex.get("description", ""),
                            key=_k(prefix, "s4_exdesc", str(i), str(j)),
                            placeholder="Descreva o efeito customizado...",
                        )
                with c_e2:
                    max_r = edata.get("max_ranks", 10)
                    if edata.get("cost_type") in ("flat_per_rank",) or max_r > 1:
                        ex["ranks"] = st.number_input(
                            "Ranks", 1, max_r, int(ex.get("ranks", 1)),
                            key=_k(prefix, "s4_exr", str(i), str(j)),
                        )
                with c_e3:
                    if st.button("X", key=_k(prefix, "s4_exdel", str(i), str(j))):
                        extras_to_remove.append(j)

            # Remover extras marcados
            for idx in sorted(extras_to_remove, reverse=True):
                comp.extras.pop(idx)

            # Adicionar extra (filtrado por compatibilidade com o efeito)
            existing_extra_keys = {e["key"] for e in comp.extras}
            compatible_extras = _get_compatible_extras(comp.effect_key)
            available_extras = [k for k in compatible_extras if k not in existing_extra_keys]
            if available_extras:
                c_add1, c_add2 = st.columns([4, 1])
                with c_add1:
                    new_ex = st.selectbox(
                        "Adicionar extra",
                        ["---"] + available_extras,
                        format_func=lambda k: EXTRAS[k]["label_pt"] if k in EXTRAS else "---",
                        key=_k(prefix, "s4_new_ex", str(i)),
                    )
                with c_add2:
                    if st.button("+", key=_k(prefix, "s4_add_ex", str(i))) and new_ex != "---":
                        comp.extras.append({"key": new_ex, "ranks": 1, "description": ""})
                        _set_draft(prefix, draft)
                        st.rerun()

            st.markdown("---")

            # ── Falhas deste componente ──
            st.markdown("**Falhas:**")
            flaws_to_remove: List[int] = []
            for j, fl in enumerate(comp.flaws):
                fdata = FLAWS.get(fl["key"], {})
                c_f1, c_f2, c_f3 = st.columns([4, 2, 1])
                with c_f1:
                    st.write(f"{fdata.get('label_pt', fl['key'])}")
                    st.caption(fdata.get("desc_pt", ""))
                with c_f2:
                    if fdata.get("has_description"):
                        fl["description"] = st.text_input(
                            "Detalhe",
                            value=fl.get("description", ""),
                            key=_k(prefix, "s4_fldesc", str(i), str(j)),
                        )
                with c_f3:
                    if st.button("X", key=_k(prefix, "s4_fldel", str(i), str(j))):
                        flaws_to_remove.append(j)

            for idx in sorted(flaws_to_remove, reverse=True):
                comp.flaws.pop(idx)

            # Adicionar falha (filtrada por compatibilidade com o efeito)
            existing_flaw_keys = {f["key"] for f in comp.flaws}
            compatible_flaws = _get_compatible_flaws(comp.effect_key)
            available_flaws = [k for k in compatible_flaws if k not in existing_flaw_keys]
            if available_flaws:
                c_add1, c_add2 = st.columns([4, 1])
                with c_add1:
                    new_fl = st.selectbox(
                        "Adicionar falha",
                        ["---"] + available_flaws,
                        format_func=lambda k: FLAWS[k]["label_pt"] if k in FLAWS else "---",
                        key=_k(prefix, "s4_new_fl", str(i)),
                    )
                with c_add2:
                    if st.button("+", key=_k(prefix, "s4_add_fl", str(i))) and new_fl != "---":
                        comp.flaws.append({"key": new_fl, "ranks": 1, "description": ""})
                        _set_draft(prefix, draft)
                        st.rerun()

    # Remover componentes marcados
    for idx in sorted(components_to_remove, reverse=True):
        draft.components.pop(idx)

    st.divider()

    # ── Adicionar componente ──
    st.markdown("### Adicionar Componente (Linked)")
    available_effects = [k for k in EFFECTS if k not in {c.effect_key for c in draft.components}]
    if available_effects:
        c_add1, c_add2 = st.columns([4, 1])
        with c_add1:
            new_eff = st.selectbox(
                "Novo efeito",
                ["---"] + available_effects,
                format_func=lambda k: EFFECTS[k]["label_pt"] if k in EFFECTS else "---",
                key=_k(prefix, "s4_new_comp"),
            )
        with c_add2:
            if st.button("+ Adicionar", key=_k(prefix, "s4_add_comp")) and new_eff != "---":
                main_rank = draft.components[0].rank if draft.components else 1
                draft.components.append(PowerComponent(
                    effect_key=new_eff,
                    rank=main_rank,
                    is_linked=True,
                ))
                _set_draft(prefix, draft)
                st.rerun()

    st.divider()

    # ── Vantagens de Combate ──
    st.markdown("### Vantagens de Combate")
    st.caption("Selecione vantagens de combate para o golpe (opcionais).")

    current_adv_keys = {a["key"] for a in draft.advantages}
    adv_to_remove: List[int] = []

    # Mostrar vantagens atuais
    for j, adv in enumerate(draft.advantages):
        adv_data = next((a for a in COMBAT_ADVANTAGES if a["key"] == adv["key"]), None)
        if adv_data:
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                st.write(f"{adv_data['label_pt']}")
                st.caption(adv_data["desc_pt"])
            with c2:
                if adv_data.get("has_ranks"):
                    adv["ranks"] = st.number_input(
                        "Ranks", 1, adv_data.get("max_ranks", 10), int(adv.get("ranks", 1)),
                        key=_k(prefix, "s4_advr", str(j)),
                    )
            with c3:
                if st.button("X", key=_k(prefix, "s4_advdel", str(j))):
                    adv_to_remove.append(j)

    for idx in sorted(adv_to_remove, reverse=True):
        draft.advantages.pop(idx)

    # Adicionar vantagem
    available_advs = [a["key"] for a in COMBAT_ADVANTAGES if a["key"] not in current_adv_keys]
    if available_advs:
        c_add1, c_add2 = st.columns([4, 1])
        with c_add1:
            new_adv = st.selectbox(
                "Adicionar vantagem",
                ["---"] + available_advs,
                format_func=lambda k: next((a["label_pt"] for a in COMBAT_ADVANTAGES if a["key"] == k), k) if k != "---" else "---",
                key=_k(prefix, "s4_new_adv"),
            )
        with c_add2:
            if st.button("+", key=_k(prefix, "s4_add_adv")) and new_adv != "---":
                draft.advantages.append({"key": new_adv, "ranks": 1})
                _set_draft(prefix, draft)
                st.rerun()

    st.divider()

    # ── Preview e PP ──
    st.markdown("### Preview")
    build = generate_build_string(draft)
    st.code(build, language="text")

    pp, explanations = calculate_total_pp(draft)
    st.metric("PP Total", f"{int(pp)}")

    with st.expander("Detalhes do cálculo"):
        for exp in explanations:
            st.write(exp)

    with st.expander("Explicação detalhada"):
        for item in generate_explanation(draft):
            st.markdown(f"**{item['component']}** — {item['cost']} PP")
            st.caption(item["explanation_pt"])

    warnings = validate_draft(draft)
    for w in warnings:
        st.warning(w)

    st.divider()

    # Salvar draft atualizado
    _set_draft(prefix, draft)

    # Botões de navegação
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Confirmar e Salvar", key=_k(prefix, "s4_confirm"), type="primary"):
            draft.description_pt = generate_description_pt(draft)
            _set_draft(prefix, draft)
            st.session_state[_k(prefix, "final_move")] = draft_to_move_json(draft)
            _set_step(prefix, 5)
            st.rerun()
    with c2:
        if st.button("Voltar ao Passo 3", key=_k(prefix, "s4_back3")):
            _set_step(prefix, 3)
            st.rerun()
    with c3:
        if st.button("Recomeçar", key=_k(prefix, "s4_restart")):
            _set_step(prefix, 1)
            st.rerun()


# ─────────────────────────────────────────────
# Passo 5: Salvar
# ─────────────────────────────────────────────

def _render_step5(prefix: str, trainer_name: str):
    """Salvar o golpe no formato JSON + Firestore."""
    st.subheader("Passo 5: Salvar Golpe")

    draft = _get_draft(prefix)
    final_move = st.session_state.get(_k(prefix, "final_move"))

    if final_move:
        st.markdown("### Golpe Finalizado")
        st.write(f"**Nome:** {final_move['name']}")
        st.write(f"**Rank:** {final_move['rank']}")
        st.write(f"**PP:** {final_move['pp_cost']}")
        st.write(f"**Categoria:** {final_move['meta']['category']}")
        st.code(final_move["build"], language="text")

    # Descrição natural
    st.markdown("### Descrição em Linguagem Natural")
    auto_desc = generate_description_pt(draft) if draft.components else ""
    description = st.text_area(
        "Edite a descrição (gerada automaticamente, mas você pode modificar)",
        value=draft.description_pt or auto_desc,
        height=120,
        key=_k(prefix, "s5_desc"),
    )
    draft.description_pt = description
    _set_draft(prefix, draft)

    # Accuracy
    accuracy = st.number_input("Accuracy (0 = automático)", 0, 20, 0, key=_k(prefix, "s5_accuracy"))

    st.divider()

    # ── Template e Histórico ──────────────────────────────────────────────
    col_tpl, col_hist = st.columns([1, 1])
    with col_tpl:
        if st.button("📋 Salvar como Template", key=_k(prefix, "s5_save_tpl"), use_container_width=True):
            _save_template(draft)
            st.success(f"Template '{draft.name}' salvo! Disponível no Passo 1.")
    with col_hist:
        st.caption(f"Histórico: {len(_get_history())}/{_MAX_HISTORY} golpes salvos")

    st.divider()

    # ── Botões de salvamento ──────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Adicionar à ficha (cg_moves)", key=_k(prefix, "s5_save_session"), type="primary"):
            move_json = final_move or draft_to_move_json(draft, accuracy=accuracy)
            move_json["accuracy"] = int(accuracy)
            if "cg_moves" not in st.session_state:
                st.session_state["cg_moves"] = []
            st.session_state["cg_moves"].append(move_json)
            _save_to_history(draft)  # ← salva no histórico
            st.success(f"Golpe '{move_json['name']}' adicionado à ficha!")
            st.json(move_json)

    with c2:
        if st.button("Salvar na nuvem (Firestore)", key=_k(prefix, "s5_save_firestore")):
            try:
                db, _ = _init_firebase_safe()
                if db is None:
                    st.error("Firebase não configurado. Configure st.secrets['firebase_service_account'].")
                else:
                    doc = draft_to_firestore_doc(draft, trainer_name)
                    trainer_id = trainer_name.replace(" ", "_").replace("/", "_")
                    move_id = f"{draft.name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

                    db.collection("trainers") \
                      .document(trainer_id) \
                      .collection("custom_moves") \
                      .document(move_id) \
                      .set(doc)

                    # Também salvar no session_state e histórico
                    move_json = final_move or draft_to_move_json(draft, accuracy=accuracy)
                    move_json["accuracy"] = int(accuracy)
                    if "cg_moves" not in st.session_state:
                        st.session_state["cg_moves"] = []
                    st.session_state["cg_moves"].append(move_json)
                    _save_to_history(draft)  # ← salva no histórico

                    st.success(f"Golpe '{draft.name}' salvo na nuvem e adicionado à ficha!")
                    st.json(doc)
            except Exception as e:
                st.error(f"Erro ao salvar no Firestore: {e}")

    st.divider()

    # ── Live preview do golpe finalizado ─────────────────────────────────
    _render_live_preview(draft, compact=True)

    # ── JSON final para copiar ────────────────────────────────────────────
    if final_move:
        with st.expander("JSON do golpe (para copiar)"):
            st.json(final_move)

    # ── Navegação ─────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Voltar ao Editor", key=_k(prefix, "s5_back")):
            _set_step(prefix, 4)
            st.rerun()
    with c2:
        if st.button("🔄 Criar outro golpe", key=_k(prefix, "s5_new")):
            # Limpar state (preserva histórico e templates que estão em chaves globais)
            for key in list(st.session_state.keys()):
                if key.startswith(prefix):
                    del st.session_state[key]
            _set_step(prefix, 1)
            st.rerun()


def _init_firebase_safe():
    """Inicializa Firebase de forma segura (retorna None se não configurado)."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage

        if not firebase_admin._apps:
            raw = st.secrets.get("firebase_service_account")
            if not raw:
                return None, None
            cred_dict = {k: raw[k] for k in raw.keys()}
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                "projectId": cred_dict.get("project_id", ""),
                "storageBucket": "batalhas-de-gaal.firebasestorage.app",
            })

        db = firestore.client()
        bucket = storage.bucket()
        return db, bucket
    except Exception:
        return None, None


# ─────────────────────────────────────────────
# Função principal (ponto de entrada)
# ─────────────────────────────────────────────

def render_golpe_builder(
    excel_path: str = "golpes_pokemon_MM_reescritos.xlsx",
    state_key_prefix: str = "gb",
    return_to_view: str | None = None,
    trainer_name: str = "Treinador",
):
    """
    Renderiza o Golpe Builder completo.

    Parâmetros:
        excel_path: caminho para o Excel com golpes existentes
        state_key_prefix: prefixo para chaves do session_state
        return_to_view: se definido, mostra botão para voltar à view
        trainer_name: nome do treinador para salvar no Firestore
    """
    prefix = state_key_prefix

    st.title("Golpe Builder")
    st.caption("Crie golpes customizados usando o sistema M&M adaptado para Pokémon.")

    # Carregar MoveDB (se disponível)
    move_db = None
    try:
        from move_db import MoveDB
        if os.path.exists(excel_path):
            move_db = _load_move_db(excel_path)
    except ImportError:
        pass

    # Indicador de progresso
    step = _get_step(prefix)
    step_labels = ["Buscar", "Efeitos", "Preview", "Editor", "Salvar"]
    cols = st.columns(5)
    for idx, label in enumerate(step_labels):
        with cols[idx]:
            if idx + 1 == step:
                st.markdown(f"**:orange[{idx+1}. {label}]**")
            elif idx + 1 < step:
                st.markdown(f":green[{idx+1}. {label}]")
            else:
                st.markdown(f":gray[{idx+1}. {label}]")

    st.divider()

    # Renderizar passo atual
    if step == 1:
        _render_step1(prefix, move_db)
    elif step == 2:
        _render_step2(prefix)
    elif step == 3:
        _render_step3(prefix)
    elif step == 4:
        _render_step4(prefix)
    elif step == 5:
        _render_step5(prefix, trainer_name)

    # Botão voltar à view (para integração com app.py)
    if return_to_view:
        st.divider()
        if st.button("Voltar", key=_k(prefix, "return_view")):
            st.session_state["cg_view"] = return_to_view
            st.rerun()


@st.cache_resource
def _load_move_db(excel_path: str):
    """Carrega MoveDB com cache."""
    from move_db import MoveDB
    return MoveDB.from_excel(excel_path, sheet_name="Golpes_MM")


# ─────────────────────────────────────────────
# Standalone mode: streamlit run golpe_builder_ui.py
# ─────────────────────────────────────────────
# Quando importado como módulo, nada acontece.
# Quando rodado diretamente pelo Streamlit, renderiza o builder.

def _main():
    st.set_page_config(page_title="Golpe Builder", layout="wide")
    render_golpe_builder()

if __name__ == "__main__":
    _main()
