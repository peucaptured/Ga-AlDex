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
# Passo 1: Busca por Nome
# ─────────────────────────────────────────────

def _render_step1(prefix: str, move_db):
    """Busca o golpe pelo nome na base existente."""
    st.subheader("Passo 1: Buscar golpe existente")
    st.caption("Digite o nome do golpe. Se existir na base, você pode usá-lo como está ou personalizá-lo.")

    name = st.text_input("Nome do golpe", key=_k(prefix, "s1_name"))
    rank = st.slider("Rank base", 1, 20, 10, key=_k(prefix, "s1_rank"))

    col1, col2 = st.columns(2)
    with col1:
        search_clicked = st.button("Buscar", key=_k(prefix, "s1_search"), type="primary")
    with col2:
        skip_clicked = st.button("Pular para criador (golpe novo)", key=_k(prefix, "s1_skip"))

    if skip_clicked:
        draft = GolpeDraft(name=name)
        _set_draft(prefix, draft)
        _set_step(prefix, 2)
        st.rerun()
        return

    result_key = _k(prefix, "s1_result")
    if search_clicked and move_db is not None:
        mv = move_db.get_by_name(name)
        if mv:
            st.session_state[result_key] = mv.name
        else:
            st.session_state[result_key] = None
            st.warning("Golpe não encontrado na base. Use o botão acima para criar um novo.")

    # Mostrar resultado persistente
    found_name = st.session_state.get(result_key)
    if found_name and move_db is not None:
        mv = move_db.get_by_name(found_name)
        if mv:
            st.success(f"Encontrado: **{mv.name}** ({mv.tipo} / {mv.categoria})")
            build = mv.render_build(rank)
            st.code(build, language="text")
            st.write(f"**Descrição:** {mv.descricao or '—'}")
            pp, why = mv.pp_cost(rank)
            if pp:
                st.info(f"PP: **{pp}** — {why}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Usar como está", key=_k(prefix, "s1_use_asis")):
                    # Salva diretamente como JSON no formato existente
                    move_json = {
                        "name": mv.name,
                        "rank": int(rank),
                        "build": build,
                        "pp_cost": int(pp) if pp else 0,
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
                if st.button("Personalizar este golpe", key=_k(prefix, "s1_customize")):
                    # Criar draft a partir do golpe existente
                    draft = _existing_move_to_draft(mv, rank)
                    _set_draft(prefix, draft)
                    _set_step(prefix, 4)
                    st.rerun()


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
# Passo 2: Seleção de Palavras-Chave
# ─────────────────────────────────────────────

def _render_step2(prefix: str):
    """Sopa de letrinhas: seleção de efeitos, extras, falhas por categoria."""
    st.subheader("Passo 2: Selecione os efeitos do golpe")
    st.caption("Marque tudo que o seu golpe faz. Depois o sistema monta a build automaticamente.")

    draft = _get_draft(prefix)

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

    # Organizar por tipo
    effects_by_type = {"attack": [], "support": [], "utility": [], "movement": [], "sensory": []}
    for key, eff in EFFECTS.items():
        effects_by_type.setdefault(eff["type"], []).append((key, eff))

    selected_effects: List[str] = []

    col_atk, col_sup, col_util = st.columns(3)
    with col_atk:
        st.markdown("**Ataque**")
        for key, eff in effects_by_type.get("attack", []):
            if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=eff["desc_pt"]):
                selected_effects.append(key)

    with col_sup:
        st.markdown("**Suporte**")
        for key, eff in effects_by_type.get("support", []):
            if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=eff["desc_pt"]):
                selected_effects.append(key)

    with col_util:
        st.markdown("**Utilidade**")
        for key, eff in effects_by_type.get("utility", []):
            if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=eff["desc_pt"]):
                selected_effects.append(key)

    col_mov, col_sens = st.columns(2)
    with col_mov:
        st.markdown("**Movimento**")
        for key, eff in effects_by_type.get("movement", []):
            if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=eff["desc_pt"]):
                selected_effects.append(key)
    with col_sens:
        st.markdown("**Sensorial**")
        for key, eff in effects_by_type.get("sensory", []):
            if st.checkbox(eff["label_pt"], key=_k(prefix, "s2_eff", key), help=eff["desc_pt"]):
                selected_effects.append(key)

    st.divider()

    # ── Configurações por efeito ──
    effect_configs: Dict[str, Dict] = {}
    for eff_key in selected_effects:
        eff = EFFECTS[eff_key]
        with st.expander(f"Configurar: {eff['label_pt']}", expanded=True):
            config: Dict[str, Any] = {}

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
                    descs = [""] + [o["desc_pt"] for o in deg_data["options"]]
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

    st.divider()

    # ── Extras Rápidos ──
    st.markdown("### Modificadores Rápidos (Extras)")
    st.caption("Marque os extras que se aplicam ao golpe todo.")

    quick_extras: List[str] = []
    # Organizar por grupo
    extras_range = [(k, v) for k, v in EXTRAS.items() if v.get("group") == "range"]
    extras_area = [(k, v) for k, v in EXTRAS.items() if v.get("group") == "area"]
    extras_other = [(k, v) for k, v in EXTRAS.items() if "group" not in v]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Alcance**")
        for key, ex in extras_range:
            if st.checkbox(ex["label_pt"], key=_k(prefix, "s2_qex", key), help=ex["desc_pt"]):
                quick_extras.append(key)
    with c2:
        st.markdown("**Área**")
        for key, ex in extras_area:
            if st.checkbox(ex["label_pt"], key=_k(prefix, "s2_qex", key), help=ex["desc_pt"]):
                quick_extras.append(key)
    with c3:
        st.markdown("**Outros Extras**")
        # Mostrar primeiros 6 extras mais comuns
        common_extras = ["selective", "accurate", "counter", "reaction", "cumulative", "improved_critical"]
        for key in common_extras:
            if key in EXTRAS:
                ex = EXTRAS[key]
                if st.checkbox(ex["label_pt"], key=_k(prefix, "s2_qex", key), help=ex["desc_pt"]):
                    quick_extras.append(key)

    # Extras adicionais em expander
    remaining_extras = [
        (k, v) for k, v in extras_other
        if k not in common_extras
    ]
    if remaining_extras:
        with st.expander("Mais extras..."):
            for key, ex in remaining_extras:
                if st.checkbox(ex["label_pt"], key=_k(prefix, "s2_qex", key), help=ex["desc_pt"]):
                    quick_extras.append(key)

    st.divider()

    # ── Falhas Rápidas ──
    st.markdown("### Falhas (reduzem custo)")
    st.caption("Falhas tornam o golpe mais barato mas com restrições.")

    quick_flaws: List[Dict] = []
    common_flaws = ["limited", "tiring", "unreliable", "grab_based", "concentration", "side_effect"]
    c1, c2 = st.columns(2)
    flaw_keys_list = list(FLAWS.keys())
    half = len(common_flaws) // 2

    with c1:
        for key in common_flaws[:half + 1]:
            if key in FLAWS:
                fl = FLAWS[key]
                if st.checkbox(fl["label_pt"], key=_k(prefix, "s2_qfl", key), help=fl["desc_pt"]):
                    desc = ""
                    if fl.get("has_description"):
                        desc = st.text_input(
                            f"Descreva a limitação ({fl['label_en']})",
                            key=_k(prefix, "s2_qfl_desc", key),
                        )
                    quick_flaws.append({"key": key, "ranks": 1, "description": desc})
    with c2:
        for key in common_flaws[half + 1:]:
            if key in FLAWS:
                fl = FLAWS[key]
                if st.checkbox(fl["label_pt"], key=_k(prefix, "s2_qfl", key), help=fl["desc_pt"]):
                    desc = ""
                    if fl.get("has_description"):
                        desc = st.text_input(
                            f"Descreva a limitação ({fl['label_en']})",
                            key=_k(prefix, "s2_qfl_desc", key),
                        )
                    quick_flaws.append({"key": key, "ranks": 1, "description": desc})

    remaining_flaws = [k for k in FLAWS if k not in common_flaws]
    if remaining_flaws:
        with st.expander("Mais falhas..."):
            for key in remaining_flaws:
                fl = FLAWS[key]
                if st.checkbox(fl["label_pt"], key=_k(prefix, "s2_qfl", key), help=fl["desc_pt"]):
                    desc = ""
                    if fl.get("has_description"):
                        desc = st.text_input(
                            f"Descreva ({fl['label_en']})",
                            key=_k(prefix, "s2_qfl_desc", key),
                        )
                    quick_flaws.append({"key": key, "ranks": 1, "description": desc})

    st.divider()

    # ── Gerar Build ──
    if st.button("Gerar Build", key=_k(prefix, "s2_generate"), type="primary", disabled=not selected_effects):
        # Montar componentes
        components: List[PowerComponent] = []
        for i, eff_key in enumerate(selected_effects):
            cfg = effect_configs.get(eff_key, {})
            comp = PowerComponent(
                effect_key=eff_key,
                rank=int(cfg.get("rank", base_rank)),
                is_linked=(i > 0),
                conditions=cfg.get("conditions"),
                stat_targets=cfg.get("stat_targets", []),
                resist_override=cfg.get("resist"),
            )
            # Aplicar extras rápidos ao primeiro componente de ataque
            if i == 0 or eff_key in ("damage", "affliction", "weaken"):
                for ex_key in quick_extras:
                    comp.extras.append({"key": ex_key, "ranks": 1, "description": ""})

            # Aplicar falhas rápidas ao primeiro componente
            if i == 0:
                for fl in quick_flaws:
                    comp.flaws.append(dict(fl))

            components.append(comp)

        draft = GolpeDraft(
            name=draft.name,
            category=draft.category,
            components=components,
            pokemon_type=draft.pokemon_type,
        )
        _set_draft(prefix, draft)
        _set_step(prefix, 3)
        st.rerun()

    if not selected_effects:
        st.info("Selecione pelo menos um efeito para gerar a build.")

    # Botão voltar
    if st.button("Voltar ao Passo 1", key=_k(prefix, "s2_back")):
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

    # Botões de salvamento
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Adicionar à ficha (cg_moves)", key=_k(prefix, "s5_save_session"), type="primary"):
            move_json = final_move or draft_to_move_json(draft, accuracy=accuracy)
            move_json["accuracy"] = int(accuracy)
            if "cg_moves" not in st.session_state:
                st.session_state["cg_moves"] = []
            st.session_state["cg_moves"].append(move_json)
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

                    # Também salvar no session_state
                    move_json = final_move or draft_to_move_json(draft, accuracy=accuracy)
                    move_json["accuracy"] = int(accuracy)
                    if "cg_moves" not in st.session_state:
                        st.session_state["cg_moves"] = []
                    st.session_state["cg_moves"].append(move_json)

                    st.success(f"Golpe '{draft.name}' salvo na nuvem e adicionado à ficha!")
                    st.json(doc)
            except Exception as e:
                st.error(f"Erro ao salvar no Firestore: {e}")

    st.divider()

    # JSON final para copiar
    if final_move:
        with st.expander("JSON do golpe (para copiar)"):
            st.json(final_move)

    # Navegação
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Voltar ao Editor", key=_k(prefix, "s5_back")):
            _set_step(prefix, 4)
            st.rerun()
    with c2:
        if st.button("Criar outro golpe", key=_k(prefix, "s5_new")):
            # Limpar state
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
