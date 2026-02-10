# -*- coding: utf-8 -*-
"""
Dex UI (modularizado)
- Mantém o comportamento do bloco original da página 'Pokédex (Busca)'.
"""
from __future__ import annotations

import os
import json
import re
import html
import base64
import gzip
from typing import Any, Dict

import streamlit as st


# ============================================================
# Helpers locais (eram funções globais no app.py monolítico)
# ============================================================

def _split_types(raw: Any) -> list[str]:
    """Divide o campo 'Tipo' em lista de tipos limpos.

    Aceita formatos comuns como:
    - "Water/Flying"
    - "Water, Flying"
    - "Water | Flying"
    - "Water" (apenas 1)
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    # Normaliza separadores
    parts = re.split(r"\s*[/,|;+]\s*", s)
    out: list[str] = []
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        # Mantém o texto como veio, mas sem espaços duplicados
        p = re.sub(r"\s+", " ", p)
        out.append(p)
    return out


def _type_color(t: str) -> str:
    """Retorna uma cor (hex) para o tipo Pokémon, para badges/UI."""
    key = (t or "").strip().lower()
    palette = {
        "normal": "#a8a77a",
        "fire": "#ee8130",
        "water": "#6390f0",
        "electric": "#f7d02c",
        "grass": "#7ac74c",
        "ice": "#96d9d6",
        "fighting": "#c22e28",
        "poison": "#a33ea1",
        "ground": "#e2bf65",
        "flying": "#a98ff3",
        "psychic": "#f95587",
        "bug": "#a6b91a",
        "rock": "#b6a136",
        "ghost": "#735797",
        "dragon": "#6f35fc",
        "dark": "#705746",
        "steel": "#b7b7ce",
        "fairy": "#d685ad",
    }
    return palette.get(key, "#94a3b8")

def render_pokedex_busca_page(*, df, user_data: dict, trainer_name: str, api_name_map: dict, get_pokemon_image_url):
    """
    Renderiza a página 'Pokédex (Busca)'.
    Dependências externas necessárias:
      - df: DataFrame da Pokédex
      - user_data: dict do treinador logado
      - trainer_name: nome do treinador
      - api_name_map: mapa oficial de nomes (pokeapi)
      - get_pokemon_image_url: função existente no app para obter sprites/artworks
    """
    st.markdown("""
    <style>
    /* ============================================================
       1. ESTILO GLOBAL E GERAL (MANTIDO)
       ============================================================ */
    [data-testid="stAppViewContainer"] {
        animation: pageFade 0.35s ease-in;
    }
    [data-testid="stMainBlockContainer"] {
        animation: contentSlide 0.35s ease-in;
    }
    h1, h2, h3 {
        color: #f8fafc;
        text-shadow: 0 1px 2px rgba(15, 23, 42, 0.6);
    }
    .pokedex-shell {
        border-radius: 18px;
        padding: 18px 18px 8px 18px;
        border: 2px solid rgba(148, 163, 184, 0.55);
        box-shadow: inset 0 0 18px rgba(15, 23, 42, 0.55);
        background: rgba(15, 23, 42, 0.65);
    }

    /* Fontes P2 */
    .pokedex-shell, .pokedex-card, .pokedex-info-value, .pokedex-info-title, .pokedex-header {
        font-family: "Press Start 2P", cursive !important;
    }

    .pokedex-header {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        background: rgba(226, 232, 240, 0.9);
        padding: 6px 16px;
        border-radius: 16px;
        font-size: 12px;
        color: #0f172a;
    }
    .pokedex-grid-note {
        font-size: 12px;
        color: #e2e8f0;
        text-align: center;
        margin: 6px 0 10px 0;
    }
    .pokedex-card {
        background: rgba(15, 23, 42, 0.82);
        color: #f8fafc;
        padding: 18px;
        border-radius: 16px;
        border: 2px solid rgba(255,255,255,0.35);
        margin-top: 18px;
    }
    .pokedex-detail-grid { display: grid; gap: 12px; }

    .pokedex-info-card {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        color: #f8fafc !important;
        padding: 10px;
        border-radius: 8px;
    }
    .pokedex-info-card--dark { background: #e2e8f0; }

    .pokedex-info-title {
        font-size: 12px;
        color: #38bdf8 !important;
        margin-bottom: 4px;
    }
    .pokedex-info-value {
        font-size: 14px;
        color: #0f172a;
        line-height: 1.6;
    }
    .pokedex-info-card--wide { padding: 12px 14px; }
    .pokedex-info-card--wide .pokedex-info-value { font-size: 12px; }
    .pokedex-info-card--wide .section-title { margin-top: 0; }

    .pokedex-tags span {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10px;
        margin-right: 6px;
        margin-bottom: 4px;
        background: rgba(0,0,0,0.35);
        color: #ffffff;
    }
    .pokedex-carousel {
        display: flex; gap: 12px; overflow-x: auto; padding: 10px 4px;
    }
    .pokedex-carousel img {
        width: 72px; height: 72px; image-rendering: pixelated;
        background: rgba(255,255,255,0.25); border-radius: 10px; padding: 6px;
    }

    /* CARROSSEL INFERIOR */
    .pokedex-footer-carousel {
        display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 12px; padding: 14px;
        background: rgba(0, 0, 0, 0.30); border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.18); scroll-behavior: smooth;
    }
    .pokedex-footer-carousel::-webkit-scrollbar { height: 8px; }
    .pokedex-footer-carousel::-webkit-scrollbar-thumb { background: #FFCC00; border-radius: 10px; }

    .carousel-item {
        flex: 0 0 auto; width: 70px; height: 70px; border-radius: 12px;
        display: grid; place-items: center; cursor: pointer;
        background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
        transition: transform 0.15s;
    }
    .carousel-item:hover { transform: scale(1.12); }
    .carousel-item img { width: 54px; height: 54px; image-rendering: pixelated; }
    .carousel-item-active { border: 2px solid #FFCC00; background: rgba(255, 204, 0, 0.10); }

    .info-label { color: #ffd166; font-weight: 800; }
    .section-title { color: #80ed99; font-weight: 900; margin-top: 10px; }
    .hi-red { color: #ff5c5c; font-weight: 900; }
    .hi-cyan { color: #4dd6ff; font-weight: 900; }
    .hi-purple { color: #b197ff; font-weight: 900; }
    .power-badge {
        display: block; width: fit-content; margin: 10px auto 0 auto;
        padding: 6px 12px; border-radius: 999px;
        background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.25);
        color: #ffd166; font-weight: 900; text-align: center;
    }
    @keyframes pageFade { from { opacity: 0.92; } to { opacity: 1; } }
    @keyframes contentSlide { from { transform: translateY(8px); opacity: 0.92; } to { transform: translateY(0); opacity: 1; } }


    /* ============================================================
       2. O NOVO CSS DOS TILES (CORRIGIDO PARA BATER COM O PYTHON)
       ============================================================ */

    /* MOLDURA DO CARD */
    .dex-card-frame {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 110px;        
        width: 100%;
        border-radius: 12px;
        margin-bottom: 8px;
        transition: transform 0.2s ease;
        background: rgba(15, 23, 42, 0.6);
        position: relative;
    }

    .dex-card-frame:hover {
        transform: scale(1.02);
    }

    /* A IMAGEM */
    .dex-sprite-img {
        max-width: 80px;
        max-height: 80px;
        width: auto;
        height: auto;
        object-fit: contain;
        image-rendering: pixelated;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.5));
        z-index: 1;
    }

    /* --- AQUI ESTAVA O ERRO: NOMES DAS CLASSES --- */
    /* Antes estava .frame-caught, agora é .dex-frame--caught */

    /* 🟢 CAPTURADO (Verde) */
    .dex-frame--caught {
        border: 2px solid #00ff41;
        box-shadow: 0 0 12px rgba(0, 255, 65, 0.25), inset 0 0 15px rgba(0, 60, 20, 0.6);
        background: rgba(0, 60, 20, 0.4);
    }

    /* 🔵 VISTO (Azul) */
    .dex-frame--seen {
        border: 2px solid #00d0ff;
        box-shadow: 0 0 12px rgba(0, 208, 255, 0.25), inset 0 0 15px rgba(0, 40, 60, 0.6);
        background: rgba(0, 40, 60, 0.4);
    }

    /* ⭐ WISHLIST (Dourado) */
    .dex-frame--wish {
        border: 2px solid #ffd700;
        box-shadow: 0 0 12px rgba(255, 215, 0, 0.25), inset 0 0 15px rgba(60, 50, 0, 0.6);
        background: rgba(60, 50, 0, 0.4);
    }

    /* ⚪ PADRÃO (Cinza) */
    .dex-frame--default {
        border: 2px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.03);
    }

    .dex-card-link {
        display: block;
        text-decoration: none;
        color: inherit;
        cursor: pointer;
    }


    /* ============================================================
       3. CARDS "TCG" DA POKEDEX (NOVO)
       ============================================================ */
    .dex-tcg-card{
        border-radius: 14px;
        overflow: hidden;
        position: relative;
        height: 170px;
        width: 100%;
        cursor: pointer;
        box-shadow: 0 10px 22px rgba(0,0,0,0.35);
        border: 4px solid rgba(255,255,255,0.12); /* será "dominada" pela classe de status */
        transition: transform .15s ease, filter .15s ease;
    }
    .dex-tcg-card:hover{
        transform: translateY(-2px) scale(1.01);
        filter: brightness(1.03);
    }

    .dex-tcg-header{
        height: 30px;
        display: grid;
        grid-template-columns: 26px 1fr auto;
        align-items: center;
        gap: 8px;
        padding: 6px 8px;
        background: rgba(15,23,42,0.62);
        border-bottom: 1px solid rgba(255,255,255,0.12);
    }
    .dex-tcg-statusicon{
        width: 22px; height: 22px;
        display:flex; align-items:center; justify-content:center;
        opacity: .95;
        filter: drop-shadow(0 2px 3px rgba(0,0,0,.45));
    }
    .dex-tcg-name{
        font-size: 9px;
        color: #e2e8f0;
        text-shadow: 0 2px 2px rgba(0,0,0,.55);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
    }
    /* GRID: sempre no mínimo 6 por fileira */
    .pokedex-grid{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
    }

    /* Evita esmagar demais em telas menores: cria scroll horizontal */
    .pokedex-grid{
      overflow-x: auto;
      padding-bottom: 8px;
      -webkit-overflow-scrolling: touch;
    }

    /* Opcional: garante um tamanho mínimo agradável pros cards */
    .dex-tcg-card{
      min-width: 165px;
    }

    .dex-tcg-np{
        font-size: 8px;
        color: #e2e8f0;
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 999px;
        padding: 2px 7px;
        text-shadow: 0 1px 2px rgba(0,0,0,.55);
    }

    .dex-tcg-body{
        height: 110px;
        display:flex;
        align-items:center;
        justify-content:center;
        background: rgba(2,6,23,0.18);
    }
    .dex-tcg-sprite{
        width: 86px;
        height: 86px;
        object-fit: contain;
        image-rendering: pixelated;
        filter: drop-shadow(0 7px 10px rgba(0,0,0,0.55));
    }

    .dex-tcg-footer{
        height: 30px;
        display:flex;
        align-items:center;
        justify-content:center;
        background: rgba(15,23,42,0.62);
        border-top: 1px solid rgba(255,255,255,0.12);
    }
    .dex-tcg-viab{
        font-size: 14px;
        letter-spacing: 2px;
        color: #ffd166;
        text-shadow: 0 2px 2px rgba(0,0,0,.55);
    }

    </style>
    """, unsafe_allow_html=True)


    dex_param = st.query_params.get("dex", None)
    if dex_param:
        st.session_state["pokedex_selected"] = str(dex_param)
        st.query_params.clear() # Limpa a URL após capturar
        st.rerun()
    st.sidebar.header("🔍 Filtros")
    search_query = st.sidebar.text_input("Buscar (Nome ou Nº)", "")

    # --- FIX REAL: remove a “moldura preta” do click_detector sem afetar o iframe do Compendium ---
    st.markdown("""
    <style>
    /* Só dentro da Pokédex: wrapper que a gente vai colocar em volta dos click_detector */
    .pokedex-clickwrap,
    .pokedex-clickwrap *{
      background: transparent !important;
    }

    /* Remove fundo/borda do frame do componente (é isso que costuma aparecer como “contorno preto”) */
    .pokedex-clickwrap div[data-testid="stComponentFrame"],
    .pokedex-clickwrap div[data-testid="stComponentFrame"] > div,
    .pokedex-clickwrap div[data-testid="stElementContainer"],
    .pokedex-clickwrap div[data-testid="stElementContainer"] > div{
      background: transparent !important;
      border: none !important;
      box-shadow: none !important;
      outline: none !important;
      padding: 0 !important;
      margin: 0 !important;
    }

    /* O iframe real do st_click_detector */
    .pokedex-clickwrap iframe[title^="st_click_detector"],
    .pokedex-clickwrap iframe[title*="click_detector"]{
      background: transparent !important;
      border: none !important;
      box-shadow: none !important;
      outline: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 1) FILTRO DE REGIÃO
    all_regions = sorted(list(set([r.strip() for region in df["Região"].unique() for r in str(region).split("/")])) )
    selected_regions = st.sidebar.multiselect("Região", all_regions)

    # 2) FILTRO DE BIOMA (CASCATA)
    if selected_regions:
        df_for_biomes = df[df["Região"].apply(lambda x: any(reg in str(x) for reg in selected_regions))]
        raw_biomes = df_for_biomes["Biomas"].unique()
    else:
        raw_biomes = df["Biomas"].unique()

    all_biomes = sorted(list(set([b.strip() for biome in raw_biomes for b in str(biome).split("/")])) )
    biomes_clean = [b for b in all_biomes if "toda" not in b.lower() and "ga" not in b.lower()]
    selected_biomes = st.sidebar.multiselect("Bioma", biomes_clean)

    # 3) FILTRO DE TIPO (COMBINAÇÃO)
    all_types = sorted(list(set([t.strip() for t_str in df["Tipo"].unique() for t in str(t_str).split("/")])) )
    selected_types = st.sidebar.multiselect("Tipo Elementar (Combinação)", all_types)

    # 4) NÍVEL DE PODER
    min_p, max_p = int(df["Nivel_Poder"].min()), int(df["Nivel_Poder"].max())
    power_range = st.sidebar.slider("⚡ Nível de Poder", min_p, max_p, (min_p, max_p))

    # 5) ESTRATÉGIA
    st.sidebar.subheader("⚔️ Estratégia")
    sel_func = st.sidebar.selectbox("Função", ["Todos", "C - Controlador", "F - Finalizador", "S - Suporte"])
    sel_style = st.sidebar.selectbox("Estilo", ["Todos", "O - Ofensivo", "D - Defensivo", "F - Furtivo", "I - Incompleto", "C - Completo"])
    sel_speed = st.sidebar.selectbox("Velocidade", ["Todos", "R - Rápido", "L - Lento"])

    l1 = sel_func[0] if sel_func != "Todos" else ""
    l2 = sel_style[0] if sel_style != "Todos" else ""
    l3 = sel_speed[0] if sel_speed != "Todos" else ""

    # -----------------------------
    # APLICAÇÃO DOS FILTROS
    # -----------------------------
    filtered_df = df.copy()

    if search_query:
        filtered_df = filtered_df[
            filtered_df["Nome"].str.contains(search_query, case=False, na=False)
            | filtered_df["Nº"].astype(str).str.contains(search_query, case=False, na=False)
        ]

    if selected_regions:
        filtered_df = filtered_df[filtered_df["Região"].apply(lambda x: any(region in str(x) for region in selected_regions))]

    if selected_biomes:
        filtered_df = filtered_df[
            filtered_df["Biomas"].apply(
                lambda x: ("toda" in str(x).lower() and "ga" in str(x).lower())
                or any(b in str(x) for b in selected_biomes)
            )
        ]

    if selected_types:
        # “Combinação”: precisa conter TODOS os tipos marcados
        filtered_df = filtered_df[filtered_df["Tipo"].apply(lambda x: all(t in str(x) for t in selected_types))]

    filtered_df = filtered_df[
        (filtered_df["Nivel_Poder"] >= power_range[0]) & (filtered_df["Nivel_Poder"] <= power_range[1])
    ]

    if l1 or l2 or l3:
        def _match_codes(codes):
            if not isinstance(codes, list):
                codes = [c.strip() for c in str(codes).split(",") if c.strip()]
            for c in codes:
                if len(c) >= 3:
                    ok = (not l1 or c[0] == l1) and (not l2 or c[1] == l2) and (not l3 or c[2] == l3)
                    if ok:
                        return True
            return False

        filtered_df = filtered_df[filtered_df["Codigos_Estrategia"].apply(_match_codes)]

    # -----------------------------
    # CSS DO CARROSSEL INFERIOR
    # -----------------------------
    st.markdown(
        """
        <style>
        .footer-carousel {
            display: flex;
            flex-wrap: nowrap;
            overflow-x: auto;
            gap: 15px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .footer-carousel::-webkit-scrollbar { height: 8px; }
        .footer-carousel::-webkit-scrollbar-thumb { background: #FFCC00; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # SESSION STATE (garante que existe)
    # -----------------------------
    if "pokedex_selected" not in st.session_state:
        st.session_state["pokedex_selected"] = None

    def select_pokedex_entry(pid: str) -> None:
        st.session_state["pokedex_selected"] = str(pid)


    # Se veio por query param (?dex=XXX), seleciona o Pokémon na MESMA ABA
    try:
        _dex_param = st.query_params.get("dex")
        if isinstance(_dex_param, list):
            _dex_param = _dex_param[0] if _dex_param else None
        if _dex_param:
            select_pokedex_entry(str(_dex_param))
            try:
                del st.query_params["dex"]
            except Exception:
                st.query_params.clear()
    except Exception:
        pass

    selected_id = st.session_state.get("pokedex_selected")

    # ==============================================================================
    # VISÃO DE FOCO (selecionado)
    # ==============================================================================
    if selected_id:
    

        selected_df = df[df["Nº"].astype(str) == str(selected_id)]
        if selected_df.empty:
            st.session_state["pokedex_selected"] = None
            st.rerun()

        row = selected_df.iloc[0]
        dex_num = str(row["Nº"])
        p_name = row["Nome"]
        codes = row.get("Codigos_Estrategia", [])
        if not isinstance(codes, list):
            codes = [c.strip() for c in str(codes).split(",") if c.strip()]

        if "wishlist" not in user_data:
            user_data["wishlist"] = []

        # Botão sair
        if st.button("⬅️ Sair da Visão de Foco"):
            st.session_state["pokedex_selected"] = None
            st.rerun()

        # Helpers locais (não depende do resto do arquivo)
        def build_info_entries():
            hidden = {
                "Estágio",
                "Tipo de Evolução",
                "Nivel_Poder",
                "Nível de Poder",
            }
    
            entries = []
            for col in row.index:
                if col in {"Nome", "Nº", "Codigos_Estrategia"}:
                    continue
                if col in hidden:
                    continue
    
                value = row[col]
                if pd.isna(value):
                    continue
                value_str = str(value).strip()
                if not value_str or value_str.lower() == "nan":
                    continue
                entries.append((col, value_str))
            return entries


        def render_info_columns(entries):
            st.markdown("<div class='pokedex-detail-grid'>", unsafe_allow_html=True)
            for label, value in entries:
                # Título: Descrição
                if label == "Descrição da Pokedex":
                    value_html = str(value).replace("\n", "<br>")
                    st.markdown(
                        f"""
                        <div class='pokedex-info-card pokedex-info-card--wide'>
                            <div class='section-title'>📘 Descrição da Pokédex</div>
                            <div class='pokedex-info-value'>{value_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    continue

                # Título: Viabilidade (texto grande)
                if label == "Viabilidade":
                    viab = (
                        str(value)
                        .replace("PARCEIROS:", "<br><br><strong>👥 PARCEIROS:</strong>")
                        .replace("Explicação:", "<br><br><strong>💡 EXPLICAÇÃO:</strong>")
                        .replace("Habilidade:", "<strong>✨ Habilidade:</strong>")
                    )

                    viab = viab.replace("\n", "<br>")

                    # pinta o FIR (e outros códigos se quiser)
                    viab = viab.replace("FIR", "<span class='hi-red'>FIR</span>")

                    # destaca os códigos de estratégia no texto (mantém o que você já tinha)
                    for code in codes:
                        viab = re.sub(rf"\b{re.escape(code)}\b", f"<span class='hi-purple'>{code}</span>", viab)

                    st.markdown(
                        f"""
                        <div class='pokedex-info-card pokedex-info-card--wide pokedex-info-card--dark'>
                            <div class='section-title'>🧠 Viabilidade</div>
                            <div class='pokedex-info-value'>{viab}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    continue

                # Campos normais (Tipo, Raridade, Biomas, Região etc.)
                value_html = str(value).replace("\n", "<br>")
                st.markdown(
                    f"""
                    <div class='pokedex-info-card'>
                        <div class='pokedex-info-title'>{label}</div>
                        <div class='pokedex-info-value'>{value_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)


        def render_info_tags():
            tags_html = "".join([f"<span>{c}</span>" for c in codes])
            st.markdown(f"<div class='pokedex-tags'>{tags_html}</div>", unsafe_allow_html=True)

        def render_status_controls():
            # 🛡️ garante que IDs antigos sejam migrados para a dex atual
            try:
                _dex_guard_once(user_data, df)
            except Exception:
                pass
            # precisa existir no save
            if "seen" not in user_data:
                user_data["seen"] = []
            if "caught" not in user_data:
                user_data["caught"] = []

            is_seen = dex_num in user_data["seen"]
            is_caught = dex_num in user_data["caught"]
            is_wished = dex_num in user_data["wishlist"]

            s1, s2, s3 = st.columns(3)

            with s1:
                label = "👁️ Visto" if not is_seen else "✅ Visto"
                if st.button(label, key=f"seen_{dex_num}"):
                    if dex_num not in user_data["seen"]:
                        user_data["seen"].append(dex_num)
                    save_data_cloud(trainer_name, user_data)
                    st.rerun()

            with s2:
                label = "🔴 Capturar" if not is_caught else "✅ Capturado"
                if st.button(label, key=f"caught_{dex_num}"):
                    if dex_num in user_data["caught"]:
                        user_data["caught"].remove(dex_num)
                    else:
                        user_data["caught"].append(dex_num)
                        if dex_num not in user_data["seen"]:
                            user_data["seen"].append(dex_num)
                    save_data_cloud(trainer_name, user_data)
                    st.rerun()

            with s3:
                label = "⭐ Desejar" if not is_wished else "✅ Na Lista"
                if st.button(label, key=f"wish_{dex_num}"):
                    if dex_num in user_data["wishlist"]:
                        user_data["wishlist"].remove(dex_num)
                    else:
                        user_data["wishlist"].append(dex_num)
                    save_data_cloud(trainer_name, user_data)
                    st.rerun()

        # --- LAYOUT DO FOCO (seu estilo) ---
        st.markdown("<div class='pokedex-card'>", unsafe_allow_html=True)
        st.markdown(f"### #{dex_num} • {p_name}")

        info_entries = build_info_entries()
        midpoint = (len(info_entries) + 1) // 2
        top_left, top_center, top_right = st.columns([1.3, 1.7, 1.3])

        with top_left:
            render_info_columns(info_entries[:midpoint])

        with top_center:
            # --- LÓGICA DE FORMAS VISUAIS (Dex) ---
            # Por padrão, usa o nome do Excel
            target_visual_name = p_name 
        
            # Se for Lycanroc, exibe o seletor
            if "lycanroc" in p_name.lower().strip():
                lyc_dex_form = st.radio(
                    "Visualizar Forma:",
                    ["Midday", "Midnight", "Dusk"],
                    horizontal=True,
                    key="dex_lyc_visual_selector"
                )
            
                # Mapeia a escolha para o nome da API
                if lyc_dex_form == "Midnight":
                    target_visual_name = "lycanroc-midnight"
                elif lyc_dex_form == "Dusk":
                    target_visual_name = "lycanroc-dusk"
                else:
                    target_visual_name = "lycanroc-midday"

            # --- RENDERIZA A IMAGEM PRINCIPAL ---
            # Usa o nome específico da forma para buscar a arte oficial
            final_img_url = get_pokemon_image_url(target_visual_name, api_name_map, mode="artwork", shiny=False)
            st.image(final_img_url, use_container_width=True)
    
            # Nível de Poder (mantido)
            np = row.get("Nivel_Poder", row.get("Nível de Poder", ""))
            if str(np).strip() != "" and str(np).lower() != "nan":
                st.markdown(
                    f"<div class='power-badge'>⚡ Nível de Poder: {np}</div>",
                    unsafe_allow_html=True
                )

        with top_right:
            render_info_columns(info_entries[midpoint:])

        render_status_controls()
        render_info_tags()

        # --- CARROSSEL DE SPRITES (Atualizado para seguir a forma) ---
        st.markdown("#### 🎞️ Variações (Sprites)")
    
        # Gera URLs baseadas no nome da forma (target_visual_name) em vez do número da Dex
        sprite_normal = get_pokemon_image_url(target_visual_name, api_name_map, mode="sprite", shiny=False)
        sprite_shiny  = get_pokemon_image_url(target_visual_name, api_name_map, mode="sprite", shiny=True)
    
        sprites_html = f"""
            <div class='pokedex-carousel'>
                <div style="text-align:center; font-size:10px;">
                    <img src='{sprite_normal}' style='width:70px; image-rendering: pixelated;'><br>Normal
                </div>
                <div style="text-align:center; font-size:10px;">
                    <img src='{sprite_shiny}' style='width:70px; image-rendering: pixelated;'><br>Shiny
                </div>
            </div>
        """
        st.markdown(sprites_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # --- CARROSSEL INFERIOR (navegação) ---
        from st_click_detector import click_detector
        items_html_list = []
        carousel_id_map = {}
        for idx, (_, r_car) in enumerate(filtered_df.iterrows()):
            pid = str(r_car["Nº"])
            # Em versão modularizada, evitamos depender de funções globais do app.py.
            # Usamos o nome do Pokémon do próprio DataFrame para buscar o sprite.
            try:
                _nm = str(r_car.get("Nome") or "").strip()
            except Exception:
                _nm = ""
            sprite = get_pokemon_image_url(_nm or pid, api_name_map, mode="sprite", shiny=False)
            is_active = "carousel-item-active" if pid == str(st.session_state.get("pokedex_selected")) else ""
        
            safe_id = f"carousel_{idx}"
            carousel_id_map[safe_id] = pid

            item_node = (
                f'<a href="javascript:void(0)" id="{safe_id}" class="dex-card-link">'
                f'  <div class="carousel-item {is_active}">'
                f'    <img src="{sprite}" alt="{pid}">'
                "  </div>"
                "</a>"
            )
        
            items_html_list.append(item_node)
    
        all_items_string = "".join(items_html_list)
    
        html_template = '''
        <style>
            .pokedex-footer-carousel {
                display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 12px; padding: 14px;
                background: rgba(0, 0, 0, 0.30); border-radius: 15px;
                border: 1px solid rgba(255,255,255,0.18); scroll-behavior: smooth;
            }
            .pokedex-footer-carousel::-webkit-scrollbar { height: 8px; }
            .pokedex-footer-carousel::-webkit-scrollbar-thumb { background: #FFCC00; border-radius: 10px; }
        
            .carousel-item {
                flex: 0 0 auto; width: 70px; height: 70px; border-radius: 12px;
                display: grid; place-items: center; cursor: pointer;
                background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
                transition: transform 0.15s;
            }
            .carousel-item:hover { transform: scale(1.12); border-color: #FFCC00; }
            .carousel-item-active { border: 2px solid #FFCC00; background: rgba(255, 204, 0, 0.10); }
            .carousel-item img { width: 54px; height: 54px; image-rendering: pixelated; }
        </style>
    
        <div class="pokedex-footer-carousel" id="pokedex-footer-carousel" tabindex="0">
            REPLACE_ME
        </div>
        <script>
            const carousel = document.getElementById("pokedex-footer-carousel");
            if (carousel) {
                carousel.addEventListener("wheel", (event) => {
                    event.preventDefault();
                    const delta = Math.abs(event.deltaY) > Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
                    carousel.scrollLeft += delta;
                }, { passive: false });

                const centerActiveItem = () => {
                    const activeItem = carousel.querySelector(".carousel-item-active");
                    if (!activeItem) return;
                    const targetLeft =
                        activeItem.offsetLeft - (carousel.clientWidth - activeItem.offsetWidth) / 2;
                    carousel.scrollLeft = Math.max(0, targetLeft);
                };

                requestAnimationFrame(centerActiveItem);
                window.addEventListener("resize", centerActiveItem);
            }
        </script>
        '''
    
        final_carousel_html = html_template.replace("REPLACE_ME", all_items_string)
        st.markdown("<div class='pokedex-clickwrap'>", unsafe_allow_html=True)
        clicked_carousel_id = click_detector(final_carousel_html, key="pokedex_carousel")
        st.markdown("</div>", unsafe_allow_html=True)
    
        if clicked_carousel_id is not None:
            selected_pid = carousel_id_map.get(str(clicked_carousel_id))
            if selected_pid and selected_pid != st.session_state.get("pokedex_selected"):
                st.session_state["pokedex_selected"] = selected_pid
                st.rerun()                
           
    

    # ==============================================================================
    # GRID (visão geral)
    # ==============================================================================
    else:
        if filtered_df.empty:
            st.warning("Nenhum Pokémon encontrado.")
        else:
            st.title("📕 Pokédex Universal")
            st.markdown(f"**Resultados:** {len(filtered_df)}")

            # contadores (igual seu arquivo)
            obtained_count = len(user_data.get("caught", []))
            seen_count = len(user_data.get("seen", []))

            st.markdown("<div class='pokedex-shell'>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='pokedex-header'><span>Pokémon Obtidos {obtained_count}</span><span>Pokémon Vistos {seen_count}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='pokedex-grid-note'>Clique em um Pokémon para ver os detalhes.</div>", unsafe_allow_html=True)

            grid_cols = 6 # Reduzi para 6 para as bordas não ficarem espremidas
            rows = list(filtered_df.iterrows())

        
            # Renderiza o grid com o visual original + clique (click_detector roda em iframe, então precisa CSS inline)
            from st_click_detector import click_detector

            POKEDEX_IFRAME_CSS = r"""@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
       
            html, body {
              background: transparent !important;
              margin: 0 !important;
              padding: 0 !important;
            }
            /* o st_click_detector costuma ter um root/wrapper */
            #root, .root, .container, .main, .app {
              background: transparent !important;
            }

            .pokedex-grid-note {
                    font-size: 12px;
                    color: #e2e8f0;
                    text-align: center;
                    margin: 6px 0 10px 0;
                }

                /* GRID (sempre 6 colunas; em telas pequenas vira scroll horizontal) */
                .pokedex-grid-wrap { overflow-x: auto; padding-bottom: 6px; }
                .pokedex-grid {
                    display: grid;
                    grid-template-columns: repeat(var(--cols, 6), minmax(140px, 1fr));
                    gap: 14px;
                    align-items: stretch;
                    min-width: calc(var(--cols, 6) * 150px);
                }
                .dex-tcg-card { height: 160px; } /* ajuda a caber mais na viewport */
                .dex-tcg-body { display:flex; justify-content:center; align-items:center; height: 96px; }
                .dex-tcg-sprite { width: 64px; height: 64px; image-rendering: pixelated; }


                .pokedex-card {
                    background: rgba(15, 23, 42, 0.82);
                    color: #f8fafc;
                    padding: 18px;
                    border-radius: 16px;
                    border: 2px solid rgba(255,255,255,0.35);
                    margin-top: 18px;
                }
                .pokedex-detail-grid { display: grid; gap: 12px; }

                .pokedex-info-card {
                    background: rgba(15, 23, 42, 0.9) !important;
                    border: 1px solid rgba(56, 189, 248, 0.4) !important;
                    color: #f8fafc !important;
                    padding: 10px;
                    border-radius: 8px;
                }
                .pokedex-info-card--dark { background: #e2e8f0; }

                .pokedex-info-title {
                    font-size: 12px;
                    color: #38bdf8 !important;
                    margin-bottom: 4px;
                }
                .pokedex-info-value {
                    font-size: 14px;
                    color: #0f172a;
                    line-height: 1.6;
                }
                .pokedex-info-card--wide { padding: 12px 14px; }
                .pokedex-info-card--wide .pokedex-info-value { font-size: 12px; }
                .pokedex-info-card--wide .section-title { margin-top: 0; }

                .pokedex-tags span {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 999px;
                    font-size: 10px;
                    margin-right: 6px;
                    margin-bottom: 4px;
                    background: rgba(0,0,0,0.35);
                    color: #ffffff;
                }
                .pokedex-carousel {
                    display: flex; gap: 12px; overflow-x: auto; padding: 10px 4px;
                }
                .pokedex-carousel img {
                    width: 72px; height: 72px; image-rendering: pixelated;
                    background: rgba(255,255,255,0.25); border-radius: 10px; padding: 6px;
                }

                /* CARROSSEL INFERIOR */
                .pokedex-footer-carousel {
                    display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 12px; padding: 14px;
                    background: rgba(0, 0, 0, 0.30); border-radius: 15px;
                    border: 1px solid rgba(255,255,255,0.18); scroll-behavior: smooth;
                }
                .pokedex-footer-carousel::-webkit-scrollbar { height: 8px; }
                .pokedex-footer-carousel::-webkit-scrollbar-thumb { background: #FFCC00; border-radius: 10px; }

                .carousel-item {
                    flex: 0 0 auto; width: 70px; height: 70px; border-radius: 12px;
                    display: grid; place-items: center; cursor: pointer;
                    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
                    transition: transform 0.15s;
                }
                .carousel-item:hover { transform: scale(1.12); }
                .carousel-item img { width: 54px; height: 54px; image-rendering: pixelated; }
                .carousel-item-active { border: 2px solid #FFCC00; background: rgba(255, 204, 0, 0.10); }

                .info-label { color: #ffd166; font-weight: 800; }
                .section-title { color: #80ed99; font-weight: 900; margin-top: 10px; }
                .hi-red { color: #ff5c5c; font-weight: 900; }
                .hi-cyan { color: #4dd6ff; font-weight: 900; }
                .hi-purple { color: #b197ff; font-weight: 900; }
                .power-badge {
                    display: block; width: fit-content; margin: 10px auto 0 auto;
                    padding: 6px 12px; border-radius: 999px;
                    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.25);
                    color: #ffd166; font-weight: 900; text-align: center;
                }
                @keyframes pageFade { from { opacity: 0.92; } to { opacity: 1; } }
                @keyframes contentSlide { from { transform: translateY(8px); opacity: 0.92; } to { transform: translateY(0); opacity: 1; } }


                /* ============================================================
                   2. O NOVO CSS DOS TILES (CORRIGIDO PARA BATER COM O PYTHON)
                   ============================================================ */

                /* MOLDURA DO CARD */
                .dex-card-frame {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 110px;        
                    width: 100%;
                    border-radius: 12px;
                    margin-bottom: 8px;
                    transition: transform 0.2s ease;
                    background: rgba(15, 23, 42, 0.6);
                    position: relative;
                }

                .dex-card-frame:hover {
                    transform: scale(1.02);
                }

                /* A IMAGEM */
                .dex-sprite-img {
                    max-width: 80px;
                    max-height: 80px;
                    width: auto;
                    height: auto;
                    object-fit: contain;
                    image-rendering: pixelated;
                    filter: drop-shadow(0 4px 6px rgba(0,0,0,0.5));
                    z-index: 1;
                }

                /* --- AQUI ESTAVA O ERRO: NOMES DAS CLASSES --- */
                /* Antes estava .frame-caught, agora é .dex-frame--caught */

                /* 🟢 CAPTURADO (Verde) */
                .dex-frame--caught {
                    border: 2px solid #00ff41;
                    box-shadow: 0 0 12px rgba(0, 255, 65, 0.25), inset 0 0 15px rgba(0, 60, 20, 0.6);
                    background: rgba(0, 60, 20, 0.4);
                }

                /* 🔵 VISTO (Azul) */
                .dex-frame--seen {
                    border: 2px solid #00d0ff;
                    box-shadow: 0 0 12px rgba(0, 208, 255, 0.25), inset 0 0 15px rgba(0, 40, 60, 0.6);
                    background: rgba(0, 40, 60, 0.4);
                }

                /* ⭐ WISHLIST (Dourado) */
                .dex-frame--wish {
                    border: 2px solid #ffd700;
                    box-shadow: 0 0 12px rgba(255, 215, 0, 0.25), inset 0 0 15px rgba(60, 50, 0, 0.6);
                    background: rgba(60, 50, 0, 0.4);
                }

                /* ⚪ PADRÃO (Cinza) */
                .dex-frame--default {
                    border: 2px solid rgba(255, 255, 255, 0.15);
                    background: rgba(255, 255, 255, 0.03);
                }

                .dex-card-link {
                    display: block;
                    text-decoration: none;
                    color: inherit;
                    cursor: pointer;
                }


                /* ============================================================
                   3. CARDS "TCG" DA POKEDEX (NOVO)
                   ============================================================ */
                .dex-tcg-card{
                    border-radius: 14px;
                    overflow: hidden;
                    position: relative;
                    height: 170px;
                    width: 100%;
                    cursor: pointer;
                    box-shadow: 0 10px 22px rgba(0,0,0,0.35);
                    border: 4px solid rgba(255,255,255,0.12); /* será "dominada" pela classe de status */
                    transition: transform .15s ease, filter .15s ease;
                }
                .dex-tcg-card:hover{
                    transform: translateY(-2px) scale(1.01);
                    filter: brightness(1.03);
                }

                .dex-tcg-header{
                    height: 30px;
                    display: grid;
                    grid-template-columns: 26px 1fr auto;
                    align-items: center;
                    gap: 8px;
                    padding: 6px 8px;
                    background: rgba(15,23,42,0.62);
                    border-bottom: 1px solid rgba(255,255,255,0.12);
                }
                .dex-tcg-statusicon{
                    width: 22px; height: 22px;
                    display:flex; align-items:center; justify-content:center;
                    opacity: .95;
                    filter: drop-shadow(0 2px 3px rgba(0,0,0,.45));
                }
                .dex-tcg-name{
                    font-size: 9px;
                    color: #e2e8f0;
                    text-shadow: 0 2px 2px rgba(0,0,0,.55);
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    text-align: center;
                }
                .dex-tcg-np{
                    font-size: 8px;
                    color: #e2e8f0;
                    background: rgba(15,23,42,0.7);
                    border: 1px solid rgba(255,255,255,0.14);
                    border-radius: 999px;
                    padding: 2px 7px;
                    text-shadow: 0 1px 2px rgba(0,0,0,.55);
                }

                .dex-tcg-body{
                    height: 110px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background: rgba(2,6,23,0.18);
                }
                .dex-tcg-sprite{
                    width: 86px;
                    height: 86px;
                    object-fit: contain;
                    image-rendering: pixelated;
                    filter: drop-shadow(0 7px 10px rgba(0,0,0,0.55));
                }

                .dex-tcg-footer{
                    height: 30px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background: rgba(15,23,42,0.62);
                    border-top: 1px solid rgba(255,255,255,0.12);
                }
                .dex-tcg-viab{
                    font-size: 14px;
                    letter-spacing: 2px;
                    color: #ffd166;
                    text-shadow: 0 2px 2px rgba(0,0,0,.55);
                }

            """

            card_nodes = []
            grid_id_map = {}

            for _, row_g in filtered_df.iterrows():
                dex_num = str(row_g["Nº"])
                p_name = row_g["Nome"]

                # Sprite baseado no nome (mais robusto com formas/IDs regionais)
                sprite_url = get_pokemon_image_url(p_name, api_name_map, mode="sprite", shiny=False)

                is_caught = dex_num in user_data.get("caught", [])
                is_seen = dex_num in user_data.get("seen", [])
                is_wished = dex_num in user_data.get("wishlist", [])

                if is_caught:
                    status_class = "dex-frame--caught"
                elif is_wished:
                    status_class = "dex-frame--wish"
                elif is_seen:
                    status_class = "dex-frame--seen"
                else:
                    status_class = "dex-frame--default"

                types = _split_types(row_g.get("Tipo", ""))
                t1 = types[0] if len(types) > 0 else ""
                t2 = types[1] if len(types) > 1 else ""
                c1 = _type_color(t1)
                c2 = _type_color(t2) if t2 else ""

                if c2:
                    bg_style = f"background: linear-gradient(135deg, {c1} 0%, {c1} 50%, {c2} 50%, {c2} 100%);"
                else:
                    bg_style = f"background: {c1};"

                viab_code = _extract_viab_code_from_text(row_g.get("Viabilidade", ""))
                try:
                    np_val = int(row_g.get("Nivel_Poder", 0) or 0)
                except Exception:
                    np_val = 0

                if is_caught:
                    status_svg = _SVG_POKEBALL
                elif is_wished:
                    status_svg = _SVG_STAR
                elif is_seen:
                    status_svg = _SVG_EYE
                else:
                    status_svg = ""

                safe_id = f"dex_{dex_num}"
                safe_id_card = f"{safe_id}__card"
                safe_id_img = f"{safe_id}__img"

                grid_id_map[safe_id] = dex_num
                grid_id_map[safe_id_card] = dex_num
                grid_id_map[safe_id_img] = dex_num

                card_nodes.append("\n".join([
                    f'<a href="javascript:void(0)" id="{safe_id}" '
                    f'style="text-decoration:none;color:inherit;display:block;">',
            
                    f'  <div id="{safe_id_card}" class="dex-tcg-card {status_class}" '
                    f'style="{bg_style}; cursor:pointer;" role="button" tabindex="0">',
            
                    '    <div class="dex-tcg-header" style="pointer-events:none;">',
                    f'      <div class="dex-tcg-statusicon">{status_svg}</div>',
                    f'      <div class="dex-tcg-name">{p_name}</div>',
                    f'      <div class="dex-tcg-np">NP {np_val}</div>',
                    '    </div>',
            
                    '    <div class="dex-tcg-body" style="pointer-events:none;">',
                    f'      <img id="{safe_id_img}" src="{sprite_url}" class="dex-tcg-sprite" alt="{p_name}" />',
                    '    </div>',
            
                    '    <div class="dex-tcg-footer" style="pointer-events:none;">',
                    f'      <div class="dex-tcg-viab" title="{viab_code}">{viab_code}</div>',
                    '    </div>',
            
                    '  </div>',
                    '</a>',
                ]))

            grid_html = f"<style>{POKEDEX_IFRAME_CSS}</style><div class='pokedex-grid-wrap'><div class='pokedex-grid' style='--cols:{grid_cols};'>" + "".join(card_nodes) + "</div></div>"
            st.markdown("<div class='pokedex-clickwrap'>", unsafe_allow_html=True)
            clicked_id = click_detector(grid_html, key="pokedex_grid")
            st.markdown("</div>", unsafe_allow_html=True)

        
            if clicked_id is not None:
                clicked_id = str(clicked_id)
                selected_pid = grid_id_map.get(clicked_id)

                if selected_pid is None and "__" in clicked_id:
                    selected_pid = grid_id_map.get(clicked_id.split("__", 1)[0])

                if selected_pid:
                    st.session_state["pokedex_selected"] = str(selected_pid)
                    st.rerun()


    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

