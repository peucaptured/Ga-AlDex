# -*- coding: utf-8 -*-
"""
PvP Arena UI (modularizado)
- Mantém o comportamento do bloco original da página 'PvP – Arena Tática'.
"""
from __future__ import annotations

import time
import random
import re
import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from ga_aldex.utils_state import set_busy, request_rerun, flush_pending_rerun, pvp_in_action
from streamlit_image_coordinates import streamlit_image_coordinates

from ga_aldex.firebase_bundle import *
from ga_aldex.moves_bundle import *

def render_pvp_arena_page(*, trainer_name: str, user_data: dict, df):
    """
    Renderiza a página 'PvP – Arena Tática'.
    """
    st.title("⚔️ PvP – Arena Tática")
    st.caption(
        "Base multiplayer: criar/abrir arena, entrar por código, espectadores "
    )

    db, bucket = init_firebase()
    view = st.session_state.get("pvp_view", "lobby")
    rid = st.session_state.get("active_room_id")  # ✅ sempre existe ou None
    room = get_room(db, rid) if rid else None     # ✅ evita NameError
    role = get_role(room, trainer_name) if room else "spectator"
    is_player = role in ["owner", "challenger"]
    battle_sheets_map = {}
    if is_player:
        try:
            for sh in list_sheets(db, trainer_name) or []:
                p = (sh.get("pokemon") or {})
                pid = p.get("id")
                if pid is None:
                    continue
                battle_sheets_map[str(pid)] = sh
        except Exception:
            battle_sheets_map = {}



    # =========================
    # VIEW: BATTLE (CÓDIGO CONSOLIDADO E CORRIGIDO)
    # =========================
    if view == "battle":
        sync_watchdog(db, rid)
        flush_pending_rerun()
        if not rid or not room:
            st.session_state["pvp_view"] = "lobby"
            request_rerun("pvp", force=True)
    
        # --- AQUI: INICIA O SISTEMA DE SYNC AUTOMÁTICO ---
        # Isso cria a thread que fica "dormindo" até o Firebase avisar de uma mudança.
        # -------------------------------------------------

        if "last_click_processed" not in st.session_state:
            st.session_state["last_click_processed"] = None
        if "placing_trainer" not in st.session_state:
            st.session_state["placing_trainer"] = None

        # --- 1. SINCRONIZAÇÃO DE DADOS ---
        current_party = user_data.get("party") or []
        db.collection("rooms").document(rid).collection("public_state").document("players").set(
            {trainer_name: current_party}, merge=True
        )

        if "stats" in user_data:
            # Prepara um dicionário estruturado para o merge funcionar direito
            # Estrutura: { "NomeTreinador": { "ID_Pokemon": { "stats": {...} } } }
            nested_update = {}
        
            for pid in current_party:
                is_shiny = pid in user_data.get("shinies", [])
                hub_stats = user_data["stats"].get(pid, {})
                saved_form = user_data.get("forms", {}).get(str(pid))
                if hub_stats:
                    # Se o treinador ainda não está no dicionário, cria
                    if trainer_name not in nested_update:
                        nested_update[trainer_name] = {}
                
                    clean_stats = {
                        "dodge": int(hub_stats.get("dodge", 0)),
                        "parry": int(hub_stats.get("parry", 0)),
                        "will": int(hub_stats.get("will", 0)),
                        "fort": int(hub_stats.get("fort", 0)),
                        "thg": int(hub_stats.get("thg", 0)),
                    }
                    # Adiciona os dados do Pokémon
                    nested_update[trainer_name][str(pid)] = {
                        "stats": hub_stats,
                        "shiny": is_shiny,
                        "form": saved_form, # ✅ ENVIANDO PARA O BANCO
                        "updatedAt": str(datetime.now())
                    }
        
            if nested_update:
                # Agora o .set(..., merge=True) vai entender a estrutura aninhada corretamente!
                db.collection("rooms").document(rid).collection("public_state").document("party_states").set(nested_update, merge=True)

        # --- 2. CARREGAMENTO DO ESTADO ---
        state = get_state(db, rid)
        seed = state.get("seed")
        no_water_state = bool(state.get("noWater", False))
    
        all_pieces = state.get("pieces") or []
        seen_pids = state.get("seen") or []
        field_effects = state.get("effects") or []

        # --- 3. HELPERS LOCAIS ---
        ps_doc = db.collection("rooms").document(rid).collection("public_state").document("party_states").get()
        party_states_data = ps_doc.to_dict() or {}

        # --- FUNÇÃO DE LEITURA INTELIGENTE (CORRIGIDA) ---
        def get_poke_data(t_name, p_id):
            user_dict = party_states_data.get(t_name, {})
            p_data = user_dict.get(str(p_id), {})
        
            hp = p_data.get("hp", 6)
            cond = p_data.get("cond", [])
            stats = p_data.get("stats", {})
            shiny_status = p_data.get("shiny", False)
            saved_form = p_data.get("form", None)  # ✅ LER DO BANCO
        
            if t_name == trainer_name:
                # CORREÇÃO: Converte para string, remove espaços e trata vazio como '0'
                # Além disso, ignora a chave 'notes' que é texto puro.
                stats_is_bad = not stats or all(
                    int(str(v).strip() or 0) == 0 
                    for k, v in stats.items() if k != "notes"
                )
            
                if stats_is_bad:
                    if "stats" in user_data:
                        local_s = user_data["stats"].get(str(p_id)) or user_data["stats"].get(p_id)
                        if local_s:
                            stats = local_s
        
            return hp, cond, stats, shiny_status, saved_form
    
        def get_poke_display_name(pid):
            row = df[df['Nº'].astype(str) == str(pid)]
            if not row.empty: return row.iloc[0]['Nome']
            return str(pid)

        # Definição da Função de Renderização da Coluna (DEFINIDA ANTES DE USAR)
        # Definição da Função de Renderização da Coluna (MELHORADA)
        def render_player_column(p_name, p_label, is_me):

            # --- Painel visual (conceito) ---
            with st.container():
                st.markdown("<div class='pvp-team-panel-marker'></div>", unsafe_allow_html=True)

                if is_me:
                    photo_data = get_trainer_photo_thumb(user_data)
                    if photo_data:
                        st.markdown(
                            f"""### <span class='pvp-team-title'><img src='{photo_data}' class='pvp-mini'/> {p_label}</span>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"### {p_label}")
                else:
                    st.markdown(f"### {p_label}")

                # Busca party e estado público
    # Busca party e estado público
            p_doc_data = db.collection("rooms").document(rid).collection("public_state").document("players").get().to_dict() or {}
            party_list = p_doc_data.get(p_name, [])[:8] 
        
            # Variáveis de Estado de Ação
            moving_piece_id = st.session_state.get("moving_piece_id")
            placing_pid = st.session_state.get("placing_pid")
            placing_trainer = st.session_state.get("placing_trainer")
        
            state = get_state(db, rid)
            all_pieces = state.get("pieces") or []
            seen_pids = state.get("seen") or []
        
            if not party_list:
                st.caption("Aguardando...")
                return
    
            trainer_piece = next(
                (p for p in all_pieces if p.get("owner") == p_name and p.get("kind") == "trainer"),
                None,
            )
            p_pieces_on_board = [
                p for p in all_pieces if p.get("owner") == p_name and p.get("kind") != "trainer"
            ]

            if is_me:
                with st.container():
                    st.markdown("<div class='pvp-avatar-card-marker'></div>", unsafe_allow_html=True)
                    st.markdown("#### 🙂 Meu Avatar")
                avatar_choice, avatar_path = get_selected_trainer_avatar(user_data)
                if avatar_path:
                    st.image(avatar_path, width=96)
                else:
                    st.caption("Escolha um avatar na aba Meu Treinador.")

                is_busy = (moving_piece_id is not None) or (placing_pid is not None) or bool(placing_trainer)

                if placing_trainer:
                    st.info("Clique no mapa para posicionar seu avatar.")
                    if st.button("🔙 Cancelar avatar", key="cancel_place_trainer"):
                        st.session_state["placing_trainer"] = None
                        request_rerun("pvp", force=True)
                else:
                    if trainer_piece:
                        c_avatar_1, c_avatar_2, c_avatar_3 = st.columns(3)
                        with c_avatar_1:
                            if st.button("🚶 Mover", key="move_trainer", disabled=is_busy):
                                st.session_state["moving_piece_id"] = trainer_piece.get("id")
                                set_busy(1.2)
                                st.session_state["placing_pid"] = None
                                st.session_state["placing_trainer"] = None
                                request_rerun("pvp", force=True)
                        with c_avatar_2:
                            trainer_revealed = trainer_piece.get("revealed", True)
                            if st.button("👁️" if trainer_revealed else "✅", key="toggle_trainer"):
                                trainer_piece["revealed"] = not trainer_revealed
                                upsert_piece(db, rid, trainer_piece)
                                request_rerun("pvp", force=True)
                        with c_avatar_3:
                            if st.button("❌", key="remove_trainer"):
                                delete_piece(db, rid, trainer_piece.get("id"))
                                if st.session_state.get("moving_piece_id") == trainer_piece.get("id"):
                                    st.session_state["moving_piece_id"] = None
                                request_rerun("pvp", force=True)
                    else:
                        if st.button("📍 Colocar avatar", key="place_trainer", disabled=not avatar_choice or is_busy):
                            st.session_state["placing_trainer"] = True
                            set_busy(1.2)
                            st.session_state["placing_pid"] = None
                            st.session_state["moving_piece_id"] = None
                            request_rerun("pvp", force=True)

            # ==========================
            # UI compacta: lista + detalhes do selecionado
            # ==========================
            sel_key = f"pvp_sel_{rid}_{p_name}"
            if sel_key not in st.session_state:
                st.session_state[sel_key] = str(party_list[0]) if party_list else None
            selected_pid = st.session_state.get(sel_key)

            def _hp_color(hp_val: int) -> str:
                try:
                    v = int(hp_val)
                except Exception:
                    v = 0
                if v <= 0:
                    return "#64748b"  # slate
                if v <= 2:
                    return "#ef4444"  # red
                if v <= 4:
                    return "#f59e0b"  # amber
                return "#22c55e"      # green

            def _hp_bar_html(hp_val: int, max_hp: int = 6) -> str:
                try:
                    v = int(hp_val)
                except Exception:
                    v = 0
                v = max(0, min(max_hp, v))
                pct = int(round((v / max_hp) * 100)) if max_hp else 0
                col = _hp_color(v)
                return f"""<div style='display:flex;align-items:center;gap:10px;margin-top:4px;'>
                    <div style='flex:1;height:10px;background:rgba(148,163,184,0.25);border:1px solid rgba(148,163,184,0.22);border-radius:999px;overflow:hidden;'>
                      <div style='width:{pct}%;height:100%;background:{col};'></div>
                    </div>
                    <div style='min-width:48px;text-align:right;font-family:monospace;color:#e2e8f0;'>{v}/{max_hp}</div>
                </div>"""

            def _badges_html(conds: list[str], limit: int = 3) -> str:
                conds = conds or []
                show = conds[:limit]
                extra = max(0, len(conds) - len(show))
                chips = "".join([
                    f"<span class='pvp-badge' style='font-size:11px;padding:2px 8px;border-radius:999px;margin-right:6px;display:inline-block;'>{{c}}</span>"
                    for c in show
                ])
                if extra:
                    chips += f"<span class='pvp-badge' style='font-size:11px;padding:2px 8px;border-radius:999px;opacity:0.85;'>+{extra}</span>"
                return chips


            for i, pid in enumerate(party_list):
                # 1. Agora recuperamos 5 valores (incluindo p_form)
                cur_hp, cur_cond, cur_stats, is_shiny, p_form = get_poke_data(p_name, pid)
            
                is_on_map = any(str(p["pid"]) == str(pid) for p in p_pieces_on_board)
                p_obj = next((p for p in p_pieces_on_board if str(p["pid"]) == str(pid)), None)
                already_seen = str(pid) in seen_pids
            
                # Ícone de HP
                if cur_hp >= 5: hpi = "💚"
                elif cur_hp >= 3: hpi = "🟡"
                elif cur_hp >= 1: hpi = "🔴"
                else: hpi = "💀"
            
                # 2. Lógica da imagem com prioridade para a Forma
                if p_form:
                     sprite_url = pokemon_pid_to_image(f"EXT:{p_form}", mode="sprite", shiny=is_shiny)
                else:
                     sprite_url = pokemon_pid_to_image(pid, mode="sprite", shiny=is_shiny)
    
                # Checa se ESTE Pokémon específico está realizando uma ação
                is_moving_this = (p_obj and moving_piece_id == p_obj.get("id"))
                is_placing_this = (placing_pid == pid)

                # Estilo da Borda: Amarelo se movendo, Azul se colocando, Padrão caso contrário
                border_color = "#FFCC00" if is_moving_this else ("#38bdf8" if is_placing_this else None)
            
                # Container Visual
                with st.container():
                    card_cls = "pvp-mon-card-marker" + (" pvp-selected" if (str(pid) == str(selected_pid)) else "")
                    st.markdown(f"<div class='{card_cls}'></div>", unsafe_allow_html=True)
                    if is_me:
                        # --- Linha compacta (sempre visível) ---
                        is_busy = (moving_piece_id is not None) or (placing_pid is not None) or bool(placing_trainer)
                        is_selected = (str(pid) == str(selected_pid))
                        p_real_name = get_poke_display_name(pid)
                        loc_lbl = "No campo" if is_on_map else "Mochila"

                        # Destaque sutil se estiver em ação
                        if is_moving_this or is_placing_this:
                            border_color = "#FFCC00" if is_moving_this else "#38bdf8"
                            st.markdown(
                                f"<div style='height:2px;background:{border_color};border-radius:999px;margin:4px 0 10px 0;'></div>",
                                unsafe_allow_html=True,
                            )

                        c_row_img, c_row_mid, c_row_act = st.columns([0.85, 2.85, 1.30])

                        with c_row_img:
                            if cur_hp == 0:
                                st.markdown(
                                    f'<img src="{sprite_url}" style="width:56px;filter:grayscale(100%);opacity:0.65;">',
                                    unsafe_allow_html=True,
                                )
                                st.caption("FAINTED")
                            else:
                                st.image(sprite_url, width=56)

                        with c_row_mid:
                            st.markdown(
                                f"**{p_real_name}** <span style='color:#94a3b8;font-size:12px;margin-left:8px;'>{loc_lbl}</span>",
                                unsafe_allow_html=True,
                            )
                            if cur_cond:
                                st.markdown(_badges_html(cur_cond, limit=3), unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    "<span style='color:#94a3b8;font-size:12px;'>Sem status negativos.</span>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(_hp_bar_html(cur_hp, 6), unsafe_allow_html=True)

                        with c_row_act:
                            a1, a2 = st.columns(2)
                            with a1:
                                if st.button(
                                    "🔍",
                                    key=f"sel_{p_name}_{pid}_{i}",
                                    help="Detalhes",
                                    use_container_width=True,
                                ):
                                    st.session_state[sel_key] = str(pid)
                                    request_rerun("pvp", force=True)

                            with a2:
                                main_icon = "🚶" if is_on_map else "📍"
                                main_help = "Mover" if is_on_map else "Colocar no campo"
                                disabled_main = is_busy and not (is_moving_this or is_placing_this)
                                if st.button(
                                    main_icon,
                                    key=f"main_{p_name}_{pid}_{i}",
                                    help=main_help,
                                    disabled=disabled_main or (cur_hp <= 0),
                                    use_container_width=True,
                                ):
                                    if is_on_map and p_obj:
                                        st.session_state["moving_piece_id"] = p_obj.get("id")
                                        set_busy(1.2)
                                        st.session_state["placing_pid"] = None
                                        st.session_state["placing_trainer"] = None
                                        request_rerun("pvp", force=True)
                                    else:
                                        st.session_state["placing_pid"] = pid
                                        set_busy(1.2)
                                        st.session_state["placing_effect"] = None
                                        st.session_state["moving_piece_id"] = None
                                        st.session_state["placing_trainer"] = None
                                        request_rerun("pvp", force=True)

                        # --- Detalhes (apenas do selecionado) ---
                        if is_selected:
                            st.markdown(
                                "<div style='height:8px;'></div>",
                                unsafe_allow_html=True,
                            )

                            if is_moving_this:
                                st.info("📍 Selecione um quadrado vazio no mapa.")
                                if st.button("🔙 Cancelar mover", key=f"cncl_move_{p_name}_{pid}_{i}"):
                                    st.session_state["moving_piece_id"] = None
                                    request_rerun("pvp", force=True)

                            elif is_placing_this:
                                st.info("📍 Clique no mapa para posicionar.")
                                if st.button("🔙 Cancelar posicionamento", key=f"cncl_place_{p_name}_{pid}_{i}"):
                                    st.session_state["placing_pid"] = None
                                    request_rerun("pvp", force=True)

                            else:
                                # Controles do Pokémon no campo (revelar/remover)
                                if is_on_map and p_obj:
                                    c_vis, c_del = st.columns(2)
                                    with c_vis:
                                        is_rev = p_obj.get("revealed", True)
                                        if st.button(
                                            "👁️" if is_rev else "✅",
                                            key=f"v_{p_name}_{pid}_{i}_detail",
                                            help="Revelar/Esconder",
                                            use_container_width=True,
                                        ):
                                            p_obj["revealed"] = not is_rev
                                            upsert_piece(db, rid, p_obj)
                                            if p_obj["revealed"]:
                                                mark_pid_seen(db, rid, pid)
                                            request_rerun("pvp", force=True)
                                    with c_del:
                                        if st.button(
                                            "❌",
                                            key=f"r_{p_name}_{pid}_{i}_detail",
                                            help="Remover do Mapa",
                                            use_container_width=True,
                                        ):
                                            delete_piece(db, rid, p_obj.get("id"))
                                            add_public_event(db, rid, "pokemon_removed", p_name, {"pid": pid})
                                            st.session_state["moving_piece_id"] = None
                                            request_rerun("pvp", force=True)

                                # Editor de HP / Status (mesmas chaves e callback)
                                st.slider(
                                    "HP",
                                    0,
                                    6,
                                    value=int(cur_hp),
                                    key=f"hp_{p_name}_{pid}_{i}",
                                    label_visibility="collapsed",
                                    on_change=update_poke_state_callback,
                                    args=(db, rid, p_name, pid, i),
                                )

                                st.multiselect(
                                    "Status",
                                    ["⚡", "❄️", "🔥", "💤", "☠️", "💓"],
                                    default=cur_cond,
                                    key=f"cond_{p_name}_{pid}_{i}",
                                    label_visibility="collapsed",
                                    on_change=update_poke_state_callback,
                                    args=(db, rid, p_name, pid, i),
                                )

                                # Botão principal grande (opcional) para conforto
                                if cur_hp > 0:
                                    if is_on_map and p_obj:
                                        if st.button(
                                            "🚶 Mover",
                                            key=f"m_{p_name}_{pid}_{i}_detail",
                                            disabled=is_busy,
                                            use_container_width=True,
                                        ):
                                            st.session_state["moving_piece_id"] = p_obj.get("id")
                                            set_busy(1.2)
                                            request_rerun("pvp", force=True)
                                    else:
                                        if st.button(
                                            "📍 Colocar no Campo",
                                            key=f"p_{p_name}_{pid}_{i}_detail",
                                            disabled=is_busy,
                                            use_container_width=True,
                                        ):
                                            st.session_state["placing_pid"] = pid
                                            set_busy(1.2)
                                            st.session_state["placing_effect"] = None
                                            request_rerun("pvp", force=True)

                            st.markdown("<hr style='opacity:0.15;margin:10px 0;'>", unsafe_allow_html=True)

                    else:
                        # ==========================
                        # VISÃO DO OPONENTE (Corrigida)
                        # ==========================
                        piece_obj = next((p for p in p_pieces_on_board if str(p["pid"]) == str(pid)), None)
                        is_revealed = piece_obj.get("revealed", True) if piece_obj else False
                    
                        # Se estiver no campo e revelado, OU se já foi visto antes (está na bench mas conhecido)
                        show_full = (piece_obj and is_revealed) or already_seen
                    
                        status_txt = "(Mochila)" if not piece_obj else ("(Escondido)" if not is_revealed else "")
    
                        if show_full:
                            p_real_name = get_poke_display_name(pid)
                            c1, c2 = st.columns([1, 2])
                        
                            with c1: 
                                # ✅ CORREÇÃO 1: Imagem em escala de cinza se HP for 0
                                # Antes usava st.image direto, que sempre é colorido.
                                if cur_hp == 0:
                                    st.markdown(f'<img src="{sprite_url}" style="width:50px; filter:grayscale(100%); opacity:0.6;">', unsafe_allow_html=True)
                                else:
                                    st.image(sprite_url, width=50)
    
                            with c2:
                                st.markdown(f"**{p_real_name}**")
                                st.caption(f"{hpi} HP: {cur_hp}/6 {status_txt}")
                            
                                # ✅ CORREÇÃO 2: Exibir ícones de status para o oponente
                                if cur_cond:
                                    # Junta os ícones em uma string (ex: "🔥 ☠️")
                                    cond_str = " ".join(cur_cond)
                                    st.markdown(f"Status: {cond_str}")
    
                                if cur_hp == 0: 
                                    st.caption("**FAINTED**")
                        else:
                            # Pokémon desconhecido / escondido
                            c1, c2 = st.columns([1, 2])
                            with c1: st.image("https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg", width=40)
                            with c2: st.caption(f"??? {status_txt}")
    
        # --- 4. PREPARAÇÃO DE TIMES E VARIÁVEIS (UNIFICADO) ---
        owner_name = (room.get("owner") or {}).get("name", "Host")
        challengers = room.get("challengers") or []
        challenger_names = [c.get("name") for c in challengers]
    
        # Lista total de quem está na arena para a lógica de cores e calculadora
        all_players = [owner_name] + challenger_names

        # 1. MECÂNICA DE INTERFACE: Define quem é "Você" e as etiquetas (do código SEU)
        if trainer_name == owner_name:
            p1_name = owner_name
            p1_label = f"🎒 {owner_name} (Você)"
            viewer_is_p1 = True
            p2_name = challenger_names[0] if challenger_names else None
            p2_label = f"🆚 {p2_name}" if p2_name else "🆚 Aguardando..."
        elif trainer_name in challenger_names:
            p1_name = trainer_name
            p1_label = f"🎒 {trainer_name} (Você)"
            viewer_is_p1 = True 
            p2_name = owner_name
            p2_label = f"🆚 {owner_name}"
        else:
            p1_name = owner_name
            p1_label = f"🔴 {owner_name}"
            viewer_is_p1 = False
            p2_name = challenger_names[0] if challenger_names else None
            p2_label = f"🔵 {p2_name}" if p2_name else "🔵 Aguardando..."

        # 2. MECÂNICA DE MAPA E CALCULADORA: Processa peças e visibilidade (do código MEU)
        pieces_to_draw = []
        player_pieces_map = {name: [] for name in all_players}

        for p in all_pieces:
            if p.get("kind") == "trainer":
                if p.get("owner") == trainer_name:
                    pieces_to_draw.append(p)
                elif p.get("revealed", True):
                    pieces_to_draw.append(p)
                if p.get("owner") in player_pieces_map:
                    player_pieces_map[p.get("owner")].append(p)
                continue
            # 1. Captura a forma (5ª variável) do banco de dados
            hp_check, _, _, _, p_form = get_poke_data(p.get("owner"), p.get("pid"))
        
            p["status"] = "fainted" if hp_check == 0 else "active"
        
            # ✅ O TRUQUE: Se tiver forma salva, substituímos o PID temporariamente
            # Isso força o renderizador do mapa a baixar o sprite da forma correta (ex: EXT:lycanroc-midnight)
            if p_form:
                p["pid"] = f"EXT:{p_form}"

            # Lógica de Visibilidade: Dono vê tudo, outros veem apenas revelados 
            if p.get("owner") == trainer_name: 
                pieces_to_draw.append(p)
            elif p.get("revealed", True): 
                pieces_to_draw.append(p)
        
            # Popula o mapa para a calculadora encontrar os alvos corretamente
            if p.get("owner") in player_pieces_map:
                player_pieces_map[p.get("owner")].append(p)

        theme_key = room.get("theme", "cave_water")
        grid = int(state.get("gridSize") or room.get("gridSize") or 10)

        # --- 5. INTERFACE DO TOPO ---
        last_events = list_public_events(db, rid, limit=1)
        last_dice = next((e for e in last_events if e.get("type") == "dice"), None)

        top = st.columns([1, 1, 1, 1, 4])
        with top[0]:
            if st.button("⬅️ Lobby"):
                st.session_state["pvp_view"] = "lobby"
                request_rerun("pvp", force=True)
        with top[1]:
            if st.button("🔄 Atualizar"): request_rerun("pvp", force=True)
        with top[2]:
            if st.button("🎲 d20", disabled=not is_player): roll_die(db, rid, trainer_name, sides=20); request_rerun("pvp", force=True)
        with top[3]:
            if st.button("🎲 d6", disabled=not is_player): roll_die(db, rid, trainer_name, sides=6); request_rerun("pvp", force=True)
        with top[4]:
            if last_dice:
                pl = last_dice.get("payload", {})
                dice_line = f"🎲 {last_dice.get('by')}: <strong>{pl.get('result')}</strong> (d{pl.get('sides')})"
                dice_bg = "rgba(34, 197, 94, 0.18)"
                dice_border = "rgba(34, 197, 94, 0.5)"
            else:
                dice_line = "🎲 Aguardando rolagem..."
                dice_bg = "rgba(148, 163, 184, 0.15)"
                dice_border = "rgba(148, 163, 184, 0.35)"
            st.markdown(
                f"""
                <div style="
                    min-height: 42px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 8px;
                    padding: 6px 10px;
                    border-radius: 10px;
                    border: 1px solid {dice_border};
                    background: {dice_bg};
                    font-weight: 600;">
                    {dice_line}
                </div>
                """,
                unsafe_allow_html=True,
            )
            # ✅ PEDIDO: Mostrar Código da Sala aqui em cima
            st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 10px;'>
        <span style='font-family: "Press Start 2P"; font-size: 18px;'>🏟️ Arena:</span>
        <span style='font-family: "Press Start 2P"; font-size: 24px; color: #FFCC00; background: #333; padding: 5px 10px; border-radius: 5px;'>{rid}</span>
    </div>
    """, unsafe_allow_html=True) 

        # ==========================================
        # 🧮 6. CALCULADORA DE COMBATE
        # ==========================================
        battle_ref = db.collection("rooms").document(rid).collection("public_state").document("battle")
        battle_doc = battle_ref.get()
        b_data = battle_doc.to_dict() or {"status": "idle", "logs": []}

        # =========================
        # 🎛️ HUD / Barra de Status
        # =========================
        b_status = (b_data.get("status") or "idle")
        b_attacker = b_data.get("attacker") or "—"
        sync_badge = "Atualizando…" if st.session_state.get("pvp_sync_pending") else "Sincronizado ✓"

        # CSS leve para cards/badges (aplicado só na Arena PvP)
        st.markdown("""
        <style>
        .pvp-statusbar{
            display:flex; gap:10px; align-items:center; flex-wrap:wrap;
            padding:10px 12px; border-radius:14px;
            border:1px solid rgba(148,163,184,0.35);
            background: rgba(15,23,42,0.55);
            box-shadow: inset 0 0 18px rgba(15,23,42,0.35);
            margin: 6px 0 10px 0;
        }
        .pvp-pill{
            display:inline-flex; gap:8px; align-items:center;
            padding:6px 10px; border-radius:999px;
            border:1px solid rgba(148,163,184,0.35);
            background: rgba(2,6,23,0.35);
            font-weight:600; font-size:12px;
        }
        .pvp-badge{
            display:inline-flex; align-items:center; justify-content:center;
            padding:2px 8px; border-radius:999px;
            background: rgba(56,189,248,0.18);
            border: 1px solid rgba(56,189,248,0.35);
            font-size: 12px; font-weight: 700;
            margin-right: 6px;
        }
    
        /* =========================
           PvP / Arena - Team Panel (conceito)
           ========================= */
        [data-testid="stVerticalBlock"]:has(.pvp-team-panel-marker){
            border-radius: 18px;
            padding: 14px 14px 10px 14px !important;
            border: 2px solid rgba(148,163,184,0.35);
            background: radial-gradient(1200px 500px at 20% 0%, rgba(56,189,248,0.12), transparent 60%),
                        rgba(15,23,42,0.58);
            box-shadow: inset 0 0 18px rgba(15,23,42,0.40);
        }
        .pvp-team-title{
            display:flex; align-items:center; gap:10px;
            font-weight:800;
            letter-spacing: 0.5px;
        }
        .pvp-team-title .pvp-mini{
            width:32px;height:32px;border-radius:10px;object-fit:cover;
            border:1px solid rgba(226,232,240,0.28);
            box-shadow: 0 6px 14px rgba(2,6,23,0.35);
        }

        /* Card de Avatar (topo) */
        [data-testid="stVerticalBlock"]:has(.pvp-avatar-card-marker){
            margin-top: 6px;
            border-radius: 18px;
            padding: 12px 12px 12px 12px !important;
            border: 1px solid rgba(148,163,184,0.28);
            background: rgba(2,6,23,0.28);
            box-shadow: inset 0 0 14px rgba(2,6,23,0.22);
        }

        /* Card de Pokémon (cada linha) */
        [data-testid="stVerticalBlock"]:has(.pvp-mon-card-marker){
            border-radius: 18px;
            padding: 12px 12px 10px 12px !important;
            border: 1px solid rgba(148,163,184,0.22);
            background: rgba(2,6,23,0.22);
            box-shadow: 0 10px 20px rgba(2,6,23,0.18);
            margin-bottom: 10px;
        }
        [data-testid="stVerticalBlock"]:has(.pvp-mon-card-marker.pvp-selected){
            border: 2px solid rgba(56,189,248,0.55);
            background: radial-gradient(900px 380px at 15% 0%, rgba(56,189,248,0.12), transparent 60%),
                        rgba(2,6,23,0.22);
        }

        /* Botões dentro dos cards (aproxima o conceito) */
        [data-testid="stVerticalBlock"]:has(.pvp-mon-card-marker) [data-testid="stButton"] button,
        [data-testid="stVerticalBlock"]:has(.pvp-avatar-card-marker) [data-testid="stButton"] button{
            border-radius: 14px !important;
            border: 1px solid rgba(148,163,184,0.28) !important;
            background: linear-gradient(180deg, rgba(56,189,248,0.55) 0%, rgba(14,165,233,0.35) 100%) !important;
            box-shadow: 0 8px 18px rgba(2,6,23,0.22);
            color: #e2e8f0 !important;
            font-weight: 800 !important;
        }
        [data-testid="stVerticalBlock"]:has(.pvp-mon-card-marker) [data-testid="stButton"] button:hover,
        [data-testid="stVerticalBlock"]:has(.pvp-avatar-card-marker) [data-testid="stButton"] button:hover{
            filter: brightness(1.08);
            transform: translateY(-1px);
        }

        /* Botões "ícone" (🔍 / 📍 / 🚶 / 👁️ / ❌) */
        [data-testid="stVerticalBlock"]:has(.pvp-mon-card-marker) [data-testid="stButton"] button:has(span){
            min-height: 40px;
        }

        /* Remove padding extra do bloco interno quando usamos marcadores */
        [data-testid="stVerticalBlock"] > div:has(.pvp-team-panel-marker),
        [data-testid="stVerticalBlock"] > div:has(.pvp-avatar-card-marker),
        [data-testid="stVerticalBlock"] > div:has(.pvp-mon-card-marker){
            padding: 0 !important;
            margin: 0 !important;
        }

        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="pvp-statusbar">
              <span class="pvp-pill">🧩 Sala <b>{rid}</b></span>
              <span class="pvp-pill">🎚️ Fase: <b>{b_status}</b></span>
              <span class="pvp-pill">⚔️ Agindo: <b>{b_attacker}</b></span>
              <span class="pvp-pill">📡 {sync_badge}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        tab_arena, tab_combate, tab_inic, tab_log = st.tabs(["🗺️ Arena", "⚔️ Combate", "🧭 Iniciativa", "📜 Log"])
        import math

        with tab_combate:
            with st.expander("⚔️ Calculadora de Combate", expanded=(b_status != "idle")):
        
                # [FASE 0] IDLE
                if b_status == "idle":
                    if is_player:
                        if st.button("Nova Batalha (Atacar)"):
                            battle_ref.set({"status": "setup", "attacker": trainer_name, "attack_move": None, "logs": []})
                            request_rerun("pvp", force=True)
                    else:
                        st.caption("Aguardando combate...")
        
                # [FASE 1] CONFIGURAR ATAQUE
                elif b_status == "setup":
                    # CORREÇÃO: Parêntese fechado corretamente aqui
                    st.caption(f"**Atacante:** {b_data.get('attacker')}")
            
                    if b_data.get("attacker") == trainer_name:
                        attacker_pid = None
                        attacker_sheet = None
                        attacker_stats = {}
                        if current_party:
                            attacker_pid = st.selectbox(
                                "Seu Pokémon (Atacante)",
                                options=current_party,
                                format_func=get_poke_display_name,
                                key=f"atk_self_{rid}",
                            )
                            attacker_sheet = battle_sheets_map.get(str(attacker_pid))
                            if attacker_sheet:
                                attacker_stats = attacker_sheet.get("stats") or {}
                            else:
                                _, _, attacker_stats, _, _ = get_poke_data(trainer_name, attacker_pid)
                        else:
                            st.info("Sua party está vazia para selecionar o atacante.")
                        # Busca peças de TODOS os outros jogadores que não são você
                        target_options = {}
                        for p_name, p_pieces in player_pieces_map.items():
                            if p_name != trainer_name:
                                for p in p_pieces:
                                    # Nome do Pokemon + Dono para identificar no 2v1 ou 2v2
                                    label = f"{get_poke_display_name(p['pid'])} ({p_name})"
                                    target_options[p['id']] = label
                
                        c_atk1, c_atk2, c_atk3 = st.columns(3)
                
                        # CORREÇÃO: Indentação alinhada com o bloco acima (4 espaços dentro do IF)
                        with c_atk1:
                            target_id = st.selectbox("Alvo", options=list(target_options.keys()), 
                                        format_func=lambda x: target_options[x],
                                        key=f"atk_target_{rid}") if target_options else None
                
                        with c_atk2:
                            attack_mode = st.radio("Modo", ["Normal", "Área"], horizontal=True, key=f"atk_mode_{rid}")
                
                        # Se for Área
                        if attack_mode == "Área":
                            st.info("Ataque em Área: Dodge (CD 10 + Nível) reduz dano pela metade.")
                            lvl_effect = st.number_input("Nível do Efeito / Dano", min_value=1, value=1)
                            is_eff_area = st.checkbox("É Efeito? (Affliction)", key=f"area_eff_{rid}")

                            if st.button("🚀 Lançar Área"):
                                if target_id:
                                    t_p = next((p for p in all_pieces if p['id'] == target_id), None)
                                    battle_ref.update({
                                        "status": "aoe_defense",
                                        "target_id": target_id,
                                        "target_owner": t_p['owner'],
                                        "target_pid": t_p['pid'],
                                        "aoe_dc": lvl_effect + 10,
                                        "dmg_base": lvl_effect,
                                        "is_effect": is_eff_area,
                                        "logs": [f"{trainer_name} lançou Área (Nv {lvl_effect}). Defensor rola Dodge (CD {lvl_effect+10})."]
                                    })
                                    request_rerun("pvp", force=True)
                        else:
                            # Normal
                            with c_atk3:
                                atk_type = st.selectbox("Alcance", ["Distância (Dodge)", "Corpo-a-corpo (Parry)"])
                    

                            move_payload = None
                            selected_accuracy = None
                            selected_damage = None
                            if attacker_sheet:
                                moves = attacker_sheet.get("moves") or []
                            else:
                                moves = []

                            if moves:
                                def _move_label(idx: int) -> str:
                                    mv = moves[idx]
                                    name = mv.get("name", "Golpe")
                                    rank = int(mv.get("rank", 0) or 0)
                                    based, stat_val = _move_stat_value(mv.get("meta") or {}, attacker_stats)
                                    damage = rank + stat_val
                                    acc = int(mv.get("accuracy", 0) or 0)
                                    return f"{name}. Acerto: {acc}. Dano {damage}"

                                move_options = ["manual"] + list(range(len(moves)))
                                move_choice = st.selectbox(
                                    "Golpe (opcional)",
                                    options=move_options,
                                    format_func=lambda x: "Manual (sem golpe)" if x == "manual" else _move_label(x),
                                    key=f"atk_move_{rid}",
                                )

                                if move_choice != "manual":
                                    selected_move = moves[int(move_choice)]
                                    selected_accuracy = int(selected_move.get("accuracy", 0) or 0)
                                    rank = int(selected_move.get("rank", 0) or 0)
                                    based, stat_val = _move_stat_value(selected_move.get("meta") or {}, attacker_stats)
                                    selected_damage = rank + stat_val
                                    move_payload = {
                                        "name": selected_move.get("name", "Golpe"),
                                        "accuracy": selected_accuracy,
                                        "damage": selected_damage,
                                        "rank": rank,
                                        "based_stat": based,
                                        "stat_value": stat_val,
                                    }
                            else:
                                st.caption("Sem golpes disponíveis para este Pokémon.")

                            ac1, ac2 = st.columns([2, 1])
                            with ac1:
                                atk_mod = st.number_input("Acerto (Modificador)", value=0, step=1)
                            with ac2:
                                if selected_accuracy is not None:
                                    st.markdown(f"**Acerto sugerido:** {selected_accuracy}")
                                else:
                                    st.caption("Sem acerto sugerido.")
                    
                            if st.button("⚔️ Rolar Ataque"):
                                if target_id:
                                    d20 = random.randint(1, 20)
                                    t_p = next((p for p in all_pieces if p['id'] == target_id), None)
                            
                                    # Pega stats do alvo
                                    _, _, t_stats, _, _ = get_poke_data(t_p['owner'], t_p['pid'])
                                    dodge = int(t_stats.get("dodge", 0))
                                    parry = int(t_stats.get("parry", 0))
                            
                                    defense_val = dodge if "Distância" in atk_type else parry
                                    needed = defense_val + 10
                                    total_atk = atk_mod + d20
                            
                                    hit = total_atk >= needed
                                    result_msg = "ACERTOU! ✅" if hit else "ERROU! ❌"
                            
                                    battle_ref.update({
                                        "status": "hit_confirmed" if hit else "missed",
                                        "target_id": target_id,
                                        "target_owner": t_p['owner'],
                                        "target_pid": t_p['pid'],
                                        "attack_move": move_payload,
                                        "logs": [f"{trainer_name} rolou {d20}+{atk_mod}=**{total_atk}** (vs Def {needed} [{defense_val}+10])... {result_msg}"]
                                    })
                                    request_rerun("pvp", force=True)
                    else:
                        st.info(f"Aguardando {b_data.get('attacker')}...")

                # [FASE 1.5] DEFESA DE ÁREA
                elif b_status == "aoe_defense":
                    st.info(b_data["logs"][-1])
                    if b_data.get("target_owner") == trainer_name:
                        st.markdown("### 🏃 Rolar Esquiva (Dodge)")
                        if st.button("Rolar Dodge"):
                            d20 = random.randint(1, 20)
                            _, _, t_stats, _, _ = get_poke_data(trainer_name, b_data.get('target_pid'))
                            dodge_val = int(t_stats.get("dodge", 0))
                    
                            total_roll = d20 + dodge_val
                            dc = b_data.get("aoe_dc", 10)
                            base_rank = b_data.get("dmg_base", 0)
                    
                            if total_roll >= dc:
                                final_rank = math.floor(base_rank / 2)
                                msg = f"Sucesso! ({total_roll} vs {dc}). Rank reduzido: {base_rank} -> {final_rank}."
                            else:
                                final_rank = base_rank
                                msg = f"Falha! ({total_roll} vs {dc}). Rank total: {final_rank}."
                    
                            battle_ref.update({
                                "status": "waiting_defense",
                                "dmg_base": final_rank,
                                "logs": firestore.ArrayUnion([msg + " Escolha a resistência agora."])
                            })
                            request_rerun("pvp", force=True)
                    else:
                        st.warning("Aguardando defensor...")

                # [FASE 2] INSERIR DANO (Se acertou)
                elif b_status == "hit_confirmed":
                    st.success(b_data["logs"][-1])
            
                    if b_data.get("attacker") == trainer_name:
                        move_payload = b_data.get("attack_move") or {}
                        suggested_damage = move_payload.get("damage")

                        c1, c2, c3 = st.columns([2, 1, 1])
                        with c1:
                            dmg_input = st.number_input("Dano Base / Rank", min_value=0, value=0)
                        with c2:
                            if suggested_damage is not None:
                                st.markdown(f"**Dano sugerido:** {suggested_damage}")
                            else:
                                st.caption("Sem dano sugerido.")
                        with c3:
                            is_eff_check = st.checkbox("É Efeito?", value=False, key=f"norm_eff_{rid}", help="Se marcado, CD base será 10. Se não, 15.")
                
                        if st.button("Enviar Dano/Efeito"):
                            battle_ref.update({
                                "status": "waiting_defense",
                                "dmg_base": dmg_input,
                                "is_effect": is_eff_check,
                                "logs": firestore.ArrayUnion([f"Rank/Dano: {dmg_input} ({'Efeito' if is_eff_check else 'Dano'}). Aguardando resistência..."])
                            })
                            request_rerun("pvp", force=True)
                    else:
                        st.info("Aguardando atacante definir o dano...")

                elif b_status == "missed":
                    st.error(b_data["logs"][-1])
                    if b_data.get("attacker") == trainer_name:
                        if st.button("Encerrar"):
                            battle_ref.update({"status": "idle", "logs": []})
                            request_rerun("pvp", force=True)

                # [FASE 3] RESISTÊNCIA FINAL
                elif b_status == "waiting_defense":
                    is_eff = b_data.get("is_effect", False)
                    base_val = 10 if is_eff else 15
                    rank = int(b_data.get("dmg_base", 0))
                    dc_total = base_val + rank
            
                    st.info(f"Resistir contra: **CD {dc_total}** ({base_val} + {rank})")
            
                    if b_data.get("target_owner") == trainer_name:
                        st.markdown("### 🛡️ Resistir com:")
                        c1, c2, c3, c4 = st.columns(4)
                        res_type = None
                        if c1.button("Dodge"): res_type = "dodge"
                        if c2.button("Parry"): res_type = "parry"
                        if c3.button("Fort"): res_type = "fort"
                        if c4.button("Will"): res_type = "will"
                        if st.button("THG (Toughness)"): res_type = "thg"

                        if res_type:
                            def_die = random.randint(1, 20)
                            _, _, t_stats, _, _ = get_poke_data(trainer_name, b_data.get('target_pid'))
                            stat_val = int(t_stats.get(res_type, 0))
                    
                            check_total = def_die + stat_val
                    
                            # Cálculo M&M: Falha = CD - Check
                            diff = dc_total - check_total
                    
                            if diff <= 0:
                                bars_lost = 0
                                res_msg = "SUCESSO (Nenhum dano)"
                            else:
                                # 1 grau a cada 5 pontos
                                bars_lost = math.ceil(diff / 5)
                                res_msg = f"FALHA por {diff}"
                    
                            final_msg = f"🛡️ Defensor rolou {def_die} + {stat_val} = **{check_total}** ({res_type.upper()}). {res_msg}. Perdeu **{bars_lost}** barras."
                    
                            battle_ref.update({
                                "status": "finished",
                                "final_bars": bars_lost,
                                "logs": firestore.ArrayUnion([final_msg])
                            })
                            request_rerun("pvp", force=True)
                    else:
                        st.warning("Aguardando defesa...")

                # [FASE 4] FIM / SECUNDÁRIO
                elif b_status == "finished":
                    st.markdown(f"## 🩸 Resultado: -{b_data.get('final_bars')} Barras")
                    for log in b_data.get("logs", []): st.text(log)
            
                    if b_data.get("attacker") == trainer_name:
                        c_end1, c_end2 = st.columns(2)
                        if c_end1.button("Encerrar Combate"):
                            battle_ref.update({"status": "idle", "logs": []})
                            request_rerun("pvp", force=True)
                
                        target_name = get_poke_display_name(b_data.get('target_pid'))
                        if c_end2.button(f"⚡ Efeito Secundário em {target_name}"):
                            battle_ref.update({
                                "status": "hit_confirmed", 
                                "is_effect": False, 
                                "logs": [f"⚡ Efeito Secundário ativado em {target_name}!"]
                            })
                            request_rerun("pvp", force=True)
                    else:
                        st.info("Aguardando atacante encerrar...")


            # =========================
        with tab_arena:
            # 7. LAYOUT DAS COLUNAS (EQUILIBRADO PARA 4 JOGADORES)
            # =========================
            if seed is None:
                st.warning("Sem mapa.")
                st.stop()

            # Proporção [1.8, 4.6, 1.8] dá leve redução do mapa e mantém fôlego nas laterais
            opponents_list = sorted(list(set([p for p in all_players if p != trainer_name])))
            if opponents_list:
                c_me, c_map, c_opps = st.columns([1.9, 5.2, 1.9])
            else:
                c_me, c_map = st.columns([2.2, 6.0])
                c_opps = None

            with c_me:
                render_player_column(trainer_name, "🎒 Sua Equipe", is_me=True)

            with c_map:
                st.markdown(f"### 🗺️ Arena (Sala {rid})")
        
                # --- ALERTA VISUAL DE AÇÃO NO MAPA ---
                #if st.session_state.get("moving_piece_id"):
                #    st.warning("🏃 MODO MOVIMENTO: Clique em um quadrado vazio para mover o Pokémon.", icon="📍")
                #elif st.session_state.get("placing_pid"):
                #    st.info("📍 MODO POSICIONAMENTO: Clique no mapa para colocar o Pokémon.", icon="⬇️")
                #elif st.session_state.get("placing_effect"):
                #    eff_icon = st.session_state.get("placing_effect")
                #    st.info(f"✨ MODO TERRENO: Clique para adicionar {eff_icon}.", icon="✨")
                # -------------------------------------

                            # Ferramentas de Campo (UX melhorada)
                moving_piece_id = st.session_state.get("moving_piece_id")
                placing_pid = st.session_state.get("placing_pid")
                placing_trainer = st.session_state.get("placing_trainer")
                placing_eff = st.session_state.get("placing_effect")

                # Banner do "modo" atual + botão de cancelar
                if moving_piece_id or placing_pid or placing_trainer or placing_eff:
                    bmsg, bbtn = st.columns([6, 1])
                    with bmsg:
                        if moving_piece_id:
                            st.warning("🏃 MODO MOVIMENTO: clique em um quadrado vazio para mover.", icon="📍")
                        elif placing_pid:
                            st.info("📍 MODO POSICIONAMENTO: clique no mapa para colocar o Pokémon.", icon="⬇️")
                        elif placing_trainer:
                            st.info("🧍 MODO AVATAR: clique no mapa para posicionar seu treinador.", icon="⬇️")
                        elif placing_eff == "__erase__":
                            st.warning("🧽 MODO BORRACHA: clique em um quadrado para REMOVER o efeito.", icon="🧽")
                        elif placing_eff:
                            st.info(f"✨ MODO TERRENO: clique para adicionar {placing_eff}.", icon="✨")
                    with bbtn:
                        if st.button("✖", key=f"cancel_action_{rid}", help="Cancelar ação atual", use_container_width=True):
                            st.session_state["moving_piece_id"] = None
                            st.session_state["placing_pid"] = None
                            st.session_state["placing_trainer"] = None
                            st.session_state["placing_effect"] = None
                            request_rerun("pvp", force=True)

                # Ferramentas de Campo
                with st.expander("🛠️ Itens e Terrenos", expanded=False):
                    if is_player:
                        st.checkbox(
                            "🖌️ Pincel (não desarmar ao clicar)",
                            value=bool(st.session_state.get(f"effect_brush_{rid}", True)),
                            key=f"effect_brush_{rid}",
                            help="Se ligado, você pode clicar várias vezes no mapa sem precisar re-selecionar o ícone.",
                        )

                        top_tools = st.columns([1.1, 1.1, 2.8])
                        with top_tools[0]:
                            erase_curr = (st.session_state.get("placing_effect") == "__erase__")
                            if st.button(
                                "🧽 Apagar",
                                key=f"effect_erase_{rid}",
                                type=("primary" if erase_curr else "secondary"),
                                use_container_width=True,
                                help="Ativa um modo para remover o efeito clicando no mapa.",
                            ):
                                st.session_state["placing_effect"] = None if erase_curr else "__erase__"
                                st.session_state["moving_piece_id"] = None
                                st.session_state["placing_pid"] = None
                                st.session_state["placing_trainer"] = None
                                request_rerun("pvp", force=True)
                        with top_tools[1]:
                            if st.button(
                                "🧼 Limpar Tudo",
                                key=f"effect_clear_{rid}",
                                use_container_width=True,
                                help="Remove todos os efeitos do mapa.",
                            ):
                                db.collection("rooms").document(rid).collection("public_state").document("state").update({
                                    "effects": [],
                                    "updatedAt": firestore.SERVER_TIMESTAMP,
                                })
                                request_rerun("pvp", force=True)
                        with top_tools[2]:
                            st.caption("Selecione um ícone e clique no mapa. (Pincel mantém o modo ativo.)")

                        effects_map = {"Fogo":"🔥", "Gelo":"🧊", "Água":"💧", "Rocha":"🪨", "Nuvem":"☁️", "Sol":"☀️", "Grama":"🍃", "Raio":"⚡"}
                        curr = st.session_state.get("placing_effect")

                        # Contagem dos efeitos atuais (opcional, ajuda a visualizar densidade)
                        counts = {}
                        for e in (field_effects or []):
                            ic = (e or {}).get("icon")
                            if ic:
                                counts[ic] = counts.get(ic, 0) + 1

                        cols = st.columns(8)
                        for i, (k, v) in enumerate(effects_map.items()):
                            btn_type = "primary" if curr == v else "secondary"
                            label = f"{v}" + (f" {counts[v]}" if counts.get(v) else "")
                            if cols[i].button(label, key=f"ef_{rid}_{k}", type=btn_type, use_container_width=True):
                                st.session_state["placing_effect"] = None if curr == v else v
                                # Se ativar efeito, cancela outras ações para evitar bugs
                                st.session_state["moving_piece_id"] = None
                                st.session_state["placing_pid"] = None
                                st.session_state["placing_trainer"] = None
                                request_rerun("pvp", force=True)

                # Ajustes de visualização do mapa (zoom automático por tamanho para manter proporção)
                toolbar = st.columns([1.15, 2.35, 1.0])
                with toolbar[0]:
                    show_grid = st.checkbox("Grade Tática", value=bool(st.session_state.get(f"grid_{rid}", True)), key=f"grid_{rid}")

                # Menor mapa => maior zoom, com limite para não ficar desproporcional.
                auto_zoom_by_grid = {
                    6: 1.15,
                    8: 1.00,
                    10: 0.90,
                    12: 0.82,
                }
                map_zoom = float(auto_zoom_by_grid.get(int(grid), 0.90))
                st.session_state[f"map_zoom_{rid}"] = map_zoom

                with toolbar[1]:
                    st.markdown("**Zoom do mapa:** automático")
                with toolbar[2]:
                    st.caption(f"{int(map_zoom * 100)}%")

                # ... (Restante do código de renderização do mapa permanece igual) ...
                state_updated_at = state.get("updatedAt")
                map_signature = json.dumps({
                    "seed": seed,
                    "theme": theme_key,
                    "noWater": no_water_state,
                    "pieces": pieces_to_draw,
                    "effects": field_effects,
                    "grid": show_grid,
                    "updatedAt": state_updated_at,
                }, sort_keys=True, default=str)
                if st.session_state.get("map_cache_sig") != map_signature:
                    st.session_state["map_cache_sig"] = map_signature
                    st.session_state["map_cache_img"] = render_map_with_pieces(
                        grid,
                        theme_key,
                        seed,
                        no_water_state,
                        pieces_to_draw,
                        trainer_name,
                        room,
                        effects=field_effects,
                        show_grid=show_grid,
                    )
                img = st.session_state.get("map_cache_img")

                # Zoom do mapa (ajuda a encaixar melhor em 100%/50% de zoom do navegador)
                map_zoom = float(st.session_state.get(f"map_zoom_{rid}", 0.90) or 0.90)
                tile_px = max(1, int(TILE_SIZE * map_zoom))
                st.session_state[f"_tile_px_{rid}"] = tile_px

                img_to_show = img
                if img_to_show and map_zoom != 1.0:
                    try:
                        img_to_show = img_to_show.resize(
                            (max(1, int(img_to_show.width * map_zoom)), max(1, int(img_to_show.height * map_zoom))),
                            resample=Image.NEAREST,
                        )
                    except Exception:
                        img_to_show = img
                        tile_px = TILE_SIZE
                        st.session_state[f"_tile_px_{rid}"] = tile_px

                with st.container():
                    click = streamlit_image_coordinates(img_to_show, key=f"map_{rid}")
            if c_opps is not None:
                with c_opps:
                    st.markdown("### 🆚 Oponentes")
                    opponents = opponents_list

                    if not opponents:
                        st.caption("Aguardando...")
                    else:
                        for idx, opp_name in enumerate(opponents):
                            icons = ["🔴", "🟡", "🌸"]
                            icon = icons[idx] if idx < len(icons) else "⚪"

                            with st.expander(f"{icon} {opp_name}", expanded=True):
                                render_player_column(opp_name, "", is_me=False)


            # =========================
            # 8. LÓGICA DE CLIQUE
            # =========================
            if click and "x" in click and "y" in click:
                tile_px_eff = int(st.session_state.get(f"_tile_px_{rid}", TILE_SIZE) or TILE_SIZE)
                col = int(click["x"] // tile_px_eff)
                row = int(click["y"] // tile_px_eff)
                if 0 <= row < grid and 0 <= col < grid:
                    ppid = st.session_state.get("placing_pid")
                    peff = st.session_state.get("placing_effect")
                    moving_piece_id = st.session_state.get("moving_piece_id")
                    placing_trainer = st.session_state.get("placing_trainer")
                    if peff:
                        curr = state.get("effects") or []
                        brush_on = bool(st.session_state.get(f"effect_brush_{rid}", True))

                        if peff == "__erase__":
                            # Remove efeito (se existir) no quadrado clicado
                            new = [
                                e for e in curr
                                if not (int((e or {}).get("row", -1)) == row and int((e or {}).get("col", -1)) == col)
                            ]
                            if len(new) != len(curr):
                                db.collection("rooms").document(rid).collection("public_state").document("state").update({
                                    "effects": new,
                                    "updatedAt": firestore.SERVER_TIMESTAMP,
                                })
                                add_public_event(db, rid, "effect_removed", trainer_name, {"to": [row, col]})

                            if not brush_on:
                                st.session_state["placing_effect"] = None
                            request_rerun("pvp", force=True)

                        # Aplica/atualiza efeito
                        new = [
                            e for e in curr
                            if not (int((e or {}).get("row", -1)) == row and int((e or {}).get("col", -1)) == col)
                        ]
                        new.append({"icon": peff, "row": row, "col": col, "id": str(uuid.uuid4())[:8]})
                        db.collection("rooms").document(rid).collection("public_state").document("state").update({
                            "effects": new,
                            "updatedAt": firestore.SERVER_TIMESTAMP,
                        })
                        add_public_event(db, rid, "effect", trainer_name, {"icon": peff, "to": [row, col]})

                        if not brush_on:
                            st.session_state["placing_effect"] = None
                        request_rerun("pvp", force=True)

                    elif ppid:
                        new_id = str(uuid.uuid4())[:8]
                        # Stats já estão no banco, não precisa passar aqui
                        am_i_shiny = ppid in user_data.get("shinies", [])
                        new_piece = {
                            "id": new_id, 
                            "pid": ppid, 
                            "owner": trainer_name, 
                            "row": row, 
                            "col": col, 
                            "revealed": True, 
                            "status": "active",
                            "shiny": am_i_shiny # <--- SALVA NA PEÇA
                        }
                        upsert_piece(db, rid, new_piece)
                        mark_pid_seen(db, rid, ppid)
                        add_public_event(db, rid, "piece_placed", trainer_name, {"pid": ppid, "to": [row, col]})
                        st.session_state.pop("placing_pid", None)
                        request_rerun("pvp", force=True)
                    elif placing_trainer:
                        s_now = get_state(db, rid)
                        all_p = s_now.get("pieces") or []
                        avatar_choice = user_data.get("trainer_profile", {}).get("avatar_choice")
                        existing_trainer = next(
                            (p for p in all_p if p.get("owner") == trainer_name and p.get("kind") == "trainer"),
                            None,
                        )
                        if avatar_choice:
                            if existing_trainer:
                                existing_trainer["row"] = row
                                existing_trainer["col"] = col
                                existing_trainer["avatar"] = avatar_choice
                                existing_trainer["revealed"] = True
                                upsert_piece(db, rid, existing_trainer)
                            else:
                                new_id = str(uuid.uuid4())[:8]
                                new_piece = {
                                    "id": new_id,
                                    "kind": "trainer",
                                    "avatar": avatar_choice,
                                    "owner": trainer_name,
                                    "row": row,
                                    "col": col,
                                    "revealed": True,
                                    "status": "active",
                                }
                                upsert_piece(db, rid, new_piece)
                            add_public_event(db, rid, "trainer_placed", trainer_name, {"to": [row, col]})
                        st.session_state["placing_trainer"] = None
                        request_rerun("pvp", force=True)
                    elif moving_piece_id and is_player:
                        s_now = get_state(db, rid)
                        all_p = s_now.get("pieces") or []
                        mover = next((p for p in all_p if p["id"] == moving_piece_id), None)
                        if mover:
                            # 1. Guarda a posição antiga para o Log
                            old_pos = [mover["row"], mover["col"]]

                            # 2. Atualiza para a nova posição
                            mover["row"] = row
                            mover["col"] = col

                            # 3. Salva a peça no Firebase
                            upsert_piece(db, rid, mover)

                            # 4. Registra o movimento publicamente NO LOG
                            add_public_event(db, rid, "move", trainer_name, {
                                "pid": mover["pid"],
                                "from": old_pos,
                                "to": [row, col]
                            })



                            # 5. Limpa a seleção e recarrega
                            st.session_state["moving_piece_id"] = None
                            request_rerun("pvp", force=True)


        with tab_inic:
            st.markdown("### 🧭 Iniciativa")
            st.caption("Informe apenas o valor **final** da iniciativa dos Pokémon em campo (e do avatar, se estiver em campo).")

            s_now = get_state(db, rid)
            _pieces = s_now.get("pieces") or []
            init_store = (b_data.get("initiative") or {})

            rows = []
            for p in _pieces:
                p_kind = str(p.get("kind") or "piece")
                if p_kind not in {"trainer", "piece"}:
                    continue
                pid_label = str(p.get("pid") or "")
                if p_kind == "trainer":
                    label = f"🧍 Treinador • {p.get('owner') or '—'}"
                else:
                    label = f"🐾 {get_poke_display_name(pid_label)} • {p.get('owner') or '—'}"

                key = f"{p_kind}:{p.get('id')}"
                saved = init_store.get(key) or {}
                rows.append({
                    "key": key,
                    "kind": "Avatar" if p_kind == "trainer" else "Pokémon",
                    "item": label,
                    "initiative": int(saved.get("initiative") or 0),
                })

            if rows:
                import pandas as _pd
                df_init = _pd.DataFrame(rows)
                st.markdown("#### ✍️ Entrada rápida")
                edited = st.data_editor(
                    df_init,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "key": st.column_config.TextColumn("ID", disabled=True),
                        "kind": st.column_config.TextColumn("Tipo", disabled=True),
                        "item": st.column_config.TextColumn("Item", disabled=True),
                        "initiative": st.column_config.NumberColumn("Iniciativa", min_value=0, max_value=999, step=1),
                    },
                )

                out = {}
                for rec in edited.to_dict("records"):
                    k = str(rec.get("key") or "").strip()
                    if not k:
                        continue
                    out[k] = {
                        "initiative": int(rec.get("initiative") or 0),
                        "note": "",
                    }

                st.markdown("#### 🏁 Ordem automática")
                ordered_df = (
                    edited.sort_values(by=["initiative", "item"], ascending=[False, True])
                    .reset_index(drop=True)
                )
                ordered_df.insert(0, "ordem", ordered_df.index + 1)
                st.dataframe(
                    ordered_df[["ordem", "kind", "item", "initiative"]],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "ordem": st.column_config.NumberColumn("Ordem", format="%d"),
                        "kind": st.column_config.TextColumn("Tipo"),
                        "item": st.column_config.TextColumn("Em campo"),
                        "initiative": st.column_config.NumberColumn("Iniciativa", format="%d"),
                    },
                )

                if st.button("💾 Salvar iniciativa", use_container_width=True):
                    battle_ref.update({"initiative": out})
                    st.success("Iniciativa salva ✅")

            else:
                st.info("Sem Pokémon ou avatares em campo para registrar iniciativa.")

        with tab_log:
            st.markdown("### 📜 Log")
            st.caption("Eventos públicos em tempo real (movimentos, efeitos, rolagens, etc.).")
            render_public_log_fragment(db, rid, title=None, height=320, show_divider=False, limit=60)

            st.markdown("### ⚔️ Logs do combate")
            logs = list(reversed(b_data.get("logs") or []))[:60]
            if not logs:
                st.caption("Sem logs de combate ainda.")
            else:
                for line in logs:
                    st.markdown(f"- {line}")

        st.stop()
    elif view == "lobby":
            stop_pvp_sync_listener()
            # --- MAPA DE NOMES (Para exibição amigável) ---
            THEME_NAMES = {
                "cave_water": "Caverna (com água)",
                "forest": "Floresta (padrão)",
                "mountain_slopes": "Montanha (padrão)",
                "plains": "Pradaria",
                "dirt": "Terra Batida",
                "river": "Rio",
                "sea_coast": "Costa Marítima",
                "center_lake": "Lago Central",

                # --- BIOMAS (novos) ---
                "biome_grass": "Campos / Rotas gramadas",
                "biome_forest": "Floresta (densidade)",
                "biome_meadow": "Meadow / Campo florido",
                "biome_desert": "Deserto / Árido",
                "biome_mountain": "Montanha / Rochoso (bioma)",
                "biome_snow": "Neve / Gelo",
                "biome_water": "Água (rio/lago/mar) (bioma)",
                "biome_cave": "Caverna / Dungeon (bioma)",
                "biome_mix": "Mix (rotas variadas) (bioma)",
            }

            # ==========================================
            # 1. PAINEL SUPERIOR (CRIAR / LISTAR / ENTRAR)
            # ==========================================
        
            # --- Criar nova arena ---
            st.subheader("➕ Criar nova arena")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                grid = st.selectbox("Tamanho do grid", [6, 8, 10, 12], index=0)
            with c2:
                    # --- Seletor reduzido: agrupa biomas tematicamente iguais ---
                    SELECTOR_THEMES = {
                        "Floresta": "biome_forest",
                        "Campos / Rotas": "biome_grass",
                        "Deserto / Costa": "biome_desert",
                        "Montanha / Neve": "biome_mountain",
                        "Caverna": "biome_cave",
                        "Água (rio / lago / mar)": "biome_water",
                    }
                    theme_label = st.selectbox("Tema / Bioma", list(SELECTOR_THEMES.keys()), index=0)
                    theme = SELECTOR_THEMES[theme_label]

                    # (Opcional) detalhar o tipo de água sem poluir a lista principal
                    if theme == "biome_water":
                        with st.expander("Opções avançadas (água)", expanded=False):
                            water_kind = st.selectbox("Tipo de água", ["Rio", "Lago", "Costa/Mar"], index=0)
                            theme = {"Rio": "river", "Lago": "center_lake", "Costa/Mar": "sea_coast"}.get(water_kind, theme)
            with c3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🆕 Criar arena", type="primary"):
                    rid, err = create_room(db, trainer_name, grid, theme, max_active=5)
                    if err:
                        st.error(err)
                    else:
                        st.success(f"Arena criada! Código: **{rid}**")
                        st.session_state["active_room_id"] = rid
                        request_rerun("pvp", force=True)
    
            st.markdown("---")
    
            # --- Minhas arenas ---
            st.subheader("📌 Minhas arenas")
            my_rooms = list_my_rooms(db, trainer_name)
        
            if not my_rooms:
                st.info("Você ainda não tem arenas ativas. Crie uma acima.")
            else:
                map_choice = {} 
                for rid in my_rooms[:20]:
                    info = get_room(db, rid)
                    if info:
                        gs = info.get("gridSize", "?")
                        th_key = info.get("theme", "cave_water")
                        th_nice = THEME_NAMES.get(th_key, th_key)
                        chal = (info.get("challenger") or {})
                        chal_name = chal.get("name") if isinstance(chal, dict) else (chal or "Ninguém")
                    
                        label = f"{th_nice} vs {chal_name} ({gs}x{gs}) [ID: {rid}]"
                        map_choice[label] = rid
    
                if map_choice:
                    chosen_label = st.selectbox("Selecionar Arena", list(map_choice.keys()))
                    chosen_rid = map_choice[chosen_label]
    
                    b1, b2 = st.columns([1, 4])
                    with b1:
                        if st.button("📂 Abrir Selecionada"):
                            st.session_state["active_room_id"] = chosen_rid
                            request_rerun("pvp", force=True)
                    with b2:
                        if st.button("🗄️ Arquivar"):
                            remove_room_from_user(db, trainer_name, chosen_rid)
                            if st.session_state.get("active_room_id") == chosen_rid:
                                st.session_state.pop("active_room_id", None)
                            request_rerun("pvp", force=True)
    
            st.markdown("---")
    
            # --- Entrar por código ---
            st.subheader("🔑 Entrar por código")
            cc1, cc2, cc3 = st.columns([2, 1, 1])
            with cc1:
                code = st.text_input("Código da arena (roomId)", value="")
            with cc2:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🥊 Desafiante"):
                    if code.strip():
                        res = join_room_as_challenger(db, code.strip(), trainer_name)
                        if res == "OK":
                            st.session_state["active_room_id"] = code.strip()
                            request_rerun("pvp", force=True)
                        elif res == "ALREADY_OWNER":
                            st.warning("Você é o dono desta sala.")
                            st.session_state["active_room_id"] = code.strip()
                            request_rerun("pvp", force=True)
                        else:
                            st.error(res)
                            pass
            with cc3:
                # Adiciona um espaço para alinhar com o input que tem label
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("👀 Espectador"):
                    if code.strip():
                        res = join_room_as_spectator(db, code.strip(), trainer_name)
                        if res in ["OK", "PLAYER"]:
                            st.session_state["active_room_id"] = code.strip()
                            request_rerun("pvp", force=True)
                        else:
                            st.error(res)
                            pass

            st.markdown("---")

            # ==========================================
            # 2. PAINEL INFERIOR (PRÉ-VISUALIZAÇÃO DA ARENA ATIVA)
            # ==========================================
            rid = st.session_state.get("active_room_id")
            st.subheader("🎮 Arena Ativa (Pré-visualização)")
        
            # Botões de Navegação Básica
            c_nav1, c_nav2 = st.columns([1, 5])
            with c_nav1:
                if st.button("🔄 Atualizar"): request_rerun("pvp", force=True)
            with c_nav2:
                 if st.button("❌ Fechar Prévia"): 
                    st.session_state["active_room_id"] = None
                    request_rerun("pvp", force=True)

            if not rid:
                st.info("Nenhuma arena selecionada. Abra uma na lista acima.")
            else:
                room = get_room(db, rid)
                if not room:
                    st.error("Arena não encontrada.")
                else:
                    owner = (room.get("owner") or {}).get("name")
                    chal = room.get("challenger") or {}
                    chal_name = chal.get("name") if isinstance(chal, dict) else (chal or "Aguardando...")
                
                    st.info(f"📍 **Arena {rid}** | {room.get('theme')} | {owner} vs {chal_name}")

                    state_ref = db.collection("rooms").document(rid).collection("public_state").document("state")
                    state_doc = state_ref.get()
                    state = state_doc.to_dict() if state_doc.exists else {}

                    grid = int(room.get("gridSize") or 6)
                    theme_key = room.get("theme") or "cave_water"
                    seed = state.get("seed")
                    no_water_state = bool(state.get("noWater", False))
                    all_pieces = state.get("pieces") or []
                    pieces = visible_pieces_for(room, trainer_name, all_pieces)
                
                    role = get_role(room, trainer_name)
                    is_player = role in ["owner", "challenger"]
                
                    no_water = st.checkbox("🚫 Gerar sem água", value=no_water_state, disabled=not is_player)
                
                    if seed is None:
                        if st.button("🗺️ Gerar mapa (pixel art)", disabled=not is_player):
                            seed = generate_biome_seed()
                            state_ref.set({
                                "gridSize": grid, "theme": theme_key, "seed": seed, 
                                "tilesPacked": None, "noWater": bool(no_water),
                                "updatedAt": firestore.SERVER_TIMESTAMP,
                            }, merge=True)
                            st.session_state["pvp_view"] = "battle"
                            request_rerun("pvp", force=True)
                    else:
                        show_grid = st.checkbox("Mostrar grade tática", value=False, key=f"show_grid_preview_{rid}")
                        img = render_map_with_pieces(grid, theme_key, seed, no_water_state, pieces, trainer_name, room, show_grid=show_grid)
                        st.image(img, caption="Prévia do Campo")
                    
                        if st.button("⚔️ IR PARA O CAMPO DE BATALHA", type="primary"):
                            st.session_state["pvp_view"] = "battle"
                            request_rerun("pvp", force=True)
                    
                        if st.button("🔁 Regerar Mapa", disabled=not is_player):
                             seed = generate_biome_seed()
                             state_ref.set({"seed": seed, "tilesPacked": None, "noWater": bool(no_water)}, merge=True)
                             request_rerun("pvp", force=True)

