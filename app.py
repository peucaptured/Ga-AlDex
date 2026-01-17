import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import requests
import unicodedata
import os
import re
import plotly.graph_objects as go
import random
import base64
import math

def calculate_damage(damage_val, base_const, defense_roll, defense_stat):
    # Fórmula: (Dano + Constante) - (Dado + Resistência)
    # Dividido por 5 e arredondado pra cima
    
    attack_total = damage_val + base_const
    defense_total = defense_roll + defense_stat
    
    final_val = (attack_total - defense_total) / 5
    
    # Se o resultado for negativo (defendeu tudo), é 0
    if final_val < 0: final_val = 0
    
    return math.ceil(final_val)

# Configuração da Página
st.set_page_config(
    page_title="Pokedex RPG - Battle Engine V2",
    page_icon="⚔️",
    layout="wide"
)

# --- CONEXÃO COM GOOGLE SHEETS (Mesma de antes) ---
def get_google_sheet(sheet_name="SaveData_RPG", tab_index=0):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(sheet_name)
        return sheet.get_worksheet(tab_index)
    except Exception as e:
        st.error(f"Erro de Conexão: {e}")
        st.stop()

# --- NOVA LÓGICA DE MAPAS PROCEDURAIS ---

def generate_procedural_map(size, biome_type):
    """
    Gera terrenos complexos baseados no bioma escolhido.
    Retorna uma lista de obstáculos/terrenos.
    """
    obstacles = []
    center = size // 2
    
    # 1. ARENA DE TERRA BATIDA (Limpa, poucos obstáculos nas bordas)
    if biome_type == "Terra Batida":
        # Apenas alguns caixotes ou pedras nos cantos
        for _ in range(int(size * 0.5)):
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if x != center or y != center: # Deixa o centro livre
                obstacles.append({"x": x, "y": y, "icon": "🪨", "name": "Pedra", "type": "wall"})

    # 2. ARENA DE GRAMA (Muitos arbustos para se esconder)
    elif biome_type == "Arena de Grama":
        for x in range(size):
            for y in range(size):
                # 20% de chance de ter grama alta
                if random.random() < 0.2:
                    obstacles.append({"x": x, "y": y, "icon": "🌾", "name": "Grama Alta", "type": "cover"})

    # 3. LAGO NO CENTRO (Água no meio, terra em volta)
    elif biome_type == "Lago Central":
        radius = size / 3.5
        for x in range(size):
            for y in range(size):
                dist = math.sqrt((x - center)**2 + (y - center)**2)
                if dist < radius:
                    obstacles.append({"x": x, "y": y, "icon": "🌊", "name": "Água Profunda", "type": "water"})
                elif dist < radius + 1:
                    # Margem do lago
                    obstacles.append({"x": x, "y": y, "icon": "🔹", "name": "Água Rasa", "type": "water"})

    # 4. RIO (Corta o mapa de um lado ao outro)
    elif biome_type == "Rio":
        # Sorteia se o rio é Vertical ou Horizontal
        orientation = random.choice(['vert', 'horiz'])
        bridge_pos = random.randint(1, size-2) # Posição da ponte
        
        river_line = center + random.randint(-1, 1) # Onde o rio passa
        
        for i in range(size):
            # Cria a água
            if orientation == 'vert':
                obstacles.append({"x": river_line, "y": i, "icon": "🌊", "name": "Rio", "type": "water"})
                obstacles.append({"x": river_line+1, "y": i, "icon": "🌊", "name": "Rio", "type": "water"})
                # Ponte
                if i == bridge_pos:
                    obstacles.append({"x": river_line, "y": i, "icon": "🌉", "name": "Ponte", "type": "ground"})
                    obstacles.append({"x": river_line+1, "y": i, "icon": "🌉", "name": "Ponte", "type": "ground"})
            else:
                obstacles.append({"x": i, "y": river_line, "icon": "🌊", "name": "Rio", "type": "water"})
                obstacles.append({"x": i, "y": river_line+1, "icon": "🌊", "name": "Rio", "type": "water"})
                # Ponte
                if i == bridge_pos:
                    obstacles.append({"x": i, "y": river_line, "icon": "🌉", "name": "Ponte", "type": "ground"})
                    obstacles.append({"x": i, "y": river_line+1, "icon": "🌉", "name": "Ponte", "type": "ground"})

    # 5. MONTANHA (Elevação central)
    elif biome_type == "Montanha":
        for x in range(size):
            for y in range(size):
                dist = math.sqrt((x - center)**2 + (y - center)**2)
                if dist < 1.5:
                    obstacles.append({"x": x, "y": y, "icon": "🏔️", "name": "Pico (Bloqueio)", "type": "wall"})
                elif dist < 3.0:
                    obstacles.append({"x": x, "y": y, "icon": "⛰️", "name": "Declive", "type": "rough"})
                elif random.random() < 0.1:
                    obstacles.append({"x": x, "y": y, "icon": "🪨", "name": "Pedra Solta", "type": "cover"})

    # 6. PADRÃO (Caverna/Floresta Antiga)
    else:
        icon = "🌲" if biome_type == "Floresta" else "🪨"
        num_obstacles = int((size * size) * 0.15)
        for _ in range(num_obstacles):
            obstacles.append({
                "x": random.randint(0, size-1),
                "y": random.randint(0, size-1),
                "icon": icon, "name": "Obstáculo", "type": "wall"
            })
            
    return obstacles

def render_battle_grid(size, biome, units, obstacles):
    """Renderiza o Grid com cores temáticas"""
    
    # Paleta de Cores por Bioma
    colors = {
        "Floresta": "#4CAF50", "Caverna": "#3E2723", "Mar": "#0288D1",
        "Montanha": "#795548", "Pradaria": "#8BC34A", "Terra Batida": "#D7CCC8",
        "Lago Central": "#AED581", "Rio": "#C5E1A5", "Arena de Grama": "#33691E"
    }
    bg_color = colors.get(biome, "#F0F0F0")

    fig = go.Figure()

    # Layout Base
    fig.update_layout(
        title=f"🏟️ {biome}",
        xaxis=dict(range=[-0.5, size-0.5], showgrid=True, dtick=1, gridcolor='rgba(0,0,0,0.2)', showticklabels=False),
        yaxis=dict(range=[-0.5, size-0.5], showgrid=True, dtick=1, gridcolor='rgba(0,0,0,0.2)', showticklabels=False),
        plot_bgcolor=bg_color,
        width=600, height=600,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='closest'
    )

    # Camada 1: Terreno/Obstáculos
    if obstacles:
        obs_x, obs_y, obs_txt = [], [], []
        for o in obstacles:
            obs_x.append(o['x'])
            obs_y.append(o['y'])
            obs_txt.append(o['icon'])
            
        fig.add_trace(go.Scatter(
            x=obs_x, y=obs_y,
            mode='text',
            text=obs_txt,
            textfont=dict(size=30), # Ícones grandes
            hoverinfo='text',
            hovertext=[o['name'] for o in obstacles],
            name='Terreno'
        ))

    # Camada 2: Unidades (Treinadores e Pokemons)
    if units:
        # Separa Time A e Time B (Futuramente)
        # Por enquanto, pinta todos de vermelho, Avatar de Azul
        u_x, u_y, u_txt, u_color = [], [], [], []
        
        for u in units:
            u_x.append(u['x'])
            u_y.append(u['y'])
            # Se for Treinador, usa o ícone de Avatar, senão Bola
            is_trainer = u.get('is_trainer', False)
            marker = '👤' if is_trainer else '👾'
            color = 'blue' if is_trainer else 'red'
            
            u_txt.append(marker)
            u_color.append(color)

        fig.add_trace(go.Scatter(
            x=u_x, y=u_y,
            mode='markers+text',
            marker=dict(size=40, color=u_color, opacity=0.5, line=dict(width=2, color='white')),
            text=u_txt,
            textfont=dict(size=20),
            hoverinfo='text',
            hovertext=[u['name'] for u in units],
            name='Unidades'
        ))

    return fig

# --- CÓDIGO DE LOGIN E DADOS MANTIDO IGUAL ---
# (Copie aqui as funções find_user_row, authenticate_user, register_new_user, 
# save_data_cloud, get_image_from_name, etc. do código anterior)

# ... (Carregamento de Excel igual) ...

# --- INÍCIO DA INTERFACE (MANTIDA) ---

if 'trainer_name' not in st.session_state:
    # (Tela de Login igual)
    st.title("Login")
    # ...
    st.stop()

# ... (Menu Lateral e Avatar mantidos) ...

page = st.sidebar.radio("Navegação", ["Pokédex", "Trainer Hub", "⚔️ Arena de Batalha (PvP)"])

if page == "Pokédex":
    st.title("Pokédex")
    # (Código da Pokedex)

elif page == "Trainer Hub":
    st.title("Hub")

# ==============================================================================
# PÁGINA DE BATALHA COM CALCULADORA RPG E DADOS
# ==============================================================================
elif page == "⚔️ Arena de Batalha (PvP)":
    st.title("⚔️ Arena Tática V3 (RPG System)")
    
    if 'battle_state' not in st.session_state:
        st.session_state['battle_state'] = {
            "active": False, "grid_size": 8, "units": [], "obstacles": [],
            "combat_log": ["Batalha iniciada!"]
        }
    battle = st.session_state['battle_state']

    # --- LOBBY DE CRIAÇÃO (IGUAL AO ANTERIOR) ---
    if not battle['active']:
        st.subheader("🛠️ Configurar Mapa")
        c1, c2 = st.columns(2)
        biomes_list = ["Terra Batida", "Arena de Grama", "Lago Central", "Rio", "Montanha", "Floresta", "Caverna"]
        size = c1.select_slider("Tamanho", options=[6, 8, 10, 12, 14], value=8)
        biome = c2.selectbox("Estilo de Terreno", biomes_list)
        
        if st.button("🎲 Gerar Campo de Batalha", type="primary"):
            obs = generate_procedural_map(size, biome)
            st.session_state['battle_state']['active'] = True
            st.session_state['battle_state']['grid_size'] = size
            st.session_state['battle_state']['biome'] = biome
            st.session_state['battle_state']['obstacles'] = obs
            st.session_state['battle_state']['units'] = []
            st.rerun()
            
    else:
        # --- HUD DE BATALHA ---
        st.markdown(f"""
        <div style="background:#222; border-radius:10px; padding:10px; border:2px solid #444; color:white; display:flex; justify-content:space-between;">
            <div><h3 style="margin:0; color:#4facfe">{st.session_state['trainer_name']}</h3></div>
            <div style="font-weight:bold; color:red;">VS</div>
            <div><h3 style="margin:0; color:#ff6b6b">Oponente</h3></div>
        </div>
        """, unsafe_allow_html=True)

        # LOG DE COMBATE (Mostra rolagens para todos)
        with st.expander("📜 Log de Combate (Últimas Ações)", expanded=True):
            for msg in reversed(battle['combat_log'][-5:]):
                st.write(msg)

        c_map, c_panel = st.columns([1.5, 1])
        
        with c_map:
            fig = render_battle_grid(battle['grid_size'], battle['biome'], battle['units'], battle['obstacles'])
            st.plotly_chart(fig, use_container_width=True)
            
        with c_panel:
            # ABAS DE CONTROLE
            tab_calc, tab_move, tab_add, tab_gm = st.tabs(["🧮 Combate", "👣 Mover", "➕ Invocar", "⚙️ Mestre"])
            
            # --- ABA 1: CALCULADORA DE COMBATE ---
            with tab_calc:
                st.subheader("⚔️ Fase de Ataque")
                
                # Seleção de Quem Bate e Quem Apanha
                my_units = [u for u in battle['units']]
                if not my_units:
                    st.warning("Adicione unidades no campo primeiro!")
                else:
                    col_atk, col_def = st.columns(2)
                    attacker_name = col_atk.selectbox("Atacante", [u['name'] for u in my_units], key="sel_atk")
                    defender_name = col_def.selectbox("Alvo", [u['name'] for u in my_units], key="sel_def")
                    
                    # Busca os dados dos selecionados
                    attacker = next((u for u in battle['units'] if u['name'] == attacker_name), None)
                    defender = next((u for u in battle['units'] if u['name'] == defender_name), None)

                    st.divider()
                    
                    # MODO 1: ACERTO (Hit Chance)
                    st.markdown("🎯 **Teste de Acerto**")
                    type_atk = st.radio("Tipo de Ataque", ["Distância (vs Dodge)", "Perto (vs Parry)"], horizontal=True)
                    hit_mod = st.number_input("Bônus de Acerto do Golpe", value=0)
                    
                    if st.button("🎲 Rolar Ataque (D20)"):
                        d20 = random.randint(1, 20)
                        total_hit = d20 + hit_mod
                        
                        # Pega a defesa escondida do alvo
                        def_stat = defender['stats']['dodge'] if "Distância" in type_atk else defender['stats']['parry']
                        target_dc = def_stat + 10
                        
                        result_msg = "ACERTOU! ✅" if total_hit >= target_dc else "ERROU! ❌"
                        
                        # Log público (Sem revelar os números exatos do defensor)
                        log = f"⚔️ {attacker_name} atacou {defender_name}! (Rolou {d20} + {hit_mod} = {total_hit}). Resultado: **{result_msg}**"
                        battle['combat_log'].append(log)
                        st.rerun()

                    st.divider()

                    # MODO 2: DANO (Damage Calc)
                    st.markdown("💥 **Cálculo de Dano**")
                    dmg_val = st.number_input("Dano do Golpe", min_value=0, value=10)
                    res_type = st.selectbox("Resistência do Alvo", ["Toughness (+15)", "Will (+10)", "Fortitude (+10)"])
                    
                    # Aqui o sistema roda o dado do defensor automaticamente para ser justo e rápido
                    if st.button("🎲 Calcular Dano (Defensor Rola D20)"):
                        def_d20 = random.randint(1, 20)
                        
                        if "Toughness" in res_type:
                            const = 15
                            stat_val = defender['stats']['thg']
                            res_name = "Toughness"
                        elif "Will" in res_type:
                            const = 10
                            stat_val = defender['stats']['will']
                            res_name = "Will"
                        else:
                            const = 10
                            stat_val = defender['stats']['fort']
                            res_name = "Fortitude"
                            
                        # Fórmula aplicada
                        bars_lost = calculate_damage(dmg_val, const, def_d20, stat_val)
                        
                        # Atualiza o HP do alvo (opcional, visual)
                        defender['hp'] = max(0, defender['hp'] - bars_lost)
                        
                        log = f"🛡️ {defender_name} defendeu com {res_name} (Rolou {def_d20} + {stat_val}). {attacker_name} causou **{bars_lost} BARRAS DE DANO** 💔."
                        battle['combat_log'].append(log)
                        st.rerun()

            # --- ABA 3: INVOCAR (ATUALIZADA COM STATUS) ---
            with tab_add:
                st.write("📋 **Registrar Novo Combatente**")
                
                # Tipo de entrada
                add_type = st.radio("Tipo", ["Pokémon", "Treinador"], horizontal=True)
                
                c_name, c_hp = st.columns([2, 1])
                new_name = c_name.text_input("Nome/Apelido")
                
                st.caption("🛡️ Atributos Defensivos (Insira Manualmente)")
                c1, c2, c3, c4, c5 = st.columns(5)
                v_dodge = c1.number_input("Dodge", value=2)
                v_parry = c2.number_input("Parry", value=2)
                v_thg = c3.number_input("Thg", value=2)
                v_will = c4.number_input("Will", value=2)
                v_fort = c5.number_input("Fort", value=2)
                
                if st.button("📥 Colocar em Campo"):
                    if new_name:
                        new_unit = {
                            "name": new_name,
                            "x": 0 if add_type == "Treinador" else 1,
                            "y": 0,
                            "is_trainer": (add_type == "Treinador"),
                            "hp": 6, # Padrão 6 barras
                            "stats": {
                                "dodge": v_dodge, "parry": v_parry,
                                "thg": v_thg, "will": v_will, "fort": v_fort
                            }
                        }
                        battle['units'].append(new_unit)
                        st.success(f"{new_name} adicionado!")
                        st.rerun()

            # --- OUTRAS ABAS (Mover e Mestre - Manter código anterior) ---
            with tab_move:
                # (Copie o código da aba Mover da resposta anterior)
                my_units = battle['units']
                if my_units:
                    u_name = st.selectbox("Unidade", [u['name'] for u in my_units], key="mv_sel")
                    dx = st.number_input("X", 0, battle['grid_size']-1, key="mx")
                    dy = st.number_input("Y", 0, battle['grid_size']-1, key="my")
                    if st.button("Mover"):
                        for u in battle['units']:
                            if u['name'] == u_name: u['x'], u['y'] = dx, dy
                        st.rerun()

            with tab_gm:
                if st.button("🔥 Fogo"): battle['obstacles'].append({"x":4,"y":4,"icon":"🔥","name":"Fogo","type":"hazard"}); st.rerun()
                if st.button("🧹 Limpar Tudo"): st.session_state['battle_state']['active'] = False; st.rerun()
