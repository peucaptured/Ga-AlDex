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
# --- FUNÇÕES DE LOGIN/DADOS ---

def find_user_row(sheet, name):
    try:
        all_names = sheet.col_values(1)
        row_index = all_names.index(name) + 1 
        return row_index
    except ValueError:
        return None

def authenticate_user(name, password):
    try:
        sheet = get_google_sheet("SaveData_RPG", 0) # Aba 0 é Save
        row_num = find_user_row(sheet, name)
        if row_num is None: return "NOT_FOUND"
        row_values = sheet.row_values(row_num)
        if len(row_values) < 3: return "WRONG_PASS"
        if str(row_values[2]) == str(password): return json.loads(row_values[1])
        else: return "WRONG_PASS"
    except Exception as e:
        return None

def register_new_user(name, password):
    try:
        sheet = get_google_sheet("SaveData_RPG", 0)
        if find_user_row(sheet, name) is not None: return "EXISTS"
        empty_data = json.dumps({"seen": [], "caught": [], "party": [], "notes": {}, "avatar": ""})
        sheet.append_row([name, empty_data, str(password)])
        return "SUCCESS"
    except: return "ERROR"

def save_data_cloud(trainer_name, data):
    try:
        sheet = get_google_sheet("SaveData_RPG", 0)
        json_str = json.dumps(data)
        row_num = find_user_row(sheet, trainer_name)
        if row_num:
            sheet.update_cell(row_num, 2, json_str)
            return True
        return False
    except: return False

def normalize_text(text):
    if not isinstance(text, str): return str(text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def get_image_from_name(user_name, name_map):
    if not isinstance(user_name, str): return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
    pre_clean = user_name.replace('♀', '-f').replace('♂', '-m')
    clean = normalize_text(pre_clean).replace('.', '').replace("'", '').replace(' ', '-')
    
    # Exceções comuns
    exceptions = {
        'mimikyu': 'mimikyu-disguised', 'aegislash': 'aegislash-blade', 'giratina': 'giratina-origin',
        'wishiwashi': 'wishiwashi-solo', 'toxtricity': 'toxtricity-amped', 'eiscue': 'eiscue-ice',
        'indeedee': 'indeedee-male', 'morpeko': 'morpeko-full-belly', 'urshifu': 'urshifu-single-strike',
        'basculegion': 'basculegion-male', 'enamorus': 'enamorus-incarnate'
    }
    if clean in exceptions: clean = exceptions[clean]
    
    # Sufixos Regionais
    if clean.endswith('-a'): clean = clean[:-2] + '-alola'
    if clean.endswith('-g'): clean = clean[:-2] + '-galar'
    if clean.endswith('-h'): clean = clean[:-2] + '-hisui'
    if clean.endswith('-p'): clean = clean[:-2] + '-paldea'
    
    p_id = name_map.get(clean)
    if not p_id: 
        base_name = clean.split('-')[0]
        p_id = name_map.get(base_name)
    
    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p_id}.png" if p_id else "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"

@st.cache_data
def get_official_pokemon_map():
    try:
        url = "https://pokeapi.co/api/v2/pokemon?limit=10000"
        response = requests.get(url, timeout=5)
        data = response.json()
        return {p['name']: p['url'].split('/')[-2] for p in data['results']}
    except: return {}

def load_excel_data():
    file_name = "pokedex.xlsx"
    if not os.path.exists(file_name): return None, None
    try:
        df = pd.read_excel(file_name)
        df.columns = [c.strip() for c in df.columns]
        # Tratamento de erros básico
        df['Região'] = df['Região'].fillna('Desconhecida').astype(str)
        df['Biomas'] = df['Biomas'].fillna('Desconhecido').astype(str)
        df['Nome'] = df['Nome'].fillna('Desconhecido')
        df['Viabilidade'] = df['Viabilidade'].fillna('Sem dados.')
        if 'Nº' in df.columns: df['Nº'] = df['Nº'].astype(str).str.replace('#', '')
        df['Nivel_Poder'] = pd.to_numeric(df.get('Nivel_Poder', 1), errors='coerce').fillna(1)
        df['Codigos_Estrategia'] = df['Viabilidade'].apply(lambda x: re.findall(r'([CFS][ODFIC][RL])', str(x)))
        return df, {}
    except: return None, None

api_name_map = get_official_pokemon_map()
if 'df_data' not in st.session_state:
    st.session_state['df_data'], st.session_state['cols_map'] = load_excel_data()
df = st.session_state['df_data']

if 'trainer_name' not in st.session_state:
    st.title("🔒 Login Pokedex RPG")
    
    l_user = st.text_input("Usuário")
    l_pass = st.text_input("Senha", type="password")
    
    col_entrar, col_criar = st.columns(2)
    
    with col_entrar:
        if st.button("Entrar"):
            if not l_user or not l_pass:
                st.warning("Preencha tudo.")
            else:
                res = authenticate_user(l_user, l_pass)
                if res == "NOT_FOUND":
                    st.error("Usuário não encontrado.")
                elif res == "WRONG_PASS":
                    st.error("Senha incorreta.")
                elif isinstance(res, dict):
                    st.session_state['trainer_name'] = l_user
                    st.session_state['user_data'] = res
                    st.rerun()
    
    with col_criar:
        if st.button("Criar Conta"):
            if not l_user or not l_pass:
                st.warning("Digite nome e senha para criar.")
            else:
                res = register_new_user(l_user, l_pass)
                if res == "EXISTS":
                    st.warning("Usuário já existe.")
                elif res == "SUCCESS":
                    st.success("Conta criada! Clique em Entrar.")
                else:
                    st.error("Erro ao criar.")
    st.stop()

user_data = st.session_state['user_data']
trainer_name = st.session_state['trainer_name']

st.sidebar.title(f"👤 {trainer_name}")
if st.sidebar.button("💾 Salvar Tudo"):
    save_data_cloud(trainer_name, user_data)
    st.sidebar.success("Salvo!")

# --- MENU DE AVATAR ---
with st.sidebar.expander("📸 Meu Avatar"):
    uploaded_avatar = st.file_uploader("Enviar Foto", type=['png', 'jpg', 'jpeg'])
    if uploaded_avatar:
        bytes_data = uploaded_avatar.getvalue()
        base64_str = base64.b64encode(bytes_data).decode()
        user_data['avatar'] = f"data:image/png;base64,{base64_str}"
        save_data_cloud(trainer_name, user_data)
        st.success("Avatar atualizado!")
    
    if user_data.get('avatar'):
        st.image(user_data['avatar'], width=100, caption="Você em Campo")

st.sidebar.markdown("---")

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

