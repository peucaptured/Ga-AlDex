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

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
st.set_page_config(
    page_title="Pokedex RPG - Battle Engine V4",
    page_icon="⚔️",
    layout="wide"
)

# ==============================================================================
# 1. FUNÇÕES DE DADOS (BLINDADAS)
# ==============================================================================

def normalize_text(text):
    if not isinstance(text, str): return str(text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

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
        st.error(f"Erro de Conexão com Google Sheets: {e}")
        st.stop()

def find_user_row(sheet, name):
    try:
        all_names = sheet.col_values(1)
        row_index = all_names.index(name) + 1 
        return row_index
    except ValueError:
        return None

def authenticate_user(name, password):
    try:
        sheet = get_google_sheet("SaveData_RPG", 0)
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

def get_image_from_name(user_name, name_map):
    if not isinstance(user_name, str): return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
    pre_clean = user_name.replace('♀', '-f').replace('♂', '-m')
    clean = normalize_text(pre_clean).replace('.', '').replace("'", '').replace(' ', '-')
    
    # Exceções Manuais
    exceptions = {
        'mimikyu': 'mimikyu-disguised', 'aegislash': 'aegislash-blade', 'giratina': 'giratina-origin',
        'wishiwashi': 'wishiwashi-solo', 'pumpkaboo': 'pumpkaboo-average', 'gourgeist': 'gourgeist-average',
        'lycanroc': 'lycanroc-midday', 'deoxys': 'deoxys-normal', 'wormadam': 'wormadam-plant',
        'shaymin': 'shaymin-land', 'toxtricity': 'toxtricity-amped', 'eiscue': 'eiscue-ice',
        'indeedee': 'indeedee-male', 'morpeko': 'morpeko-full-belly', 'urshifu': 'urshifu-single-strike',
        'basculegion': 'basculegion-male', 'enamorus': 'enamorus-incarnate', 'keldeo': 'keldeo-ordinary',
        'meloetta': 'meloetta-aria'
    }
    if clean in exceptions: clean = exceptions[clean]
    
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

# --- FUNÇÃO DE CARREGAMENTO ULTRA-SEGURA ---
def load_excel_data():
    file_name = "pokedex.xlsx"
    if not os.path.exists(file_name):
        st.error(f"🚨 Arquivo {file_name} não encontrado no repositório.")
        return None, None
    try:
        # Lê o Excel e converte tudo pra string inicialmente para não quebrar
        df = pd.read_excel(file_name, dtype=str)
        
        # Limpa espaços nos nomes das colunas
        df.columns = [c.strip() for c in df.columns]
        
        # Garante colunas mínimas
        cols_needed = ['Região', 'Biomas', 'Nome', 'Nº', 'Tipo', 'Nivel_Poder', 'Viabilidade']
        for c in cols_needed:
            if c not in df.columns: df[c] = "Desconhecido"
        
        # Tratamento de Nulos
        df.fillna("Desconhecido", inplace=True)
        
        # Limpeza Específica
        df['Nº'] = df['Nº'].astype(str).str.replace('#', '').str.strip()
        
        # Converte Poder para Número (Força bruta)
        # Qualquer coisa que não for número vira 0
        df['Nivel_Poder'] = pd.to_numeric(df['Nivel_Poder'], errors='coerce').fillna(0).astype(int)
        
        # Processa Estratégia
        df['Codigos_Estrategia'] = df['Viabilidade'].apply(lambda x: re.findall(r'([CFS][ODFIC][RL])', str(x)))
        
        return df, {}
    except Exception as e:
        st.error(f"Erro fatal ao ler Excel: {e}")
        return None, None

def calculate_damage(damage_val, base_const, defense_roll, defense_stat):
    attack_total = damage_val + base_const
    defense_total = defense_roll + defense_stat
    final_val = (attack_total - defense_total) / 5
    if final_val < 0: final_val = 0
    return math.ceil(final_val)

# ==============================================================================
# 2. ENGINE DE BATALHA
# ==============================================================================
def generate_procedural_map(size, biome_type):
    obstacles = []
    center = size // 2
    
    if biome_type == "Terra Batida":
        for _ in range(int(size * 0.5)):
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if x != center or y != center:
                obstacles.append({"x": x, "y": y, "icon": "🪨", "name": "Pedra", "type": "wall"})

    elif biome_type == "Arena de Grama":
        for x in range(size):
            for y in range(size):
                if random.random() < 0.2:
                    obstacles.append({"x": x, "y": y, "icon": "🌾", "name": "Grama Alta", "type": "cover"})

    elif biome_type == "Lago Central":
        radius = size / 3.5
        for x in range(size):
            for y in range(size):
                dist = math.sqrt((x - center)**2 + (y - center)**2)
                if dist < radius:
                    obstacles.append({"x": x, "y": y, "icon": "🌊", "name": "Água Profunda", "type": "water"})
                elif dist < radius + 1:
                    obstacles.append({"x": x, "y": y, "icon": "🔹", "name": "Água Rasa", "type": "water"})

    elif biome_type == "Rio":
        orientation = random.choice(['vert', 'horiz'])
        bridge_pos = random.randint(1, size-2)
        river_line = center + random.randint(-1, 1)
        for i in range(size):
            if orientation == 'vert':
                obstacles.append({"x": river_line, "y": i, "icon": "🌊", "name": "Rio", "type": "water"})
                obstacles.append({"x": river_line+1, "y": i, "icon": "🌊", "name": "Rio", "type": "water"})
                if i == bridge_pos:
                    obstacles.append({"x": river_line, "y": i, "icon": "🌉", "name": "Ponte", "type": "ground"})
                    obstacles.append({"x": river_line+1, "y": i, "icon": "🌉", "name": "Ponte", "type": "ground"})
            else:
                obstacles.append({"x": i, "y": river_line, "icon": "🌊", "name": "Rio", "type": "water"})
                obstacles.append({"x": i, "y": river_line+1, "icon": "🌊", "name": "Rio", "type": "water"})
                if i == bridge_pos:
                    obstacles.append({"x": i, "y": river_line, "icon": "🌉", "name": "Ponte", "type": "ground"})
                    obstacles.append({"x": i, "y": river_line+1, "icon": "🌉", "name": "Ponte", "type": "ground"})

    elif biome_type == "Montanha":
        for x in range(size):
            for y in range(size):
                dist = math.sqrt((x - center)**2 + (y - center)**2)
                if dist < 1.5:
                    obstacles.append({"x": x, "y": y, "icon": "🏔️", "name": "Pico", "type": "wall"})
                elif dist < 3.0:
                    obstacles.append({"x": x, "y": y, "icon": "⛰️", "name": "Declive", "type": "rough"})
                elif random.random() < 0.1:
                    obstacles.append({"x": x, "y": y, "icon": "🪨", "name": "Pedra", "type": "cover"})

    else:
        icon = "🌲" if biome_type == "Floresta" else "🪨"
        num_obstacles = int((size * size) * 0.15)
        for _ in range(num_obstacles):
            obstacles.append({"x": random.randint(0, size-1), "y": random.randint(0, size-1), "icon": icon, "name": "Obstáculo", "type": "wall"})
            
    return obstacles

def render_battle_grid(size, biome, units, obstacles):
    colors = {
        "Floresta": "#4CAF50", "Caverna": "#3E2723", "Mar": "#0288D1",
        "Montanha": "#795548", "Pradaria": "#8BC34A", "Terra Batida": "#D7CCC8",
        "Lago Central": "#AED581", "Rio": "#C5E1A5", "Arena de Grama": "#33691E"
    }
    bg_color = colors.get(biome, "#F0F0F0")
    fig = go.Figure()
    fig.update_layout(
        title=f"🏟️ {biome}",
        xaxis=dict(range=[-0.5, size-0.5], showgrid=True, dtick=1, gridcolor='rgba(0,0,0,0.2)', showticklabels=False),
        yaxis=dict(range=[-0.5, size-0.5], showgrid=True, dtick=1, gridcolor='rgba(0,0,0,0.2)', showticklabels=False),
        plot_bgcolor=bg_color,
        width=600, height=600,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='closest'
    )
    if obstacles:
        obs_x = [o['x'] for o in obstacles]
        obs_y = [o['y'] for o in obstacles]
        obs_txt = [o['icon'] for o in obstacles]
        fig.add_trace(go.Scatter(x=obs_x, y=obs_y, mode='text', text=obs_txt, textfont=dict(size=30), hoverinfo='text', hovertext=[o['name'] for o in obstacles], name='Terreno'))
    if units:
        u_x = [u['x'] for u in units]
        u_y = [u['y'] for u in units]
        u_txt = ['👤' if u.get('is_trainer') else '👾' for u in units]
        u_color = ['blue' if u.get('is_trainer') else 'red' for u in units]
        fig.add_trace(go.Scatter(x=u_x, y=u_y, mode='markers+text', marker=dict(size=40, color=u_color, opacity=0.5, line=dict(width=2, color='white')), text=u_txt, textfont=dict(size=20), hoverinfo='text', hovertext=[u['name'] for u in units], name='Unidades'))
    return fig

# ==============================================================================
# 3. CARREGAMENTO E LOGIN
# ==============================================================================
api_name_map = get_official_pokemon_map()
if 'df_data' not in st.session_state:
    st.session_state['df_data'], st.session_state['cols_map'] = load_excel_data()
df = st.session_state['df_data']

if 'trainer_name' not in st.session_state:
    st.title("🔒 Login Pokedex RPG")
    l_user = st.text_input("Usuário")
    l_pass = st.text_input("Senha", type="password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Entrar"):
            res = authenticate_user(l_user, l_pass)
            if isinstance(res, dict):
                st.session_state['trainer_name'] = l_user
                st.session_state['user_data'] = res
                st.rerun()
            else: st.error("Login falhou.")
    with c2:
        if st.button("Criar Conta"):
            if register_new_user(l_user, l_pass) == "SUCCESS": st.success("Criado! Entre agora.")
            else: st.error("Erro ao criar.")
    st.stop()

# ==============================================================================
# 4. INTERFACE PRINCIPAL
# ==============================================================================
user_data = st.session_state['user_data']
trainer_name = st.session_state['trainer_name']

st.sidebar.title(f"👤 {trainer_name}")
if st.sidebar.button("💾 Salvar Tudo"):
    save_data_cloud(trainer_name, user_data)
    st.sidebar.success("Salvo!")

with st.sidebar.expander("📸 Meu Avatar"):
    uploaded_avatar = st.file_uploader("Enviar Foto", type=['png', 'jpg', 'jpeg'])
    if uploaded_avatar:
        bytes_data = uploaded_avatar.getvalue()
        base64_str = base64.b64encode(bytes_data).decode()
        user_data['avatar'] = f"data:image/png;base64,{base64_str}"
        save_data_cloud(trainer_name, user_data)
        st.success("Avatar salvo!")
    if user_data.get('avatar'): st.image(user_data['avatar'], width=100)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navegação", ["Pokédex", "Trainer Hub", "⚔️ Arena de Batalha (PvP)"])

# --- PÁGINA POKEDEX ---
if page == "Pokédex":
    st.sidebar.header("🔍 Filtros")
    search_query = st.sidebar.text_input("Buscar (Nome ou Nº)", "")
    
    if df is not None and not df.empty:
        # TRATAMENTO DE ERRO NA COLUNA REGIÃO (BLINDAGEM)
        all_regions = []
        if 'Região' in df.columns:
            # Força str() em tudo para evitar TypeError
            raw_regions = [str(r) for r in df['Região'].unique()]
            all_regions = sorted(list(set([r.strip() for region in raw_regions for r in region.split('/')])))
        
        selected_regions = st.sidebar.multiselect("Região", all_regions)
        
        if selected_regions:
            df_biomes = df[df['Região'].apply(lambda x: any(reg in str(x) for reg in selected_regions))]
            raw_biomes = df_biomes['Biomas'].unique()
        else:
            raw_biomes = df['Biomas'].unique()
            
        all_biomes = sorted(list(set([b.strip() for biome in raw_biomes for b in str(biome).split('/')])))
        selected_biomes = st.sidebar.multiselect("Bioma", [b for b in all_biomes if len(b)>1])

        all_types = sorted(list(set([t.strip() for t_str in df['Tipo'].unique() for t in str(t_str).split('/')])))
        selected_types = st.sidebar.multiselect("Tipo", all_types)
        
        # TRATAMENTO DE ERRO NO SLIDER (BLINDAGEM)
        try:
            min_p = int(df['Nivel_Poder'].min())
            max_p = int(df['Nivel_Poder'].max())
        except:
            min_p, max_p = 0, 100 # Fallback se der erro
            
        if min_p >= max_p: max_p = min_p + 10 # Evita erro de min=max
        
        power_range = st.sidebar.slider("⚡ Nível de Poder", min_p, max_p, (min_p, max_p))
        
        # Aplicação
        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df['Nome'].str.contains(search_query, case=False) | filtered_df['Nº'].str.contains(search_query, case=False)]
        if selected_regions:
            filtered_df = filtered_df[filtered_df['Região'].apply(lambda x: any(r in str(x) for r in selected_regions))]
        if selected_biomes:
            filtered_df = filtered_df[filtered_df['Biomas'].apply(lambda x: any(b in str(x) for b in selected_biomes))]
        if selected_types:
            filtered_df = filtered_df[filtered_df['Tipo'].apply(lambda x: all(t in str(x) for t in selected_types))]
        filtered_df = filtered_df[(filtered_df['Nivel_Poder'] >= power_range[0]) & (filtered_df['Nivel_Poder'] <= power_range[1])]
        
        st.markdown(f"**Resultados:** {len(filtered_df)}")
        for index, row in filtered_df.iterrows():
            dex_num = str(row['Nº'])
            p_name = str(row['Nome'])
            img_url = get_image_from_name(p_name, api_name_map)
            power = row['Nivel_Poder']
            key_s = f"s_{dex_num}_{index}"
            key_c = f"c_{dex_num}_{index}"
            
            with st.container():
                c1, c2, c3 = st.columns([0.5, 3, 1.5])
                with c1: st.image(img_url, width=80)
                with c2:
                    st.markdown(f"### #{dex_num} {p_name}")
                    if power >= 13: clr = "#D32F2F"
                    elif power >= 8: clr = "#F57C00"
                    else: clr = "#388E3C"
                    st.markdown(f"<span style='background:{clr};color:white;padding:2px 8px;border-radius:10px'>⚡ {power}</span> **{row['Tipo']}**", unsafe_allow_html=True)
                    with st.expander("Detalhes"):
                        st.markdown(f"**📍** {row['Região']} | **🌿** {row['Biomas']}")
                        st.info(row['Viabilidade'])
                with c3:
                    is_seen = dex_num in user_data["seen"]
                    is_caught = dex_num in user_data["caught"]
                    if st.checkbox("👁️ Visto", is_seen, key=key_s):
                        if dex_num not in user_data["seen"]: 
                            user_data["seen"].append(dex_num)
                            save_data_cloud(trainer_name, user_data)
                    else:
                        if dex_num in user_data["seen"]:
                            user_data["seen"].remove(dex_num)
                            save_data_cloud(trainer_name, user_data)
                    
                    if st.checkbox("🔴 Capturado", is_caught, key=key_c):
                        if dex_num not in user_data["caught"]:
                            user_data["caught"].append(dex_num)
                            if dex_num not in user_data["seen"]: user_data["seen"].append(dex_num)
                            save_data_cloud(trainer_name, user_data)
                            st.rerun()
                    else:
                        if dex_num in user_data["caught"]:
                            user_data["caught"].remove(dex_num)
                            save_data_cloud(trainer_name, user_data)
                st.divider()
    else:
        st.warning("Carregando banco de dados...")

# --- PÁGINA HUB ---
elif page == "Trainer Hub":
    st.title("🏕️ Hub do Treinador")
    st.subheader(f"🐺 Seu Time ({len(user_data.get('party', []))})")
    
    if "party" not in user_data: user_data["party"] = []
    current_party = user_data["party"]
    
    if current_party:
        cols_per_row = 6
        for i in range(0, len(current_party), cols_per_row):
            cols = st.columns(cols_per_row)
            batch = current_party[i:i+cols_per_row]
            for j, p_dex_num in enumerate(batch):
                with cols[j]:
                    # CORREÇÃO DO ERRO DE TYPE NO HUB (Força tudo para string)
                    pokemon_rows = df[df['Nº'] == str(p_dex_num)]
                    if not pokemon_rows.empty:
                        p_name = pokemon_rows.iloc[0]['Nome']
                        img = get_image_from_name(p_name, api_name_map)
                        st.image(img, use_container_width=True)
                        st.caption(f"**{p_name}**")
                    else:
                        st.image("https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg", width=50)
                        st.caption(f"#{p_dex_num}")
    else: st.info("Time vazio.")
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📥 Adicionar")
        caught = user_data.get("caught", [])
        avail = [p for p in caught if p not in current_party]
        if avail:
            opts = []
            opt_map = {}
            for n in avail:
                r = df[df['Nº'] == str(n)]
                nm = r.iloc[0]['Nome'] if not r.empty else "?"
                lb = f"#{n} - {nm}"
                opts.append(lb)
                opt_map[lb] = n
            sel = st.selectbox("PC", opts)
            if st.button("Adicionar"):
                user_data["party"].append(opt_map[sel])
                save_data_cloud(trainer_name, user_data)
                st.rerun()
        else: st.info("PC Vazio.")
        
    with c2:
        st.markdown("### 📤 Remover")
        if current_party:
            rm_opts = []
            rm_map = {}
            for n in current_party:
                r = df[df['Nº'] == str(n)]
                nm = r.iloc[0]['Nome'] if not r.empty else "?"
                lb = f"#{n} - {nm}"
                rm_opts.append(lb)
                rm_map[lb] = n
            sel_rm = st.selectbox("Remover", rm_opts)
            if st.button("Tirar do Time"):
                user_data["party"].remove(rm_map[sel_rm])
                save_data_cloud(trainer_name, user_data)
                st.rerun()
    
    st.divider()
    with st.expander("📝 Notas"):
        nts = user_data.get("notes", "")
        if isinstance(nts, dict): nts = ""
        new_nts = st.text_area("Anotações", nts, height=150)
        if st.button("Salvar Notas"):
            user_data["notes"] = new_nts
            save_data_cloud(trainer_name, user_data)
            st.success("Salvo!")

# --- PÁGINA BATALHA ---
elif page == "⚔️ Arena de Batalha (PvP)":
    st.title("⚔️ Arena Tática V4")
    if 'battle_state' not in st.session_state:
        st.session_state['battle_state'] = {"active": False, "grid_size": 8, "units": [], "obstacles": [], "combat_log": []}
    battle = st.session_state['battle_state']
    
    if not battle['active']:
        st.subheader("Configurar")
        c1, c2 = st.columns(2)
        sz = c1.select_slider("Tamanho", options=[6, 8, 10, 12, 14], value=8)
        bio = c2.selectbox("Terreno", ["Terra Batida", "Arena de Grama", "Lago Central", "Rio", "Montanha", "Floresta"])
        if st.button("Gerar", type="primary"):
            battle.update({"active": True, "grid_size": sz, "biome": bio, "obstacles": generate_procedural_map(sz, bio), "units": []})
            st.rerun()
    else:
        with st.expander("📜 Log", expanded=True):
            for m in reversed(battle['combat_log'][-5:]): st.write(m)
            
        c_map, c_painel = st.columns([1.5, 1])
        with c_map:
            fig = render_battle_grid(battle['grid_size'], battle['biome'], battle['units'], battle['obstacles'])
            st.plotly_chart(fig, use_container_width=True)
            
        with c_painel:
            t1, t2, t3, t4 = st.tabs(["⚔️", "👣", "➕", "⚙️"])
            with t1: # Combate
                units = battle['units']
                if units:
                    at = st.selectbox("Atacante", [u['name'] for u in units], key='at')
                    df_u = st.selectbox("Alvo", [u['name'] for u in units], key='df')
                    atk_unit = next((u for u in units if u['name']==at), None)
                    def_unit = next((u for u in units if u['name']==df_u), None)
                    
                    st.divider()
                    st.write("🎯 Acerto")
                    mod = st.number_input("Mod. Ataque", value=0)
                    tipo = st.radio("Tipo", ["Distancia", "Perto"], horizontal=True)
                    if st.button("Rolar D20 Ataque"):
                        d20 = random.randint(1, 20)
                        tot = d20 + mod
                        stat = def_unit['stats']['dodge'] if tipo == "Distancia" else def_unit['stats']['parry']
                        res = "ACERTOU!" if tot >= (stat+10) else "ERROU!"
                        battle['combat_log'].append(f"⚔️ {at} atacou {df_u} (D20: {d20}+{mod}={tot}). **{res}**")
                        st.rerun()
                        
                    st.divider()
                    st.write("💥 Dano")
                    dmg = st.number_input("Dano Base", value=10)
                    res_t = st.selectbox("Resistência", ["Toughness", "Will", "Fortitude"])
                    if st.button("Calcular Dano"):
                        d20_def = random.randint(1, 20)
                        if res_t == "Toughness": const, st_val = 15, def_unit['stats']['thg']
                        elif res_t == "Will": const, st_val = 10, def_unit['stats']['will']
                        else: const, st_val = 10, def_unit['stats']['fort']
                        
                        lost = calculate_damage(dmg, const, d20_def, st_val)
                        def_unit['hp'] = max(0, def_unit['hp'] - lost)
                        battle['combat_log'].append(f"🛡️ {df_u} defendeu ({res_t} {d20_def}+{st_val}). Tomou **{lost} dano**.")
                        st.rerun()
            
            with t2: # Mover
                if battle['units']:
                    u_mv = st.selectbox("Quem", [u['name'] for u in battle['units']], key='mv')
                    mx = st.number_input("X", 0, battle['grid_size']-1)
                    my = st.number_input("Y", 0, battle['grid_size']-1)
                    if st.button("Ir"):
                        for u in battle['units']:
                            if u['name'] == u_mv: u['x'], u['y'] = mx, my
                        st.rerun()
            
            with t3: # Invocar
                nm = st.text_input("Nome")
                tp = st.radio("Classe", ["Pokemon", "Treinador"], horizontal=True)
                c1, c2, c3, c4, c5 = st.columns(5)
                vd = c1.number_input("Dodge", 0)
                vp = c2.number_input("Parry", 0)
                vt = c3.number_input("Thg", 0)
                vw = c4.number_input("Will", 0)
                vf = c5.number_input("Fort", 0)
                if st.button("Adicionar"):
                    battle['units'].append({
                        "name": nm, "x":0, "y":0, "is_trainer": (tp=="Treinador"), "hp":6,
                        "stats": {"dodge":vd, "parry":vp, "thg":vt, "will":vw, "fort":vf}
                    })
                    st.rerun()
            
            with t4:
                if st.button("Limpar"):
                    battle['active'] = False
                    st.rerun()
