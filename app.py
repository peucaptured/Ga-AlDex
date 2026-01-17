import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import requests
import unicodedata
import os
import re

# Configuração da Página
st.set_page_config(
    page_title="Pokedex RPG Cloud",
    page_icon="🔒",
    layout="wide"
)

# --- CONEXÃO COM GOOGLE SHEETS ---
def get_google_sheet():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("SaveData_RPG").sheet1
        return sheet
    except Exception as e:
        st.error(f"Erro de Conexão: {e}")
        st.stop()

# --- SISTEMA DE LOGIN SEGURO (CORRIGIDO) ---

def find_user_row(sheet, name):
    """
    Procura o usuário APENAS na Coluna 1 (Coluna A).
    Isso evita bugs de achar o nome em outros lugares.
    Retorna o número da linha ou None.
    """
    try:
        # Pega todos os valores da primeira coluna
        all_names = sheet.col_values(1)
        # Tenta achar o índice na lista (Python começa em 0, Sheets em 1)
        # O index lança erro se não achar, então usamos try/except
        row_index = all_names.index(name) + 1 
        return row_index
    except ValueError:
        return None

def authenticate_user(name, password):
    try:
        sheet = get_google_sheet()
        row_num = find_user_row(sheet, name)
        
        if row_num is None:
            return "NOT_FOUND"
        
        # Pega a linha exata
        row_values = sheet.row_values(row_num)
        
        # Validação de segurança se a linha estiver quebrada
        if len(row_values) < 3:
            return "WRONG_PASS"
            
        stored_password = str(row_values[2]) # Coluna C
        stored_data = row_values[1]          # Coluna B
        
        if stored_password == str(password):
            return json.loads(stored_data)
        else:
            return "WRONG_PASS"
            
    except Exception as e:
        st.error(f"Erro na autenticação: {e}")
        return None

def register_new_user(name, password):
    try:
        sheet = get_google_sheet()
        
        # Verifica APENAS na coluna 1
        if find_user_row(sheet, name) is not None:
            return "EXISTS"
            
        empty_data = json.dumps({"seen": [], "caught": [], "party": [], "notes": {}})
        # Adiciona: Coluna A (Nome), Coluna B (Dados), Coluna C (Senha)
        sheet.append_row([name, empty_data, str(password)])
        return "SUCCESS"
    except Exception as e:
        st.error(f"Erro ao criar usuário: {e}")
        return "ERROR"

def save_data_cloud(trainer_name, data):
    try:
        sheet = get_google_sheet()
        json_str = json.dumps(data)
        
        row_num = find_user_row(sheet, trainer_name)
        
        if row_num:
            # Atualiza apenas a coluna 2 (Dados) dessa linha
            sheet.update_cell(row_num, 2, json_str)
            return True
        else:
            st.error("Erro crítico: Usuário sumiu da planilha enquanto salvava.")
            return False
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
        return False

# --- TELA DE LOGIN ---
if 'trainer_name' not in st.session_state:
    st.title("🔒 Acesso Seguro à Pokédex RPG")
    
    tab_login, tab_register = st.tabs(["🔑 Entrar", "📝 Criar Conta"])
    
    # ABA DE LOGIN
    with tab_login:
        l_user = st.text_input("Nome do Treinador", key="l_user")
        l_pass = st.text_input("Senha", type="password", key="l_pass")
        
        if st.button("Entrar", type="primary"):
            if l_user and l_pass:
                with st.spinner("Verificando credenciais..."):
                    result = authenticate_user(l_user, l_pass)
                    
                    if result == "WRONG_PASS":
                        st.error("🚫 Senha incorreta!")
                    elif result == "NOT_FOUND":
                        st.warning("Usuário não encontrado. Crie uma conta na aba ao lado.")
                    elif isinstance(result, dict):
                        st.session_state['trainer_name'] = l_user
                        st.session_state['user_data'] = result
                        st.rerun()
    
    # ABA DE REGISTRO
    with tab_register:
        st.info("Crie um novo usuário. Se apagou o antigo no Excel, pode recriar aqui.")
        r_user = st.text_input("Escolha seu Nome", key="r_user")
        r_pass = st.text_input("Escolha sua Senha", type="password", key="r_pass")
        
        if st.button("Criar Conta"):
            if r_user and r_pass:
                with st.spinner("Registrando..."):
                    res = register_new_user(r_user, r_pass)
                    if res == "SUCCESS":
                        st.success("Conta criada! Vá na aba 'Entrar' para fazer login.")
                    elif res == "EXISTS":
                        st.error("Esse nome de treinador já existe na Coluna A da planilha.")
                    else:
                        st.error("Erro ao criar conta.")
            else:
                st.warning("Preencha nome e senha.")

    st.stop() 

# --- APP PRINCIPAL ---

user_data = st.session_state['user_data']
trainer_name = st.session_state['trainer_name']

# --- FUNÇÕES DO APP ---

def normalize_text(text):
    if not isinstance(text, str): return str(text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def get_image_from_name(user_name, name_map):
    if not isinstance(user_name, str): return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
    pre_clean = user_name.replace('♀', '-f').replace('♂', '-m')
    clean = normalize_text(pre_clean).replace('.', '').replace("'", '').replace(' ', '-')
    
    if clean == 'mimikyu': clean = 'mimikyu-disguised'
    if clean == 'aegislash': clean = 'aegislash-blade'
    if clean == 'giratina': clean = 'giratina-origin'
    if clean == 'wishiwashi': clean = 'wishiwashi-solo'
    if clean == 'pumpkaboo': clean = 'pumpkaboo-average'
    if clean == 'gourgeist': clean = 'gourgeist-average'
    if clean == 'lycanroc': clean = 'lycanroc-midday'
    if clean == 'deoxys': clean = 'deoxys-normal'
    if clean == 'wormadam': clean = 'wormadam-plant'
    if clean == 'shaymin': clean = 'shaymin-land'

    if clean.endswith('-a'): clean = clean[:-2] + '-alola'
    if clean.endswith('-g'): clean = clean[:-2] + '-galar'
    if clean.endswith('-h'): clean = clean[:-2] + '-hisui'
    if clean.startswith('g-'): clean = clean[2:] + '-galar'
    if clean.startswith('a-'): clean = clean[2:] + '-alola'
    if clean.startswith('h-'): clean = clean[2:] + '-hisui'

    p_id = name_map.get(clean)
    if not p_id:
        base_name = clean.split('-')[0]
        p_id = name_map.get(base_name)

    if p_id:
        return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p_id}.png"
    else:
        return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"

@st.cache_data
def get_official_pokemon_map():
    try:
        url = "https://pokeapi.co/api/v2/pokemon?limit=10000"
        response = requests.get(url, timeout=5)
        data = response.json()
        name_map = {}
        for p in data['results']:
            p_id = p['url'].split('/')[-2]
            name_map[p['name']] = p_id
        return name_map
    except:
        return {}

def extract_strategies(text):
    if not isinstance(text, str): return []
    pattern = r'(?:^|\n)\s*(?:\*\*|[\-\>])?\s*([CFS][ODFIC][RL])\b'
    matches = re.findall(pattern, text)
    return matches

def calculate_power_level(row, cols_map):
    score = 0
    rarity = normalize_text(row.get('Raridade', ''))
    if 'trio' in rarity: score += 10
    elif 'fóssil' in rarity or 'fossil' in rarity: score += 7
    elif 'ultra' in rarity or 'super' in rarity: score += 5
    elif 'raro' in rarity: score += 3
    else: score += 1
    
    types = str(row.get('Tipo', '')).split('/')
    for t in types:
        t = normalize_text(t)
        if t in ['psychic', 'ghost', 'dragon', 'psiquico', 'fantasma', 'dragao']: score += 3
        elif t in ['steel', 'ice', 'dark', 'fighting', 'fairy', 'poison', 'metal', 'gelo', 'noturno', 'lutador', 'fada', 'veneno']: score += 2
        else: score += 1
        
    try:
        if cols_map.get('estagio'):
            val = pd.to_numeric(row.get(cols_map.get('estagio'), 0), errors='coerce')
            score += 0 if pd.isna(val) else val
    except: pass

    try:
        if cols_map.get('evolucao'):
            val = pd.to_numeric(row.get(cols_map.get('evolucao'), 0), errors='coerce')
            score += 0 if pd.isna(val) else val
    except: pass
    
    final_score = int(score)
    if final_score > 15: final_score = 15
    return final_score

def load_excel_data():
    file_name = "pokedex.xlsx"
    if not os.path.exists(file_name): return None, None
    try:
        df = pd.read_excel(file_name)
        df.columns = [c.strip() for c in df.columns]
        cols_map = {}
        for col in df.columns:
            norm_col = normalize_text(col)
            if 'estagio' in norm_col: cols_map['estagio'] = col
            if 'evolucao' in norm_col or 'evolution' in norm_col: cols_map['evolucao'] = col

        df['Região'] = df['Região'].fillna('Desconhecida').astype(str)
        df['Biomas'] = df['Biomas'].fillna('Desconhecido').astype(str)
        df['Nome'] = df['Nome'].fillna('Desconhecido')
        df['Viabilidade'] = df['Viabilidade'].fillna('Sem dados.')
        if 'Nº' in df.columns:
            df['Nº'] = df['Nº'].astype(str).str.replace('#', '')
            
        df['Codigos_Estrategia'] = df['Viabilidade'].apply(extract_strategies)
        df['Nivel_Poder'] = df.apply(lambda row: calculate_power_level(row, cols_map), axis=1)
        return df, cols_map
    except Exception as e:
        st.error(f"Erro ao ler Excel: {e}")
        return None, None

api_name_map = get_official_pokemon_map()

if 'df_data' not in st.session_state:
    st.session_state['df_data'], st.session_state['cols_map'] = load_excel_data()

df = st.session_state['df_data']
cols_map = st.session_state.get('cols_map', {})

# --- INTERFACE PRINCIPAL ---

st.sidebar.title("📱 Menu")
st.sidebar.markdown(f"**Treinador:** {trainer_name}")

if st.sidebar.button("💾 Salvar na Nuvem"):
    if save_data_cloud(trainer_name, user_data):
        st.sidebar.success("Salvo com sucesso!")

if st.sidebar.button("🚪 Sair (Logout)"):
    del st.session_state['trainer_name']
    st.rerun()

if st.sidebar.button("🔄 Recarregar Excel"):
    st.session_state['df_data'], st.session_state['cols_map'] = load_excel_data()
    st.rerun()

st.sidebar.markdown("---")
page = st.sidebar.radio("Ir para:", ["Pokédex (Busca)", "Trainer Hub (Meus Pokémons)"])

# ==============================================================================
# PÁGINA 1: POKEDEX
# ==============================================================================
if page == "Pokédex (Busca)":
    st.sidebar.header("🔍 Filtros")
    search_query = st.sidebar.text_input("Buscar (Nome ou Nº)", "")
    
    all_regions = sorted(list(set([r.strip() for region in df['Região'].unique() for r in region.split('/')])))
    selected_regions = st.sidebar.multiselect("Região", all_regions)
    
    all_biomes = sorted(list(set([b.strip() for biome in df['Biomas'].unique() for b in biome.split('/')])))
    biomes_clean = [b for b in all_biomes if "toda" not in b.lower()]
    selected_biomes = st.sidebar.multiselect("Bioma", biomes_clean)
    
    min_p, max_p = int(df['Nivel_Poder'].min()), int(df['Nivel_Poder'].max())
    power_range = st.sidebar.slider("⚡ Nível de Poder", min_p, max_p, (min_p, max_p))
    
    st.sidebar.subheader("⚔️ Estratégia")
    sel_func = st.sidebar.selectbox("Função", ["Todos", "C - Controlador", "F - Finalizador", "S - Suporte"])
    sel_style = st.sidebar.selectbox("Estilo", ["Todos", "O - Ofensivo", "D - Defensivo", "F - Furtivo", "I - Incompleto", "C - Completo"])
    sel_speed = st.sidebar.selectbox("Velocidade", ["Todos", "R - Rápido", "L - Lento"])
    
    l1 = sel_func[0] if sel_func != "Todos" else ""
    l2 = sel_style[0] if sel_style != "Todos" else ""
    l3 = sel_speed[0] if sel_speed != "Todos" else ""

    filtered_df = df.copy()
    
    if search_query:
        filtered_df = filtered_df[filtered_df['Nome'].str.contains(search_query, case=False, na=False) | filtered_df['Nº'].str.contains(search_query, case=False, na=False)]
    if selected_regions:
        filtered_df = filtered_df[filtered_df['Região'].apply(lambda x: any(region in x for region in selected_regions))]
    if selected_biomes:
        filtered_df = filtered_df[filtered_df['Biomas'].apply(lambda x: ("toda" in str(x).lower() and "ga" in str(x).lower()) or any(b in x for b in selected_biomes))]
    
    filtered_df = filtered_df[
        (filtered_df['Nivel_Poder'] >= power_range[0]) & 
        (filtered_df['Nivel_Poder'] <= power_range[1])
    ]
    
    if l1 or l2 or l3:
            filtered_df = filtered_df[filtered_df['Codigos_Estrategia'].apply(lambda codes: any(((not l1 or c[0]==l1) and (not l2 or c[1]==l2) and (not l3 or c[2]==l3)) for c in codes))]

    st.title("📕 Pokédex Universal")
    st.markdown(f"**Resultados:** {len(filtered_df)}")
    
    if filtered_df.empty: st.warning("Nenhum Pokémon encontrado.")

    for index, row in filtered_df.iterrows():
        dex_num = row['Nº']
        p_name = row['Nome']
        img_url = get_image_from_name(p_name, api_name_map)
        power = row['Nivel_Poder']
        
        with st.container():
            c_img, c_info, c_check = st.columns([0.5, 3, 1.5])
            with c_img: st.image(img_url, width=80)
            with c_info:
                st.markdown(f"### #{dex_num} {p_name}")
                tags_html = "".join([f"<span style='background-color:#444;color:white;padding:2px 5px;border-radius:4px;margin-right:5px;font-size:0.8em'>{c}</span>" for c in row['Codigos_Estrategia']])
                
                if power >= 13: p_color = "#D32F2F"
                elif power >= 8: p_color = "#F57C00"
                else: p_color = "#388E3C"

                power_badge = f"<span style='background-color:{p_color};color:white;padding:2px 8px;border-radius:10px;font-weight:bold;font-size:0.8em'>⚡ NP: {power}</span>"
                st.markdown(f"**{row['Tipo']}** | {power_badge} {tags_html}", unsafe_allow_html=True)
                
                with st.expander("📖 Detalhes"):
                    st.markdown(f"**📍 Região:** {row['Região']} | **🌿 Bioma:** {row['Biomas']}")
                    st.info(row['Descrição da Pokedex'])
                    
                    viab = str(row['Viabilidade']).replace("PARCEIROS:", "\n\n**👥 PARCEIROS:**").replace("Explicação:", "\n\n**💡 EXPLICAÇÃO:**").replace("Habilidade:", "**✨ Habilidade:**")
                    for code in row['Codigos_Estrategia']:
                        viab = re.sub(rf'\b{code}\b', f":red[**{code}**]", viab)
                    st.write(viab)

            with c_check:
                st.write("") 
                is_seen = dex_num in user_data["seen"]
                is_caught = dex_num in user_data["caught"]
                
                if st.checkbox("👁️ Visto", value=is_seen, key=f"seen_{dex_num}"):
                    if dex_num not in user_data["seen"]:
                        user_data["seen"].append(dex_num)
                        save_data_cloud(trainer_name, user_data)
                else:
                    if dex_num in user_data["seen"]:
                        user_data["seen"].remove(dex_num)
                        save_data_cloud(trainer_name, user_data)

                if st.checkbox("🔴 Capturado", value=is_caught, key=f"caught_{dex_num}"):
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

# ==============================================================================
# PÁGINA 2: TRAINER HUB
# ==============================================================================
elif page == "Trainer Hub (Meus Pokémons)":
    st.title("🏕️ Trainer Hub")
    tab1, tab2, tab3 = st.tabs(["🎒 Minha Party", "🔴 Capturados", "👁️ Pokedex (Vistos)"])
    
    with tab1:
        with st.expander("➕ Adicionar Pokémon à Equipe", expanded=True):
            col_add1, col_add2 = st.columns(2)
            with col_add1:
                st.subheader("Da Dex")
                options_all = df.apply(lambda x: f"#{x['Nº']} - {x['Nome']}", axis=1).tolist()
                current_pc_in_party = [m for m in user_data['party'] if not str(m).startswith("EXT:")]
                current_ext_in_party = [m for m in user_data['party'] if str(m).startswith("EXT:")]
                default_names = []
                for pid in current_pc_in_party:
                        res = df[df['Nº'] == pid]['Nome'].values
                        if len(res) > 0: default_names.append(f"#{pid} - {res[0]}")
                selected_names = st.multiselect("Selecione para Equipe", options=options_all, default=default_names)
                new_pc_ids = [n.split(" - ")[0].replace("#", "") for n in selected_names]
                full_new_party = new_pc_ids + current_ext_in_party
                if set(full_new_party) != set(user_data['party']) or len(full_new_party) != len(user_data['party']):
                        for pid in new_pc_ids:
                            if pid not in user_data['caught']: user_data['caught'].append(pid)
                            if pid not in user_data['seen']: user_data['seen'].append(pid)
                        user_data['party'] = full_new_party
                        save_data_cloud(trainer_name, user_data)
                        st.rerun()
            with col_add2:
                st.subheader("Visitante")
                external_name = st.text_input("Nome (ex: Sawsbuck)")
                if st.button("Adicionar"):
                    if external_name:
                        ext_id = f"EXT:{external_name}"
                        user_data['party'].append(ext_id)
                        if ext_id not in user_data['caught']: user_data['caught'].append(ext_id)
                        if ext_id not in user_data['seen']: user_data['seen'].append(ext_id)
                        save_data_cloud(trainer_name, user_data)
                        st.rerun()
        st.markdown("---")
        if user_data['party']:
            cols = st.columns(3)
            for i, member in enumerate(user_data['party']):
                is_ext = str(member).startswith("EXT:")
                if is_ext:
                    p_name = member.replace("EXT:", "")
                    p_img = get_image_from_name(p_name, api_name_map)
                    p_subtitle = "Visitante"
                else:
                    p_search = df[df['Nº'] == member]
                    if not p_search.empty:
                        r = p_search.iloc[0]
                        p_name = r['Nome']
                        p_img = get_image_from_name(p_name, api_name_map)
                        p_subtitle = f"⚡ NP: {r['Nivel_Poder']} | {r['Tipo']}"
                    else:
                        p_name, p_subtitle = f"ID: {member}", "?"
                        p_img = ""
                with cols[i % 3]:
                    with st.container(border=True):
                        c_p1, c_p2 = st.columns([3, 1])
                        with c_p1: st.markdown(f"**{p_name}**")
                        with c_p2:
                            if st.button("❌", key=f"rem_{i}"):
                                user_data['party'].pop(i)
                                save_data_cloud(trainer_name, user_data)
                                st.rerun()
                        st.image(p_img, width=120)
                        st.caption(p_subtitle)
                        nk = f"note_party_{i}_{member}"
                        curr = user_data["notes"].get(nk, "")
                        new = st.text_area("Notas", value=curr, height=80, key=nk)
                        if new != curr:
                            user_data["notes"][nk] = new
                            save_data_cloud(trainer_name, user_data)
        else: st.info("Sua equipe está vazia.")

    with tab2:
        st.markdown(f"### Total Capturados: {len(user_data['caught'])}")
        if not user_data['caught']: st.info("Sua caixa está vazia.")
        else:
            for p_id in user_data['caught']:
                is_in_party = p_id in user_data['party']
                status_text = "Na Equipe" if is_in_party else "No PC"
                if str(p_id).startswith("EXT:"):
                    p_name = p_id.replace("EXT:", "")
                    icon = "🌐"
                    with st.expander(f"{icon} {p_name} ({status_text})"):
                            st.image(get_image_from_name(p_name, api_name_map), width=100)
                            st.write("**Origem:** Visitante de fora de Ga'al.")
                else:
                    p_search = df[df['Nº'] == p_id]
                    if p_search.empty: continue
                    p_row = p_search.iloc[0]
                    icon = "🎒" if is_in_party else "🖥️"
                    header = f"{icon} #{p_id} - {p_row['Nome']} (NP: {p_row['Nivel_Poder']})"
                    with st.expander(header):
                        c1, c2 = st.columns([1, 4])
                        with c1: st.image(get_image_from_name(p_row['Nome'], api_name_map), width=100)
                        with c2:
                            st.write(f"**Status:** {status_text}")
                            st.write(f"**Estratégia:** {p_row['Viabilidade'][:150]}...")
                            nk = f"pc_note_{p_id}"
                            curr = user_data["notes"].get(p_id, "")
                            note = st.text_area("Notas", value=curr, key=nk)
                            if note != curr:
                                user_data["notes"][p_id] = note
                                save_data_cloud(trainer_name, user_data)

    with tab3:
        total = len(df)
        vistos = len(user_data['seen'])
        st.markdown(f"### Progresso da Pokédex")
        st.progress(min(vistos / total, 1.0))
        st.write(f"**{vistos}** de **{total}** Pokémons registrados.")
