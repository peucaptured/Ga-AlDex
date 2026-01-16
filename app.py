import streamlit as st
import pandas as pd
import os
import re
import requests
import json
import unicodedata
from io import StringIO

# Configuração da Página
st.set_page_config(
    page_title="Pokedex 2.0 - Trainer Companion",
    page_icon="⚡",
    layout="wide"
)

# --- ARQUIVO DE SAVE LOCAL (Temporário da Sessão) ---
SAVE_FILE = "save_data.json"

def load_user_data():
    if not os.path.exists(SAVE_FILE):
        default_data = {"seen": [], "caught": [], "party": [], "notes": {}}
        with open(SAVE_FILE, "w") as f:
            json.dump(default_data, f)
        return default_data
    with open(SAVE_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return {"seen": [], "caught": [], "party": [], "notes": {}}

def save_user_data(data):
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f)

# Carrega dados iniciais
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = load_user_data()

user_data = st.session_state['user_data']

# --- FUNÇÕES DE AJUDA ---

def normalize_text(text):
    """Remove acentos e deixa minúsculo"""
    if not isinstance(text, str): return str(text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def get_type_color(tipo):
    tipo = str(tipo).lower()
    colors = {
        'grass': '#78C850', 'fire': '#F08030', 'water': '#6890F0', 'bug': '#A8B820', 
        'normal': '#A8A878', 'poison': '#A040A0', 'electric': '#F8D030', 'ground': '#E0C068', 
        'fairy': '#EE99AC', 'fighting': '#C03028', 'psychic': '#F85888', 'rock': '#B8A038', 
        'ghost': '#705898', 'ice': '#98D8D8', 'dragon': '#7038F8', 'steel': '#B8B8D0', 
        'dark': '#705848', 'flying': '#A890F0'
    }
    for t, color in colors.items():
        if t in tipo: return color
    return '#A8A878'

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

def get_image_from_name(user_name, name_map):
    if not isinstance(user_name, str): return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
    
    # 1. TRATAMENTO DE GÊNERO (CRUCIAL: Fazer ANTES de normalizar)
    # Substitui os símbolos por texto antes que a normalização os apague
    pre_clean = user_name.replace('♀', '-f').replace('♂', '-m')
    
    # 2. Limpeza padrão
    clean = normalize_text(pre_clean).replace('.', '').replace("'", '').replace(' ', '-')
    
    # 3. Exceções Manuais (Nomes Oficiais da API)
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

    # 4. Sufixos Regionais
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
        col_estagio = cols_map.get('estagio')
        if col_estagio:
            val = pd.to_numeric(row.get(col_estagio, 0), errors='coerce')
            score += 0 if pd.isna(val) else val
    except: pass

    try:
        col_evolucao = cols_map.get('evolucao')
        if col_evolucao:
            val = pd.to_numeric(row.get(col_evolucao, 0), errors='coerce')
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

# --- INTERFACE ---
if df is None:
    st.error("Erro: Arquivo 'pokedex.xlsx' não encontrado.")
    if st.button("Tentar Ler Novamente"):
        st.session_state['df_data'], st.session_state['cols_map'] = load_excel_data()
        st.rerun()
else:
    st.sidebar.title("📱 Menu")
    
    if st.sidebar.button("🔄 Recarregar Excel"):
        st.session_state['df_data'], st.session_state['cols_map'] = load_excel_data()
        st.rerun()

    page = st.sidebar.radio("Ir para:", ["Pokédex (Busca)", "Trainer Hub (Meus Pokémons)"])
    st.sidebar.markdown("---")
    
    # --- SISTEMA DE SAVE/LOAD ---
    st.sidebar.subheader("💾 Gerenciar Save")
    
    # 1. Upload de Save
    uploaded_save = st.sidebar.file_uploader("Carregar Save (.json)", type=["json"])
    if uploaded_save is not None:
        try:
            # Lê o arquivo enviado
            loaded_json = json.load(uploaded_save)
            # Atualiza a sessão e o arquivo local
            st.session_state['user_data'] = loaded_json
            user_data = st.session_state['user_data'] # Atualiza variável local
            save_user_data(user_data)
            st.sidebar.success("Save carregado com sucesso!")
        except Exception as e:
            st.sidebar.error("Erro ao carregar save.")

    # 2. Download de Save
    st.sidebar.download_button(
        label="⬇️ Baixar Save Atual",
        data=json.dumps(st.session_state['user_data'], indent=4),
        file_name="meu_save_pokemon.json",
        mime="application/json",
    )
    st.sidebar.markdown("---")

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
                            save_user_data(user_data)
                    else:
                        if dex_num in user_data["seen"]:
                            user_data["seen"].remove(dex_num)
                            save_user_data(user_data)
                    if st.checkbox("🔴 Capturado", value=is_caught, key=f"caught_{dex_num}"):
                        if dex_num not in user_data["caught"]:
                            user_data["caught"].append(dex_num)
                            if dex_num not in user_data["seen"]: user_data["seen"].append(dex_num)
                            save_user_data(user_data)
                            st.rerun()
                    else:
                        if dex_num in user_data["caught"]:
                            user_data["caught"].remove(dex_num)
                            save_user_data(user_data)
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
                         save_user_data(user_data)
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
                            save_user_data(user_data)
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
                                    save_user_data(user_data)
                                    st.rerun()
                            st.image(p_img, width=120)
                            st.caption(p_subtitle)
                            nk = f"note_party_{i}_{member}"
                            curr = user_data["notes"].get(nk, "")
                            new = st.text_area("Notas", value=curr, height=80, key=nk)
                            if new != curr:
                                user_data["notes"][nk] = new
                                save_user_data(user_data)
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
                                    save_user_data(user_data)

        with tab3:
            total = len(df)
            vistos = len(user_data['seen'])
            st.markdown(f"### Progresso da Pokédex")
            st.progress(min(vistos / total, 1.0))
            st.write(f"**{vistos}** de **{total}** Pokémons registrados.")
