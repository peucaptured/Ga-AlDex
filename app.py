
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import requests
import unicodedata
import os
import re
import uuid
from datetime import datetime
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import random
import gzip
import base64
from io import BytesIO
from PIL import ImageFont
if "carousel_click" not in st.session_state:
    st.session_state["carousel_click"] = None




# ================================
# FIREBASE - TESTE DE CONEXÃO
# ================================
import firebase_admin
from firebase_admin import credentials, firestore, storage



def init_firebase():
    if not firebase_admin._apps:
        raw = st.secrets["firebase_service_account"]
        cred_dict = {k: raw[k] for k in raw.keys()}  # <-- dict puro

        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {
            "projectId": cred_dict["project_id"],
            "storageBucket": "batalhas-de-gaal.firebasestorage.app",
        })

    db = firestore.client()
    bucket = storage.bucket()
    return db, bucket




# Configuração da Página
st.set_page_config(
    page_title="Pokedex RPG Cloud",
    page_icon="🔒",
    layout="wide"
)
# ==========================================
# 🎨 ESTILO VISUAL GLOBAL (POKÉMON RETRÔ)
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    /* 1. Aplica a fonte retrô apenas a textos de conteúdo */
    .stApp, .stMarkdown p, .stButton button, .stTab p, h1, h2, h3, .stWidget label {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 13px !important;
        line-height: 1.6;
    }

    /* 🛡️ 2. PROTEÇÃO TOTAL CONTRA O BUG 'keyboard_arrow_right' */
    /* Remove a fonte pixelada de qualquer elemento que contenha ícones do Material Design */
    [data-testid="stExpander"] summary, 
    [data-testid="stExpander"] svg,
    [data-testid="stHeader"] svg,
    .stSelectbox svg,
    .stMultiSelect svg,
    div[data-baseweb="icon"],
    span[class*="icon"],
    i[class*="icon"] {
        font-family: sans-serif !important;
    }

    /* Esconde especificamente o texto que vaza dos ícones */
    .st-emotion-cache-1vt4yqh, .st-emotion-cache-p5msec {
        font-family: sans-serif !important;
        color: transparent !important; /* Torna o texto invisível se ele vazar */
    }

    /* 🏟️ 3. Estilo do Título da Arena */
    .arena-header {
        display: flex; 
        align-items: center; 
        gap: 15px; 
        margin-bottom: 25px;
        font-family: 'Press Start 2P', cursive;
    }
    .arena-title { font-size: 20px; color: white; }
    .arena-id { 
        font-size: 28px; 
        color: #FFCC00; 
        background: #1E1E1E; 
        padding: 8px 15px; 
        border: 2px solid #FFCC00;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


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

def pack_tiles(tiles: list[list[str]]) -> str:
    # list -> json -> gzip -> base64 (string)
    raw = json.dumps(tiles, separators=(",", ":")).encode("utf-8")
    gz = gzip.compress(raw, compresslevel=9)
    return base64.b64encode(gz).decode("ascii")

def unpack_tiles(packed: str) -> list[list[str]]:
    gz = base64.b64decode(packed.encode("ascii"))
    raw = gzip.decompress(gz).decode("utf-8")
    return json.loads(raw)


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
        
def coord_to_notation(row, col):
    # Converte coluna em letra (0=A, 1=B...) e linha em número (0=1, 1=2...)
    letter = chr(65 + int(col)) 
    number = int(row) + 1
    return f"{letter}{number}"

@st.fragment(run_every="3s") # Atualiza o log sozinho a cada 3 segundos
def render_public_log_fragment(db, rid):
    st.markdown("---")
    st.subheader("📜 Log de Batalha (Tempo Real)")
    
    with st.container(height=300):
        # Busca os eventos mais recentes do Firestore 
        events = list_public_events(db, rid, limit=25) 
        
        if not events:
            st.caption("Aguardando ações na arena...")
        else:
            for ev in events:
                et = ev.get("type", "?") #
                by = ev.get("by", "?") #
                pl = ev.get("payload", {}) # 
                
                # Traduz o ID do Pokemon para Nome 
                p_id = pl.get("pid")
                p_name = get_poke_display_name(p_id) if p_id else by

                if et == "move":
                    f_coord = coord_to_notation(*pl.get("from", [0,0]))
                    t_coord = coord_to_notation(*pl.get("to", [0,0]))
                    st.write(f"👣 **{p_name}** se moveu de **({f_coord})** para **({t_coord})**")
                
                elif et == "dice":
                    st.write(f"🎲 **{by}** rolou d{pl.get('sides')}: **{pl.get('result')}**") # 
                
                elif et == "hit_confirmed":
                    st.success(f"⚔️ **{p_name}** ({by}) ACERTOU o ataque!") # 
                
                elif et == "missed":
                    st.error(f"🛡️ **{p_name}** ({by}) ERROU o ataque!") # 
                
                elif et == "finished":
                    # Pega a mensagem de dano final [
                    log_msg = pl.get("logs", [""])[0] if isinstance(pl.get("logs"), list) else ""
                    st.info(f"🩸 {log_msg}")
                
                else:
                    st.write(f"🔹 **{by}** ({et}): {pl}") # 


FLOOR_PREFIXES = ("agua", "areia", "grama", "pedra", "terra", "slope")

@st.cache_resource
def load_map_assets():
    base_path = "Assets/Texturas"
    asset_names = [
        "agua_1", "agua_2", "agua_3", "areia_1", "areia_2", "areia_3",
        "brush_1", "brush_2", "estalagmite_1", "grama_1", "grama_2", "grama_3",
        "pedra_1", "pedra_2", "pedra_3", "pico_1", "rochas", "rochas_2",
        "slope_1", "slope_2", "slope_3", "slope_4", "terra_1", "terra_2", "terra_3",
        "tree_1", "tree_2", "tree_3", "wall_1"
    ]

    def pick_solid_color(img: Image.Image) -> tuple[int, int, int]:
        counts = {}
        for r, g, b, a in img.getdata():
            if a > 0:
                counts[(r, g, b)] = counts.get((r, g, b), 0) + 1
        if counts:
            return max(counts, key=counts.get)
        return (0, 0, 0)

    def crop_to_alpha(img: Image.Image) -> Image.Image:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bbox = img.getchannel("A").getbbox()
        if bbox:
            img = img.crop(bbox)
        return img

    def normalize_floor(img: Image.Image) -> Image.Image:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        alpha = img.getchannel("A")
        if alpha.getextrema()[0] < 255:
            solid = pick_solid_color(img)
            base = Image.new("RGBA", img.size, (*solid, 255))
            base.alpha_composite(img)
            img = base
        return img

    assets = {}
    for name in asset_names:
        path = f"{base_path}/{name}.png"
        if os.path.exists(path):
            img = Image.open(path).convert("RGBA")
            if name.startswith(FLOOR_PREFIXES):
                img = crop_to_alpha(img)
            if img.size != (TILE_SIZE, TILE_SIZE):
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.NEAREST)
            if name.startswith(FLOOR_PREFIXES):
                img = normalize_floor(img)
            assets[name] = img
    return assets

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
    st.title("Bem Vindo(a) à Ga'Al")
    
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
import random

def roll_die(db, rid: str, by: str, sides: int = 20):
    result = random.randint(1, int(sides))
    add_public_event(db, rid, "dice", by, {"sides": int(sides), "result": int(result)})
    return result
    
def get_role(room: dict, trainer_name: str) -> str:
    owner = (room.get("owner") or {}).get("name")
    
    # CORREÇÃO: Pega a lista de desafiantes (plural)
    challengers = room.get("challengers") or []
    # Cria uma lista apenas com os nomes dos desafiantes
    challenger_names = [c.get("name") for c in challengers]

    if trainer_name == owner:
        return "owner"
    if trainer_name in challenger_names:
        return "challenger"
    return "spectator"

def safe_doc_id(name: str) -> str:
    # Evita caracteres problemáticos no Firestore doc id
    if not isinstance(name, str):
        name = str(name)
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name).strip("_")[:80] or "user"

def get_item_image_url(item_name):
    if not item_name:
        return "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/question-mark.png"
    
    # Normalização para Pokébolas e TMs
    name = normalize_text(item_name).replace(" ", "-")
    if "ball" in name and "-" not in name:
        name = name.replace("ball", "-ball")
    if "tm" in name and "-" not in name:
        name = name.replace("tm", "tm-")

    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/{name}.png"

def render_item_row(category_key, index, item_data, show_image=True):
    # Layout das colunas
    cols = st.columns([2.5, 1, 1.5]) if show_image else st.columns([3, 1])
    
    with cols[0]:
        new_name = st.text_input("Item", value=item_data.get("name", ""), 
                                key=f"it_nm_{category_key}_{index}", label_visibility="collapsed")
    
    with cols[-1]:
        new_qty = st.number_input("Qtd", min_value=0, value=item_data.get("qty", 0), 
                                 key=f"it_qt_{category_key}_{index}", label_visibility="collapsed")

    if show_image:
        with cols[1]:
            if new_name:
                img_url = get_item_image_url(new_name)
                # Fallback se a imagem não carregar
                st.markdown(f'<img src="{img_url}" width="35" style="image-rendering: pixelated;">', unsafe_allow_html=True)

    return {"name": new_name, "qty": new_qty}


def room_id_new() -> str:
    # curto e fácil de digitar
    return str(random.randint(100, 999))

def mark_pid_seen(db, rid, pid):
    # Adiciona o ID do Pokémon à lista de "vistos" no banco de dados
    # Usa ArrayUnion para não duplicar se já estiver lá
    ref = db.collection("rooms").document(rid).collection("public_state").document("state")
    # Tenta atualizar, se o documento não tiver o campo 'seen', o firestore cria na hora se usarmos set com merge, 
    # mas aqui vamos assumir que o state existe.
    try:
        ref.update({"seen": firestore.ArrayUnion([str(pid)])})
    except:
        # Fallback caso o campo não exista ainda
        ref.set({"seen": [str(pid)]}, merge=True)
def get_user_doc_ref(db, trainer_name: str):
    return db.collection("users").document(safe_doc_id(trainer_name))

def list_my_rooms(db, trainer_name: str):
    uref = get_user_doc_ref(db, trainer_name)
    udoc = uref.get()
    if not udoc.exists:
        return []
    data = udoc.to_dict() or {}
    return data.get("active_rooms", []) or []

def add_room_to_user(db, trainer_name: str, rid: str):
    uref = get_user_doc_ref(db, trainer_name)
    uref.set(
        {"active_rooms": firestore.ArrayUnion([rid]),
         "updatedAt": firestore.SERVER_TIMESTAMP},
        merge=True
    )

def remove_room_from_user(db, trainer_name: str, rid: str):
    uref = get_user_doc_ref(db, trainer_name)
    uref.set(
        {"active_rooms": firestore.ArrayRemove([rid]),
         "updatedAt": firestore.SERVER_TIMESTAMP},
        merge=True
    )
# --- FUNÇÃO DE CALLBACK CORRIGIDA (CORREÇÃO DO BUG DE STATS 0) ---
def update_poke_state_callback(db, rid, trainer_name, pid, index):
    # CHAVE CORRIGIDA: Agora inclui o nome do treinador e o índice da party
    key_hp = f"hp_{trainer_name}_{pid}_{index}"
    key_cond = f"cond_{trainer_name}_{pid}_{index}"
    
    new_hp = st.session_state.get(key_hp)
    new_cond = st.session_state.get(key_cond)
    
    if new_hp is None: return

    ref = db.collection("rooms").document(rid).collection("public_state").document("party_states")
    data = {
        trainer_name: {
            str(pid): {
                "hp": int(new_hp),
                "cond": new_cond,
                "updatedAt": str(datetime.now())
            }
        }
    }
    ref.set(data, merge=True)
    
    # Lógica de Fainted no Mapa (Visual) - Mantém igual
    if new_hp == 0:
        state_ref = db.collection("rooms").document(rid).collection("public_state").document("state")
        stt = state_ref.get().to_dict() or {}
        pieces = stt.get("pieces") or []
        dirty = False
        for p in pieces:
            if p.get("owner") == trainer_name and str(p.get("pid")) == str(pid):
                if p.get("status") != "fainted":
                    p["status"] = "fainted"
                    dirty = True
        if dirty: state_ref.update({"pieces": pieces})
            
    elif new_hp > 0:
        state_ref = db.collection("rooms").document(rid).collection("public_state").document("state")
        stt = state_ref.get().to_dict() or {}
        pieces = stt.get("pieces") or []
        dirty = False
        for p in pieces:
            if p.get("owner") == trainer_name and str(p.get("pid")) == str(pid):
                if p.get("status") == "fainted":
                    p["status"] = "active"
                    dirty = True
        if dirty: state_ref.update({"pieces": pieces})


def create_room(db, trainer_name: str, grid_size: int, theme: str, max_active: int = 5):
    my_rooms = list_my_rooms(db, trainer_name)
    if len(my_rooms) >= max_active:
        return None, f"Você já tem {len(my_rooms)} arenas ativas (limite {max_active}). Finalize/arquive uma para criar outra."

    # --- LÓGICA DE CÓDIGO ÚNICO (Tenta 5 vezes achar um livre) ---
    rid = None
    room_ref = None
    
    for _ in range(5):
        candidate = room_id_new()
        ref = db.collection("rooms").document(candidate)
        if not ref.get().exists:
            rid = candidate
            room_ref = ref
            break
    
    if not rid:
        return None, "Servidor cheio ou erro ao gerar código. Tente novamente."
    # -------------------------------------------------------------

    room_ref.set({
        "createdAt": firestore.SERVER_TIMESTAMP,
        "status": "lobby",
        "gridSize": int(grid_size),
        "theme": theme,
        "owner": {"name": trainer_name},
        "challenger": None,
        "spectators": [],
        "turn": "owner",
        "turnNumber": 1,
    })

    # estado público inicial
    room_ref.collection("public_state").document("state").set({
        "tilesPacked": None,
        "seed": None,
        "pieces": [],
        "effects": [],
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

    add_room_to_user(db, trainer_name, rid)
    return rid, None

def get_room(db, rid: str):
    ref = db.collection("rooms").document(rid)
    doc = ref.get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    data["_id"] = rid
    return data

def get_perspective_color(viewer_name, player_name, room_data):
    """
    Retorna a cor baseada na perspectiva do visualizador.
    Azul: Você | Vermelho: Oponente 1 | Amarelo: Oponente 2 | Rosa: Oponente 3
    """
    if viewer_name == player_name:
        return (0, 150, 255) # Azul (Sempre você)
    
    owner = (room_data.get("owner") or {}).get("name")
    challengers = [c.get("name") for c in (room_data.get("challengers") or [])]
    
    # Criar lista de "outros" (todos menos o espectador)
    others = [owner] + challengers
    if viewer_name in others:
        others.remove(viewer_name)
    
    # Atribui cores aos oponentes na ordem em que aparecem
    opp_colors = [
        (255, 50, 50),   # Vermelho (Opp 1)
        (255, 215, 0),  # Amarelo (Opp 2)
        (255, 105, 180)  # Rosa (Opp 3)
    ]
    
    try:
        idx = others.index(player_name)
        return opp_colors[idx] if idx < len(opp_colors) else (200, 200, 200)
    except ValueError:
        return (200, 200, 200) # Cor neutra para espectadores


def join_room_as_challenger(db, rid: str, trainer_name: str, max_challengers: int = 4):
    ref = db.collection("rooms").document(rid)
    doc = ref.get()
    if not doc.exists:
        return "NOT_FOUND"

    data = doc.to_dict() or {}
    owner = (data.get("owner") or {}).get("name")
    # Agora lidamos com uma lista de desafiantes
    challengers = data.get("challengers") or []
    
    # Se já estiver na lista ou for o dono
    if owner == trainer_name:
        add_room_to_user(db, trainer_name, rid)
        return "ALREADY_OWNER"
    
    if any(c.get("name") == trainer_name for c in challengers):
        add_room_to_user(db, trainer_name, rid)
        return "ALREADY_CHALLENGER"

    # Verifica se ainda há vaga (até 4 desafiantes)
    if len(challengers) < max_challengers:
        new_challenger = {"name": trainer_name}
        ref.update({
            "challengers": firestore.ArrayUnion([new_challenger]),
            "status": "running",
        })
        add_room_to_user(db, trainer_name, rid)
        return "OK"

    return "ARENA_FULL"




def join_room_as_spectator(db, rid: str, trainer_name: str):
    ref = db.collection("rooms").document(rid)
    doc = ref.get()
    if not doc.exists:
        return "NOT_FOUND"

    data = doc.to_dict() or {}
    owner = (data.get("owner") or {}).get("name")
    challenger = (data.get("challenger") or {}).get("name") if isinstance(data.get("challenger"), dict) else data.get("challenger")

    if trainer_name in [owner, challenger]:
        add_room_to_user(db, trainer_name, rid)
        return "PLAYER"

    ref.update({
        "spectators": firestore.ArrayUnion([trainer_name]),
    })
    # evento público
    ref.collection("public_events").add({
        "type": "join_spectator",
        "by": trainer_name,
        "payload": {"room": rid},
        "ts": firestore.SERVER_TIMESTAMP,
    })
    add_room_to_user(db, trainer_name, rid)
    return "OK"

def add_public_event(db, rid: str, event_type: str, by: str, payload: dict):
    db.collection("rooms").document(rid).collection("public_events").add({
        "type": event_type,
        "by": by,
        "payload": payload or {},
        "ts": firestore.SERVER_TIMESTAMP,
    })
def state_ref_for(db, rid: str):
    return (
        db.collection("rooms")
          .document(rid)
          .collection("public_state")
          .document("state")
    )

def get_state(db, rid: str) -> dict:
    doc = state_ref_for(db, rid).get()
    return doc.to_dict() if doc.exists else {}

def update_party_state(db, rid, trainer_name, pid, hp, conditions):
    # Salva HP e Status no documento 'party_states'
    ref = db.collection("rooms").document(rid).collection("public_state").document("party_states")
    
    # Estrutura: { "NomeTreinador": { "PID": { "hp": 6, "cond": [...] } } }
    # Usamos merge=True com notação de ponto para não apagar os outros
    key = f"{trainer_name}.{pid}"
    
    data = {
        key: {
            "hp": int(hp),
            "cond": conditions,
            "updatedAt": str(datetime.now())
        }
    }
    ref.set(data, merge=True)
    
    # Se o HP for 0, precisamos atualizar a peça no tabuleiro para 'fainted' (se ela estiver lá)
    if hp == 0:
        # Busca peças desse treinador e desse PID
        state_doc = db.collection("rooms").document(rid).collection("public_state").document("state").get()
        if state_doc.exists:
            pieces = state_doc.to_dict().get("pieces", [])
            for p in pieces:
                if p.get("owner") == trainer_name and str(p.get("pid")) == str(pid):
                    p["status"] = "fainted"
                    upsert_piece(db, rid, p)
    # Se HP > 0 e estava fainted, revive
    elif hp > 0:
        state_doc = db.collection("rooms").document(rid).collection("public_state").document("state").get()
        if state_doc.exists:
            pieces = state_doc.to_dict().get("pieces", [])
            for p in pieces:
                if p.get("owner") == trainer_name and str(p.get("pid")) == str(pid) and p.get("status") == "fainted":
                    p["status"] = "active"
                    upsert_piece(db, rid, p)

def upsert_piece(db, rid: str, piece: dict):
    # piece precisa ter id único
    sref = state_ref_for(db, rid)
    stt = get_state(db, rid)
    pieces = stt.get("pieces") or []

    # substitui se já existe
    new_pieces = [p for p in pieces if p.get("id") != piece.get("id")]
    new_pieces.append(piece)

    sref.set({
        "pieces": new_pieces,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)

def delete_piece(db, rid: str, piece_id: str):
    sref = state_ref_for(db, rid)
    stt = get_state(db, rid)
    pieces = stt.get("pieces") or []
    new_pieces = [p for p in pieces if p.get("id") != piece_id]
    sref.set({
        "pieces": new_pieces,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)

def find_piece_at(pieces: list[dict], row: int, col: int) -> dict | None:
    for p in pieces or []:
        if int(p.get("row", -1)) == int(row) and int(p.get("col", -1)) == int(col):
            return p
    return None


def visible_pieces_for(room: dict, viewer_name: str, pieces: list[dict]) -> list[dict]:
    # Jogador vê tudo dele; vê do oponente só o que estiver "revealed" (no campo)
    role = get_role(room, viewer_name)
    owner = (room.get("owner") or {}).get("name")
    chal = room.get("challenger") or {}
    chal_name = chal.get("name") if isinstance(chal, dict) else (chal or None)

    if role == "spectator":
        # espectador vê somente o que está no campo (revealed)
        return [p for p in pieces if p.get("revealed") is True]

    # jogador: vê os dele sempre; e do outro só se revealed
    me = owner if role == "owner" else chal_name
    out = chal_name if role == "owner" else owner

    result = []
    for p in pieces:
        if p.get("owner") == me:
            result.append(p)
        elif p.get("owner") == out and p.get("revealed") is True:
            result.append(p)
    return result

def list_public_events(db, rid: str, limit: int = 30):
    q = (db.collection("rooms").document(rid)
         .collection("public_events")
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [d.to_dict() for d in q.stream()]
# =========================
# MAPA TÁTICO (3.1 / 3.2 / 3.3)
# =========================

TILE_SIZE = 64

THEMES = {
    "cave_water": {"base": "rock", "border": "wall"},
    "forest": {"base": "grass", "border": "tree"},
    "mountain_slopes": {"base": "stone", "border": "peak"},
    "plains": {"base": "grass", "border": "bush"},
    "dirt": {"base": "dirt", "border": "rock"},
    "river": {"base": "grass", "border": "tree"},
    "sea_coast": {"base": "sand", "border": "sea"},
    "center_lake": {"base": "grass", "border": "tree"},
}

def gen_tiles(grid: int, theme_key: str, seed: int | None = None, no_water: bool = False):
    # REGRA: Bloqueia água em 6x6, exceto se o tema tiver "water", "river", "lake" ou "sea" no nome
    themes_com_agua = ["water", "river", "lake", "sea", "coast"]
    if grid <= 6 and not any(word in theme_key.lower() for word in themes_com_agua):
        no_water = True
    if seed is None:
        seed = random.randint(1, 999999999)

    rng = random.Random(seed)
    theme = THEMES.get(theme_key, THEMES["cave_water"])
    base = theme["base"]
    tiles = [[base for _ in range(grid)] for _ in range(grid)]

    def inside(r, c):
        return 1 <= r <= grid - 2 and 1 <= c <= grid - 2

   # pedras leves em todos os temas (dá textura)
    for _ in range(rng.randint(grid, grid * 2)):
        rr = rng.randint(1, grid - 2)
        cc = rng.randint(1, grid - 2)
        if inside(rr, cc) and rng.random() > 0.75:
            if tiles[rr][cc] in ["grass", "dirt", "sand", "trail"]:
                tiles[rr][cc] = "rock"

    # --- features por tema ---
    if theme_key == "cave_water":
        if not no_water:
            pools = rng.randint(1, 2)
            for _ in range(pools):
                cr = rng.randint(2, grid - 3)
                cc = rng.randint(2, grid - 3)
                rad = rng.randint(1, 2)

                for rr in range(cr - rad, cr + rad + 1):
                    for cc2 in range(cc - rad, cc + rad + 1):
                        if inside(rr, cc2) and rng.random() > 0.25:
                            tiles[rr][cc2] = "water"

        # elementos sólidos continuam existindo
        spikes = rng.randint(1, max(2, grid - 3))
        for _ in range(spikes):
            rr = rng.randint(1, grid - 2)
            cc = rng.randint(1, grid - 2)
            if inside(rr, cc) and tiles[rr][cc] == base:
                tiles[rr][cc] = "stalagmite"

    elif theme_key == "forest":
        # “carpete” de grama
        for r in range(1, grid - 1):
            for c in range(1, grid - 1):
                if inside(r, c):
                    tiles[r][c] = "grass" if rng.random() > 0.15 else "bush"
    
        # árvores espalhadas
        trees = rng.randint(grid, grid * 2)
        for _ in range(trees):
            rr = rng.randint(1, grid - 2)
            cc = rng.randint(1, grid - 2)
            if inside(rr, cc) and tiles[rr][cc] in ["grass", "bush"] and rng.random() > 0.35:
                tiles[rr][cc] = "tree"

        # caminho opcional
        if rng.random() > 0.35:
            r = rng.randint(2, grid - 3)
            for c in range(1, grid - 1):
                if inside(r, c) and tiles[r][c] != "tree":
                    tiles[r][c] = "path"

        # água só se permitido
        if not no_water:
            ponds = rng.randint(0, 2)
            for _ in range(ponds):
                cr = rng.randint(2, grid - 3)
                cc = rng.randint(2, grid - 3)
                rad = rng.randint(1, 2)
                for rr in range(cr - rad, cr + rad + 1):
                    for cc2 in range(cc - rad, cc + rad + 1):
                        if inside(rr, cc2) and rng.random() > 0.35:
                            tiles[rr][cc2] = "water"
    
    elif theme_key == "mountain_slopes":
        # base rochosa
        for r in range(1, grid - 1):
            for c in range(1, grid - 1):
                if inside(r, c):
                    tiles[r][c] = "stone" if rng.random() > 0.25 else "rock"

        # faixas de declive (diagonais)
        bands = rng.randint(2, 4)
        for _ in range(bands):
            start_r = rng.randint(1, grid - 2)
            for c in range(1, grid - 1):
                rr = start_r + (c // 2)
                if inside(rr, c) and rng.random() > 0.25:
                    tiles[rr][c] = "slope1" if rng.random() > 0.5 else "slope2"

        # picos
        peaks = rng.randint(1, 3)
        for _ in range(peaks):
            rr = rng.randint(2, grid - 3)
            cc = rng.randint(2, grid - 3)
            tiles[rr][cc] = "peak"

    elif theme_key == "plains":
        for _ in range(rng.randint(2, grid * 2)):
            rr = rng.randint(1, grid - 2)
            cc = rng.randint(1, grid - 2)
            if inside(rr, cc) and rng.random() > 0.5:
                tiles[rr][cc] = "flower"

        if rng.random() > 0.5:
            c = rng.randint(2, grid - 3)
            for r in range(1, grid - 1):
                if inside(r, c):
                    tiles[r][c] = "trail"

    elif theme_key == "dirt":
        for r in range(1, grid - 1):
            for c in range(1, grid - 1):
                if inside(r, c) and rng.random() > 0.85:
                    tiles[r][c] = "stone"

        if rng.random() > 0.4:
            r = rng.randint(2, grid - 3)
            for c in range(1, grid - 1):
                if inside(r, c) and tiles[r][c] == base:
                    tiles[r][c] = "rut"

    elif theme_key == "river":
        r = rng.randint(1, grid - 2)
        width = 2 if grid >= 8 else 1
    
        for c in range(1, grid - 1):
            for w in range(width):
                rr = r + w
                if 1 <= rr <= grid - 2:
                    tiles[rr][c] = ("water" if not no_water else "trail")
    
            # margens
            if r - 1 >= 1 and rng.random() > 0.35:
                tiles[r - 1][c] = "sand" if not no_water else "stone"
            if r + width <= grid - 2 and rng.random() > 0.35:
                tiles[r + width][c] = "sand" if not no_water else "stone"
    
            step = rng.choice([-1, 0, 1])
            r = max(1, min(grid - 2 - (width - 1), r + step))

        
    elif theme_key == "sea_coast":
        for r in range(grid):
            if not no_water:
                tiles[r][0] = "sea"
                if grid > 4:
                    tiles[r][1] = "sea" if rng.random() > 0.25 else "sand"
            else:
                tiles[r][0] = "sand"
                if grid > 4:
                    tiles[r][1] = "sand" if rng.random() > 0.25 else "stone"
    
        for _ in range(rng.randint(1, 4)):
            rr = rng.randint(1, grid - 2)
            cc = rng.randint(2, grid - 2)
            if inside(rr, cc) and rng.random() > 0.45:
                tiles[rr][cc] = "rock"
    
    elif theme_key == "center_lake":
        cr = grid // 2
        cc = grid // 2
        rad = 2 if grid >= 8 else 1
        for rr in range(cr - rad, cr + rad + 1):
            for cc2 in range(cc - rad, cc + rad + 1):
                if inside(rr, cc2) and (abs(rr - cr) + abs(cc2 - cc) <= rad + 1):
                    tiles[rr][cc2] = ("water" if not no_water else "stone")

   
    # --- limpeza final: garante zero água se no_water=True ---
    if no_water:
        for r in range(grid):
            for c in range(grid):
                if tiles[r][c] == "water":
                    # substitui por algo coerente com o tema
                    tiles[r][c] = "path" if theme_key in ["forest", "cave_water"] else "trail"
                elif tiles[r][c] == "sea":
                    tiles[r][c] = "sand"

    return tiles, seed

def draw_tile_asset(img, r, c, tiles, assets, rng):
    grid = len(tiles)
    t = tiles[r][c]
    x, y = c * TILE_SIZE, r * TILE_SIZE
    
    asset_key = None

    # LÓGICA DE SELEÇÃO DE ASSET
    if t == "water":
        # Verifica se há terra/grama/areia ao redor (Margem)
        is_shore = False
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid and 0 <= nc < grid:
                if tiles[nr][nc] not in ["water", "sea"]:
                    is_shore = True
                    break
        asset_key = "agua_2" if is_shore else rng.choice(["agua_1", "agua_3"])

    elif t == "grass":
        asset_key = rng.choice(["grama_1", "grama_2", "grama_3"])
    
    elif t == "sand":
        asset_key = rng.choice(["areia_1", "areia_2", "areia_3"])

    elif t == "tree":
        asset_key = rng.choice(["tree_1", "tree_2", "tree_3"])

    elif t == "rock":
        asset_key = "rochas" if rng.random() > 0.5 else "rochas_2"

    elif t.startswith("slope"):
        # Mapeia slope1/slope2 do gerador para os assets slope_1 a slope_4
        # Exemplo: slope1 -> slope_1 (subida), slope2 -> slope_2 (descida)
        asset_key = t.replace("slope", "slope_") 

    elif t == "stone":
        asset_key = rng.choice(["pedra_1", "pedra_2", "pedra_3"])

    # Fallbacks genéricos para outros nomes
    else:
        mapping = {
            "wall": "wall_1",
            "stalagmite": "estalagmite_1",
            "peak": "pico_1",
            "dirt": "terra_1",
            "bush": "brush_1"
        }
        asset_key = mapping.get(t, "terra_1")

    # DESENHO NA IMAGEM
    if asset_key in assets:
        tile_img = assets[asset_key]
        img.paste(tile_img, (x, y), tile_img)
        

# ==========================================
# 🛠️ CORREÇÃO: FUNÇÕES DE DESENHO (CACHE + EFEITOS)
# ==========================================

@st.cache_data(show_spinner=False)
def fetch_image_pil(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGBA")
        return img
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def render_map_png(tiles: list[list[str]], theme_key: str, seed: int, show_grid: bool = True):
    grid = len(tiles)
    # Criamos a imagem base. RGBA é essencial para a transparência das árvores
    img = Image.new("RGBA", (grid * TILE_SIZE, grid * TILE_SIZE))
    assets = load_map_assets() # Carrega seus PNGs 64x64
    rng = random.Random(int(seed or 0) + 1337)
    floor_variants = {}
    for key in assets:
        base = key.split("__", 1)[0]
        if base.startswith(FLOOR_PREFIXES):
            floor_variants.setdefault(base, []).append(key)

    # 1. Definimos o "Chão Base" do tema para não haver buracos pretos
    theme_floors = {
        "forest": "grama_1", "cave_water": "pedra_1", "mountain_slopes": "pedra_1",
        "plains": "grama_1", "dirt": "terra_1", "sea_coast": "areia_1"
    }
    base_floor = theme_floors.get(theme_key, "grama_1")

    for r in range(grid):
        for c in range(grid):
            x, y = c * TILE_SIZE, r * TILE_SIZE
            t_type = tiles[r][c]

            # --- CAMADA 1: O CHÃO SEMPRE PRESENTE ---
            # Colamos a grama ou pedra base primeiro em TODOS os tiles
            img.paste(assets[base_floor], (x, y))

            # --- CAMADA 2: TERRENOS ESPECÍFICOS E TRANSIÇÃO ---
            asset_to_draw = None
            
            if t_type == "water" or t_type == "sea":
                # Lógica de Suavização: Se houver terra vizinha, usa agua_2 (margem)
                is_margin = False
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    if 0 <= r+dr < grid and 0 <= c+dc < grid:
                        if tiles[r+dr][c+dc] not in ["water", "sea"]:
                            is_margin = True
                            break
                asset_to_draw = "agua_2" if is_margin else rng.choice(["agua_1", "agua_3"])
            
            elif t_type in ["sand", "stone", "dirt", "grass"]:
                # Variação aleatória do próprio chão
                prefix = {"sand":"areia", "stone":"pedra", "dirt":"terra", "grass":"grama"}[t_type]
                asset_to_draw = f"{prefix}_{rng.randint(1,3)}"

            if asset_to_draw in assets:
                choices = floor_variants.get(asset_to_draw, [asset_to_draw])
                asset_choice = rng.choice(choices)
                img.alpha_composite(assets[asset_choice], (x, y))

            # --- CAMADA 3: OBJETOS (Árvores e Rochas em vários mapas) ---
            obj_asset = None
            if t_type == "tree":
                obj_asset = rng.choice(["tree_1", "tree_2", "tree_3"])
            elif t_type == "stalagmite":
                obj_asset = "estalagmite_1"
            elif t_type == "peak":
                obj_asset = "pico_1"
            
            # Adiciona ROCHAS aleatórias em qualquer terreno (conforme pedido)
            # 10% de chance de aparecer uma rocha de detalhe em tiles de chão
            if t_type in ["grass", "stone", "dirt", "sand"] and rng.random() < 0.10:
                obj_asset = rng.choice(["rochas", "rochas_2"])
            elif t_type == "rock": # Se o tile for nominalmente rocha
                obj_asset = rng.choice(["rochas", "rochas_2"])

            if obj_asset in assets:
                img.alpha_composite(assets[obj_asset], (x, y))

    # --- CAMADA 4: GRID TÁTICO FINO ---
    if show_grid:
        draw = ImageDraw.Draw(img)
        # Cor branca com baixa opacidade (40/255) para ser sutil
        grid_color = (255, 255, 255, 40)
        for i in range(grid + 1):
            pos = i * TILE_SIZE
            draw.line([(0, pos), (grid * TILE_SIZE, pos)], fill=grid_color, width=1)
            draw.line([(pos, 0), (pos, grid * TILE_SIZE)], fill=grid_color, width=1)

    return img.convert("RGB")

def render_map_with_pieces(tiles, theme_key, seed, pieces, viewer_name, room, effects=None, show_grid: bool = True):
    
    # 1. Base do Mapa (Cacheada)␊
    img = render_map_png(tiles, theme_key, seed, show_grid=show_grid).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    # 2. CAMADA DE EFEITOS (Agora usando Imagens Reais)
    if effects:
        # Mapeamento: Emoji -> Caminho do Arquivo (Deve coincidir com a variável usada abaixo)
        EMOJI_TO_PATH = {
            "🔥": "Assets/fogo.png",
            "🧊": "Assets/gelo.png",
            "💧": "Assets/agua.png",
            "🪨": "Assets/rocha.png",
            "☁️": "Assets/nuvem.png",
            "☀️": "Assets/sol.png",
            "🍃": "Assets/terrenograma.png",
            "⚡": "Assets/terrenoeletrico.png",
        }

        for eff in effects:
            try:
                r, c = int(eff.get("row")), int(eff.get("col"))
                icon_char = eff.get("icon", "?")
                x, y = c * TILE_SIZE, r * TILE_SIZE
                
                path = EMOJI_TO_PATH.get(icon_char)
                
                if path and os.path.exists(path):
                    icon_img = Image.open(path).convert("RGBA")
                    # Ajusta o tamanho do ícone para 70% do tile
                    icon_img.thumbnail((int(TILE_SIZE * 0.7), int(TILE_SIZE * 0.7)))
                    
                    # Centraliza o ícone no tile
                    ix = x + (TILE_SIZE - icon_img.size[0]) // 2
                    iy = y + (TILE_SIZE - icon_img.size[1]) // 2
                    img.alpha_composite(icon_img, (ix, iy))
                else:
                    # Fallback visual caso o arquivo não seja encontrado
                    draw.ellipse([x+16, y+16, x+TILE_SIZE-16, y+TILE_SIZE-16], fill=(255, 255, 255, 150))
            except Exception as e:
                # Opcional: imprimir o erro no console para debug
                print(f"Erro ao renderizar efeito {icon_char}: {e}")
                continue

    # 3. CAMADA DE POKÉMONS
    local_cache = {}
    
    for p in pieces or []:
        r = int(p.get("row", -1))
        c = int(p.get("col", -1))
        if r < 0 or c < 0: continue

        owner = p.get("owner")
        border_color = get_perspective_color(viewer_name, owner, room)

        x = c * TILE_SIZE
        y = r * TILE_SIZE
        
        # Borda
        draw.rectangle([x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1], outline=border_color, width=4)

        pid = str(p.get("pid", ""))
        is_p_shiny = p.get("shiny", False) #
        url = pokemon_pid_to_image(pid, mode="sprite", shiny=is_p_shiny)

        if url not in local_cache:
            local_cache[url] = fetch_image_pil(url)

        sprite = local_cache[url]
        if sprite is None: continue

        sp = sprite.copy()
        sp.thumbnail((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
        
        x0 = x + (TILE_SIZE - sp.size[0]) // 2
        y0 = y + (TILE_SIZE - sp.size[1]) // 2
        img.alpha_composite(sp, (x0, y0))

    return img.convert("RGB")
    
def normalize_text(text):
    if not isinstance(text, str): return str(text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def get_pid_from_name(user_name: str, name_map: dict) -> str | None:
    if not isinstance(user_name, str):
        return None

    pre_clean = user_name.replace('♀', '-f').replace('♂', '-m')
    clean = normalize_text(pre_clean).replace('.', '').replace("'", '').replace(' ', '-')

    # exceções / formas (as mesmas que você já usa)
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

    if clean == 'toxtricity': clean = 'toxtricity-amped'
    if clean == 'eiscue': clean = 'eiscue-ice'
    if clean == 'indeedee': clean = 'indeedee-male'
    if clean == 'morpeko': clean = 'morpeko-full-belly'
    if clean == 'urshifu': clean = 'urshifu-single-strike'

    if clean == 'basculegion': clean = 'basculegion-male'
    if clean == 'enamorus': clean = 'enamorus-incarnate'
    if clean == 'keldeo': clean = 'keldeo-ordinary'
    if clean == 'meloetta': clean = 'meloetta-aria'

    # regionais (as mesmas)
    if clean.endswith('-a'): clean = clean[:-2] + '-alola'
    if clean.endswith('-g'): clean = clean[:-2] + '-galar'
    if clean.endswith('-h'): clean = clean[:-2] + '-hisui'
    if clean.endswith('-p'): clean = clean[:-2] + '-paldea'
    if clean.startswith('g-'): clean = clean[2:] + '-galar'
    if clean.startswith('a-'): clean = clean[2:] + '-alola'
    if clean.startswith('h-'): clean = clean[2:] + '-hisui'
    if clean.startswith('p-'): clean = clean[2:] + '-paldea'

    p_id = name_map.get(clean)
    if not p_id:
        base_name = clean.split('-')[0]
        p_id = name_map.get(base_name)

    return p_id
    
def get_pokemon_image_url(user_name: str, name_map: dict, mode: str = "artwork", shiny: bool = False) -> str:
    # Tenta achar o ID da National Dex pelo nome
    p_id = get_pid_from_name(user_name, name_map)
    
    if not p_id:
        return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"

    if mode == "sprite":
        return get_pokemon_sprite_url(p_id, shiny=shiny)
    return get_pokemon_artwork_url(p_id, shiny=shiny)


def get_image_from_name(user_name, name_map):
    if not isinstance(user_name, str): return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
    pre_clean = user_name.replace('♀', '-f').replace('♂', '-m')
    clean = normalize_text(pre_clean).replace('.', '').replace("'", '').replace(' ', '-')
    
    # --- LISTA DE EXCEÇÕES E FORMAS (ATUALIZADA) ---
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
    
    # Correções da 8ª Geração (Seus pedidos)
    if clean == 'toxtricity': clean = 'toxtricity-amped'
    if clean == 'eiscue': clean = 'eiscue-ice'
    if clean == 'indeedee': clean = 'indeedee-male'
    if clean == 'morpeko': clean = 'morpeko-full-belly'
    if clean == 'urshifu': clean = 'urshifu-single-strike'
    
    # Outras correções úteis
    if clean == 'basculegion': clean = 'basculegion-male'
    if clean == 'enamorus': clean = 'enamorus-incarnate'
    if clean == 'keldeo': clean = 'keldeo-ordinary'
    if clean == 'meloetta': clean = 'meloetta-aria'

    # Sufixos Regionais
    if clean.endswith('-a'): clean = clean[:-2] + '-alola'
    if clean.endswith('-g'): clean = clean[:-2] + '-galar'
    if clean.endswith('-h'): clean = clean[:-2] + '-hisui'
    if clean.endswith('-p'): clean = clean[:-2] + '-paldea' # Paldea
    if clean.startswith('g-'): clean = clean[2:] + '-galar'
    if clean.startswith('a-'): clean = clean[2:] + '-alola'
    if clean.startswith('h-'): clean = clean[2:] + '-hisui'
    if clean.startswith('p-'): clean = clean[2:] + '-paldea'
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
        
def get_pokemon_artwork_url(p_id: str, shiny: bool = False) -> str:
    # Garante que é numérico para API
    try:
        n = int(str(p_id).lstrip("0") or "0")
    except ValueError:
        return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
        
    if shiny:
        return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/shiny/{n}.png"
    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{n}.png"


def get_pokemon_sprite_url(p_id: str, shiny: bool = False) -> str:
    try:
        n = int(str(p_id).lstrip("0") or "0")
    except ValueError:
        return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"

    if shiny:
        return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/shiny/{n}.png"
    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{n}.png"


    
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

def pokemon_pid_to_image(pid: str, mode: str = "artwork", shiny: bool = False) -> str:
    if not pid:
        return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
    pid_str = str(pid).strip()
    
    # Caso 1: Visitante (EXT)
    if pid_str.startswith("EXT:"):
        name = pid_str.replace("EXT:", "")
        return get_pokemon_image_url(name, api_name_map, mode=mode, shiny=shiny)
        
    # Caso 2: Busca no EXCEL (Correção do ID Regional)
    # Procura o ID no Excel para pegar o NOME correto
    if 'df' in globals() or 'df' in st.session_state:
        # Tenta pegar o df de onde estiver disponível
        local_df = st.session_state.get('df_data') if 'df_data' in st.session_state else df
        
        row = local_df[local_df["Nº"].astype(str) == pid_str]
        if not row.empty:
            # Pega o nome (ex: "MyStarter")
            real_name = row.iloc[0]["Nome"]
            # Busca a imagem pelo NOME, não pelo número
            return get_pokemon_image_url(real_name, api_name_map, mode=mode, shiny=shiny)

    # Fallback: Se não achou no Excel, retorna erro ou tenta direto (mas evita erro de imagem quebrada)
    return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"


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
page = st.sidebar.radio("Ir para:", ["Pokédex (Busca)", "Trainer Hub (Meus Pokémons)", "PvP – Arena Tática", "Mochila"])


# ==============================================================================
# PÁGINA 1: POKEDEX (VISÃO DE FOCO + CARROSSEL INFERIOR)
# ==============================================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #1f4e79 0%, #3b7ca6 45%, #5fb2cf 100%);
}
h1, h2, h3 {
    color: #0b1f2a;
    text-shadow: 0 1px 0 rgba(255, 255, 255, 0.35);
}
.pokedex-shell {
    border-radius: 18px;
    padding: 18px 18px 8px 18px;
    border: 3px solid rgba(255,255,255,0.65);
    box-shadow: inset 0 0 15px rgba(255,255,255,0.35);
    background: rgba(10, 36, 58, 0.45);
}
.pokedex-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    background: rgba(255,255,255,0.65);
    padding: 6px 16px;
    border-radius: 16px;
    font-size: 12px;
    color: #0b1f2a;
}
.pokedex-grid-note {
    font-size: 11px;
    color: #e7f5ff;
    text-align: center;
    margin: 6px 0 10px 0;
}
.pokedex-card {
    background: rgba(10, 25, 40, 0.65);
    color: #f5f5f5;
    padding: 18px;
    border-radius: 16px;
    border: 2px solid rgba(255,255,255,0.35);
    margin-top: 18px;
}
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
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding: 10px 4px;
}
.pokedex-carousel img {
    width: 72px;
    height: 72px;
    image-rendering: pixelated;
    background: rgba(255,255,255,0.25);
    border-radius: 10px;
    padding: 6px;
}
.pokedex-grid img { image-rendering: pixelated; }

/* CARROSSEL INFERIOR (o seu estilo) */
.pokedex-footer-carousel {
    display: flex;
    flex-wrap: nowrap;
    overflow-x: auto;
    gap: 12px;
    padding: 14px;
    background: rgba(0, 0, 0, 0.30);
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.18);
    scroll-behavior: smooth;
}
.pokedex-footer-carousel::-webkit-scrollbar { height: 8px; }
.pokedex-footer-carousel::-webkit-scrollbar-thumb { background: #FFCC00; border-radius: 10px; }

.carousel-item {
    flex: 0 0 auto;
    width: 70px;
    height: 70px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    cursor: pointer;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    transition: transform 0.15s;
}
.carousel-item:hover { transform: scale(1.12); }

.carousel-item img {
    width: 54px;
    height: 54px;
    image-rendering: pixelated;
}
.carousel-item-active {
    border: 2px solid #FFCC00;
    background: rgba(255, 204, 0, 0.10);
}
.pokedex-tile button {
    width: 100%;
    max-width: 90px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center;
    font-size: 11px;
    line-height: 1.2;
}
.info-label {
  color: #ffd166;             /* amarelo */
  font-weight: 800;
}

.section-title {
  color: #80ed99;             /* verde */
  font-weight: 900;
  margin-top: 10px;
}

.hi-red { color: #ff5c5c; font-weight: 900; }    /* FIR / alertas */
.hi-cyan { color: #4dd6ff; font-weight: 900; }   /* palavras-chave */
.hi-purple { color: #b197ff; font-weight: 900; } /* tags extras */

.power-badge {
  display: inline-block;
  margin-top: 10px;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.25);
  color: #ffd166;
  font-weight: 900;
}

</style>
""", unsafe_allow_html=True)




if page == "Pokédex (Busca)":
    st.sidebar.header("🔍 Filtros")
    search_query = st.sidebar.text_input("Buscar (Nome ou Nº)", "")

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
            for label, value in entries:
                # Título: Descrição
                if label == "Descrição da Pokedex":
                    st.markdown("<div class='section-title'>📘 Descrição da Pokédex</div>", unsafe_allow_html=True)
                    st.write(value)
                    continue
        
                # Título: Viabilidade (texto grande)
                if label == "Viabilidade":
                    st.markdown("<div class='section-title'>🧠 Viabilidade</div>", unsafe_allow_html=True)
        
                    viab = (
                        str(value)
                        .replace("PARCEIROS:", "\n\n**👥 PARCEIROS:**")
                        .replace("Explicação:", "\n\n**💡 EXPLICAÇÃO:**")
                        .replace("Habilidade:", "**✨ Habilidade:**")
                    )
        
                    # pinta o FIR (e outros códigos se quiser)
                    viab = viab.replace("FIR", "<span class='hi-red'>FIR</span>")
        
                    # destaca os códigos de estratégia no texto (mantém o que você já tinha)
                    for code in codes:
                        viab = re.sub(rf"\b{re.escape(code)}\b", f"<span class='hi-purple'>{code}</span>", viab)
        
                    st.markdown(viab, unsafe_allow_html=True)
                    continue
        
                # Campos normais (Tipo, Raridade, Biomas, Região etc.)
                st.markdown(
                    f"<span class='info-label'>{label}:</span> {value}",
                    unsafe_allow_html=True
                )


        def render_info_tags():
            tags_html = "".join([f"<span>{c}</span>" for c in codes])
            st.markdown(f"<div class='pokedex-tags'>{tags_html}</div>", unsafe_allow_html=True)

        def render_status_controls():
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
            st.image(pokemon_pid_to_image(dex_num, mode="artwork", shiny=False), use_container_width=True)

    # Nível de Poder abaixo da imagem
    np = row.get("Nivel_Poder", "")
    if str(np).strip() != "" and str(np).lower() != "nan":
        st.markdown(f"<div class='power-badge'>⚡ Nível de Poder: {np}</div>", unsafe_allow_html=True)


        with top_right:
            render_info_columns(info_entries[midpoint:])

        render_status_controls()
        render_info_tags()

        st.markdown("#### 🎞️ Variações")
        sprite_urls = [pokemon_pid_to_image(dex_num, mode="sprite", shiny=s) for s in [False, True]]
        sprites_html = "".join([f"<img src='{url}' style='width:70px; image-rendering: pixelated;'>" for url in sprite_urls])
        st.markdown(f"<div class='pokedex-carousel'>{sprites_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # --- CARROSSEL INFERIOR (navegação) ---
        st.subheader("🔄 Navegar pela Dex")
        
        # recebe clique via JS
        clicked = st.session_state.get("carousel_click", None)
        if clicked:
            st.session_state["pokedex_selected"] = clicked
            st.session_state["carousel_click"] = None
            st.rerun()
        
        items_html = []
        for _, r_car in filtered_df.iterrows():
            pid = str(r_car["Nº"])
            sprite = pokemon_pid_to_image(pid, mode="sprite", shiny=False)
            active = "carousel-item-active" if pid == dex_num else ""
        
            items_html.append(
                f"""
                <div class="carousel-item {active}"
                     onclick="selectDex('{pid}')">
                    <img src="{sprite}">
                </div>
                """
            )
        
        st.markdown(
            f"""
            <div id="dex-carousel" class="pokedex-footer-carousel">
                {''.join(items_html)}
            </div>
        
            <script>
            // converte scroll vertical em horizontal
            const carousel = document.getElementById("dex-carousel");
            carousel.addEventListener("wheel", (evt) => {{
                evt.preventDefault();
                carousel.scrollLeft += evt.deltaY;
            }});
        
            // envia clique para o Streamlit (mesma aba)
            function selectDex(pid) {{
                const input = document.createElement("input");
                input.type = "hidden";
                input.name = "carousel_click";
                input.value = pid;
        
                const form = document.createElement("form");
                form.method = "POST";
                form.appendChild(input);
        
                document.body.appendChild(form);
                form.submit();
            }}
            </script>
            """,
            unsafe_allow_html=True,
        )
        

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
            st.markdown(
                "<div class='pokedex-grid-note'>Passe o mouse sobre o Pokémon para ver o nome. Clique em um Pokémon para ver os detalhes completos.</div>",
                unsafe_allow_html=True,
            )

            grid_cols = 8
            rows = list(filtered_df.iterrows())

            st.markdown("<div class='pokedex-grid'>", unsafe_allow_html=True)
            for start in range(0, len(rows), grid_cols):
                cols = st.columns(grid_cols)
                for col, (index, row_g) in zip(cols, rows[start : start + grid_cols]):
                    dex_num = str(row_g["Nº"])
                    p_name = row_g["Nome"]
                    sprite_url = pokemon_pid_to_image(dex_num, mode="sprite", shiny=False)

                    with col:
                        st.markdown("<div class='pokedex-tile'>", unsafe_allow_html=True)
                        st.image(sprite_url, width=64)

                        # botão “invisível” (clique no tile)
                        st.button(
                            f"{p_name}",
                            key=f"dex_tile_{dex_num}_{index}",
                            help=f"#{dex_num} • {p_name}",
                            on_click=select_pokedex_entry,
                            args=(dex_num,),
                        )

                        st.markdown("</div>", unsafe_allow_html=True)
                st.write("")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 2: TRAINER HUB
# ==============================================================================
if page == "Trainer Hub (Meus Pokémons)":
    st.title("🏕️ Trainer Hub")
    # --- INICIALIZAÇÃO DE DADOS NOVOS ---
    if "stats" not in user_data: user_data["stats"] = {}
    if "wishlist" not in user_data: user_data["wishlist"] = [] # Nova Lista de Desejo
    if "shinies" not in user_data: user_data["shinies"] = []   # Nova Lista de Shinies
    
    # Adicionei a nova aba "Lista de Desejo" aqui
    tab1, tab2, tab3, tab4 = st.tabs(["🎒 Minha Party", "🔴 Capturados", "🌟 Lista de Desejo", "👁️ Pokedex (Vistos)"])
    # Garante que existe o dicionário de stats no save
    if "stats" not in user_data:
        user_data["stats"] = {}
        
    with tab1:
        with st.expander("➕ Adicionar Pokémon à Equipe", expanded=False):
            col_add1, col_add2 = st.columns(2)
            with col_add1:
                st.subheader("Da Dex (Apenas Capturados)")
                
                # --- NOVO FILTRO: APENAS CAPTURADOS ---
                # Filtra o dataframe para pegar apenas IDs que estão em user_data['caught']
                caught_ids = [str(c) for c in user_data['caught'] if not str(c).startswith("EXT:")]
                df_caught = df[df['Nº'].astype(str).isin(caught_ids)]
                
                options_all = df_caught.apply(lambda x: f"#{x['Nº']} - {x['Nome']}", axis=1).tolist()
                # --------------------------------------

                current_pc_in_party = [m for m in user_data['party'] if not str(m).startswith("EXT:")]
                current_ext_in_party = [m for m in user_data['party'] if str(m).startswith("EXT:")]
                
                default_names = []
                for pid in current_pc_in_party:
                        res = df[df['Nº'].astype(str) == str(pid)]['Nome'].values
                        if len(res) > 0: default_names.append(f"#{pid} - {res[0]}")
                
                selected_names = st.multiselect("Selecione para Equipe", options=options_all, default=default_names)
                
                # ... (Lógica de salvar mantém igual, pode manter o resto do bloco if set(full_new_party)...)
                new_pc_ids = [n.split(" - ")[0].replace("#", "") for n in selected_names]
                full_new_party = new_pc_ids + current_ext_in_party
                
                if set(full_new_party) != set(user_data['party']) or len(full_new_party) != len(user_data['party']):
                    # (Lógica de adicionar mantém a mesma)
                    user_data['party'] = full_new_party
                    save_data_cloud(trainer_name, user_data)
                    st.rerun()

            # ... (Código da col_add2 mantém igual) ...
            with col_add2:
                st.subheader("Visitante")
                external_name = st.text_input("Nome (ex: Sawsbuck)")
                if st.button("Adicionar"):
                    if external_name:
                        ext_id = f"EXT:{external_name}"
                        user_data['party'].append(ext_id)
                        save_data_cloud(trainer_name, user_data)
                        st.rerun()
        
        st.markdown("---")
        
        if user_data['party']:
            cols = st.columns(3)
            for i, member in enumerate(user_data['party']):
                # Checa se é shiny
                is_shiny = member in user_data.get("shinies", [])

                is_ext = str(member).startswith("EXT:")
                if is_ext:
                    p_name = member.replace("EXT:", "")
                    p_img = get_image_from_name(p_name, api_name_map) # Visitantes sem shiny por enquanto
                    p_subtitle = "Visitante"
                else:
                    p_search = df[df['Nº'].astype(str) == str(member)]
                    if not p_search.empty:
                        r = p_search.iloc[0]
                        p_name = r['Nome']
                        # --- USA A NOVA FUNÇÃO COM SHINY ---
                        p_img = pokemon_pid_to_image(member, mode="artwork", shiny=is_shiny)
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
                        
                        # --- CHECKBOX SHINY ---
                        if not is_ext:
                            # Se marcar, adiciona na lista de shinies. Se desmarcar, remove.
                            shiny_check = st.checkbox("✨ Shiny", value=is_shiny, key=f"shiny_{member}_{i}")
                            if shiny_check != is_shiny:
                                if shiny_check:
                                    if member not in user_data["shinies"]: user_data["shinies"].append(member)
                                else:
                                    if member in user_data["shinies"]: user_data["shinies"].remove(member)
                                save_data_cloud(trainer_name, user_data)
                                st.rerun()
                        # ----------------------

                        st.caption(p_subtitle)
                        
                        # ... (O resto do código de Stats e Notas mantém igual) ...
                        my_stats = user_data["stats"].get(member, {})
                        with st.expander("📊 Ficha de Combate"):
                             # (Código dos inputs Dodge, Parry etc mantém igual)
                             s1, s2 = st.columns(2)
                             d = s1.number_input("Dodge", value=int(my_stats.get("dodge", 0)), key=f"hub_dod_{member}")
                             p = s2.number_input("Parry", value=int(my_stats.get("parry", 0)), key=f"hub_par_{member}")
                             w = s1.number_input("Will", value=int(my_stats.get("will", 0)), key=f"hub_wil_{member}")
                             f = s2.number_input("Fort", value=int(my_stats.get("fort", 0)), key=f"hub_for_{member}")
                             t = st.number_input("THG", value=int(my_stats.get("thg", 0)), key=f"hub_thg_{member}")
                             
                             current_vals = {"dodge": d, "parry": p, "will": w, "fort": f, "thg": t}
                             if current_vals != my_stats:
                                 user_data["stats"][member] = current_vals
                                 
                        nk = f"note_party_{i}_{member}"
                        curr = user_data["notes"].get(nk, "")
                        new = st.text_area("Notas", value=curr, height=60, key=nk)
                        if new != curr:
                            user_data["notes"][nk] = new

    
        else: 
            st.info("Sua equipe está vazia.")
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
        st.header("🌟 Lista de Desejo")
        wishlist = user_data.get("wishlist", [])
        if not wishlist:
            st.info("Sua lista de desejos está vazia. Marque pokémons na aba Pokédex.")
        else:
            # Mostra os pokémons desejados
            for p_id in wishlist:
                # Busca dados no Excel
                p_search = df[df['Nº'].astype(str) == str(p_id)]
                if p_search.empty: continue
                p_row = p_search.iloc[0]
                
                with st.expander(f"🌟 #{p_id} - {p_row['Nome']}"):
                    c1, c2 = st.columns([1, 4])
                    with c1: 
                        # CORREÇÃO: Usa a função inteligente que converte ID Regional -> Imagem Real
                        img_url = pokemon_pid_to_image(p_id, mode="artwork", shiny=False)
                        st.image(img_url, width=100)
                    with c2:
                        st.write(f"**Tipo:** {p_row['Tipo']}")
                        st.write(f"**Região:** {p_row['Região']}")
                        if st.button("Remover da Lista", key=f"rm_wish_{p_id}"):
                            user_data["wishlist"].remove(p_id)
                            save_data_cloud(trainer_name, user_data)
                            st.rerun()

    with tab4:
        total = len(df)
        vistos = len(user_data['seen'])
        st.markdown(f"### Progresso da Pokédex")
        st.progress(min(vistos / total, 1.0))
        st.write(f"**{vistos}** de **{total}** Pokémons registrados.")
        
# PVP ARENA

elif page == "PvP – Arena Tática":
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


# =========================
    # VIEW: BATTLE (CÓDIGO CONSOLIDADO E CORRIGIDO)
    # =========================
    if view == "battle":
        if not rid or not room:
            st.session_state["pvp_view"] = "lobby"
            st.rerun()
            click = None

        if "last_click_processed" not in st.session_state:
            st.session_state["last_click_processed"] = None

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
                        "updatedAt": str(datetime.now())
                    }
            
            if nested_update:
                # Agora o .set(..., merge=True) vai entender a estrutura aninhada corretamente!
                db.collection("rooms").document(rid).collection("public_state").document("party_states").set(nested_update, merge=True)

        # --- 2. CARREGAMENTO DO ESTADO ---
        state = get_state(db, rid)
        seed = state.get("seed")
        tiles_packed = state.get("tilesPacked")
        tiles = unpack_tiles(tiles_packed) if tiles_packed else None
        
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
            
            # VERIFICAÇÃO DE SEGURANÇA:
            # Se for o MEU pokemon e o banco estiver zerado, puxa do meu Hub local
            if t_name == trainer_name:
                # Checa se stats é None, vazio ou só tem zeros
                stats_is_bad = not stats or all(int(v) == 0 for v in stats.values())
                
                if stats_is_bad:
                    # Tenta achar no user_data local (Hub)
                    if "stats" in user_data:
                        local_s = user_data["stats"].get(str(p_id)) or user_data["stats"].get(p_id)
                        if local_s:
                            stats = local_s
            
            return hp, cond, stats, shiny_status
        
        def get_poke_display_name(pid):
            row = df[df['Nº'].astype(str) == str(pid)]
            if not row.empty: return row.iloc[0]['Nome']
            return str(pid)

        # Definição da Função de Renderização da Coluna (DEFINIDA ANTES DE USAR)
        def render_player_column(p_name, p_label, is_me):
            st.markdown(f"### {p_label}")
            
            # Busca party e estado público
            p_doc_data = db.collection("rooms").document(rid).collection("public_state").document("players").get().to_dict() or {}
            party_list = p_doc_data.get(p_name, [])[:8] 
            
            state = get_state(db, rid)
            all_pieces = state.get("pieces") or []
            seen_pids = state.get("seen") or []
            
            if not party_list:
                st.caption("Aguardando...")
                return
        
            p_pieces_on_board = [p for p in all_pieces if p.get("owner") == p_name]
        
            for i, pid in enumerate(party_list):
                cur_hp, cur_cond, cur_stats, is_shiny = get_poke_data(p_name, pid)
                is_on_map = any(str(p["pid"]) == str(pid) for p in p_pieces_on_board)
                already_seen = str(pid) in seen_pids
                
                if cur_hp >= 5: hpi = "💚"
                elif cur_hp >= 3: hpi = "🟡"
                elif cur_hp >= 1: hpi = "🔴"
                else: hpi = "💀"
                
                sprite_url = pokemon_pid_to_image(pid, mode="sprite", shiny=is_shiny)
        
                with st.container(border=True):
                    if is_me:
                        c_img, c_ctrl = st.columns([1, 2.5])
                        with c_img:
                            if cur_hp == 0:
                                st.markdown(f'<img src="{sprite_url}" style="width:100%; filter:grayscale(100%); opacity:0.6;">', unsafe_allow_html=True)
                                st.caption("**FAINTED**")
                            else:
                                st.image(sprite_url, use_container_width=True)
        
                            if is_on_map:
                                p_obj = next((p for p in p_pieces_on_board if str(p["pid"]) == str(pid)), None)
                                if p_obj:
                                    is_rev = p_obj.get("revealed", True)
                                    if st.button("👁️" if is_rev else "✅", key=f"v_{p_name}_{pid}_{i}"):
                                        p_obj["revealed"] = not is_rev
                                        upsert_piece(db, rid, p_obj)
                                        if p_obj["revealed"]: mark_pid_seen(db, rid, pid)
                                        st.rerun()
                                    if st.button("❌", key=f"r_{p_name}_{pid}_{i}"):
                                        delete_piece(db, rid, p_obj["id"])
                                        add_public_event(db, rid, "pokemon_removed", p_name, {"pid": pid})
                                        st.rerun()
                            elif cur_hp > 0:
                                if st.button("📍 Por", key=f"p_{p_name}_{pid}_{i}"):
                                    st.session_state["placing_pid"] = pid
                                    st.session_state["placing_effect"] = None
                                    st.rerun()
                        
                        with c_ctrl:
                            if st.session_state.get("placing_pid") == pid:
                                st.info("Clique no mapa!")
                            else:
                                st.markdown(f"**{hpi} HP: {cur_hp}/6**")
                                # CHAVES E ARGS ATUALIZADOS PARA O CALLBACK
                                st.slider("HP", 0, 6, value=int(cur_hp), 
                                         key=f"hp_{p_name}_{pid}_{i}", 
                                         label_visibility="collapsed", 
                                         on_change=update_poke_state_callback, 
                                         args=(db, rid, p_name, pid, i))
                                
                                st.multiselect("Status", ["⚡", "❄️", "🔥", "💤", "☠️", "💓"], 
                                              default=cur_cond, 
                                              key=f"cond_{p_name}_{pid}_{i}", 
                                              label_visibility="collapsed", 
                                              on_change=update_poke_state_callback, 
                                              args=(db, rid, p_name, pid, i))
        
                    else:
                        # Visão do oponente simplificada
                        piece_obj = next((p for p in p_pieces_on_board if str(p["pid"]) == str(pid)), None)
                        is_revealed = piece_obj.get("revealed", True) if piece_obj else False
                        show_full = (piece_obj and is_revealed) or already_seen
                        
                        status_txt = "(Mochila)" if not piece_obj else ("(Escondido)" if not is_revealed else "")
        
                        if show_full:
                            p_real_name = get_poke_display_name(pid)
                            c1, c2 = st.columns([1, 2])
                            with c1: st.image(sprite_url, width=50)
                            with c2:
                                st.markdown(f"**{p_real_name}**")
                                st.caption(f"{hpi} HP: {cur_hp}/6 {status_txt}")
                                if cur_hp == 0: st.caption("**FAINTED**")
                        else:
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
            # Mantém a correção do ValueError (4 valores) 
            hp_check, _, _, _ = get_poke_data(p.get("owner"), p.get("pid"))
            p["status"] = "fainted" if hp_check == 0 else "active"

            # Lógica de Visibilidade: Dono vê tudo, outros veem apenas revelados 
            if p.get("owner") == trainer_name: 
                pieces_to_draw.append(p)
            elif p.get("revealed", True): 
                pieces_to_draw.append(p)
            
            # Popula o mapa para a calculadora encontrar os alvos corretamente
            if p.get("owner") in player_pieces_map:
                player_pieces_map[p.get("owner")].append(p)

        theme_key = room.get("theme", "cave_water")
        grid = len(tiles) if tiles else 10

        # --- 5. INTERFACE DO TOPO ---
        top = st.columns([1, 1, 1, 1, 4])
        with top[0]:
            if st.button("⬅️ Lobby"):
                st.session_state["pvp_view"] = "lobby"
                st.rerun()
        with top[1]:
            if st.button("🔄 Atualizar"): st.rerun()
        with top[2]:
            if st.button("🎲 d20", disabled=not is_player): roll_die(db, rid, trainer_name, sides=20); st.rerun()
        with top[3]:
            if st.button("🎲 d6", disabled=not is_player): roll_die(db, rid, trainer_name, sides=6); st.rerun()
        with top[4]:
            # ✅ PEDIDO: Mostrar Código da Sala aqui em cima
            st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 10px;'>
        <span style='font-family: "Press Start 2P"; font-size: 18px;'>🏟️ Arena:</span>
        <span style='font-family: "Press Start 2P"; font-size: 24px; color: #FFCC00; background: #333; padding: 5px 10px; border-radius: 5px;'>{rid}</span>
    </div>
    """, unsafe_allow_html=True) 
            
            col_me, col_map, col_opps = st.columns([1.5, 3, 2])
            
            
            # Última rolagem (feedback rápido)
            last_events = list_public_events(db, rid, limit=1)
            last_dice = next((e for e in last_events if e.get("type") == "dice"), None)
            if last_dice:
                pl = last_dice.get("payload", {})
                st.warning(f"🎲 {last_dice.get('by')}: **{pl.get('result')}** (d{pl.get('sides')})")

        # ==========================================
        # 🧮 6. CALCULADORA DE COMBATE
        # ==========================================
        battle_ref = db.collection("rooms").document(rid).collection("public_state").document("battle")
        battle_doc = battle_ref.get()
        b_data = battle_doc.to_dict() or {"status": "idle", "logs": []}
        import math

        with st.expander("⚔️ Calculadora de Combate", expanded=(b_data["status"] != "idle")):
            
            # [FASE 0] IDLE
            if b_data["status"] == "idle":
                if is_player:
                    if st.button("Nova Batalha (Atacar)"):
                        battle_ref.set({"status": "setup", "attacker": trainer_name, "logs": []})
                        st.rerun()
                else:
                    st.caption("Aguardando combate...")
            
            # [FASE 1] CONFIGURAR ATAQUE
            elif b_data["status"] == "setup":
                # CORREÇÃO: Parêntese fechado corretamente aqui
                st.caption(f"**Atacante:** {b_data.get('attacker')}")
                
                if b_data.get("attacker") == trainer_name:
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
                                st.rerun()
                    else:
                        # Normal
                        with c_atk3:
                            atk_type = st.selectbox("Alcance", ["Distância (Dodge)", "Corpo-a-corpo (Parry)"])
                        
                        atk_mod = st.number_input("Acerto (Modificador)", value=0, step=1)
                        
                        if st.button("⚔️ Rolar Ataque"):
                            if target_id:
                                d20 = random.randint(1, 20)
                                t_p = next((p for p in all_pieces if p['id'] == target_id), None)
                                
                                # Pega stats do alvo
                                _, _, t_stats, _ = get_poke_data(t_p['owner'], t_p['pid'])
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
                                    "logs": [f"{trainer_name} rolou {d20}+{atk_mod}=**{total_atk}** (vs Def {needed} [{defense_val}+10])... {result_msg}"]
                                })
                                st.rerun()
                else:
                    st.info(f"Aguardando {b_data.get('attacker')}...")

            # [FASE 1.5] DEFESA DE ÁREA
            elif b_data["status"] == "aoe_defense":
                st.info(b_data["logs"][-1])
                if b_data.get("target_owner") == trainer_name:
                    st.markdown("### 🏃 Rolar Esquiva (Dodge)")
                    if st.button("Rolar Dodge"):
                        d20 = random.randint(1, 20)
                        _, _, t_stats, _ = get_poke_data(trainer_name, b_data.get('target_pid'))
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
                        st.rerun()
                else:
                    st.warning("Aguardando defensor...")

            # [FASE 2] INSERIR DANO (Se acertou)
            elif b_data["status"] == "hit_confirmed":
                st.success(b_data["logs"][-1])
                
                if b_data.get("attacker") == trainer_name:
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        dmg_input = st.number_input("Dano Base / Rank", min_value=0, value=0)
                    with c2:
                        is_eff_check = st.checkbox("É Efeito?", value=False, key=f"norm_eff_{rid}", help="Se marcado, CD base será 10. Se não, 15.")
                    
                    if st.button("Enviar Dano/Efeito"):
                        battle_ref.update({
                            "status": "waiting_defense",
                            "dmg_base": dmg_input,
                            "is_effect": is_eff_check,
                            "logs": firestore.ArrayUnion([f"Rank/Dano: {dmg_input} ({'Efeito' if is_eff_check else 'Dano'}). Aguardando resistência..."])
                        })
                        st.rerun()
                else:
                    st.info("Aguardando atacante definir o dano...")

            elif b_data["status"] == "missed":
                st.error(b_data["logs"][-1])
                if b_data.get("attacker") == trainer_name:
                    if st.button("Encerrar"):
                        battle_ref.update({"status": "idle", "logs": []})
                        st.rerun()

            # [FASE 3] RESISTÊNCIA FINAL
            elif b_data["status"] == "waiting_defense":
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
                        _, _, t_stats, _ = get_poke_data(trainer_name, b_data.get('target_pid')) 
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
                        st.rerun()
                else:
                    st.warning("Aguardando defesa...")

            # [FASE 4] FIM / SECUNDÁRIO
            elif b_data["status"] == "finished":
                st.markdown(f"## 🩸 Resultado: -{b_data.get('final_bars')} Barras")
                for log in b_data.get("logs", []): st.text(log)
                
                if b_data.get("attacker") == trainer_name:
                    c_end1, c_end2 = st.columns(2)
                    if c_end1.button("Encerrar Combate"):
                        battle_ref.update({"status": "idle", "logs": []})
                        st.rerun()
                    
                    target_name = get_poke_display_name(b_data.get('target_pid'))
                    if c_end2.button(f"⚡ Efeito Secundário em {target_name}"):
                        battle_ref.update({
                            "status": "hit_confirmed", 
                            "is_effect": False, 
                            "logs": [f"⚡ Efeito Secundário ativado em {target_name}!"]
                        })
                        st.rerun()
                else:
                    st.info("Aguardando atacante encerrar...")


        # =========================
        # 7. LAYOUT DAS COLUNAS (ME VS OPONENTES)
        # =========================
        if not tiles:
            st.warning("Sem mapa.")
            st.stop()

        # Criamos 3 áreas: Sua Equipe, Mapa, e Oponentes
        c_me, c_map, c_opps = st.columns([1.3, 3, 1.5])

        with c_me:
            # Sua coluna é sempre azul
            render_player_column(trainer_name, "🎒 Sua Equipe (Você)", is_me=True)

        with c_map:
            st.markdown("### 🗺️ Arena")
            can_edit = (trainer_name == "Ezenek" or is_player)
            with st.expander("🛠️ Itens", expanded=False):
                if can_edit:
                    effects_map = {"Fogo":"🔥", "Gelo":"🧊", "Água":"💧", "Rocha":"🪨", "Nuvem":"☁️", "Sol":"☀️", "Terreno Grama":"🍃", "Terreno Eletrico":"⚡"}
                    curr = st.session_state.get("placing_effect")
                    if curr: st.info(f"Item: {curr}")
                    cols = st.columns(6)
                    for i, (k, v) in enumerate(effects_map.items()):
                        if cols[i%6].button(v, key=f"ef_{k}"):
                            st.session_state["placing_effect"] = v if curr != v else None
                            st.session_state["placing_pid"] = None
                            st.rerun()
                    if st.button("Limpar"):
                        db.collection("rooms").document(rid).collection("public_state").document("state").update({"effects": []})
                        st.rerun()

            show_grid = st.checkbox("Mostrar grade tática", value=False, key=f"show_grid_battle_{rid}")

            if "selected_piece_id" not in st.session_state: st.session_state["selected_piece_id"] = None
            img = render_map_with_pieces(
                tiles, theme_key, seed, pieces_to_draw, trainer_name, room, effects=field_effects, show_grid=show_grid
            )
            click = streamlit_image_coordinates(img, key=f"map_{rid}")

            # Lembre-se que na renderização do mapa, a função 'get_perspective_color' 
            # deve ser usada para desenhar a borda da peça.

        with c_opps:
            st.markdown("### 🆚 Oponentes")
            # Lista todos os jogadores que não são VOCÊ
            opponents = sorted(list(set([p for p in all_players if p != trainer_name])))
            
            if not opponents:
                st.caption("Aguardando desafiantes...")
            else:
                for idx, opp_name in enumerate(opponents):
                    # Define o prefixo de cor visual para o rótulo
                    icons = ["🔴", "🟡", "🌸"]
                    icon = icons[idx] if idx < len(icons) else "⚪"
                    
                    render_player_column(opp_name, f"{icon} {opp_name}", is_me=False)

        # =========================
        # 8. LÓGICA DE CLIQUE
        # =========================
        if click and "x" in click and "y" in click:
            col = int(click["x"] // TILE_SIZE)
            row = int(click["y"] // TILE_SIZE)
            if 0 <= row < grid and 0 <= col < grid:
                ppid = st.session_state.get("placing_pid")
                peff = st.session_state.get("placing_effect")
                sel = st.session_state.get("selected_piece_id")

                if peff:
                    curr = state.get("effects") or []
                    new = [e for e in curr if not (int(e["row"])==row and int(e["col"])==col)]
                    new.append({"icon": peff, "row": row, "col": col, "id": str(uuid.uuid4())[:8]})
                    db.collection("rooms").document(rid).collection("public_state").document("state").update({"effects": new})
                    st.session_state["placing_effect"] = None
                    st.rerun()
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
                    st.session_state.pop("placing_pid", None)
                    st.rerun()
                else:
                    s_now = get_state(db, rid)
                    all_p = s_now.get("pieces") or []
                    clicked = find_piece_at(all_p, row, col)
                    if clicked:
                        if clicked["owner"] == trainer_name:
                            if sel == clicked["id"]: st.session_state["selected_piece_id"] = None
                            else: st.session_state["selected_piece_id"] = clicked["id"]
                            st.rerun()
                            
                    elif sel and is_player:
                        mover = next((p for p in all_p if p["id"] == sel), None)
                        if mover:
                            # 1. Guarda a posição antiga para o Log
                            old_pos = [mover["row"], mover["col"]]
                            
                            # 2. Atualiza para a nova posição
                            mover["row"] = row
                            mover["col"] = col
                            
                            # 3. Registra o movimento publicamente NO LOG
                            add_public_event(db, rid, "move", trainer_name, {
                                "pid": mover["pid"],
                                "from": old_pos,
                                "to": [row, col]
                            })
                            
                            # 4. Salva a peça no Firebase
                            upsert_piece(db, rid, mover)
                            
                            # 5. Limpa a seleção e recarrega
                            st.session_state["selected_piece_id"] = None
                            st.rerun()

        # Fora da lógica de clique, mas no final da View Battle
        render_public_log_fragment(db, rid)
        st.stop()
        
    elif view == "lobby":
            # --- MAPA DE NOMES (Para exibição amigável) ---
            THEME_NAMES = {
                "cave_water": "Caverna (com água)",
                "forest": "Floresta",
                "mountain_slopes": "Montanha",
                "plains": "Pradaria",
                "dirt": "Terra Batida",
                "river": "Rio",
                "sea_coast": "Costa Marítima",
                "center_lake": "Lago Central"
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
                inv_themes = {v: k for k, v in THEME_NAMES.items()}
                theme_label = st.selectbox("Tema", list(inv_themes.keys()), index=0)
                theme = inv_themes[theme_label]
            with c3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🆕 Criar arena", type="primary"):
                    rid, err = create_room(db, trainer_name, grid, theme, max_active=5)
                    if err:
                        st.error(err)
                    else:
                        st.success(f"Arena criada! Código: **{rid}**")
                        st.session_state["active_room_id"] = rid
                        st.rerun()
        
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
                            st.rerun()
                    with b2:
                        if st.button("🗄️ Arquivar"):
                            remove_room_from_user(db, trainer_name, chosen_rid)
                            if st.session_state.get("active_room_id") == chosen_rid:
                                st.session_state.pop("active_room_id", None)
                            st.rerun()
        
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
                            st.rerun()
                        elif res == "ALREADY_OWNER":
                            st.warning("Você é o dono desta sala.")
                            st.session_state["active_room_id"] = code.strip()
                            st.rerun()
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
                            st.rerun()
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
                if st.button("🔄 Atualizar"): st.rerun()
            with c_nav2:
                 if st.button("❌ Fechar Prévia"): 
                    st.session_state["active_room_id"] = None
                    st.rerun()
    
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
                    packed = state.get("tilesPacked")
                    tiles = unpack_tiles(packed) if packed else None
                    all_pieces = state.get("pieces") or []
                    pieces = visible_pieces_for(room, trainer_name, all_pieces)
                    
                    role = get_role(room, trainer_name)
                    is_player = role in ["owner", "challenger"]
                    
                    no_water = st.checkbox("🚫 Gerar sem água", value=bool(state.get("noWater", False)), disabled=not is_player)
                    
                    if not tiles:
                        if st.button("🗺️ Gerar mapa (pixel art)", disabled=not is_player):
                            tiles, seed = gen_tiles(grid, theme_key, seed=None, no_water=no_water)
                            packed = pack_tiles(tiles)
                            state_ref.set({
                                "gridSize": grid, "theme": theme_key, "seed": seed, 
                                "tilesPacked": packed, "noWater": bool(no_water),
                                "updatedAt": firestore.SERVER_TIMESTAMP,
                            }, merge=True)
                            st.session_state["pvp_view"] = "battle"
                            st.rerun()
                    else:
                        show_grid = st.checkbox("Mostrar grade tática", value=False, key=f"show_grid_preview_{rid}")
                        img = render_map_with_pieces(tiles, theme_key, seed, pieces, trainer_name, room, show_grid=show_grid)
                        st.image(img, caption="Prévia do Campo")
                        
                        if st.button("⚔️ IR PARA O CAMPO DE BATALHA", type="primary"):
                            st.session_state["pvp_view"] = "battle"
                            st.rerun()
                        
                        if st.button("🔁 Regerar Mapa", disabled=not is_player):
                             tiles, seed = gen_tiles(grid, theme_key, seed=None, no_water=no_water)
                             packed = pack_tiles(tiles)
                             state_ref.set({"seed": seed, "tilesPacked": packed, "noWater": bool(no_water)}, merge=True)
                             st.rerun()
    
            
    
    

    

elif page == "Mochila":
    if "backpack" not in user_data:
        user_data["backpack"] = {
            "money": 0, 
            "medicine": [], 
            "pokeballs": [], 
            "tms": [], 
            "key_items": []
        }

    # Cabeçalho com Dinheiro (AGORA IDENTADO CORRETAMENTE)
    st.markdown(f'<div class="money-display">💰 Dinheiro: ₽ {user_data["backpack"]["money"]}</div>', unsafe_allow_html=True)    

    col_bag, col_content = st.columns([1, 2.5])
    
    with col_bag:
        # Tenta carregar a imagem local; se não existir, usa o link reserva
        try:
            st.image("mochila.png", width=150, caption="MINHA BOLSA")
        except:
            st.image("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/main-stats/adventure-guide.png", width=150)
        
        # Ajuste de Saldo
        new_money = st.number_input("Editar Saldo", value=int(user_data["backpack"]["money"]), step=100)
        if new_money != user_data["backpack"]["money"]:
            user_data["backpack"]["money"] = new_money
            save_data_cloud(trainer_name, user_data) 
        
        if st.button("🧹 Limpar Vazios"):
            for k in ["medicine", "pokeballs", "tms", "key_items"]:
                user_data["backpack"][k] = [i for i in user_data["backpack"][k] if i["name"] and i.get("qty", 0) > 0]
            save_data_cloud(trainer_name, user_data) 
            st.rerun()

    with col_content:
        tabs = st.tabs(["💊 Med", "🔴 Pokéballs", "💿 TMs", "🔑 Chave"])
        cfg = [(tabs[0], "medicine", True), (tabs[1], "pokeballs", True), 
               (tabs[2], "tms", True), (tabs[3], "key_items", False)]

        for tab, key, use_img in cfg:
            with tab:
                current_list = user_data["backpack"].get(key, [])
                updated_items = []
                
                # Exibe itens atuais + linha extra para novo item
                for i in range(len(current_list) + 1):
                    item_data = current_list[i] if i < len(current_list) else {"name": "", "qty": 0}
                    res = render_item_row(key, i, item_data, show_image=use_img)
                    if res["name"]:
                        updated_items.append(res)
                
                if st.button(f"💾 Salvar {key.title()}", key=f"sv_{key}"):
                    user_data["backpack"][key] = updated_items
                    save_data_cloud(trainer_name, user_data) 
                    st.success("Bolsa Atualizada!")
                    st.rerun()





































