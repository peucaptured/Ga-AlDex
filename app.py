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
import random

def roll_die(db, rid: str, by: str, sides: int = 20):
    result = random.randint(1, int(sides))
    add_public_event(db, rid, "dice", by, {"sides": int(sides), "result": int(result)})
    return result
    
def get_role(room: dict, trainer_name: str) -> str:
    owner = (room.get("owner") or {}).get("name")
    chal = room.get("challenger") or {}
    chal_name = chal.get("name") if isinstance(chal, dict) else (chal or None)

    if trainer_name == owner:
        return "owner"
    if trainer_name == chal_name:
        return "challenger"
    return "spectator"

def safe_doc_id(name: str) -> str:
    # Evita caracteres problemáticos no Firestore doc id
    if not isinstance(name, str):
        name = str(name)
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name).strip("_")[:80] or "user"

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
# --- FUNÇÃO DE CALLBACK PARA SLIDERS E STATUS ---
# --- FUNÇÃO DE CALLBACK CORRIGIDA ---
def update_poke_state_callback(db, rid, trainer_name, pid):
    # 1. Pega os valores novos dos widgets
    # Usamos session_state direto para garantir que pegamos o que o usuário acabou de mexer
    new_hp = st.session_state.get(f"hp_{pid}")
    new_cond = st.session_state.get(f"cond_{pid}")
    
    if new_hp is None or new_cond is None:
        return # Evita erro se o widget não carregou

    # 2. Salva no Firestore com a ESTRUTURA CORRETA (Aninhada)
    # Antes estava salvando "Nome.ID", agora salva { "Nome": { "ID": ... } }
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
    # merge=True garante que não apaga os dados dos outros pokémons
    ref.set(data, merge=True)
    
    # 3. Atualiza status de Fainted na lista de peças (se estiver no mapa)
    if new_hp == 0:
        # Lógica rápida para atualizar o mapa sem precisar de outra leitura pesada
        state_ref = db.collection("rooms").document(rid).collection("public_state").document("state")
        stt = state_ref.get().to_dict() or {}
        pieces = stt.get("pieces") or []
        dirty = False
        for p in pieces:
            if p.get("owner") == trainer_name and str(p.get("pid")) == str(pid):
                if p.get("status") != "fainted":
                    p["status"] = "fainted"
                    dirty = True
        if dirty:
            state_ref.update({"pieces": pieces})
            
    elif new_hp > 0:
        # Revive se curou
        state_ref = db.collection("rooms").document(rid).collection("public_state").document("state")
        stt = state_ref.get().to_dict() or {}
        pieces = stt.get("pieces") or []
        dirty = False
        for p in pieces:
            if p.get("owner") == trainer_name and str(p.get("pid")) == str(pid):
                if p.get("status") == "fainted":
                    p["status"] = "active"
                    dirty = True
        if dirty:
            state_ref.update({"pieces": pieces})

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

def join_room_as_challenger(db, rid: str, trainer_name: str):
    ref = db.collection("rooms").document(rid)
    doc = ref.get()
    if not doc.exists:
        return "NOT_FOUND"

    data = doc.to_dict() or {}
    owner = (data.get("owner") or {}).get("name")
    challenger = (data.get("challenger") or {}).get("name") if isinstance(data.get("challenger"), dict) else data.get("challenger")

    if owner == trainer_name:
        add_room_to_user(db, trainer_name, rid)
        return "ALREADY_OWNER"

    if not challenger:
        ref.update({
            "challenger": {"name": trainer_name},
            "status": "running",
        })
        add_room_to_user(db, trainer_name, rid)
        # evento público
        ref.collection("public_events").add({
            "type": "join_challenger",
            "by": trainer_name,
            "payload": {"room": rid},
            "ts": firestore.SERVER_TIMESTAMP,
        })
        return "OK"

    if challenger == trainer_name:
        add_room_to_user(db, trainer_name, rid)
        return "ALREADY_CHALLENGER"

    return "CHALLENGER_TAKEN"

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
    if seed is None:
        seed = random.randint(1, 999999999)

    rng = random.Random(seed)
    theme = THEMES.get(theme_key, THEMES["cave_water"])
    base = theme["base"]
    border = theme["border"]

    tiles = [[base for _ in range(grid)] for _ in range(grid)]

    # bordas sólidas
    for r in range(grid):
        tiles[r][0] = border
        tiles[r][grid - 1] = border
    for c in range(grid):
        tiles[0][c] = border
        tiles[grid - 1][c] = border

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

def draw_tile(draw: ImageDraw.ImageDraw, x: int, y: int, t: str, rng: random.Random):
    colors = {
        "rock": (60, 60, 70),
        "wall": (45, 46, 50),
        "water": (35, 90, 140),
        "stalagmite": (90, 92, 98),
        "grass": (60, 130, 70),
        "tree": (30, 70, 35),
        "path": (120, 95, 60),
        "stone": (125, 125, 140),
        "peak": (175, 175, 190),
        "slope1": (110, 110, 125),
        "slope2": (95, 95, 110),
        "flower": (150, 80, 110),
        "trail": (105, 85, 55),
        "dirt": (110, 85, 55),
        "rut": (85, 65, 40),
        "sea": (20, 60, 120),
        "sand": (180, 165, 120),
        "bush": (40, 95, 50),        
    }
    base = colors.get(t, (200, 0, 200))

    draw.rectangle([x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1], fill=base)

    for _ in range(18):
        px = x + rng.randint(0, TILE_SIZE - 1)
        py = y + rng.randint(0, TILE_SIZE - 1)
        tweak = rng.randint(-10, 10)
        c = (max(0, min(255, base[0] + tweak)),
             max(0, min(255, base[1] + tweak)),
             max(0, min(255, base[2] + tweak)))
        draw.point((px, py), fill=c)

    if t in ["water", "sea"]:
        for _ in range(6):
            px = x + rng.randint(2, TILE_SIZE - 3)
            py = y + rng.randint(2, TILE_SIZE - 3)
            draw.point((px, py), fill=(210, 230, 255))

    if t == "tree":
        draw.rectangle([x + 10, y + 8, x + 21, y + 24], fill=(20, 55, 25))
        draw.rectangle([x + 14, y + 24, x + 17, y + 30], fill=(80, 60, 35))

    if t == "stalagmite":
        draw.polygon([(x + 16, y + 6), (x + 8, y + 26), (x + 24, y + 26)], fill=(120, 120, 125))

    if t == "path":
        draw.rectangle([x + 2, y + 12, x + TILE_SIZE - 3, y + 20], fill=(150, 120, 80))
        

# ==========================================
# 🛠️ FUNÇÕES DE DESENHO E AUXILIARES (Cole isso abaixo de draw_tile)
# ==========================================

@st.cache_data(show_spinner=False)
def render_map_png(tiles: list[list[str]], theme_key: str, seed: int):
    """Gera a imagem base do terreno (chão, pedras, árvores) com cache para não piscar."""
    grid = len(tiles)
    img = Image.new("RGB", (grid * TILE_SIZE, grid * TILE_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Usa seed fixa para as texturas não ficarem dançando
    rng = random.Random(int(seed or 0) + 1337)

    for r in range(grid):
        for c in range(grid):
            x = c * TILE_SIZE
            y = r * TILE_SIZE
            draw_tile(draw, x, y, tiles[r][c], rng)

    # Linhas da grade
    for r in range(grid + 1):
        y = r * TILE_SIZE
        draw.line([(0, y), (grid * TILE_SIZE, y)], fill=(0, 0, 0))
    for c in range(grid + 1):
        x = c * TILE_SIZE
        draw.line([(x, 0), (x, grid * TILE_SIZE)], fill=(0, 0, 0))

    return img

@st.cache_data(show_spinner=False)
def fetch_image_pil(url: str) -> Image.Image | None:
    """Baixa e converte a imagem do Pokémon (Cacheada)."""
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGBA")
        return img
    except Exception:
        return None

def render_map_with_pieces(tiles, theme_key, seed, pieces, viewer_name: str, effects=None):
    """Desenha Terreno + Efeitos + Pokémons."""
    # 1. Base do Mapa
    img = render_map_png(tiles, theme_key, seed).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    # 2. Camada de Efeitos (Itens/Terrenos)
    if effects:
        try:
            # Tenta carregar fonte arial ou padrão do sistema
            font_path = "arial.ttf" 
            if os.name == 'posix': font_path = "DejaVuSans.ttf"
            font = ImageFont.truetype(font_path, size=int(TILE_SIZE * 0.8))
        except:
            font = ImageFont.load_default()

        for eff in effects:
            try:
                r, c = int(eff.get("row")), int(eff.get("col"))
                icon = eff.get("icon", "?")
                x = c * TILE_SIZE
                y = r * TILE_SIZE
                # Centraliza o emoji
                draw.text((x + 4, y + 2), icon, fill="white", font=font)
            except:
                continue

    # 3. Camada de Peças (Pokémons)
    local_cache = {}
    
    for p in pieces or []:
        r = int(p.get("row", -1))
        c = int(p.get("col", -1))
        if r < 0 or c < 0: continue

        # Borda Colorida (Identificação)
        owner = p.get("owner")
        if owner == viewer_name:
            border_color = (0, 255, 255) # Ciano (Você)
        else:
            border_color = (255, 50, 50) # Vermelho (Inimigo)

        x = c * TILE_SIZE
        y = r * TILE_SIZE
        draw.rectangle([x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1], outline=border_color, width=3)

        # Imagem do Pokémon
        pid = str(p.get("pid", ""))
        url = pokemon_pid_to_image(pid, mode="sprite")

        if url not in local_cache:
            local_cache[url] = fetch_image_pil(url)

        sprite = local_cache[url]
        if sprite is None: continue

        # Redimensiona e Cola
        sp = sprite.copy()
        sp.thumbnail((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
        
        # Centraliza na célula
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
    
def get_pokemon_image_url(user_name: str, name_map: dict, mode: str = "artwork") -> str:
    p_id = get_pid_from_name(user_name, name_map)
    if not p_id:
        return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"

    if mode == "sprite":
        return get_pokemon_sprite_url(p_id)
    return get_pokemon_artwork_url(p_id)

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
        
def get_pokemon_artwork_url(p_id: str) -> str:
    n = int(str(p_id).lstrip("0") or "0")
    # grande (fora do PvP)
    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p_id}.png"

def get_pokemon_sprite_url(p_id: str) -> str:
    n = int(str(p_id).lstrip("0") or "0")
    # pequeno (PvP)
    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{p_id}.png"

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
def pokemon_pid_to_image(pid: str, mode: str = "artwork") -> str:
    if not pid:
        return "https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
    pid_str = str(pid).strip()
    # 1. Se for Visitante (EXT), usa o nome direto
    if pid_str.startswith("EXT:"):
        name = pid_str.replace("EXT:", "")
        return get_pokemon_image_url(name, api_name_map, mode=mode)
    # 2. Busca no EXCEL pelo ID Customizado
    # Garante que estamos comparando string com string
    row = df[df["Nº"].astype(str) == pid_str]
 
    if not row.empty:
        # Achou no Excel! Pega o nome (ex: "Nidoran")
        name = row.iloc[0]["Nome"]
        # Passa o NOME para a função que busca a URL correta na API
        return get_pokemon_image_url(name, api_name_map, mode=mode)

    # 3. Fallback: Se não achou no Excel, tenta pelo ID direto (caso raro) ou retorna pokébola
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
page = st.sidebar.radio("Ir para:", ["Pokédex (Busca)", "Trainer Hub (Meus Pokémons)", "PvP – Arena Tática"])


# ==============================================================================
# PÁGINA 1: POKEDEX (COM FILTRO DE TIPO EXCLUSIVO/COMBINADO)
# ==============================================================================
if page == "Pokédex (Busca)":
    st.sidebar.header("🔍 Filtros")
    search_query = st.sidebar.text_input("Buscar (Nome ou Nº)", "")
    
    # 1. FILTRO DE REGIÃO
    all_regions = sorted(list(set([r.strip() for region in df['Região'].unique() for r in region.split('/')])))
    selected_regions = st.sidebar.multiselect("Região", all_regions)
    
    # 2. FILTRO DE BIOMA (Cascata)
    if selected_regions:
        df_for_biomes = df[df['Região'].apply(lambda x: any(reg in x for reg in selected_regions))]
        raw_biomes = df_for_biomes['Biomas'].unique()
    else:
        raw_biomes = df['Biomas'].unique()
        
    all_biomes = sorted(list(set([b.strip() for biome in raw_biomes for b in str(biome).split('/')])))
    biomes_clean = [b for b in all_biomes if "toda" not in b.lower() and "ga" not in b.lower()]
    selected_biomes = st.sidebar.multiselect("Bioma", biomes_clean)

    # 3. FILTRO DE TIPO (COMBINADO)
    all_types = sorted(list(set([t.strip() for t_str in df['Tipo'].unique() for t in str(t_str).split('/')])))
    selected_types = st.sidebar.multiselect("Tipo Elementar (Combinação)", all_types)
    
    # 4. Outros Filtros
    min_p, max_p = int(df['Nivel_Poder'].min()), int(df['Nivel_Poder'].max())
    power_range = st.sidebar.slider("⚡ Nível de Poder", min_p, max_p, (min_p, max_p))
    
    st.sidebar.subheader("⚔️ Estratégia")
    sel_func = st.sidebar.selectbox("Função", ["Todos", "C - Controlador", "F - Finalizador", "S - Suporte"])
    sel_style = st.sidebar.selectbox("Estilo", ["Todos", "O - Ofensivo", "D - Defensivo", "F - Furtivo", "I - Incompleto", "C - Completo"])
    sel_speed = st.sidebar.selectbox("Velocidade", ["Todos", "R - Rápido", "L - Lento"])
    
    l1 = sel_func[0] if sel_func != "Todos" else ""
    l2 = sel_style[0] if sel_style != "Todos" else ""
    l3 = sel_speed[0] if sel_speed != "Todos" else ""

    # --- APLICAÇÃO DOS FILTROS ---
    filtered_df = df.copy()
    
    if search_query:
        filtered_df = filtered_df[filtered_df['Nome'].str.contains(search_query, case=False, na=False) | filtered_df['Nº'].str.contains(search_query, case=False, na=False)]
    
    if selected_regions:
        filtered_df = filtered_df[filtered_df['Região'].apply(lambda x: any(region in x for region in selected_regions))]
    
    if selected_biomes:
        filtered_df = filtered_df[filtered_df['Biomas'].apply(lambda x: ("toda" in str(x).lower() and "ga" in str(x).lower()) or any(b in x for b in selected_biomes))]

    # --- AQUI MUDOU: LÓGICA 'AND' (E) ---
    if selected_types:
        # Usa 'all' em vez de 'any'. O Pokemon precisa ter TODOS os tipos selecionados.
        filtered_df = filtered_df[filtered_df['Tipo'].apply(lambda x: all(t in str(x) for t in selected_types))]
    
    filtered_df = filtered_df[
        (filtered_df['Nivel_Poder'] >= power_range[0]) & 
        (filtered_df['Nivel_Poder'] <= power_range[1])
    ]
    
    if l1 or l2 or l3:
            filtered_df = filtered_df[filtered_df['Codigos_Estrategia'].apply(lambda codes: any(((not l1 or c[0]==l1) and (not l2 or c[1]==l2) and (not l3 or c[2]==l3)) for c in codes))]

    # --- EXIBIÇÃO ---
    st.title("📕 Pokédex Universal")
    st.markdown(f"**Resultados:** {len(filtered_df)}")
    
    if filtered_df.empty: 
        st.warning("Nenhum Pokémon encontrado.")
        if len(selected_types) > 1:
            st.caption("Dica: Você selecionou múltiplos tipos. O sistema está buscando Pokémons que tenham **TODOS** esses tipos simultaneamente.")

    for index, row in filtered_df.iterrows():
        dex_num = row['Nº']
        p_name = row['Nome']
        img_url = get_image_from_name(p_name, api_name_map)
        power = row['Nivel_Poder']
        
        # Chaves Únicas
        key_seen = f"seen_{dex_num}_{index}"
        key_caught = f"caught_{dex_num}_{index}"
        
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
                
                if st.checkbox("👁️ Visto", value=is_seen, key=key_seen):
                    if dex_num not in user_data["seen"]:
                        user_data["seen"].append(dex_num)
                        save_data_cloud(trainer_name, user_data)
                else:
                    if dex_num in user_data["seen"]:
                        user_data["seen"].remove(dex_num)
                        save_data_cloud(trainer_name, user_data)

                if st.checkbox("🔴 Capturado", value=is_caught, key=key_caught):
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
        
# PVP ARENA

elif page == "PvP – Arena Tática":
    st.title("⚔️ PvP – Arena Tática (MVP)")
    st.caption(
        "Base multiplayer: criar/abrir arena, entrar por código, espectadores "
        "e log público (dado visível para todos)."
    )

    db, bucket = init_firebase()
    view = st.session_state.get("pvp_view", "lobby")
    rid = st.session_state.get("active_room_id")  # ✅ sempre existe ou None
    room = get_room(db, rid) if rid else None     # ✅ evita NameError
    role = get_role(room, trainer_name) if room else "spectator"
    is_player = role in ["owner", "challenger"]



# =========================
    # VIEW: BATTLE (Versão "Batch Edit" - Múltiplos Cliques)
    # =========================
    if view == "battle":
        if not rid or not room:
            st.session_state["pvp_view"] = "lobby"
            st.rerun()

        # Inicia lista temporária de edição se não existir
        if "pending_effects" not in st.session_state:
            st.session_state["pending_effects"] = []

        # Sync Party
        current_party = user_data.get("party") or []
        db.collection("rooms").document(rid).collection("public_state").document("players").set(
            {trainer_name: current_party}, merge=True
        )

        # Carrega dados
        state = get_state(db, rid)
        seed = state.get("seed")
        tiles_packed = state.get("tilesPacked")
        tiles = unpack_tiles(tiles_packed) if tiles_packed else None
        
        all_pieces = state.get("pieces") or []
        seen_pids = state.get("seen") or []
        
        # --- COMBINA EFEITOS DO BANCO COM OS TEMPORÁRIOS ---
        db_effects = state.get("effects") or []
        # A gente soma as listas para mostrar na tela o que já tem + o que você está colocando agora
        visual_effects = db_effects + st.session_state["pending_effects"]

        # Helper HP
        ps_doc = db.collection("rooms").document(rid).collection("public_state").document("party_states").get()
        party_states_data = ps_doc.to_dict() or {}
        def get_poke_data(t_name, p_id):
            user_dict = party_states_data.get(t_name, {})
            p_data = user_dict.get(str(p_id), {})
            return p_data.get("hp", 6), p_data.get("cond", [])

        # Filtro de Desenho
        pieces_to_draw = []
        for p in all_pieces:
            hp_check, _ = get_poke_data(p.get("owner"), p.get("pid"))
            if hp_check == 0: p["status"] = "fainted"
            else: p["status"] = "active"

            if p.get("owner") == trainer_name:
                pieces_to_draw.append(p)
            elif p.get("revealed", True):
                pieces_to_draw.append(p)
        
        # Nomes
        owner_name = (room.get("owner") or {}).get("name", "Host")
        chal_name = (room.get("challenger") or {}).get("name", "Desafiante")
        
        if trainer_name == owner_name:
            my_label = f"🎒 {owner_name} (Você)"
            opp_label = f"🆚 {chal_name}" if chal_name else "🆚 Aguardando..."
            opp_name = chal_name
        elif trainer_name == chal_name:
            my_label = f"🎒 {chal_name} (Você)"
            opp_label = f"🆚 {owner_name}"
            opp_name = owner_name
        else:
            my_label = "🎒 Jogador 1"
            opp_label = "🆚 Jogador 2"
            opp_name = None

        theme_key = room.get("theme", "cave_water")
        grid = len(tiles) if tiles else 10 

        st.markdown("""
        <style>
          .block-container { max-width: 98% !important; padding-top: 1rem; padding-bottom: 5rem; }
          header { visibility: hidden; height: 0px; }
          .stSlider { padding-top: 0px; padding-bottom: 0px; margin-bottom: -15px; }
          .stMultiSelect { padding-bottom: 0px; }
        </style>
        """, unsafe_allow_html=True)

        # --- MENU SUPERIOR ---
        top = st.columns([1, 1, 1, 1, 4])
        with top[0]:
            if st.button("⬅️ Lobby"):
                st.session_state["pvp_view"] = "lobby"
                st.rerun()
        with top[1]:
            if st.button("🔄 Atualizar"):
                st.session_state["pending_effects"] = [] # Limpa pendências ao atualizar forçado
                st.rerun()
        with top[2]:
            if st.button("🎲 d20", disabled=not is_player):
                roll_die(db, rid, trainer_name, sides=20)
                st.rerun()
        with top[3]:
            if st.button("🎲 d6", disabled=not is_player):
                roll_die(db, rid, trainer_name, sides=6)
                st.rerun()
        with top[4]:
            last_events = list_public_events(db, rid, limit=5)
            last_dice = next((e for e in last_events if e.get("type") == "dice"), None)
            if last_dice:
                pl = last_dice.get("payload", {})
                st.info(f"🎲 **{last_dice.get('by')}** rolou **{pl.get('result')}** (d{pl.get('sides')})")

        with st.expander("📜 Histórico", expanded=False):
            events = list_public_events(db, rid, limit=25)
            for ev in events:
                icon = "🎲" if ev['type'] == 'dice' else "⚔️"
                st.write(f"{icon} **{ev.get('by')}**: {ev.get('type')} {ev.get('payload')}")

        if not tiles:
            st.warning("O mapa ainda não foi gerado.")
            st.stop()

        # =========================
        # LAYOUT
        # =========================
        c_bag, c_map, c_opp = st.columns([1.3, 3, 1.3])

        # --- 1. VOCÊ ---
        with c_bag:
            st.markdown(f"### {my_label}")
            party = user_data.get("party") or []
            party = party[:8] 
            
            my_pieces_on_board = [p for p in all_pieces if p.get("owner") == trainer_name]
            my_pids_on_board = {p["pid"] for p in my_pieces_on_board}
            
            for pid in party:
                is_on_map = pid in my_pids_on_board
                cur_hp, cur_cond = get_poke_data(trainer_name, pid)
                
                if cur_hp >= 5: hp_icon = "💚"
                elif cur_hp >= 3: hp_icon = "🟡"
                elif cur_hp >= 1: hp_icon = "🔴"
                else: hp_icon = "💀"

                sprite_url = pokemon_pid_to_image(pid, mode="sprite")
                
                with st.container(border=True):
                    c_img, c_ctrl = st.columns([1, 2.5])
                    with c_img:
                        if cur_hp == 0:
                            st.markdown(f'<img src="{sprite_url}" style="width:100%; filter:grayscale(100%); opacity:0.6;">', unsafe_allow_html=True)
                            st.caption("**FAINTED**")
                        else:
                            st.image(sprite_url, use_container_width=True)

                        if is_on_map:
                            piece_obj = next((p for p in my_pieces_on_board if p["pid"] == pid), None)
                            if piece_obj:
                                is_rev = piece_obj.get("revealed", True)
                                b_txt = "👁️" if is_rev else "✅"
                                if st.button(b_txt, key=f"v_{pid}"):
                                    piece_obj["revealed"] = not is_rev
                                    upsert_piece(db, rid, piece_obj)
                                    if piece_obj["revealed"]: mark_pid_seen(db, rid, pid)
                                    st.rerun()
                                if st.button("❌", key=f"r_{pid}"):
                                    delete_piece(db, rid, piece_obj["id"])
                                    add_public_event(db, rid, "pokemon_removed", trainer_name, {"pid": pid})
                                    st.rerun()
                        else:
                            if cur_hp > 0:
                                if st.button("📍 Por", key=f"p_{pid}"):
                                    st.session_state["placing_pid"] = pid
                                    st.session_state["placing_effect"] = None 
                                    st.rerun()

                    with c_ctrl:
                        st.markdown(f"**{hp_icon} HP: {cur_hp}/6**")
                        st.slider("HP", 0, 6, value=int(cur_hp), key=f"hp_{pid}", label_visibility="collapsed", on_change=update_poke_state_callback, args=(db, rid, trainer_name, pid))
                        options = ["⚡", "❄️", "🔥", "💤", "☠️", "💓"]
                        st.multiselect("Status", options, default=cur_cond, key=f"cond_{pid}", label_visibility="collapsed", placeholder="Status...", on_change=update_poke_state_callback, args=(db, rid, trainer_name, pid))

        # --- 2. MAPA ---
        with c_map:
            st.markdown("### 🗺️ Arena")
            
            can_edit_map = (trainer_name == "Ezenek" or is_player)
            
            with st.expander("🛠️ Itens e Terrenos", expanded=False):
                if can_edit_map:
                    effects_map = {
                        "Fogo": "🔥", "Gelo": "🧊", "Água": "💧", "Rocha": "🪨",
                        "Nuvem": "☁️", "Sol": "☀️", "Floco": "❄️", "Folha": "🍃",
                        "Fada": "✨", "Raio": "⚡", "Colher": "🥄", "Deserto": "🌵"
                    }
                    
                    # Seletor de Item Atual
                    curr_eff = st.session_state.get("placing_effect")
                    if curr_eff:
                        st.info(f"🔹 Item selecionado: {curr_eff} (Clique no mapa para adicionar)")
                    else:
                        st.caption("Selecione um item abaixo:")

                    cols_eff = st.columns(6)
                    for i, (name, icon) in enumerate(effects_map.items()):
                        with cols_eff[i % 6]:
                            # Se este for o item ativo, destaca
                            is_active = (icon == curr_eff)
                            type_btn = "primary" if is_active else "secondary"
                            
                            if st.button(f"{icon}", key=f"eff_{name}", type=type_btn, help=f"Colocar {name}"):
                                st.session_state["placing_effect"] = icon
                                st.session_state["placing_pid"] = None 
                                st.rerun()
                    
                    st.write("")
                    # --- ÁREA DE CONTROLE DE EDIÇÃO ---
                    col_save, col_undo, col_clear = st.columns(3)
                    
                    # Botão para Commit (Salvar no DB)
                    pending_count = len(st.session_state["pending_effects"])
                    with col_save:
                        if pending_count > 0:
                            if st.button(f"💾 Salvar ({pending_count})", type="primary"):
                                final_list = db_effects + st.session_state["pending_effects"]
                                db.collection("rooms").document(rid).collection("public_state").document("state").update({"effects": final_list})
                                st.session_state["pending_effects"] = [] # Limpa a fila
                                st.toast("Alterações salvas!", icon="✅")
                                st.rerun()
                        else:
                            st.button("💾 Salvar", disabled=True)
                            
                    with col_undo:
                         if pending_count > 0:
                             if st.button("↩️ Desfazer"):
                                 st.session_state["pending_effects"].pop()
                                 st.rerun()

                    with col_clear:
                        if st.button("🧹 Limpar Tudo"):
                            db.collection("rooms").document(rid).collection("public_state").document("state").update({"effects": []})
                            st.session_state["pending_effects"] = []
                            st.rerun()
                else:
                    st.caption("Apenas Jogadores e Mestre podem alterar o terreno.")

            if "selected_piece_id" not in st.session_state:
                st.session_state["selected_piece_id"] = None

            # Renderiza Mapa (Passamos Visual Effects = DB + Pending)
            img = render_map_with_pieces(tiles, theme_key, seed, pieces_to_draw, trainer_name, effects=visual_effects)
            click = streamlit_image_coordinates(img, key=f"battle_map_{rid}")

        # --- 3. INIMIGO ---
        with c_opp:
            st.markdown(f"### {opp_label}")
            players_doc = db.collection("rooms").document(rid).collection("public_state").document("players").get()
            players_data = players_doc.to_dict() or {}
            opp_party_list = players_data.get(opp_name, []) if opp_name else []
            opp_pieces_on_board = [p for p in all_pieces if p.get("owner") == opp_name]
            
            if not opp_party_list:
                st.caption("Aguardando...")
            
            temp_board_pieces = list(opp_pieces_on_board)
            for pid in opp_party_list:
                found_on_board = None
                for i, p in enumerate(temp_board_pieces):
                    if str(p.get("pid")) == str(pid):
                        found_on_board = p
                        del temp_board_pieces[i]
                        break
                
                opp_hp, opp_cond = get_poke_data(opp_name, pid)
                already_seen = str(pid) in seen_pids
                
                show_full = False
                status_msg = ""
                if found_on_board:
                    if found_on_board.get("revealed", True): show_full = True
                    elif already_seen: 
                        show_full = True
                        status_msg = "(Escondido)"
                    else:
                        show_full = False
                        status_msg = "❓ Oculto"
                elif already_seen:
                    show_full = True
                    status_msg = "(Mochila)"
                else:
                    show_full = False
                    status_msg = "(Mochila)"

                with st.container(border=True):
                    if show_full:
                        if opp_hp >= 5: hpi = "💚"
                        elif opp_hp >= 3: hpi = "🟡"
                        elif opp_hp >= 1: hpi = "🔴"
                        else: hpi = "💀"
                        
                        cols = st.columns([1, 2])
                        with cols[0]:
                            url = pokemon_pid_to_image(pid, mode="sprite")
                            st.image(url, width=50)
                        with cols[1]:
                            p_name = pid
                            row_p = df[df["Nº"].astype(str) == str(pid)]
                            if not row_p.empty: p_name = row_p.iloc[0]["Nome"]
                            st.markdown(f"**{p_name}**")
                            if status_msg: st.caption(status_msg)
                            st.markdown(f"{hpi} HP: {opp_hp}/6")
                            if opp_cond: st.caption(" ".join(opp_cond))
                            if opp_hp == 0: st.caption("**FAINTED**")
                    else:
                        st.image("https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg", width=40)
                        st.caption("Desconhecido")

        # =========================
        # CLIQUE
        # =========================
        if click and "x" in click and "y" in click:
            col = int(click["x"] // TILE_SIZE)
            row = int(click["y"] // TILE_SIZE)
            
            if 0 <= row < grid and 0 <= col < grid:
                
                placing_pid = st.session_state.get("placing_pid")
                placing_effect = st.session_state.get("placing_effect")
                sel = st.session_state.get("selected_piece_id")

                # --- 1. COLOCAR EFEITO (Modo Rápido / Batch) ---
                if placing_effect:
                    # Adiciona à lista temporária (Local)
                    st.session_state["pending_effects"].append({
                        "icon": placing_effect,
                        "row": row,
                        "col": col,
                        "id": str(uuid.uuid4())[:8]
                    })
                    # Rerun rápido (apenas local, não vai no DB ainda)
                    st.rerun()

                # --- 2. COLOCAR POKEMON ---
                elif placing_pid:
                    php, _ = get_poke_data(trainer_name, placing_pid)
                    if php == 0:
                        st.toast("Pokémon desmaiado!", icon="💀")
                        st.session_state.pop("placing_pid", None)
                        st.rerun()

                    state_now = get_state(db, rid)
                    current_all = state_now.get("pieces") or []
                    if find_piece_at(current_all, row, col):
                        st.toast("Ocupado!", icon="🚫")
                    else:
                        new_id = str(uuid.uuid4())[:8]
                        new_piece = {
                            "id": new_id, "pid": placing_pid, "owner": trainer_name,
                            "row": row, "col": col, "revealed": True, "status": "active"
                        }
                        upsert_piece(db, rid, new_piece)
                        mark_pid_seen(db, rid, placing_pid)
                        add_public_event(db, rid, "pokemon_placed", trainer_name, {"pid": placing_pid})
                        st.session_state.pop("placing_pid", None)
                        st.rerun()

                # --- 3. MOVER ---
                else:
                    state_now = get_state(db, rid)
                    current_all = state_now.get("pieces") or []
                    clicked_piece = find_piece_at(current_all, row, col)
            
                    if clicked_piece is not None:
                        if clicked_piece.get("owner") != trainer_name:
                            st.toast("Essa peça não é sua.", icon="🔒")
                        else:
                            pid_clk = clicked_piece.get("id")
                            if sel == pid_clk:
                                st.session_state.pop("selected_piece_id", None)
                                st.toast("Cancelado.")
                            else:
                                st.session_state["selected_piece_id"] = pid_clk
                                st.toast("Selecionado!", icon="✅")
                        st.rerun()
                    else:
                        if not is_player: st.warning("Espectador não move.")
                        elif not sel: st.toast("Selecione um Pokémon.", icon="👆")
                        else:
                            moving = next((p for p in current_all if p["id"] == sel), None)
                            if moving is None:
                                st.session_state.pop("selected_piece_id", None)
                                st.rerun()
                            if find_piece_at(current_all, row, col):
                                st.toast("Ocupado.", icon="🚫")
                            else:
                                mhp, _ = get_poke_data(trainer_name, moving.get("pid"))
                                if mhp == 0: st.toast("Desmaiado não anda!", icon="💀")
                                else:
                                    moving["row"] = int(row); moving["col"] = int(col)
                                    upsert_piece(db, rid, moving)
                                    add_public_event(db, rid, "moved", trainer_name, {"pid": moving.get("pid"), "to": [row, col]})
                                    st.session_state.pop("selected_piece_id", None)
                                    st.toast("Movido!", icon="💨")
                                    st.rerun()

        st.stop()
        
    elif view == "lobby":
        # --- Painel: criar arena ---
        st.subheader("➕ Criar nova arena")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            grid = st.selectbox("Tamanho do grid", [6, 8, 10], index=0)
        with c2:
            theme_label = st.selectbox(
                "Tema",
                [
                    "Caverna (com água)",
                    "Floresta",
                    "Montanha (declives)",
                    "Pradaria",
                    "Terra batida",
                    "Rio",
                    "Mar (costa)",
                    "Lago no centro",
                ],
                index=0
            )
    
            label_to_key = {
                "Caverna (com água)": "cave_water",
                "Floresta": "forest",
                "Montanha (declives)": "mountain_slopes",
                "Pradaria": "plains",
                "Terra batida": "dirt",
                "Rio": "river",
                "Mar (costa)": "sea_coast",
                "Lago no centro": "center_lake",
            }
    
            theme = label_to_key[theme_label]
    
        with c3:
            st.write("")
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
            # tenta mostrar infos básicas
            room_infos = []
            for rid in my_rooms[:20]:
                info = get_room(db, rid)
                if info:
                    status = info.get("status", "?")
                    gs = info.get("gridSize", "?")
                    th = info.get("theme", "?")
                    owner = (info.get("owner") or {}).get("name", "?")
                    chal = (info.get("challenger") or {})
                    chal_name = chal.get("name") if isinstance(chal, dict) else (chal or "")
                    room_infos.append((rid, f"{rid} | {status} | {gs}x{gs} | {th} | owner={owner} | challenger={chal_name or '-'}"))
    
            labels = [x[1] for x in room_infos] if room_infos else my_rooms
            chosen = st.selectbox("Abrir arena", labels, index=0)
            chosen_rid = chosen.split(" | ")[0] if " | " in chosen else chosen
    
            b1, b2, b3 = st.columns([1, 1, 2])
            with b1:
                if st.button("📂 Abrir"):
                    st.session_state["active_room_id"] = chosen_rid
                    st.rerun()
            with b2:
                if st.button("🗄️ Arquivar (remover da lista)"):
                    remove_room_from_user(db, trainer_name, chosen_rid)
                    if st.session_state.get("active_room_id") == chosen_rid:
                        st.session_state.pop("active_room_id", None)
                    st.rerun()
    
        st.markdown("---")
    
        # --- Entrar por código (desafiante / espectador) ---
        st.subheader("🔑 Entrar por código")
        cc1, cc2, cc3 = st.columns([2, 1, 1])
        with cc1:
            code = st.text_input("Código da arena (roomId)", value="")
        with cc2:
            if st.button("🥊 Entrar como desafiante"):
                if not code.strip():
                    st.warning("Digite um código.")
                else:
                    res = join_room_as_challenger(db, code.strip(), trainer_name)
                    if res == "OK":
                        st.success("Você entrou como desafiante!")
                        st.session_state["active_room_id"] = code.strip()
                        st.rerun()
                    elif res == "CHALLENGER_TAKEN":
                        st.error("Essa arena já tem desafiante. Entre como espectador.")
                    elif res == "NOT_FOUND":
                        st.error("Arena não encontrada.")
                    else:
                        st.info(res)
        with cc3:
            if st.button("👀 Entrar como espectador"):
                if not code.strip():
                    st.warning("Digite um código.")
                else:
                    res = join_room_as_spectator(db, code.strip(), trainer_name)
                    if res == "OK" or res == "PLAYER":
                        st.success("Você entrou na arena!")
                        st.session_state["active_room_id"] = code.strip()
                        st.rerun()
                    elif res == "NOT_FOUND":
                        st.error("Arena não encontrada.")
                    else:
                        st.info(res)
    
        st.markdown("---")
   
    else:
    
        # --- Painel da arena ativa ---
        rid = st.session_state.get("active_room_id")
        st.subheader("🎮 Arena ativa")
        if st.button("🔄 Atualizar arena"):
            st.rerun()
        if st.button("⬅️ Voltar ao lobby"):
            st.session_state["pvp_view"] = "lobby"
            st.rerun()


    if not rid:
        st.info("Nenhuma arena aberta. Crie ou abra uma arena acima.")
    else:
        room = get_room(db, rid)
        if not room:
            st.error("Arena ativa não existe mais (ou código inválido).")
            st.session_state.pop("active_room_id", None)
        else:
            owner = (room.get("owner") or {}).get("name")
            chal = room.get("challenger") or {}
            chal_name = chal.get("name") if isinstance(chal, dict) else (chal or None)

            role = get_role(room, trainer_name)
            is_player = role in ["owner", "challenger"]

            st.write(f"**Código:** `{rid}`")
            st.write(f"**Status:** {room.get('status')}")
            st.write(f"**Grid:** {room.get('gridSize')}x{room.get('gridSize')}  |  **Tema:** {room.get('theme')}")
            st.write(f"**Owner:** {owner}  |  **Challenger:** {chal_name or '-'}")
            st.write(f"**Espectadores:** {len(room.get('spectators') or [])}")
            

        
# =========================
            # 🗺️ BLOCO DO MAPA (ETAPA 2)
            # =========================
            state_ref = (
                db.collection("rooms")
                  .document(rid)
                  .collection("public_state")
                  .document("state")
            )

            state_doc = state_ref.get()
            state = state_doc.to_dict() if state_doc.exists else {}

            grid = int(room.get("gridSize") or 6)
            theme_key = room.get("theme") or "cave_water"

            seed = state.get("seed")
            packed = state.get("tilesPacked")
            tiles = unpack_tiles(packed) if packed else None
            
            # --- CORREÇÃO AQUI: Carregar as peças para o Lobby ---
            all_pieces = state.get("pieces") or []
            pieces = visible_pieces_for(room, trainer_name, all_pieces)
            # -----------------------------------------------------
            
            # checkbox SEMPRE disponível
            no_water = st.checkbox(
                "🚫 Gerar sem água",
                # ... restante do código ...
                value=bool(state.get("noWater", False)),
                disabled=not is_player
            )
            
            if not tiles:
                if st.button("🗺️ Gerar mapa (pixel art)", disabled=not is_player):
                    tiles, seed = gen_tiles(grid, theme_key, seed=None, no_water=no_water)
                    packed = pack_tiles(tiles)
            
                    state_ref.set({
                        "gridSize": grid,
                        "theme": theme_key,
                        "seed": seed,
                        "tilesPacked": packed,
                        "noWater": bool(no_water),
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                    }, merge=True)
            
                    add_public_event(
                        db, rid, "map_generated", trainer_name,
                        {"theme": theme_key, "grid": grid, "seed": seed, "noWater": bool(no_water)}
                    )
                    st.session_state["pvp_view"] = "battle"
                    st.rerun()
            
            else:
                # ✅ Botão de REGERAR fica aqui (porque o mapa já existe)
                if st.button("🔁 Regerar mapa", disabled=not is_player):
                    tiles, seed = gen_tiles(grid, theme_key, seed=None, no_water=no_water)
                    packed = pack_tiles(tiles)
            
                    state_ref.set({
                        "seed": seed,
                        "tilesPacked": packed,
                        "noWater": bool(no_water),
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                    }, merge=True)
            
                    add_public_event(
                        db, rid, "map_regenerated", trainer_name,
                        {"theme": theme_key, "grid": grid, "seed": seed, "noWater": bool(no_water)}
                    )
                    st.rerun()
                    # ... (código dos botões de gerar/regerar acima) ...

            # --- CORREÇÃO: PRÉVIA E BOTÃO ---
            
            # 1. Só mostra a prévia se o mapa (tiles) existir
            if tiles:
                # Se ainda não tem seleção na sessão, inicia como None
                if "selected_piece_id" not in st.session_state:
                    st.session_state["selected_piece_id"] = None
                
                # Renderiza a imagem estática para o Lobby (apenas visualização)
                img = render_map_with_pieces(tiles, theme_key, seed, pieces, trainer_name)
                st.image(img, caption="Prévia do Mapa Atual")
            else:
                st.info("O mapa ainda não foi gerado. Clique em 'Gerar mapa' acima para começar.")

            st.markdown("---")
            
            # 2. Botão ÚNICO e PROTEGIDO
            # Se 'tiles' for None (não tem mapa), o botão fica desabilitado (cinza)
            if st.button("⚔️ Ir para o Campo de Batalha", type="primary", disabled=not tiles):
                st.session_state["pvp_view"] = "battle"
                st.rerun()

            # --- Estado atual ---
            state = get_state(db, rid)
            all_pieces = state.get("pieces") or []
            pieces = visible_pieces_for(room, trainer_name, all_pieces)
            
            # --- montar lista da party (até 10) ---
            party = user_data.get("party") or []
            party = party[:10]
            
           
        
            # --- Última rolagem de dado ---
            last_events = list_public_events(db, rid, limit=10)
            last_dice = next((e for e in last_events if e.get("type") == "dice"), None)
            if last_dice:
                payload = last_dice.get("payload", {})
                st.info(
                    f"🎲 Última rolagem: **d{payload.get('sides')} = {payload.get('result')}** "
                    f"(por {last_dice.get('by')})"
                )

            # --- Botões de dado ---
            st.markdown("---")
            c1, c2, c3 = st.columns([1, 1, 2])

            with c1:
                if st.button("🎲 Rolar d20", disabled=not is_player):
                    r = roll_die(db, rid, trainer_name, sides=20)
                    st.success(f"Você rolou: **{r}**")
                    st.rerun()

            with c2:
                if st.button("🎲 Rolar d6", disabled=not is_player):
                    r = roll_die(db, rid, trainer_name, sides=6)
                    st.success(f"Você rolou: **{r}**")
                    st.rerun()

            with c3:
                if is_player:
                    st.caption("A rolagem aparece no **Log público** para jogadores e espectadores.")
                else:
                    st.caption("Você está como **espectador**.")

            # --- Log público ---
            st.markdown("### 📜 Log público (todos veem)")
            events = list_public_events(db, rid, limit=25)
            if not events:
                st.caption("Sem eventos ainda.")
            else:
                for ev in events:
                    et = ev.get("type", "?")
                    by = ev.get("by", "?")
                    payload = ev.get("payload", {})
                    st.write(f"- **{et}** — _{by}_ — {payload}")
