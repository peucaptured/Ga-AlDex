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
from streamlit_autorefresh import st_autorefresh
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import random
import gzip
import base64




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
    return uuid.uuid4().hex[:8]

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

def create_room(db, trainer_name: str, grid_size: int, theme: str, max_active: int = 5):
    my_rooms = list_my_rooms(db, trainer_name)
    if len(my_rooms) >= max_active:
        return None, f"Você já tem {len(my_rooms)} arenas ativas (limite {max_active}). Finalize/arquive uma para criar outra."

    rid = room_id_new()
    room_ref = db.collection("rooms").document(rid)

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

    # estado público inicial (vazio por enquanto)
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

def list_public_events(db, rid: str, limit: int = 30):
    q = (db.collection("rooms").document(rid)
         .collection("public_events")
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [d.to_dict() for d in q.stream()]
# =========================
# MAPA TÁTICO (3.1 / 3.2 / 3.3)
# =========================

TILE_SIZE = 32

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
        if inside(rr, cc) and rng.random() > 0.55:
            if tiles[rr][cc] in ["grass", "dirt", "stone", "sand"]:
                tiles[rr][cc] = "stone"

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
    else:
        # “poças secas” (pedra clara) pra dar variedade
        patches = rng.randint(1, 2)
        for _ in range(patches):
            cr = rng.randint(2, grid - 3)
            cc = rng.randint(2, grid - 3)
            rad = rng.randint(1, 2)
            for rr in range(cr - rad, cr + rad + 1):
                for cc2 in range(cc - rad, cc + rad + 1):
                    if inside(rr, cc2) and rng.random() > 0.25:
                        tiles[rr][cc2] = "stone"

    spikes = rng.randint(1, max(2, grid - 3))
    for _ in range(spikes):
        rr = rng.randint(1, grid - 2)
        cc = rng.randint(1, grid - 2)
        if inside(rr, cc) and tiles[rr][cc] == base and rng.random() > 0.5:
            tiles[rr][cc] = "stalagmite"

    elif theme_key == "forest":
    # “carpete” de grama
    for r in range(1, grid - 1):
        for c in range(1, grid - 1):
            if inside(r, c):
                tiles[r][c] = "grass" if rng.random() > 0.10 else "bush"

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
                tiles[r][c] = "stone" if rng.random() > 0.12 else "rock"

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


    return tiles, seed

def draw_tile(draw: ImageDraw.ImageDraw, x: int, y: int, t: str, rng: random.Random):
    colors = {
        "rock": (70, 72, 78),
        "wall": (45, 46, 50),
        "water": (35, 90, 140),
        "stalagmite": (90, 92, 98),
        "grass": (60, 130, 70),
        "tree": (30, 70, 35),
        "path": (120, 95, 60),
        "stone": (95, 96, 100),
        "peak": (160, 160, 165),
        "slope1": (120, 120, 125),
        "slope2": (105, 105, 110),
        "flower": (150, 80, 110),
        "trail": (105, 85, 55),
        "dirt": (110, 85, 55),
        "rut": (85, 65, 40),
        "sea": (20, 60, 120),
        "sand": (180, 165, 120),
        "bush": (40, 95, 50),
        "rock": (90, 90, 95),
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

def render_map_png(tiles: list[list[str]], theme_key: str, seed: int):
    grid = len(tiles)
    img = Image.new("RGB", (grid * TILE_SIZE, grid * TILE_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    rng = random.Random(int(seed or 0) + 1337)

    for r in range(grid):
        for c in range(grid):
            x = c * TILE_SIZE
            y = r * TILE_SIZE
            draw_tile(draw, x, y, tiles[r][c], rng)

    for r in range(grid + 1):
        y = r * TILE_SIZE
        draw.line([(0, y), (grid * TILE_SIZE, y)], fill=(0, 0, 0))
    for c in range(grid + 1):
        x = c * TILE_SIZE
        draw.line([(x, 0), (x, grid * TILE_SIZE)], fill=(0, 0, 0))

    return img

def normalize_text(text):
    if not isinstance(text, str): return str(text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

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

    # Auto-refresh só nesta aba
    st_autorefresh(interval=4500, key="pvp_refresh")

    db, bucket = init_firebase()


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
    # --- Painel da arena ativa ---
    rid = st.session_state.get("active_room_id")
    st.subheader("🎮 Arena ativa")

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

            if not tiles:
                no_water = st.checkbox(
                    "🚫 Gerar sem água",
                    value=False,
                    disabled=not is_player
                )
                
                if st.button("🗺️ Gerar mapa (pixel art)", disabled=not is_player):
                    tiles, seed = gen_tiles(grid, theme_key, seed=None)

                    packed = pack_tiles(tiles)
                    # DEBUG (temporário)
                    st.write("DEBUG no_water:", no_water)
                    st.write("DEBUG seed:", seed)
                    st.write("DEBUG packed len:", len(packed) if packed else None)
                    
                    state_ref.set({
                        "gridSize": grid,
                        "theme": theme_key,
                        "seed": seed,
                        "noWater": bool(no_water),
                        "tilesPacked": packed,
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                    }, merge=True)

                    add_public_event(
                        db,
                        rid,
                        "map_generated",
                        trainer_name,
                        {"theme": theme_key, "grid": grid, "seed": seed}
                    )

                    st.rerun()

            else:
                img = render_map_png(tiles, theme_key, seed)

                st.markdown("### 🗺️ Mapa tático")
                click = streamlit_image_coordinates(img, key=f"map_{rid}")

                if click and "x" in click and "y" in click:
                    col = int(click["x"] // TILE_SIZE)
                    row = int(click["y"] // TILE_SIZE)

                    if 0 <= row < grid and 0 <= col < grid:
                        st.success(f"Célula selecionada: **linha {row}**, **coluna {col}**")

                        add_public_event(
                            db,
                            rid,
                            "cell_click",
                            trainer_name,
                            {"row": row, "col": col}
                        )

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








