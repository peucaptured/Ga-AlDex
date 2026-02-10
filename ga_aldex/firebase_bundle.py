# firebase_bundle.py
# Extraído de app.py para modularização (Firebase + helpers relacionados)
from __future__ import annotations

import streamlit as st

import os, json, time, io, re, glob
import pandas as pd

import requests
from typing import Any, Dict, List, Optional, Tuple, Set


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
    
# ==========================
# FIREBASE SAVE/LOAD (Fichas)
# ==========================
from datetime import datetime, timezone


def safe_doc_id(name: str) -> str:
    """Sanitiza um texto para virar ID seguro no Firestore."""
    if not isinstance(name, str):
        name = str(name)
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name).strip("_")[:80] or "user"

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def upload_pdf_to_storage(bucket, pdf_bytes: bytes, storage_path: str):
    blob = bucket.blob(storage_path)
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    return storage_path

def upload_png_to_storage(bucket, png_bytes: bytes, storage_path: str):
    blob = bucket.blob(storage_path)
    blob.upload_from_string(png_bytes, content_type="image/png")
    return storage_path

@st.cache_data(show_spinner=False, ttl=3600)
def download_storage_bytes(storage_path: str) -> bytes | None:
    if not storage_path:
        return None
    try:
        _, bucket = init_firebase()
        return bucket.blob(storage_path).download_as_bytes()
    except Exception:
        return None


def save_sheet_to_firestore(db, trainer_name: str, sheet_payload: dict, sheet_id=None):
    trainer_id = safe_doc_id(trainer_name)

    if not sheet_id:
        pname = sheet_payload.get("pokemon", {}).get("name", "pokemon")
        pid = sheet_payload.get("pokemon", {}).get("id", "0")
        sheet_id = safe_doc_id(f"{pname}_{pid}_{uuid.uuid4().hex[:8]}")

    ref = (
        db.collection("trainers")
        .document(trainer_id)
        .collection("sheets")
        .document(sheet_id)
    )

    now = _utc_now_iso()
    sheet_payload.setdefault("created_at", now)
    sheet_payload["updated_at"] = now
    sheet_payload["trainer_name"] = trainer_name

    ref.set(sheet_payload, merge=True)
    return sheet_id

def save_sheet_with_pdf(db, bucket, trainer_name: str, sheet_payload: dict, pdf_bytes=None, sheet_id=None):
    storage_path = None

    if pdf_bytes:
        pname = sheet_payload.get("pokemon", {}).get("name", "pokemon")
        pid = sheet_payload.get("pokemon", {}).get("id", "0")
        storage_path = (
            f"fichas/{safe_doc_id(trainer_name)}/"
            f"{safe_doc_id(pname)}_{pid}_{uuid.uuid4().hex[:8]}.pdf"
        )
        upload_pdf_to_storage(bucket, pdf_bytes, storage_path)

        sheet_payload.setdefault("pdf", {})
        sheet_payload["pdf"].update({
            "storage_path": storage_path,
            "updated_at": _utc_now_iso(),
            "version": int(sheet_payload.get("pdf", {}).get("version", 0)) + 1,
        })

    sheet_id = save_sheet_to_firestore(db, trainer_name, sheet_payload, sheet_id=sheet_id)
    return sheet_id, storage_path

def build_sheet_pdf(
    pname: str,
    np_: int,
    types: list[str],
    abilities: list[str],
    stats: dict,
    chosen_adv: list[str],
    moves: list[dict],
) -> bytes:
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 12)
    c.drawString(40, 800, f"Ficha Pokémon - {pname} (NP {np_})")
    c.drawString(40, 780, f"Tipos: {', '.join(types)}")
    c.drawString(40, 760, f"Abilities: {', '.join(abilities)}")
    c.drawString(
        40,
        730,
        "Stgr {stgr} | Int {intellect} | Dodge {dodge} | Parry {parry} | "
        "Fort {fortitude} | Will {will}".format(**stats),
    )
    c.drawString(40, 710, f"Advantages: {', '.join(chosen_adv) if chosen_adv else '(nenhuma)'}")

    y = 680
    c.drawString(40, y, "Golpes:")
    y -= 18
    def _draw_line(text: str, indent: int = 0):
        nonlocal y
        c.drawString(50 + indent, y, text)
        y -= 14
        if y < 80:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 800
    for m in moves:
        accuracy = int(m.get("accuracy", 0) or 0)
        _draw_line(f"- {m['name']} (Rank {m['rank']}) | PP {m.get('pp_cost')} | Acerto {accuracy}")
        build_txt = (m.get("build") or "").strip()
        if build_txt:
            _draw_line("Ingredientes:", indent=10)
            for line in build_txt.splitlines():
                _draw_line(line, indent=20)


    c.showPage()
    c.save()
    return buffer.getvalue()


def parse_sheet_pdf(pdf_bytes: bytes) -> dict:
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    header_re = re.compile(r"^Ficha Pokémon - (.+) \(NP (\d+)\)$")
    stats_re = re.compile(
        r"Stgr\s+(\d+)\s*\|\s*Int\s+(\d+)\s*\|\s*Dodge\s+(\d+)\s*\|\s*Parry\s+(\d+)\s*\|\s*Fort\s+(\d+)\s*\|\s*Will\s+(\d+)"
    )
    move_re = re.compile(r"^- (.+) \(Rank (\d+)\) \| PP ([^|]+) \| Acerto (\d+)$")

    pname = ""
    np_value = 0
    types: list[str] = []
    abilities: list[str] = []
    stats: dict[str, int] = {}
    advantages: list[str] = []
    moves: list[dict] = []

    for line in lines:
        header_match = header_re.match(line)
        if header_match:
            pname = header_match.group(1).strip()
            np_value = int(header_match.group(2))
            continue
        if line.startswith("Tipos:"):
            raw_types = line.replace("Tipos:", "", 1).strip()
            types = [t.strip() for t in raw_types.split(",") if t.strip()]
            continue
        if line.startswith("Abilities:"):
            raw_abilities = line.replace("Abilities:", "", 1).strip()
            abilities = [a.strip() for a in raw_abilities.split(",") if a.strip()]
            continue
        if line.startswith("Advantages:"):
            raw_adv = line.replace("Advantages:", "", 1).strip()
            if raw_adv and raw_adv != "(nenhuma)":
                advantages = [a.strip() for a in raw_adv.split(",") if a.strip()]
            continue
        stats_match = stats_re.search(line)
        if stats_match:
            stats = {
                "stgr": int(stats_match.group(1)),
                "int": int(stats_match.group(2)),
                "dodge": int(stats_match.group(3)),
                "parry": int(stats_match.group(4)),
                "fortitude": int(stats_match.group(5)),
                "will": int(stats_match.group(6)),
            }

    in_moves = False
    current_move = None
    build_lines: list[str] = []
    collecting_build = False

    for line in lines:
        if line == "Golpes:":
            in_moves = True
            continue
        if not in_moves:
            continue

        move_match = move_re.match(line)
        if move_match:
            if current_move:
                current_move["build"] = "\n".join(build_lines).strip()
                moves.append(current_move)
            move_name = move_match.group(1).strip()
            move_rank = int(move_match.group(2))
            pp_raw = move_match.group(3).strip()
            pp_value = int(pp_raw) if pp_raw.isdigit() else None
            acc_value = int(move_match.group(4))
            current_move = {
                "name": move_name,
                "rank": move_rank,
                "pp_cost": pp_value,
                "accuracy": acc_value,
                "build": "",
            }
            build_lines = []
            collecting_build = False
            continue

        if line.startswith("Ingredientes"):
            collecting_build = True
            continue

        if collecting_build and current_move:
            build_lines.append(line)

    if current_move:
        current_move["build"] = "\n".join(build_lines).strip()
        moves.append(current_move)

    if not pname or np_value <= 0:
        raise ValueError("Não foi possível identificar nome ou NP no PDF.")

    return {
        "pokemon": {"name": pname, "types": types, "abilities": abilities},
        "np": np_value,
        "stats": stats,
        "advantages": advantages,
        "moves": moves,
    }


def apply_imported_sheet_to_session(sheet: dict):
    pokemon = sheet.get("pokemon", {}) if isinstance(sheet, dict) else {}
    pname = str(pokemon.get("name", "") or "")
    np_ = int(sheet.get("np", 0) or 0)
    stats = sheet.get("stats") or {}
    moves = sheet.get("moves") or []
    advantages = sheet.get("advantages") or []
    abilities = pokemon.get("abilities") or []
    types = pokemon.get("types") or []

    cap = 2 * np_
    dodge_val = int(stats.get("dodge", 0) or 0)

    for m in moves:
        if isinstance(m, dict) and "accuracy" not in m:
            m["accuracy"] = 0

    st.session_state["cg_edit_sheet_id"] = None
    st.session_state["cg_imported_name"] = pname
    st.session_state["cg_imported_types"] = list(types)
    st.session_state["cg_imported_abilities"] = list(abilities)

    st.session_state["cg_draft"] = {
        "pname": pname,
        "np": np_,
        "stats": {
            "stgr": int(stats.get("stgr", 0) or 0),
            "int": int(stats.get("int", 0) or 0),
            "dodge": dodge_val,
            "parry": int(stats.get("parry", dodge_val) or 0),
            "thg": int(stats.get("thg", max(0, cap - dodge_val)) or 0),
            "fortitude": int(stats.get("fortitude", 0) or 0),
            "will": int(stats.get("will", 0) or 0),
        },
        "moves": list(moves),
    }
    st.session_state["cg_moves"] = st.session_state["cg_draft"]["moves"]
    st.session_state["cg_pname"] = pname
    st.session_state["cg_np"] = np_
    st.session_state["cg_stgr"] = st.session_state["cg_draft"]["stats"]["stgr"]
    st.session_state["cg_int"] = st.session_state["cg_draft"]["stats"]["int"]
    st.session_state["cg_dodge"] = st.session_state["cg_draft"]["stats"]["dodge"]
    st.session_state["cg_parry"] = st.session_state["cg_draft"]["stats"]["parry"]
    st.session_state["cg_thg"] = st.session_state["cg_draft"]["stats"]["thg"]
    st.session_state["cg_fortitude"] = st.session_state["cg_draft"]["stats"]["fortitude"]
    st.session_state["cg_will"] = st.session_state["cg_draft"]["stats"]["will"]
    st.session_state["cg_advantages"] = list(advantages)
    st.session_state["cg_abilities"] = list(abilities)

def list_sheets(db, trainer_name: str, limit: int = 50):
    trainer_id = safe_doc_id(trainer_name)
    docs = (
        db.collection("trainers")
        .document(trainer_id)
        .collection("sheets")
        .order_by("updated_at", direction="DESCENDING")
        .limit(limit)
        .stream()
    )
    out = []
    for d in docs:
        item = d.to_dict() or {}
        item["_sheet_id"] = d.id
        out.append(item)
    return out

def load_sheet(db, trainer_name: str, sheet_id: str):
    trainer_id = safe_doc_id(trainer_name)
    ref = (
        db.collection("trainers")
        .document(trainer_id)
        .collection("sheets")
        .document(sheet_id)
    )
    snap = ref.get()
    return snap.to_dict() if snap.exists else None
def delete_sheet(db, bucket, trainer_name: str, sheet_id: str, storage_path: str | None = None):
    trainer_id = safe_doc_id(trainer_name)
    ref = (
        db.collection("trainers")
        .document(trainer_id)
        .collection("sheets")
        .document(sheet_id)
    )
    ref.delete()

    if storage_path:
        try:
            bucket.blob(storage_path).delete()
        except Exception:
            pass

def apply_sheet_to_session(sheet: dict, sheet_id: str | None = None):
    pokemon = sheet.get("pokemon", {}) if isinstance(sheet, dict) else {}
    pname = str(pokemon.get("name", "") or "")
    stats = sheet.get("stats") or {}
    moves = sheet.get("moves") or []
    advantages = sheet.get("advantages") or []
    skills = sheet.get("skills") or []
    np_ = int(sheet.get("np", 0) or 0)
    cap = 2 * np_
    abilities = pokemon.get("abilities") or []

    st.session_state["cg_edit_sheet_id"] = sheet_id
    for m in moves:
        if isinstance(m, dict) and "accuracy" not in m:
            m["accuracy"] = 0
    st.session_state["cg_draft"] = {
        "pname": pname,
        "stats": {
            "stgr": int(stats.get("stgr", 0) or 0),
            "int": int(stats.get("int", 0) or 0),
            "dodge": int(stats.get("dodge", 0) or 0),
            "parry": int(stats.get("parry", stats.get("dodge", 0)) or 0),
            "thg": int(stats.get("thg", max(0, cap - int(stats.get("dodge", 0) or 0))) or 0),
            "fortitude": int(stats.get("fortitude", 0) or 0),
            "will": int(stats.get("will", 0) or 0),
        },
        "moves": list(moves),
    }
    st.session_state["cg_moves"] = st.session_state["cg_draft"]["moves"]
    st.session_state["cg_pname"] = pname
    st.session_state["cg_np"] = np_
    st.session_state["cg_stgr"] = st.session_state["cg_draft"]["stats"]["stgr"]
    st.session_state["cg_int"] = st.session_state["cg_draft"]["stats"]["int"]
    st.session_state["cg_dodge"] = st.session_state["cg_draft"]["stats"]["dodge"]
    st.session_state["cg_parry"] = st.session_state["cg_draft"]["stats"]["parry"]
    st.session_state["cg_thg"] = st.session_state["cg_draft"]["stats"]["thg"]
    st.session_state["cg_fortitude"] = st.session_state["cg_draft"]["stats"]["fortitude"]
    st.session_state["cg_will"] = st.session_state["cg_draft"]["stats"]["will"]
    st.session_state["cg_advantages"] = list(advantages)
    st.session_state["cg_abilities"] = list(abilities)

    base_skills = {k: 0 for k in SKILLS_MM3}
    custom_skills = []
    for row in skills:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        try:
            ranks = int(row.get("ranks", 0))
        except Exception:
            ranks = 0
        if not name:
            continue
        if name in base_skills:
            base_skills[name] = ranks
        else:
            custom_skills.append({"name": name, "ranks": ranks})

    st.session_state["cg_skills"] = base_skills
    st.session_state["cg_skill_custom"] = custom_skills



def _cg_cap():
    np_ = int(st.session_state.get("cg_np", 0) or 0)
    return 2 * np_

def _cg_sync_from_dodge():
    cap = _cg_cap()
    dodge = int(st.session_state.get("cg_dodge", 0) or 0)

    # Parry espelha Dodge (como no seu código original)
    st.session_state["cg_parry"] = dodge

    # Thg fecha o cap automaticamente
    st.session_state["cg_thg"] = max(0, min(99, cap - dodge))

def _cg_sync_from_fortitude():
    cap = _cg_cap()
    fort = int(st.session_state.get("cg_fortitude", 0) or 0)

    st.session_state["cg_will"] = max(0, min(99, cap - fort))

def _cg_sync_from_np():
    # Quando NP muda, recalcula Thg e Will para manter cap
    _cg_sync_from_dodge()
    _cg_sync_from_fortitude()

def _cg_init_defenses_if_missing(dodge_base, fort_base):
    cap = _cg_cap()

    if "cg_dodge" not in st.session_state:
        st.session_state["cg_dodge"] = int(dodge_base)
    if "cg_parry" not in st.session_state:
        st.session_state["cg_parry"] = int(st.session_state["cg_dodge"])
    if "cg_fortitude" not in st.session_state:
        st.session_state["cg_fortitude"] = int(fort_base)
    if "cg_will" not in st.session_state:
        st.session_state["cg_will"] = max(0, min(99, cap - int(st.session_state["cg_fortitude"])))
    if "cg_thg" not in st.session_state:
        st.session_state["cg_thg"] = max(0, min(99, cap - int(st.session_state["cg_dodge"])))



# ----------------------------
# Helpers UX (Criação Guiada)
# ----------------------------
def _resolve_asset_path(fname: str) -> str:
    """Resolve caminhos para assets (excel, etc.) sem quebrar em Streamlit Cloud/local.

    Também tolera nomes com sufixos comuns (ex: "(2)") quando o arquivo foi enviado/baixado.
    """
    try:
        base = os.path.dirname(__file__)
    except Exception:
        base = os.getcwd()

    candidates = [
        fname,
        os.path.join(os.getcwd(), fname),
        os.path.join(base, fname),
        os.path.join(base, "assets", fname),
        os.path.join(os.getcwd(), "assets", fname),
        os.path.join(os.getcwd(), "data", fname),
        os.path.join(base, "data", fname),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c

    # fallback inteligente: procura variações com sufixo (ex: "arquivo (5).xlsx")
    try:
        import glob
        root_dirs = [os.getcwd(), base, os.path.join(base, "assets"), os.path.join(os.getcwd(), "assets"),
                     os.path.join(base, "data"), os.path.join(os.getcwd(), "data")]
        stem, ext = os.path.splitext(fname)
        if ext:
            patterns = [f"{stem}*{ext}"]
        else:
            patterns = [f"{stem}*"]
        for rd in root_dirs:
            for pat in patterns:
                hits = sorted(glob.glob(os.path.join(rd, pat)))
                if hits:
                    return hits[0]
    except Exception:
        pass

    return fname  # fallback (deixa o erro explícito se não achar)

def _pokeapi_parse_move_names(pjson: dict) -> list[str]:
    out: list[str] = []
    for it in (pjson or {}).get("moves", []) or []:
        mv = (it or {}).get("move", {}) or {}
        nm = mv.get("name")
        if nm:
            out.append(str(nm))
    # remove duplicados preservando ordem
    seen = set()
    uniq = []
    for nm in out:
        if nm not in seen:
            seen.add(nm)
            uniq.append(nm)
    return uniq

def _try_match_move_in_db(db, api_name: str):
    """Tenta casar nomes do PokeAPI (geralmente com hífen) com os nomes do Excel (geralmente com espaço)."""
    if not api_name:
        return None

    raw = str(api_name).strip()
    cands = [
        raw,
        raw.replace("-", " "),
        raw.replace("-", " ").replace("’", "'"),
        raw.replace("-", ""),
    ]
    # algumas exceções comuns (seu excel costuma usar abreviações / pontuação diferente)
    # você pode ir adicionando aqui conforme aparecerem casos novos
    EXCEPTIONS = {
        "u turn": "u-turn",
        "v create": "v-create",
    }
    for k, v in EXCEPTIONS.items():
        if _norm(cands[1]) == _norm(k):
            cands.append(v)

    for c in cands:
        try:
            mv = db.get_by_name(c)
            if mv:
                return mv
        except Exception:
            continue

    # fallback: busca por prefixo / melhor candidato
    try:
        hits = db.search_by_name_prefix(raw.replace("-", " "))
        if hits:
            return hits[0]
    except Exception:
        pass

    return None

def _summarize_build(build_txt: str) -> list[str]:
    b = (build_txt or "").strip()
    if not b:
        return []
    low = b.lower()
    bullets: list[str] = []

    def _has(p: str) -> bool:
        return p in low

    if _has("damage"):
        bullets.append("🗡️ Dano (Damage)")
    if _has("affliction"):
        bullets.append("🌀 Status (Affliction)")
    if _has("weaken"):
        bullets.append("📉 Debuff (Weaken)")
    if _has("healing"):
        bullets.append("💚 Cura (Healing)")
    if _has("create"):
        bullets.append("🧱 Barreira/Criação (Create)")
    if _has("environment"):
        bullets.append("🌦️ Ambiente/Clima (Environment)")
    if _has("nullify"):
        bullets.append("🚫 Nullify")

    if _has("area"):
        # tenta capturar tipo de área
        m = re.search(r"\[Area:\s*([^\]]+)\]", b, flags=re.IGNORECASE)
        if m:
            bullets.append(f"🧨 Área: {m.group(1).strip()}")
        else:
            bullets.append("🧨 Área")
    if _has("perception"):
        bullets.append("🎯 Perception")
    if _has("ranged") or _has("extended range"):
        bullets.append("🏹 Ranged")

    if _has("linked"):
        bullets.append("🔗 Linked (multi-efeito)")

    # deixa no máximo 6 para não poluir
    return bullets[:6]

def _default_accuracy_from_raw(mv) -> int:
    raw = getattr(mv, "raw", {}) or {}
    try:
        return int(raw.get("Accuracy") or raw.get("Acerto") or raw.get("acerto") or 0)
    except Exception:
        return 0

def _cg_confirm_move(mv, rank: int, pp_override: int | None = None, accuracy: int | None = None) -> dict:
    # PP
    pp_auto = None
    try:
        tmp = mv.pp_cost(int(rank))
        if isinstance(tmp, tuple):
            pp_auto = tmp[0]
    except Exception:
        pp_auto = None
    pp = int(pp_override) if pp_override is not None else (int(pp_auto) if pp_auto is not None else None)

    # accuracy
    acc = int(accuracy) if accuracy is not None else _default_accuracy_from_raw(mv)

    return {
        "name": mv.name,
        "rank": int(rank),
        "build": mv.render_build(int(rank)),
        "pp_cost": pp,
        "accuracy": acc,
        "meta": {
            "ranged": bool(getattr(mv, "ranged", False)),
            "perception_area": bool(getattr(mv, "perception_area", False)),
            "category": str(getattr(mv, "categoria", "") or ""),
        },
    }


def _cg_recalculate_pp(move_data: dict, rank: int, db_moves: Optional["MoveDB"]) -> Tuple[Optional[int], Optional[str]]:
    mv_name = str(move_data.get("name") or "").strip()
    if db_moves and mv_name:
        try:
            mv = db_moves.get_by_name(mv_name)
        except Exception:
            mv = None
        if mv:
            try:
                pp_auto, why = mv.pp_cost(int(rank))
                if pp_auto is not None:
                    return int(pp_auto), why
            except Exception:
                pass

    tmp_build = str(move_data.get("build") or "").strip()
    if tmp_build:
        tmp_meta = move_data.get("meta") or {}
        tmp_mv = Move(
            name=(mv_name or "Poder Personalizado"),
            tipo="—",
            categoria=str(tmp_meta.get("category") or ""),
            descricao="",
            build=tmp_build,
            how_it_works="",
            resist_stat="",
            ranged=bool(tmp_meta.get("ranged", False)),
            perception_area=bool(tmp_meta.get("perception_area", False)),
            tags=[],
            raw={},
        )
        try:
            pp_auto, why = tmp_mv.pp_cost(int(rank))
            if pp_auto is not None:
                return int(pp_auto), why
        except Exception:
            pass

    return None, None



import math
from io import BytesIO


REGION_ALIASES = {
    "alola": "alola", "alolan": "alola", "a": "alola",
    "galar": "galar", "galarian": "galar", "g": "galar",
    "hisui": "hisui", "hisuian": "hisui", "h": "hisui",
    "paldea": "paldea", "paldean": "paldea", "p": "paldea",
}

def to_pokeapi_name(user_text: str) -> str:
    s = (user_text or "").strip().lower()

    # símbolos comuns
    s = s.replace("♀", " f").replace("♂", " m")

    # troca espaços/underscore por hífen
    s = re.sub(r"[\s_]+", "-", s)

    # remove duplo hífen
    s = re.sub(r"-{2,}", "-", s).strip("-")

    # nidoran: vira nidoran-f / nidoran-m
    if s in ("nidoran", "nidoran-"):
        return "nidoran"  # deixa ambíguo e você força escolha na UI
    if s in ("nidoran-f", "nidoranf", "nidoran-female", "nidoran-fem", "nidoran-f."):
        return "nidoran-f"
    if s in ("nidoran-m", "nidoranm", "nidoran-male", "nidoran-masc", "nidoran-m."):
        return "nidoran-m"

    # formatos tipo "sandslash-a" / "weezing-g" / "g-weezing"
    if re.match(r"^[aghp]-", s):  # g-weezing
        tag, base = s.split("-", 1)
        region = REGION_ALIASES.get(tag)
        if region and base:
            return f"{base}-{region}"

    m = re.match(r"^(.+)-([aghp])$", s)  # sandslash-a
    if m:
        base, tag = m.group(1), m.group(2)
        region = REGION_ALIASES.get(tag)
        if region:
            return f"{base}-{region}"

    # formatos tipo "sandslash-alolan" / "weezing-galarian"
    parts = s.split("-")
    if len(parts) >= 2:
        last = parts[-1]
        region = REGION_ALIASES.get(last)
        if region:
            base = "-".join(parts[:-1])
            return f"{base}-{region}"

    return s


POKEAPI_BASE = "https://pokeapi.co/api/v2"

@st.cache_data(ttl=60*60)
def pokeapi_get_pokemon(name_or_id: str) -> dict:
    q = to_pokeapi_name(name_or_id)
    url = f"{POKEAPI_BASE}/pokemon/{q}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def pokeapi_parse_stats(p: dict) -> dict:
    # base stats: hp, attack, defense, special-attack, special-defense, speed
    out = {}
    for s in p.get("stats", []):
        out[s["stat"]["name"]] = int(s["base_stat"])
    return out

def pokeapi_parse_types(p: dict) -> list[str]:
    return [t["type"]["name"] for t in p.get("types", [])]

def pokeapi_parse_abilities(p: dict) -> list[str]:
    # retorna nomes das abilities
    return [a["ability"]["name"] for a in p.get("abilities", [])]

def get_np_for_pokemon(df_pokedex: pd.DataFrame, pid: str, fallback_np: int = 6) -> int:
    """
    Tenta achar NP/PL no seu DF da pokedex.
    Se não achar coluna, retorna fallback.
    """
    pid = str(pid)
    row = df_pokedex[df_pokedex["Nº"].astype(str) == pid]
    if row.empty:
        return fallback_np

    # tenta colunas comuns
    for col in ["NP", "PL", "Nivel de Poder", "Nível de Poder", "Power Level"]:
        if col in row.columns:
            try:
                return int(row.iloc[0][col])
            except:
                pass
    return fallback_np

def calc_pp_budget(np_: int) -> int:
    # sua regra: NP x 2 = PP
    return int(np_) * 15

def can_add_more_attack_points(np_: int, spent_attack_points: int) -> bool:
    # trava: quando atingir limite de 20 pontos a mais do NP (você descreveu assim)
    # ✅ limite = NP + 20
    return spent_attack_points < (int(np_) + 20)

def upload_pdf_to_bucket(bucket, pdf_bytes: bytes, dest_path: str) -> str:
    """
    Faz upload no Firebase Storage (bucket do init_firebase).
    Retorna o caminho salvo.
    """
    blob = bucket.blob(dest_path)
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    return dest_path


# ---------------------------------------------------------------------
# Exportações
#
# Observação: o app.py usa `from ga_aldex.firebase_bundle import *`.
# Em Python, `import *` NÃO importa nomes iniciados com `_` a menos que
# estejam presentes em `__all__`. Como vários helpers do seu app usam
# nomes `_cg_*`, expomos tudo aqui para manter compatibilidade.
# ---------------------------------------------------------------------
__all__ = [k for k in globals().keys() if not k.startswith("__")]
