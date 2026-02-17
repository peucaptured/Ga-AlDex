"""Sync SaveData_RPG (Google Sheets) -> Firestore.

Objetivo
- Popular/atualizar docs em:
  - users/<uid> (perfil simples)
  - users_raw/<uid> (JSON completo do SaveData, SEM senha)

Isso permite que o battle-site (HTML) consiga montar cards de equipe mesmo se o jogador não abriu o Streamlit.

Como usar
1) Garanta que você tem um service account com acesso à planilha e ao Firestore.
   - O mesmo JSON que você usa no Streamlit (st.secrets['gcp_service_account']) serve.
2) Exporte esse JSON para um arquivo (ex.: service_account.json).
3) Rode:

   python tools/sync_savedata_to_firestore.py \
     --service-account service_account.json \
     --spreadsheet "SaveData_RPG" \
     --worksheet "Sheet1" \
     --project batalhas-de-gaal

Observações
- A planilha deve ter as colunas: A=Nome, B=JSON do usuário, C=Senha.
- Este script NUNCA copia a senha para o Firestore.

"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

import firebase_admin
from firebase_admin import credentials, firestore


def safe_doc_id(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name).strip("_")[:80] or "user"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_clients(service_account_path: str):
    sa = json.load(open(service_account_path, "r", encoding="utf-8"))

    # gspread
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    gs = gspread.authorize(creds)

    # firestore
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(sa))
    db = firestore.client()

    return gs, db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-account", required=True, help="Caminho do JSON do service account")
    ap.add_argument("--spreadsheet", default="SaveData_RPG", help="Nome da planilha (arquivo)")
    ap.add_argument("--worksheet", default="Sheet1", help="Nome da aba (worksheet)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gs, db = init_clients(args.service_account)

    sh = gs.open(args.spreadsheet)
    ws = sh.worksheet(args.worksheet)

    rows = ws.get_all_values()  # inclui header se houver
    if not rows:
        raise SystemExit("Planilha vazia")

    # Heurística: se primeira linha tem 'Nome' etc, pule
    start = 1 if rows[0] and ("nome" in (rows[0][0] or "").lower()) else 0

    total = 0
    ok = 0
    for i in range(start, len(rows)):
        row = rows[i]
        if not row:
            continue
        name = (row[0] or "").strip()
        if not name:
            continue
        data_json = row[1] if len(row) > 1 else ""
        if not data_json:
            continue
        total += 1

        try:
            data = json.loads(data_json)
        except Exception:
            print(f"[WARN] Linha {i+1}: JSON inválido para {name}")
            continue

        uid = safe_doc_id(name)

        # perfil leve
        prof = data.get("trainer_profile") if isinstance(data, dict) else None
        prof = prof if isinstance(prof, dict) else {}
        payload_user = {
            "uid": uid,
            "displayName": name,
            "avatar": {
                "avatar_choice": prof.get("avatar_choice"),
                "photo_thumb_b64": prof.get("photo_thumb_b64"),
            },
            "party": (data.get("party") or []) if isinstance(data.get("party"), list) else [],
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "syncedAt": utc_now_iso(),
        }

        payload_raw = {
            "uid": uid,
            "displayName": name,
            "data": data,  # SEM senha
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "syncedAt": utc_now_iso(),
        }

        if args.dry_run:
            print(f"[DRY] {name} -> users/{uid} + users_raw/{uid}")
            ok += 1
            continue

        # grava
        db.collection("users").document(uid).set(payload_user, merge=True)
        db.collection("users_raw").document(uid).set(payload_raw, merge=True)
        ok += 1

    print(f"Done. total linhas lidas={len(rows)-start}, registros processados={total}, gravados={ok}")


if __name__ == "__main__":
    main()
