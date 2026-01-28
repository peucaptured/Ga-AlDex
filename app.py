# app_unificado.py
# Arquivo ÚNICO: app.py + move_db.py + move_creator_ui.py (com cg_draft)


import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import requests
import unicodedata
import os
import io

import re
import difflib
import uuid
from datetime import datetime
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import random
import gzip
import base64
import streamlit.components.v1 as components
from advantages_engine import suggest_advantages
import queue
import threading
import time
from streamlit.runtime import scriptrunner # <--- IMPORTANTE: Importar o módulo inteiro
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.errors import StreamlitAPIException


# ================================
# MOVE DB + MOVE CREATOR (UNIFICADO)
# ================================
# (conteúdo original do move_db.py e move_creator_ui.py foi incorporado aqui para ficar em um único arquivo)


# move_db.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Utils
# ----------------------------
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _boolish(v: Any) -> bool:
    x = _norm(_safe_str(v))
    return x in {"sim", "true", "1", "yes", "y"}

def _move_based_stat_from_meta(move_meta: dict | None) -> str:
    move_meta = move_meta or {}
    cat_meta = str(move_meta.get("category", "") or "").strip().lower()

    if move_meta.get("is_special") is True:
        return "Int"
    if move_meta.get("is_special") is False:
        return "Stgr"

    if "status" in cat_meta:
        return "—"
    if "especial" in cat_meta or "special" in cat_meta:
        return "Int"
    if "físico" in cat_meta or "fisico" in cat_meta or "physical" in cat_meta:
        return "Stgr"

    return "Stgr"


def _move_stat_value(move_meta: dict | None, stats: dict) -> tuple[str, int]:
    based = _move_based_stat_from_meta(move_meta)
    if based == "Int":
        return based, int(stats.get("int", 0) or 0)
    if based == "Stgr":
        return based, int(stats.get("stgr", 0) or 0)
    return based, 0


def _move_accuracy_limit(move: dict, np_value: int, stats: dict) -> int:
    rank = int(move.get("rank", 0) or 0)
    _, stat_val = _move_stat_value(move.get("meta") or {}, stats)
    return max(0, (2 * int(np_value)) - rank - stat_val)


def _move_accuracy_pp(move: dict) -> float:
    return float(int(move.get("accuracy", 0) or 0)) / 2

# COLOQUE NO TOPO DO ARQUIVO
st.markdown("""
<style>
/* Estilo para os slots da BOX (Grama) */
.box-slot-grass {
    background: #55a64b !important;
    background-image: 
        linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px) !important;
    background-size: 8px 8px !important;
    border: 3px solid #2d5a27 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    box-shadow: inset 0 0 15px rgba(0,0,0,0.4) !important;
    margin-bottom: 8px !important;
}

/* Estilo para os cards da Equipe (Menu Azul) */
.gba-party-slot {
    background: linear-gradient(180deg, #4d88ff 0%, #2e5cb8 100%) !important;
    border: 4px solid #f8fafc !important;
    border-radius: 15px !important;
    padding: 15px !important;
    margin-bottom: 20px !important;
    box-shadow: 6px 6px 0px rgba(0,0,0,0.3) !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

.gba-party-slot img {
    filter: drop-shadow(0 4px 6px rgba(0,0,0,0.4)) !important;
}

/* Fix para remover margens extras do Streamlit dentro dos slots */
[data-testid="stVerticalBlock"] > div:has(.box-slot-grass) {
    padding: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Data model
# ----------------------------
@dataclass
class Move:
    name: str
    tipo: str
    categoria: str
    descricao: str
    build: str
    how_it_works: str
    resist_stat: str
    ranged: bool
    perception_area: bool
    tags: List[str]
    raw: Dict[str, Any]

    def render_build(self, rank: int, sub_ranks: dict | None = None) -> str:
        b = (self.build or "").strip()
        if not b:
            return ""
    
        # 1) Trocar Rank = PL por Rank = <rank>
        b = re.sub(r"Rank\s*=\s*PL", f"Rank = {rank}", b, flags=re.IGNORECASE)
        b = re.sub(r"Rank\s*=\s*X",  f"Rank = {rank}", b, flags=re.IGNORECASE)
    
        # 2) Escalar efeitos numéricos: cada efeito pode ter rank próprio via sub_ranks
        def _scale(m):
            effect = m.group(1)               # "Damage", "Weaken"...
            eff_key = effect.strip().lower()  # "damage", "weaken"...
            if sub_ranks and eff_key in sub_ranks and int(sub_ranks[eff_key]) > 0:
                r = int(sub_ranks[eff_key])
            else:
                r = int(rank)
            return f"{effect} {r}"
    
        b = re.sub(
            r"\b(Damage|Weaken|Affliction|Healing|Nullify|Create|Environment)\s+\d+\b",
            _scale,
            b,
            flags=re.IGNORECASE
        )
    
        # 3) Deduplicar segmentos "Linked ..." idênticos (exatos)
        parts = [p.strip() for p in b.split(";") if p.strip()]
        seen = set()
        uniq = []
        for p in parts:
            key = re.sub(r"\s+", " ", p.lower()).strip()
            if key not in seen:
                seen.add(key)
                uniq.append(p)
    
        # ==========================================
        # DEFINIÇÃO DE RESISTÊNCIA DO DANO (FINAL)
        # ==========================================
        name_desc = f"{self.name} {self.descricao or ''}".lower()
        tipo = (self.tipo or "").lower()
    
        damage_resist = "Thg"  # padrão absoluto
    
        # 1) DODGE — prioridade máxima
        if any(k in name_desc for k in [
            "ohko", "one-hit", "hit kill",
            "guillotine", "horn drill", "sheer cold", "fissure",
            "diferença de velocidade", "speed difference",
            "diferença de peso", "weight difference"
        ]):
            damage_resist = "Dodge"
    
        # 2) WILL — psíquico / fantasma / redução de will/spdef/spatk
        elif tipo in {"psychic", "psíquico", "ghost", "fantasma"}:
            damage_resist = "Will"
    
        elif any(k in name_desc for k in [
            "reduce will", "reduz will",
            "special defense down", "spdef down",
            "special attack down", "spatk down"
        ]):
            damage_resist = "Will"
    
        # 3) Aplica no primeiro Damage (no resultado final)
        out = "; ".join(uniq)
        out = re.sub(
            r"(Damage\s+\d+)(?![^\[]*\])",
            rf"\1 (Resisted by {damage_resist})",
            out,
            count=1,
            flags=re.IGNORECASE
        )
    
        return out



    def pp_cost(self, rank: int) -> Tuple[Optional[float], str]:
        """
        Retorna (pp_cost, explicacao). Aqui "PP" = Power Points do M&M.
    
        - Se existir coluna PP_Custo no Excel, usa direto.
        - Caso contrário, usa um estimador simples (você pode trocar depois pela sua tabela oficial).
        Regra base do M&M: custo final por graduação = custo básico + extras - falhas.
        """
    
        # (A0) Novo: "PP por Rank" vindo do Excel -> PP total = (pp_por_rank * rank)
        if "PP por Rank" in self.raw and _safe_str(self.raw.get("PP por Rank")):
            try:
                ppr = float(str(self.raw["PP por Rank"]).replace(",", "."))
                return ppr * float(rank), 'PP total = ("PP por Rank" do Excel) × rank.'
            except Exception:
                pass
    
        # (A) override do Excel
        for key in ("PP_Custo", "PP", "Custo_PP"):
            if key in self.raw and _safe_str(self.raw.get(key)):
                try:
                    val = float(str(self.raw[key]).replace(",", "."))
                    return val, f"PP_Custo vindo do Excel ({key})."
                except Exception:
                    pass


        # (B) estimador bem simples (trocável!)
        # Ideia: golpes que têm Damage tendem a escalar mais caro; Linked/Área etc aumentam.
        build = _norm(self.build)
        base_per_rank = 0.0

        if "damage" in build:
            base_per_rank += 1.0
        if "affliction" in build:
            base_per_rank += 1.0
        if "weaken" in build:
            base_per_rank += 1.0
        if "healing" in build:
            base_per_rank += 2.0
        if "create" in build:
            base_per_rank += 1.0
        if "environment" in build:
            base_per_rank += 1.0
        if "nullify" in build:
            base_per_rank += 1.0

        # extras comuns no texto
        if "linked" in build:
            base_per_rank += 0.5
        if "area" in build:
            base_per_rank += 0.5
        if "perception" in build:
            base_per_rank += 0.5
        if "extended range" in build or "ranged" in build:
            base_per_rank += 0.25

        if base_per_rank <= 0:
            return None, "Sem Custo no Excel e não foi possível estimar por falta de palavras-chave na build."
        pp_est = base_per_rank * int(rank)
        return float(pp_est), f"PP estimado ({base_per_rank:.2f} por rank)"



class MoveDB:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # normaliza colunas esperadas
        self.df["__name_norm"] = self.df["Nome"].apply(lambda x: _norm(_safe_str(x)))

        # texto para busca por descrição
        self.df["__text"] = (
            self.df["Nome"].fillna("").astype(str)
            + " | " + self.df["Tipo"].fillna("").astype(str)
            + " | " + self.df["Categoria"].fillna("").astype(str)
            + " | " + self.df["Descricao"].fillna("").astype(str)
            + " | " + self.df.get("Como funciona (regras/condições)", pd.Series([""] * len(self.df))).fillna("").astype(str)
            + " | " + self.df.get("Build M&M (adaptado)", pd.Series([""] * len(self.df))).fillna("").astype(str)
        )

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=25000,
        )
        self._tfidf = self._vectorizer.fit_transform(self.df["__text"].tolist())

    @staticmethod
    def from_excel(path: str, sheet_name: str = "Golpes_MM") -> "MoveDB":
        df = pd.read_excel(path, sheet_name=sheet_name)
        # garante colunas mínimas
        required = ["Nome", "Tipo", "Categoria", "Descricao", "Build M&M (adaptado)"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Excel sem colunas obrigatórias: {missing}")
        return MoveDB(df)

    def _row_to_move(self, row: Dict[str, Any]) -> Move:
        return Move(
            name=_safe_str(row.get("Nome")),
            tipo=_safe_str(row.get("Tipo")),
            categoria=_safe_str(row.get("Categoria")),
            descricao=_safe_str(row.get("Descricao")),
            build=_safe_str(row.get("Build M&M (adaptado)")),
            how_it_works=_safe_str(row.get("Como funciona (regras/condições)")),
            resist_stat=_safe_str(row.get("Resist Stat")),
            ranged=_boolish(row.get("Ranged")),
            perception_area=_boolish(row.get("Perception Area")),
            tags=self._infer_tags(row),
            raw=row,
        )

    def get_by_name(self, name: str) -> Optional[Move]:
        key = _norm(name)
        hit = self.df[self.df["__name_norm"] == key]
        if hit.empty:
            return None
        row = hit.iloc[0].to_dict()
        return self._row_to_move(row)

    def get_all_by_name(self, name: str) -> List[Move]:
        key = _norm(name)
        hit = self.df[self.df["__name_norm"] == key]
        if hit.empty:
            return []
        return [self._row_to_move(row.to_dict()) for _, row in hit.iterrows()]

    def search_by_name_prefix(self, name: str) -> List[Move]:
        key = _norm(name)
        if not key:
            return []
        hit = self.df[self.df["__name_norm"].str.startswith(key)]
        if hit.empty:
            return []
        return [self._row_to_move(row.to_dict()) for _, row in hit.iterrows()]

    def suggest_by_description(self, description: str, top_k: int = 5) -> List[Tuple[Move, float]]:
        q = _safe_str(description)
        if not q:
            return []
        q_vec = self._vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self._tfidf).flatten()
        idxs = sims.argsort()[::-1][:top_k]

        out: List[Tuple[Move, float]] = []
        for i in idxs:
            row = self.df.iloc[i].to_dict()
            mv = self._row_to_move(row)
            out.append((mv, float(sims[i])))
        return out

    def _infer_tags(self, row: Dict[str, Any]) -> List[str]:
        tags: List[str] = []

        # sinalizadores do seu Excel
        if _boolish(row.get("Dano")):
            tags.append("damage")
        if _boolish(row.get("Cura")):
            tags.append("healing")
        if _boolish(row.get("Teleport")):
            tags.append("teleport")
        if _boolish(row.get("Create")):
            tags.append("create")
        if _boolish(row.get("Clima")):
            tags.append("environment")
        if _boolish(row.get("Prioridade")):
            tags.append("priority")
        if _boolish(row.get("Perception Area")):
            tags.append("perception")
        if _boolish(row.get("Ranged")):
            tags.append("ranged")

        # detecta por texto
        b = _norm(_safe_str(row.get("Build M&M (adaptado)")))
        for kw, tag in [
            ("affliction", "affliction"),
            ("weaken", "weaken"),
            ("nullify", "nullify"),
            ("immunity", "immunity"),
            ("enhanced", "enhance"),
            ("linked", "linked"),
            ("area", "area"),
        ]:
            if kw in b and tag not in tags:
                tags.append(tag)

        return tags


# --- PLANO B: VIGIA DE SINCRONIZAÇÃO ---
@st.fragment(run_every=2) # Roda esta função sozinha a cada 2 segundos
def sync_watchdog(db, rid):
    if not rid:
        return
        
    try:
        # 1. Olha apenas o timestamp do mapa no banco (leitura leve)
        doc_ref = db.collection("rooms").document(rid).collection("public_state").document("state")
        snapshot = doc_ref.get()
        
        if not snapshot.exists:
            return

        server_data = snapshot.to_dict()
        # Pega a hora da última modificação no servidor
        server_time = server_data.get("updatedAt") 
        
        # Se for a primeira vez, salva na sessão
        if "last_map_update" not in st.session_state:
            st.session_state["last_map_update"] = server_time
            return

        # 2. Compara: Se o horário do servidor for diferente do que eu tenho...
        if server_time != st.session_state["last_map_update"]:
            # Atualiza meu horário local
            st.session_state["last_map_update"] = server_time
            # Força o recarregamento da página inteira para mostrar o movimento
            st.rerun()
            
    except Exception:
        pass # Se der erro de conexão, ignora e tenta na próxima
        
import streamlit as st  # reimport seguro

try:
    from move_interpreter import interpret_effects_to_build
except Exception:
    interpret_effects_to_build = None

@st.cache_resource
def load_move_db(excel_path: str) -> "MoveDB":
    return MoveDB.from_excel(excel_path, sheet_name="Golpes_MM")

def render_move_creator(
    excel_path: str,
    state_key_prefix: str = "mc",
    return_to_view: str | None = None,
):
    """
    Tela do criador de golpes (UNIFICADO):
    - Busca por nome (persistente: não some no rerun)
    - Sugestão por descrição (persistente)
    - Criar do zero
    - Confirmação adiciona em st.session_state["cg_moves"] (alias do cg_draft["moves"])
    """
    db = load_move_db(excel_path)

    # garante draft / alias
    cg_init()
    st.session_state.setdefault("cg_moves", st.session_state["cg_draft"]["moves"])
    st.session_state["cg_draft"]["moves"] = st.session_state["cg_moves"]

    # estados persistentes da tela
    last_name_key = f"{state_key_prefix}_last_name_found"
    last_desc_key = f"{state_key_prefix}_last_desc_suggestions"
    st.session_state.setdefault(last_name_key, None)
    st.session_state.setdefault(last_desc_key, [])
    st.session_state.setdefault("cg_manual_moves", [])

    st.subheader("⚔️ Criação de Golpes (M&M 3e)")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔎 Buscar por nome", "🧩 Procurar por Descrição", "🛠️ Criar Semi Manual", "✍️ Entrada Manual"]
    )

    def _move_default_accuracy(mv) -> int:
        raw = getattr(mv, "raw", {}) or {}
        return int(raw.get("Accuracy") or raw.get("Acerto") or raw.get("acerto") or 0)

    def _confirm_move(mv, rank: int, build: str, pp):
        st.session_state["cg_moves"].append({
            "name": mv.name,
            "rank": int(rank),
            "build": build,
            "pp_cost": pp,
            "accuracy": _move_default_accuracy(mv),
            "meta": {
                "ranged": bool(getattr(mv, "ranged", False)),
                "perception_area": bool(getattr(mv, "perception_area", False)),
                "category": str(getattr(mv, "categoria", "") or ""),
            }
        })

      
    
    def _render_move_card(mv, rank: int):
        st.markdown(f"### 🌀 {mv.name}  ({getattr(mv,'tipo','—')} / {getattr(mv,'categoria','—')})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ranged", "SIM" if getattr(mv, "ranged", False) else "NÃO")
        c2.metric("Perception Area", "SIM" if getattr(mv, "perception_area", False) else "NÃO")
        c3.metric("Resist Stat", getattr(mv, "resist_stat", None) or "—")

        st.write("**Descrição:**")
        st.write(getattr(mv, "descricao", None) or "—")

        st.write("**Build M&M (rank escolhido):**")
        
        # ✅ opção de escolher ranks por sub-efeito
        custom_sub = st.checkbox(
            "Quero escolher rank por sub-efeito (Damage/Affliction/Weaken etc.)",
            key=f"{state_key_prefix}_customsub_{mv.name}_{rank}"
        )
        
        sub_ranks = None
        manual_pp = None
        
        if custom_sub:
            st.caption("Defina o rank de cada sub-efeito. Se deixar 0, ele não entra / não altera.")
            cA, cB, cC = st.columns(3)
        
            with cA:
                r_damage = st.number_input(
                    "Rank Damage", min_value=0, max_value=30, value=int(rank),
                    key=f"{state_key_prefix}_r_damage_{mv.name}_{rank}"
                )
                r_aff = st.number_input(
                    "Rank Affliction", min_value=0, max_value=30, value=int(rank),
                    key=f"{state_key_prefix}_r_aff_{mv.name}_{rank}"
                )
        
            with cB:
                r_weaken = st.number_input(
                    "Rank Weaken", min_value=0, max_value=30, value=int(rank),
                    key=f"{state_key_prefix}_r_weaken_{mv.name}_{rank}"
                )
                r_heal = st.number_input(
                    "Rank Healing", min_value=0, max_value=30, value=0,
                    key=f"{state_key_prefix}_r_heal_{mv.name}_{rank}"
                )
        
            with cC:
                r_create = st.number_input(
                    "Rank Create", min_value=0, max_value=30, value=0,
                    key=f"{state_key_prefix}_r_create_{mv.name}_{rank}"
                )
                r_env = st.number_input(
                    "Rank Environment", min_value=0, max_value=30, value=0,
                    key=f"{state_key_prefix}_r_env_{mv.name}_{rank}"
                )
        
            sub_ranks = {
                "damage": int(r_damage),
                "affliction": int(r_aff),
                "weaken": int(r_weaken),
                "healing": int(r_heal),
                "create": int(r_create),
                "environment": int(r_env),
            }
        
            manual_pp = st.number_input(
                "PP total do golpe (obrigatório quando você customiza ranks)",
                min_value=0,
                value=0,
                step=1,
                key=f"{state_key_prefix}_manualpp_{mv.name}_{rank}"
            )
        
        # ✅ build: normal ou customizado
        if sub_ranks:
            build = mv.render_build(rank, sub_ranks=sub_ranks)
        else:
            build = mv.render_build(rank)
        
        st.code(build, language="text")
        
        # --- FIX: sempre inicializa antes de ramificar ---
        pp_final = None
        pp_auto = None
        why_auto = ""
        why = ""
        
        # ✅ PP: se customizou, usa o manual; se não, usa o Excel
        if sub_ranks:
            pp_final = int(manual_pp or 0)
            why = "PP informado manualmente (porque você escolheu ranks por sub-efeito)."
        else:
            tmp = mv.pp_cost(rank)
            if tmp is None:
                pp_auto, why_auto = None, "pp_cost() retornou None (erro interno)."
            else:
                pp_auto, why_auto = tmp  # pode ser None
        
            if pp_auto is None:
                # PP obrigatório manual (não definido no banco)
                pp_manual = st.number_input(
                    "PP total do golpe (obrigatório)",
                    min_value=1,
                    value=1,
                    step=1,
                    key=f"{state_key_prefix}_pp_required_{mv.name}_{rank}"
                )
                pp_final = int(pp_manual)
                why = "PP manual obrigatório (não definido no banco)."
            else:
                pp_final = pp_auto
                why = why_auto
        
        st.info(f"PP: **{pp_final}** — {why}")


        # define o PP final a ser exibido
        if pp_final is not None:
            pp = pp_final
            why = "PP definido manualmente"
        elif pp_auto is not None:
            pp = pp_auto
            why = why_auto
        else:
            pp = None
            why = "PP não definido"

        if pp is not None:
            st.info(f"PP (estimado ou do Excel): **{pp}** — {why}")
        else:
            st.warning(f"PP: não definido — {why}")

        how = getattr(mv, "how_it_works", None)
        if how:
            st.write("**Como funciona:**")
            st.write(how)

        col_confirm, col_add = st.columns(2)
        with col_confirm:
            if st.button(f"✅ Confirmar {mv.name}", key=f"{state_key_prefix}_confirm_{mv.name}_{rank}"):
                _confirm_move(mv, rank, build, pp)
                st.success(f"Adicionado: {mv.name} (Rank {rank})")

        if return_to_view:
            with col_add:
                if st.button("➕ Adicionar golpe à ficha", key=f"{state_key_prefix}_add_to_sheet_{mv.name}_{rank}"):
                    _confirm_move(mv, rank, build, pp)
                    st.success(f"Adicionado: {mv.name} (Rank {rank})")
                    st.session_state["cg_view"] = return_to_view
                    st.rerun()

    def _manual_move_to_move(manual: dict) -> Move:
        return Move(
            name=_safe_str(manual.get("Nome")),
            tipo=_safe_str(manual.get("Tipo")),
            categoria=_safe_str(manual.get("Categoria") or "Manual"),
            descricao=_safe_str(manual.get("Descricao")),
            build=_safe_str(manual.get("Formula")),
            how_it_works="",
            # ✅ AGORA LÊ A RESISTÊNCIA DO DICIONÁRIO
            resist_stat=_safe_str(manual.get("Resist Stat") or "—"),
            ranged=bool(manual.get("Ranged", False)),
            perception_area=bool(manual.get("Area", False)),
            tags=[],
            raw=manual,
        )

    def _search_moves_by_name(name: str) -> List[Move]:
        hits = db.search_by_name_prefix(name)
        manual_hits = [
            _manual_move_to_move(mv)
            for mv in st.session_state.get("cg_manual_moves", [])
            if _norm(mv.get("Nome")).startswith(_norm(name))
        ]
        return hits + manual_hits

    with tab1:
        name = st.text_input("Nome do golpe", key=f"{state_key_prefix}_name")
        rank = st.slider("Rank", 1, 20, 10, key=f"{state_key_prefix}_rank")

        if st.button("Buscar", key=f"{state_key_prefix}_search", type="primary"):
            matches = _search_moves_by_name(name)
            if not matches:
                st.session_state[last_name_key] = None
                st.error("Não achei pelo nome. Use a aba 'Criar por descrição'.")
            else:
                st.session_state[last_name_key] = name

        if st.session_state.get(last_name_key):
            for mv in _search_moves_by_name(st.session_state[last_name_key]):
                _render_move_card(mv, rank)

    with tab2:
        desc = st.text_area("Descrição do golpe", height=120, key=f"{state_key_prefix}_desc")
        rank2 = st.slider("Rank para renderizar sugestões", 1, 20, 10, key=f"{state_key_prefix}_rank2")
        top_k = st.slider("Sugestões", 3, 10, 5, key=f"{state_key_prefix}_topk")

        if st.button("Sugerir", key=f"{state_key_prefix}_suggest"):
            sugg = db.suggest_by_description(desc, top_k=top_k)
            if not sugg:
                st.session_state[last_desc_key] = []
                st.warning("Digite uma descrição.")
            else:
                st.session_state[last_desc_key] = [mv.name for (mv, _score) in sugg]

        for i, mv_name in enumerate(st.session_state.get(last_desc_key, []), start=1):
            mv = db.get_by_name(mv_name)
            if not mv:
                continue
            with st.expander(f"{i}) {mv.name}", expanded=(i == 1)):
                _render_move_card(mv, rank2)

    with tab3:
        st.subheader("🛠️ Criar Semi Manual")

        rank3 = st.slider("Rank do golpe", 1, 20, 10, key=f"{state_key_prefix}_z_rank")
        is_special = st.checkbox("Golpe Especial (Intelect Based)", value=True, key=f"{state_key_prefix}_z_special")

        st.markdown("### Efeitos")
        effects = {
            "damage": st.checkbox("Causar Dano", key=f"{state_key_prefix}_z_damage"),
            "affliction": st.checkbox("Causar Affliction", key=f"{state_key_prefix}_z_affliction"),
            "weaken": st.checkbox("Causar Weaken", key=f"{state_key_prefix}_z_weaken"),
            "healing": st.checkbox("Cura", key=f"{state_key_prefix}_z_healing"),
            "create": st.checkbox("Create", key=f"{state_key_prefix}_z_create"),
            "environment": st.checkbox("Environment", key=f"{state_key_prefix}_z_environment"),
        }

        st.markdown("### Detalhes do Weaken")
        effects["weaken_stgr"] = st.checkbox("Weaken Strength", key=f"{state_key_prefix}_z_w_stgr")
        effects["weaken_int"] = st.checkbox("Weaken Intellect", key=f"{state_key_prefix}_z_w_int")
        effects["weaken_dodge"] = st.checkbox("Weaken Dodge", key=f"{state_key_prefix}_z_w_dodge")
        effects["weaken_will"] = st.checkbox("Weaken Will", key=f"{state_key_prefix}_z_w_will")

        st.markdown("### Ranks por sub-efeito (opcional)")
        st.caption("Se você mudar algum rank aqui, você terá que informar o PP total do golpe.")

        c1, c2, c3 = st.columns(3)
        with c1:
            r_damage = st.number_input("Rank Damage", 0, 30, int(rank3), key=f"{state_key_prefix}_z_r_damage")
            r_aff = st.number_input("Rank Affliction", 0, 30, 0, key=f"{state_key_prefix}_z_r_aff")
        with c2:
            r_weaken = st.number_input("Rank Weaken", 0, 30, 0, key=f"{state_key_prefix}_z_r_weaken")
            r_heal = st.number_input("Rank Healing", 0, 30, 0, key=f"{state_key_prefix}_z_r_heal")
        with c3:
            r_create = st.number_input("Rank Create", 0, 30, 0, key=f"{state_key_prefix}_z_r_create")
            r_env = st.number_input("Rank Environment", 0, 30, 0, key=f"{state_key_prefix}_z_r_env")

        sub_ranks = {
            "damage": int(r_damage),
            "affliction": int(r_aff),
            "weaken": int(r_weaken),
            "healing": int(r_heal),
            "create": int(r_create),
            "environment": int(r_env),
        }

        # considera "custom" se qualquer coisa ficar diferente do rank3 (ou se algum efeito ganhar rank >0 diferente)
        custom_sub = any(
            (k == "damage" and sub_ranks[k] != int(rank3)) or (k != "damage" and sub_ranks[k] > 0)
            for k in sub_ranks
        )

        manual_pp = None
        if custom_sub:
            manual_pp = st.number_input(
                "PP total do golpe (obrigatório quando customiza sub-ranks)",
                min_value=0, value=0, step=1,
                key=f"{state_key_prefix}_z_manual_pp"
            )

        st.markdown("### Modificadores")
        area = st.selectbox("Área", ["Nenhuma", "Burst", "Cone", "Line"], key=f"{state_key_prefix}_z_area")
        perception = st.checkbox("Perception Area", key=f"{state_key_prefix}_z_perception")
        ranged = st.checkbox("Ranged", key=f"{state_key_prefix}_z_ranged")

        if interpret_effects_to_build:
            build = interpret_effects_to_build(
                rank=rank3,
                is_special=is_special,
                effects=effects,
                sub_ranks=sub_ranks,
                area=None if area == "Nenhuma" else area,
                perception=perception,
                ranged=ranged,
            )
        else:
            build = "Erro: move_interpreter.py não encontrado (interpret_effects_to_build)."

        st.markdown("### Build Gerada")
        st.code(build, language="text")
                # =========================
        # PP do golpe criado do zero
        # =========================
        def _estimate_pp_from_build(build_txt: str, rank: int) -> int | None:
            b = (build_txt or "").lower()
            base = 0.0

            if "damage" in b: base += 1.0
            if "affliction" in b: base += 1.0
            if "weaken" in b: base += 1.0
            if "healing" in b: base += 2.0
            if "create" in b: base += 1.0
            if "environment" in b: base += 1.0
            if "nullify" in b: base += 1.0

            # extras comuns
            if "linked" in b: base += 0.5
            if "area" in b: base += 0.5
            if "perception" in b: base += 0.5
            if "ranged" in b: base += 0.25

            if base <= 0:
                return None

            return int(round(base * int(rank)))

        pp_auto = _estimate_pp_from_build(build, rank3)

        # Se customizou ranks, PP manual é obrigatório.
        # Se não der pra estimar, também obriga manual.
        if custom_sub or pp_auto is None:
            default_pp = 1
            if custom_sub and manual_pp:
                default_pp = max(1, int(manual_pp))
            pp_final = int(st.number_input(
                "PP total do golpe (obrigatório)",
                min_value=1,
                value=int(default_pp),
                step=1,
                key=f"{state_key_prefix}_z_pp_required"
            ))
            st.info("PP manual obrigatório." if custom_sub else "PP manual obrigatório (não foi possível estimar).")
        else:
            # Mesmo estimando, deixo você ajustar
            pp_final = int(st.number_input(
                "PP total do golpe",
                min_value=1,
                value=int(pp_auto),
                step=1,
                key=f"{state_key_prefix}_z_pp"
            ))
            st.info(f"PP sugerido: **{pp_auto}** (você pode ajustar).")


        st.markdown("### 🏷️ Nome do Golpe")
        custom_name_input = st.text_input(
            "Defina o nome do seu golpe",
            value="Golpe Customizado",
            key=f"{state_key_prefix}_z_custom_name"
        )
        # Garante que não fique vazio (se o usuário apagar tudo, volta para o padrão)
        final_custom_name = custom_name_input.strip() or "Golpe Customizado"

        col_confirm_zero, col_add_zero = st.columns(2)
        with col_confirm_zero:
            can_confirm = (pp_final is not None) and (int(pp_final) > 0)
            if st.button("✅ Confirmar golpe criado do zero", key=f"{state_key_prefix}_z_confirm", disabled=not can_confirm):
                st.session_state["cg_moves"].append({
                    "name": final_custom_name,
                    "rank": int(rank3),
                    "build": build,
                    "pp_cost": int(pp_final),
                    "accuracy": 0,
                    "meta": {
                        "custom": True,
                        "sub_ranks": sub_ranks,
                        "pp_manual": bool(custom_sub),
                        "is_special": bool(is_special),
                    }
                })
                st.success(f"Golpe '{final_custom_name}' adicionado à ficha.")

        if return_to_view:
            with col_add_zero:
                if st.button("➕ Adicionar golpe à ficha", key=f"{state_key_prefix}_z_add_sheet"):
                    st.session_state["cg_moves"].append({
                        "name": final_custom_name,
                        "rank": int(rank3),
                        "build": build,
                        "pp_cost": int(pp_final),
                        "accuracy": 0,
                        "meta": {
                            "custom": True,
                            "sub_ranks": sub_ranks,
                            "pp_manual": bool(custom_sub),
                            "is_special": bool(is_special),
                        }                    })
                    st.success(f"Golpe '{final_custom_name}' adicionado à ficha.")
                    st.session_state["cg_view"] = return_to_view
                    st.rerun()

    with tab4:
        st.subheader("✍️ Entrada Manual")
        st.caption("Preencha os campos para golpes homebrew ou que não estão no Excel.")

        c_man_1, c_man_2 = st.columns([2, 1])
        with c_man_1:
            manual_name = st.text_input("Nome do Golpe", key=f"{state_key_prefix}_m_name")
        with c_man_2:
            # ✅ CAMPO NOVO: CUSTO DE PP
            manual_pp = st.number_input("Custo (PP)", min_value=1, value=1, key=f"{state_key_prefix}_m_pp")

        c_man_3, c_man_4 = st.columns(2)
        with c_man_3:
            manual_rank = st.number_input("Rank do Efeito", min_value=1, max_value=20, value=10, key=f"{state_key_prefix}_m_rank")
        with c_man_4:
            manual_accuracy = st.number_input("Bônus de Acerto", min_value=0, max_value=30, value=0, key=f"{state_key_prefix}_m_accuracy")

        manual_formula = st.text_area("Fórmula / Build (Opcional)", height=80, key=f"{state_key_prefix}_m_formula", placeholder="Ex: Damage 10, Ranged")
        manual_desc = st.text_area("Descrição", height=100, key=f"{state_key_prefix}_m_desc")
        
        # --- Configurações Técnicas ---
        st.markdown("**Configurações:**")
        c_cfg1, c_cfg2 = st.columns(2)
        with c_cfg1:
            manual_type = st.text_input("Tipo (ex: Fogo, Psíquico)", key=f"{state_key_prefix}_m_type")
            # ✅ CAMPO NOVO: RESISTÊNCIA
            manual_resist = st.selectbox(
                "Resistência (Save do Alvo)", 
                ["Toughness", "Fortitude", "Will", "Dodge", "Parry", "—"],
                index=0, # Padrão Toughness
                key=f"{state_key_prefix}_m_resist"
            )
        with c_cfg2:
            st.caption("Flags:")
            manual_area = st.checkbox("É Área?", key=f"{state_key_prefix}_m_area")
            manual_ranged = st.checkbox("É Ranged?", key=f"{state_key_prefix}_m_ranged")
            manual_is_special = st.checkbox("É Especial (Int)?", key=f"{state_key_prefix}_m_special")

        # Define categoria baseada na flag
        cat_manual = "Manual Special" if manual_is_special else "Manual Physical"

        manual_data = {
            "Nome": manual_name.strip(),
            "Rank": int(manual_rank),
            "Formula": manual_formula.strip(),
            "Acerto": int(manual_accuracy),
            "Tipo": manual_type.strip(),
            "Descricao": manual_desc.strip(),
            "Area": bool(manual_area),
            "Ranged": bool(manual_ranged),
            "Categoria": cat_manual,
            # ✅ SALVANDO NOVOS DADOS
            "Resist Stat": manual_resist, 
            "PP_Custo": int(manual_pp)
        }

        can_save_manual = bool(manual_data["Nome"])

        col_save, col_save_add = st.columns(2)
        with col_save:
            if st.button("💾 Salvar no repertório", key=f"{state_key_prefix}_m_save", disabled=not can_save_manual):
                st.session_state["cg_manual_moves"].append(manual_data)
                st.success("Golpe manual salvo no repertório (disponível na busca).")

        with col_save_add:
            if st.button("✅ Salvar e adicionar à ficha", key=f"{state_key_prefix}_m_save_add", disabled=not can_save_manual):
                st.session_state["cg_manual_moves"].append(manual_data)
                
                # Converte para objeto Move
                mv = _manual_move_to_move(manual_data)
                
                # ✅ Passa o PP manual diretamente
                _confirm_move(mv, rank=int(manual_data["Rank"]), build=mv.build, pp=int(manual_data["PP_Custo"]))
                
                st.success("Golpe adicionado à ficha com sucesso!")

    st.divider()
    st.subheader("📦 Golpes confirmados nesta ficha")

    if not st.session_state["cg_moves"]:
        st.info("Nenhum golpe confirmado ainda.")
    else:
        for i, m in enumerate(list(st.session_state["cg_moves"])):
            c1, c2, c3 = st.columns([6, 2, 2])
            with c1:
                accuracy = int(m.get("accuracy", 0) or 0)
                st.write(f"**{m['name']}** (Rank {m['rank']}) — PP: {m.get('pp_cost')} | Acerto {accuracy}")
                build_txt = (m.get("build") or "").strip()
                if build_txt:
                    with st.expander("Ingredientes do golpe"):
                        st.code(build_txt, language="text")
            with c2:
                stats = st.session_state.get("cg_draft", {}).get("stats", {})
                np_value = int(st.session_state.get("cg_np", 0) or 0)
                
                # --- CÓDIGO NOVO (CORREÇÃO) ---
                acc_limit = _move_accuracy_limit(m, np_value, stats)
                current_acc = int(m.get("accuracy", 0) or 0)
                
                # O "pulo do gato": o máximo é o limite OU o valor atual (o que for maior)
                # Isso impede o erro se você digitou um valor alto manualmente antes
                safe_max = max(acc_limit, current_acc)

                new_acc = st.number_input(
                    "Acerto",
                    min_value=0,
                    max_value=safe_max,  # <--- Aqui usamos o safe_max
                    value=current_acc,
                    step=1,
                    key=f"{state_key_prefix}_acc_{i}",
                )
                
                st.caption(f"Limite: {acc_limit}")
                if current_acc > acc_limit:
                    st.warning("⚠️ Acima do limite!")

                if st.button("Definir acerto", key=f"{state_key_prefix}_set_acc_{i}"):
                    m["accuracy"] = int(new_acc)
                    st.rerun()
                # --- FIM DO CÓDIGO NOVO ---
            with c3:
                if st.button("❌ Remover", key=f"{state_key_prefix}_remove_{i}"):
                    st.session_state["cg_moves"].pop(i)
                    st.rerun()






from io import BytesIO
from PIL import ImageFont
if "carousel_click" not in st.session_state:
    st.session_state["carousel_click"] = None


SKILLS_MM3 = [
    "Acrobatics",
    "Athletics",
    "Close Combat (especialidade)",
    "Deception",
    "Expertise (especialidade)",
    "Insight",
    "Intimidation",
    "Investigation",
    "Perception",
    "Persuasion",
    "Ranged Combat (especialidade)",
    "Sleight of Hand",
    "Stealth",
    "Technology",
    "Treatment",
    "Vehicles",
]
# ================================
# CRIAÇÃO GUIADA - DRAFT ÚNICO
# ================================
def cg_init(reset: bool = False):
    """Inicializa (ou reseta) o rascunho único da ficha guiada."""
    if reset or ("cg_draft" not in st.session_state):
        st.session_state["cg_draft"] = {
            "pname": "",
            "stats": {
                "stgr": 0,
                "int": 0,
                "dodge": 0,
                "parry": 0,
                "thg": 0,
                "fortitude": 0,
                "will": 0,
            },
            "moves": [],
        }
    # 🔗 alias: cg_moves aponta para a MESMA lista do draft
    st.session_state.setdefault("cg_moves", st.session_state["cg_draft"]["moves"])
    st.session_state["cg_draft"]["moves"] = st.session_state["cg_moves"]

def cg_sync_from_widgets():
    """Sincroniza valores dos widgets (keys) para o dicionário cg_draft."""
    d = st.session_state.get("cg_draft")
    if not d:
        return
    if "cg_pname" in st.session_state:
        d["pname"] = st.session_state["cg_pname"]

    for k_widget, k_stat in [
        ("cg_stgr", "stgr"),
        ("cg_int", "int"),
        ("cg_dodge", "dodge"),
        ("cg_parry", "parry"),
        ("cg_thg", "thg"),
        ("cg_fortitude", "fortitude"),
        ("cg_will", "will"),
    ]:
        if k_widget in st.session_state:
            d["stats"][k_stat] = st.session_state[k_widget]





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
from datetime import timezone

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def upload_pdf_to_storage(bucket, pdf_bytes: bytes, storage_path: str):
    blob = bucket.blob(storage_path)
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    return storage_path

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

import re

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
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --gba-bg: #0f172a;
  --gba-panel: #f8fafc;
  --gba-border: #334155;
  --gba-ink: #0f172a;
  --gba-ink2: #475569;
  --gba-accent: #38bdf8;
  --gba-accent-border: #0ea5e9;
}

/* =========================================================
   1) FUNDO (pega o “miolo” do Streamlit também)
   ========================================================= */
html, body{
  background: var(--gba-bg) !important;
}
.stApp{
  background:
    radial-gradient(circle at 20% 10%, rgba(255,255,255,0.06), transparent 35%),
    radial-gradient(circle at 90% 30%, rgba(255,255,255,0.04), transparent 40%),
    var(--gba-bg) !important;
}
div[data-testid="stAppViewContainer"]{{
  background: radial-gradient(ellipse at center, #0a0a0a 0%, #000 62%, #000 100%) !important;
}}
div[data-testid="stAppViewContainer"] > .main{{ background: transparent !important; }}
div[data-testid="stSidebar"]{{ background: #000 !important; }}
div[data-testid="stHeader"]{{ background: transparent !important; }}

/* =========================================================
   2) FONTE RETRÔ E CONTRASTE GLOBAL (FORÇADO)
   ========================================================= */
/* Aplica a fonte e a cor clara em TODOS os elementos possíveis */
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown span, li,
h1, h2, h3, h4, label, .stWidgetLabel,
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="tab"], [data-baseweb="tab-list"],
.pokedex-info-value, .pokedex-info-title, .section-title {
  font-family: "Press Start 2P", cursive !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
  color: #f8fafc !important; /* Branco fosco para máximo contraste */
  text-shadow: 1px 1px 0px #000; /* Sombra leve para destacar no fundo escuro */
}

/* Força o texto claro especificamente dentro das abas e containers verticais */
div[data-testid="stVerticalBlock"] div, 
div[role="tabpanel"] p, 
div[role="tabpanel"] span,
div[data-testid="stExpander"] p {
    color: #f8fafc !important;
}

/* Ajuste para que o texto digitado nos campos de busca também seja visível */
.stTextInput input {
    color: #ffffff !important;
    background-color: rgba(0, 0, 0, 0.5) !important;
}

/* =========================================================
   3) PROTEÇÃO CONTRA BUG DE ÍCONE (keyboard_arrow_right etc.)
   ========================================================= */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] svg,
[data-testid="stHeader"] svg,
.stSelectbox svg,
.stMultiSelect svg,
div[data-baseweb="icon"],
span[class*="icon"],
i[class*="icon"]{
  font-family: sans-serif !important;
}

/* =========================================================
   4) Sprites pixelados
   ========================================================= */
img{ image-rendering: pixelated !important; image-rendering: crisp-edges !important; }

/* =========================================================
   5) “Janelas” GBA
   ========================================================= */
.gba-window{
  background: rgba(255,255,255,0.08);
  border: 2px solid rgba(148,163,184,0.35);
  border-radius: 16px;
  padding: 12px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.25);
  margin-bottom: 14px;
}
.gba-window.party{ background: rgba(59,130,246,0.10); }
.gba-window.box{ background: rgba(34,197,94,0.08); }
.gba-window.summary{ background: rgba(255,255,255,0.08); }

/* Cabeçalho */
.gba-header{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:10px;
  margin-bottom: 10px;
}
.gba-chip{
  display:inline-flex;
  align-items:center;
  gap:6px;
  white-space:nowrap;
  background: rgba(15,23,42,0.70);
  color: rgba(255,255,255,0.95);
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 11px;
}
.gba-title{
  font-family: "Press Start 2P", cursive !important;
  color: rgba(255,255,255,0.92) !important;
  font-size: 12px !important;
  line-height: 1.2 !important;
}

/* =========================================================
   6) Botões (contraste garantido)
   ========================================================= */
div.stButton > button, div.stDownloadButton > button{
  background: var(--gba-accent) !important;
  color: #0b1220 !important;
  border: 2px solid var(--gba-accent-border) !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  font-weight: 800 !important;
  box-shadow: 0 6px 12px rgba(0,0,0,0.18) !important;
}
div.stButton > button:hover{ filter: brightness(1.05); transform: translateY(-1px); }
div.stButton > button:active{ transform: translateY(0px); }

/* =========================================================
   7) Tabs (cartucho) + remove underline
   ========================================================= */
.stTabs [data-baseweb="tab-list"]{
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.75), rgba(37, 99, 235, 0.35)) !important;
  border: 2px solid rgba(148, 163, 184, 0.35) !important;
  border-radius: 12px !important;
  padding: 6px !important;
  gap: 6px !important;
  border-bottom: 0 !important;
  box-shadow: none !important;
}
div[data-baseweb="tab-border"]{ display:none !important; }
.stTabs [data-baseweb="tab"]{
  background: rgba(30, 64, 175, 0.22) !important;
  color: rgba(255,255,255,0.92) !important;
  border: 2px solid rgba(59, 130, 246, 0.40) !important;
  border-radius: 10px !important;
  font-weight: 900 !important;
}
.stTabs [aria-selected="true"]{
  background: var(--gba-accent) !important;
  color: #0b1220 !important;
  border-color: var(--gba-accent-border) !important;
}

/* =========================================================
   8) BOX com “grama” (AGORA SIM)
   ========================================================= */
.grass-box{
  border: 2px solid rgba(34,197,94,0.75);
  border-radius: 16px;
  padding: 12px;
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(180deg, rgba(16,185,129,0.20), rgba(34,197,94,0.10)),
    repeating-linear-gradient(
      135deg,
      rgba(34,197,94,0.16) 0px,
      rgba(34,197,94,0.16) 6px,
      rgba(22,163,74,0.12) 6px,
      rgba(22,163,74,0.12) 12px
    );
  box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}
.grass-box::after{
  content:"";
  position:absolute; inset:0;
  background-image: radial-gradient(rgba(255,255,255,0.06) 1px, transparent 1px);
  background-size: 10px 10px;
  opacity: .55;
  pointer-events:none;
}

/* =========================================================
   9) PARTY com slots quadrados
   ========================================================= */
.team-box{
  border: 2px solid rgba(59,130,246,0.55);
  border-radius: 16px;
  padding: 12px;
  background: rgba(15,23,42,0.35);
  box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}
.team-slots{
  display:grid;
  grid-template-columns: repeat(4, minmax(54px, 72px));
  gap: 10px;
  align-items:center;
}
.team-slot{
  aspect-ratio: 1 / 1;
  border-radius: 12px;
  border: 2px solid rgba(255,255,255,0.18);
  background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
  box-shadow: inset 0 0 0 2px rgba(0,0,0,0.12);
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:hidden;
}
.team-slot img{
  width: 85%;
  height: 85%;
  object-fit: contain;
  image-rendering: pixelated;
  filter: drop-shadow(0 4px 8px rgba(0,0,0,0.25));
}
.box-slot-grass {
  background: #55a64b; /* Cor base da grama */
  background-image: 
    linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
  background-size: 8px 8px; /* Efeito de pixels/quadriculado */
  border: 2px solid #2d5a27;
  border-radius: 8px;
  padding: 5px;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 5px;
  box-shadow: inset 0 0 10px rgba(0,0,0,0.3);
}
/* Fundo de grama individual para cada Pokémon na BOX */
.box-slot-grass {
  background: #55a64b;
  background-image: 
    linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
  background-size: 8px 8px;
  border: 2px solid #2d5a27;
  border-radius: 8px;
  padding: 5px;
  display: flex;
  justify-content: center;
  align-items: center;
  box-shadow: inset 0 0 10px rgba(0,0,0,0.3);
  margin-bottom: 5px;
}


/* Card estilo GBA para a Equipe Ativa */
.gba-party-slot {
  background: linear-gradient(180deg, #4d88ff 0%, #2e5cb8 100%);
  border: 3px solid #f8fafc;
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 15px;
  box-shadow: 4px 4px 0px rgba(0,0,0,0.2);
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  overflow: hidden;
}

/* Detalhe de luz no card da equipe */
.gba-party-slot::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 50%;
  background: rgba(255,255,255,0.1);
  pointer-events: none;
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

def stop_pvp_sync_listener():
    sync_data = st.session_state.get("pvp_sync_listener")
    if not sync_data:
        return
    for unsubscribe in sync_data.get("unsubscribers", []):
        try:
            unsubscribe()
        except Exception:
            pass
    stop_event = sync_data.get("stop_event")
    event_queue = sync_data.get("queue")
    if stop_event:
        stop_event.set()
    if event_queue:
        event_queue.put({"tag": "stop"})
    st.session_state.pop("pvp_sync_listener", None)


def render_ds_tools_nav(selected_view: str):
    opts = ["Menu", "NPCs", "Ginásios", "Locais", "Sair"]

    to_view = {
        "Menu": "home",
        "NPCs": "npcs",
        "Ginásios": "ginasios",
        "Locais": "locais",
        "Sair": "sair",
    }
    from_view = {v: k for k, v in to_view.items()}

    default_label = from_view.get(selected_view, "NPCs")
    idx = opts.index(default_label)

    st.markdown("""
    <style>
      /* ====== RESGATE (caso você tenha CSS que esconde widgets) ====== */
      .ds-topnav div[data-testid="stRadio"],
      .ds-topnav div[data-testid="stRadio"] *{
        visibility: visible !important;
        opacity: 1 !important;
        display: revert !important;
      }
      /* garante que o radio não seja escondido pelo seu CSS global */
      .ds-topnav div[data-testid="stRadio"]{
        visibility: visible !important;
        opacity: 1 !important;
      }
        
      /* radiogroup horizontal e centralizado */
      .ds-topnav div[data-testid="stRadio"] [role="radiogroup"]{
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 58px !important;
      }

      /* cada opção (label) */
      .ds-topnav div[data-testid="stRadio"] [role="radiogroup"] > label{
        position: relative;
        margin: 0 !important;
        padding: 6px 10px !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        color: rgba(255,255,255,0.70) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.24em !important;
        font-size: 12px !important;
        user-select: none !important;
        transition: all 120ms ease !important;
      }

      /* o texto do radio no Streamlit geralmente fica em p */
      .ds-topnav div[data-testid="stRadio"] label p{
        position: relative;
        padding-left: 0;
        margin: 0 !important;
        user-select: none;
        font-family: "DarkSouls", serif;
        font-size: 12px;
        letter-spacing: 0.24em;
        line-height: 1;
        text-transform: uppercase;
        color: rgba(255,255,255,0.70);
      }

      .ds-topnav div[data-testid="stRadio"] [role="radiogroup"] > label:hover{
        color: #FFD700 !important;
        text-shadow: 0 0 10px rgba(255,215,0,0.65) !important;
      }

      .ds-topnav div[data-testid="stRadio"] [role="radiogroup"] > label[data-checked="true"]{
        color: #FFD700 !important;
        text-shadow: 0 0 10px rgba(255,215,0,0.65) !important;
      }

      /* esconde o circulinho padrão do Streamlit */
      .ds-topnav div[data-testid="stRadio"] input[type="radio"]{
        display: none !important;
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='ds-topnav'>", unsafe_allow_html=True)
    choice = st.radio(
        "",
        opts,
        index=idx,
        horizontal=True,
        key="comp_topnav_radio",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    new_view = to_view[choice]

    if new_view == "sair":
        st.session_state["nav_to"] = "Pokédex (Busca)"
        st.rerun()
    elif new_view != st.session_state.get("comp_view"):
        st.session_state["comp_view"] = new_view
        if new_view != "npcs":
            st.session_state["comp_selected_npc"] = None
        st.rerun()


def render_compendium_page() -> None:
    if "comp_view" not in st.session_state:
        st.session_state["comp_view"] = "home"

    # topo só fora da home
    if st.session_state["comp_view"] != "home":
        render_ds_tools_nav(st.session_state["comp_view"])

    if st.session_state["comp_view"] == "home":
        render_compendium_home()
        return

    if st.session_state["comp_view"] == "npcs":
        render_compendium_npcs()
        return

    if st.session_state["comp_view"] == "ginasios":
        render_compendium_ginasios()
        return

    if st.session_state["comp_view"] == "locais":
        render_compendium_locais()
        return




def ensure_pvp_sync_listener(db, rid):
    """
    Inicia listeners do Firestore (Real-Time) para sincronizar o jogo 
    automaticamente APENAS quando houver interação de qualquer jogador.
    """
    if not rid:
        return
    
    # Se já existe um listener ativo para essa sala, não recria
    active = st.session_state.get("pvp_sync_listener")
    if active and active.get("rid") == rid:
        return

    # Se trocou de sala, mata o anterior
    stop_pvp_sync_listener()

    # Prepara filas e eventos para thread
    event_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    
    # Pega o contexto da sessão atual do usuário ANTES de entrar na thread
    ctx = get_script_run_ctx() 

    # Função auxiliar para colocar na fila de processamento
    def enqueue(tag: str, source: str):
        if stop_event.is_set():
            return
        # Coloca na fila para a thread principal processar
        event_queue.put({"tag": tag, "source": source, "ts": time.time()})

    # --- CALLBACKS DO FIRESTORE (Executam quando o banco muda) ---
    
    # 1. OUVINTE DE MAPA/PEÇAS
    def on_state_snapshot(doc_snapshot, changes, read_time):
        enqueue("state_change", "map")

    # 2. OUVINTE DE HP/STATUS (Party)
    def on_party_snapshot(col_snapshot, changes, read_time):
        enqueue("state_change", "party")

    # 3. OUVINTE DE EVENTOS (Dados, Logs, Ataques)
    def on_events_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == "ADDED": # Só queremos novos eventos
                enqueue("new_event", "log")

    # --- WORKER: A Thread que força o RERUN do Streamlit ---
    def rerun_worker():
        # Anexa esta thread à sessão do usuário (CRUCIAL para o st.rerun funcionar)
        if ctx:
            add_script_run_ctx(threading.current_thread(), ctx)
        
        while not stop_event.is_set():
            try:
                # Espera passiva: não gasta CPU, fica travado aqui até o Firebase mandar algo
                # Timeout de 3s para checar se deve parar
                item = event_queue.get(timeout=3) 
                
                if item.get("tag") == "stop":
                    break
                
                # Se houver contexto, solicita o Rerun
                if ctx and ctx.script_requests:
                    # Cria a solicitação de rerun (A "mágica" do refresh)
                    rerun_data = scriptrunner.RerunData(
                        query_string=ctx.query_string,
                        page_script_hash=ctx.page_script_hash,
                        cached_message_hashes=ctx.cached_message_hashes,
                    )
                    ctx.script_requests.request_rerun(rerun_data)
                    
                    # Pequena pausa para evitar "piscar" demais se vierem muitos eventos juntos
                    time.sleep(0.5) 
                else:
                    # Se perdeu o contexto (usuário fechou a aba), para a thread
                    break
                    
            except queue.Empty:
                continue # Timeout do get, volta a esperar
            except Exception as e:
                print(f"Erro no worker de sync: {e}")
                # Não damos break aqui para tentar recuperar em próximos eventos
                time.sleep(1)

    # --- REGISTRO DOS LISTENERS NO FIREBASE ---
    try:
        # Listener 1: Documento de Estado (Mapa)
        state_unsub = db.collection("rooms").document(rid).collection("public_state").document("state").on_snapshot(on_state_snapshot)
        
        # Listener 2: Coleção de Eventos (Últimos 5 eventos)
        events_query = db.collection("rooms").document(rid).collection("public_events").order_by("ts", direction=firestore.Query.DESCENDING).limit(5)
        events_unsub = events_query.on_snapshot(on_events_snapshot)

        # Listener 3: Documento de Party (HP/Status)
        party_unsub = db.collection("rooms").document(rid).collection("public_state").document("party_states").on_snapshot(on_party_snapshot)

        # Inicia a Thread Worker
        worker = threading.Thread(target=rerun_worker, daemon=True)
        worker.start()

        # Salva na sessão para podermos cancelar depois
        st.session_state["pvp_sync_listener"] = {
            "rid": rid,
            "queue": event_queue,
            "stop_event": stop_event,
            "unsubscribers": [state_unsub, events_unsub, party_unsub], # Guarda funções para desligar
        }
    except Exception as e:
        st.error(f"Erro ao conectar no sync: {e}")


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
    st.title("Bem-vindo(a) ao Ga'Al")
    
    tab_login, tab_register = st.tabs(["🔑 Acessar", "📝 Cadastrar"])
    
    # ABA DE LOGIN
    with tab_login:
        l_user = st.text_input("Nome do treinador", key="l_user")
        l_pass = st.text_input("Senha", type="password", key="l_pass")
        
        if st.button("Acessar", type="primary"):
            if l_user and l_pass:
                with st.spinner("Verificando credenciais..."):
                    result = authenticate_user(l_user, l_pass)
                    
                    if result == "WRONG_PASS":
                        st.error("🚫 Senha incorreta. Verifique e tente novamente.")
                    elif result == "NOT_FOUND":
                        st.warning("Treinador não encontrado. Cadastre-se na aba ao lado.")
                    elif isinstance(result, dict):
                        st.session_state['trainer_name'] = l_user
                        st.session_state['user_data'] = result
                        st.rerun()
    
    # ABA DE REGISTRO
    with tab_register:
        st.info("Crie sua conta para salvar seu progresso e sincronizar seus dados.")
        r_user = st.text_input("Nome do treinador", key="r_user")
        r_pass = st.text_input("Senha", type="password", key="r_pass")
        
        if st.button("Criar Conta"):
            if r_user and r_pass:
                with st.spinner("Registrando..."):
                    res = register_new_user(r_user, r_pass)
                    if res == "SUCCESS":
                        st.success("Conta criada! Vá na aba 'Entrar' para fazer login.")
                    elif res == "EXISTS":
                        st.error("Esse nome de treinador já está em uso.")
                    else:
                        st.error("Não foi possível criar a conta. Tente novamente.")
            else:
                st.warning("Preencha nome e senha para continuar.")
                
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


def get_room_player_names(room: dict) -> set:
    names = set()
    owner = (room.get("owner") or {}).get("name")
    if owner:
        names.add(owner)
    challengers = room.get("challengers") or []
    names.update([c.get("name") for c in challengers if c.get("name")])
    legacy_challenger = room.get("challenger")
    if isinstance(legacy_challenger, dict):
        legacy_name = legacy_challenger.get("name")
    else:
        legacy_name = legacy_challenger
    if legacy_name:
        names.add(legacy_name)
    return names

def delete_room(db, rid: str):
    room_ref = db.collection("rooms").document(rid)
    for doc in room_ref.collection("public_state").list_documents():
        doc.delete()
    for doc in room_ref.collection("public_events").list_documents():
        doc.delete()
    room_ref.delete()

def cleanup_room_if_orphaned(db, rid: str):
    room = get_room(db, rid)
    if not room:
        return
    player_names = get_room_player_names(room)
    if not player_names:
        return
    for name in player_names:
        if rid in list_my_rooms(db, name):
            return
    delete_room(db, rid)
            
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
    # 1. Adiciona o evento no histórico (como antes)
    db.collection("rooms").document(rid).collection("public_events").add({
        "type": event_type,
        "by": by,
        "payload": payload or {},
        "ts": firestore.SERVER_TIMESTAMP,
    })
    
    # 2. NOVO: Atualiza o timestamp do ESTADO para disparar o sync_watchdog de todos
    # Isso garante que quem está na sala veja o dado/log aparecer sozinho
    db.collection("rooms").document(rid).collection("public_state").document("state").update({
        "updatedAt": firestore.SERVER_TIMESTAMP
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
    ref = db.collection("rooms").document(rid).collection("public_state").document("party_states")
    
    key = f"{trainer_name}.{pid}"
    data = {
        key: {
            "hp": int(hp),
            "cond": conditions,
            "updatedAt": str(datetime.now()) # Timestamp local serve, mas muda o dado
        }
    }
    # Força um campo global de update para disparar o listener da coleção/documento
    data["last_update"] = firestore.SERVER_TIMESTAMP 
    
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
    # Garante que o ID existe
    sref = state_ref_for(db, rid)
    # Precisamos ler o estado atual para não perder as outras peças
    # (Ou usar arrayUnion se fosse uma lista simples, mas é lista de dicts)
    
    # Melhor abordagem para evitar race conditions em real-time: Transação
    # Mas para simplificar no seu código atual:
    stt = get_state(db, rid)
    pieces = stt.get("pieces") or []

    # Remove a versão antiga da peça, se existir
    new_pieces = [p for p in pieces if p.get("id") != piece.get("id")]
    new_pieces.append(piece)

    sref.set({
        "pieces": new_pieces,
        "updatedAt": firestore.SERVER_TIMESTAMP, # <--- GATILHO DO SYNC
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

TILE_SIZE = 96

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


# -----------------------------
# Viabilidade -> Arquétipos (presets)
# -----------------------------

def _viab_split_blocks(viab_text: str) -> list[str]:
    """Divide um texto de Viabilidade em blocos por código (ex: CDL/FIR/...)."""
    if not isinstance(viab_text, str):
        return []
    t = viab_text.strip()
    if not t:
        return []
    # split por início de linha com código de 3 letras (tolerando bullets)
    pattern = r"(?=(?:^|\n)\s*(?:\*\*|[\-•>])?\s*[CFS][ODFIC][RL]\b)"
    parts = re.split(pattern, t, flags=re.IGNORECASE)
    blocks = [p.strip() for p in parts if p.strip()]
    return blocks if blocks else [t]

def _viab_extract_code(block: str) -> str | None:
    if not isinstance(block, str):
        return None
    m = re.match(r"^\s*(?:\*\*|[\-•>])?\s*([CFS][ODFIC][RL])\b", block.strip(), flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def _viab_extract_ability(block: str) -> str | None:
    if not isinstance(block, str):
        return None
    # "Habilidade: chlorophyll" / "habilidade: Torrent" / "a habilidade Rain Dish"
    m = re.search(r"\bhabilidade\b\s*:\s*([^\.\n;]+)", block, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\ba\s+habilidade\b\s*[:\-]?\s*([A-Za-z0-9' _\-]+)", block, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

def _viab_extract_partners(block: str) -> list[str]:
    if not isinstance(block, str):
        return []
    m = re.search(r"\bparceiros\b\s*:\s*(.+?)(?:\.|\n|\bexplica[cç][aã]o\b|$)", block, flags=re.IGNORECASE)
    if not m:
        return []
    seg = m.group(1)
    # remove parênteses e descrições
    seg = re.sub(r"\(.*?\)", "", seg)
    seg = seg.replace(" e ", ", ")
    parts = [p.strip(" .;:-").upper() for p in seg.split(",") if p.strip(" .;:-")]
    # remove vazios/duplicados mantendo ordem
    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out

def _viab_extract_moves_tokens(block: str) -> list[str]:
    """Extrai uma lista 'crua' de tokens de golpes a partir do texto de Viabilidade."""
    if not isinstance(block, str):
        return []
    b = block.strip()
    if not b:
        return []

    seg = None
    m = re.search(
        r"\bcom\s+os\s+golpes\b\s*:\s*(.+?)(?:\.|\n|\be\s+a\s+habilidade\b|\bhabilidade\b\s*:|\bparceiros\b\s*:|\bexplica[cç][aã]o\b|$)",
        b,
        flags=re.IGNORECASE,
    )
    if m:
        seg = m.group(1)
    else:
        # fallback: tudo após o primeiro ":" (até habilidade/parceiros/explicação)
        if ":" in b:
            seg = b.split(":", 1)[1]
            seg = re.split(r"\b(?:e\s+a\s+habilidade|habilidade\s*:|parceiros\s*:|explica[cç][aã]o)\b", seg, flags=re.IGNORECASE)[0]

    if not seg:
        return []

    seg = re.sub(r"\(.*?\)", "", seg)
    seg = seg.replace(" e ", ", ")
    seg = seg.replace(";", ",")
    tokens = [t.strip(" .;:-") for t in seg.split(",") if t.strip(" .;:-")]
    # remove duplicados mantendo ordem
    seen = set()
    out = []
    for t in tokens:
        key = _norm(t)
        if key and key not in seen:
            out.append(t)
            seen.add(key)
    return out

def _viab_parse_archetypes(viab_text: str) -> list[dict]:
    """Retorna lista de arquétipos extraídos da coluna Viabilidade."""
    blocks = _viab_split_blocks(viab_text)
    out = []
    for b in blocks:
        code = _viab_extract_code(b)
        moves_tokens = _viab_extract_moves_tokens(b)
        ability = _viab_extract_ability(b)
        partners = _viab_extract_partners(b)
        # label amigável
        label = (b.split(":", 1)[0] if ":" in b else (code or "Arquétipo")).strip()
        if code and not label.upper().startswith(code):
            label = f"{code} — {label}"
        out.append({
            "code": code,
            "label": label[:80] + ("…" if len(label) > 80 else ""),
            "raw": b,
            "moves_tokens": moves_tokens,
            "ability": ability,
            "partners": partners,
        })
    return out

def _viab_find_moves_in_text(db: "MoveDB", text_blob: str) -> list["Move"]:
    """Fallback: encontra golpes do banco que aparecem como substring em um trecho de texto."""
    if db is None or not isinstance(text_blob, str) or not text_blob.strip():
        return []
    t = text_blob.lower()
    # normaliza pontuação -> espaço
    t = re.sub(r"[^a-z0-9'\-\s]", " ", t)
    t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()

    found: list[Move] = []
    seen = set()

    # varre nomes do banco (875 aprox. -> ok)
    try:
        names = db.df["Nome"].dropna().astype(str).tolist()
    except Exception:
        names = []

    # prioriza nomes maiores para reduzir falsos positivos
    names.sort(key=lambda x: len(str(x)), reverse=True)

    for nm in names:
        nml = str(nm).lower()
        nml = re.sub(r"[^a-z0-9'\-\s]", " ", nml).replace("-", " ")
        nml = re.sub(r"\s+", " ", nml).strip()
        if not nml:
            continue
        # boundary simples
        if re.search(r"(?<![a-z0-9])" + re.escape(nml) + r"(?![a-z0-9])", t):
            mv = db.get_by_name(nm)
            if mv:
                k = _norm(mv.name)
                if k and k not in seen:
                    found.append(mv)
                    seen.add(k)
        if len(found) >= 20:
            break

    return found

def _viab_suggest_moves(db: "MoveDB", archetype: dict, include_fulltext: bool = True) -> tuple[list["Move"], list[str]]:
    """Dado um arquétipo, sugere Moves (encontrados no banco) e lista tokens não encontrados.

    - Quando include_fulltext=True, faz um *fallback* varrendo o texto completo do arquétipo para achar nomes do banco.
    - Para o modo "núcleo", use include_fulltext=False para ficar fiel apenas aos golpes explicitamente listados.
    """
    if db is None or not archetype:
        return [], []

    tokens = archetype.get("moves_tokens", []) or []
    found: list[Move] = []
    missing: list[str] = []
    seen = set()

    for tok in tokens:
        mv = _try_match_move_in_db(db, tok)
        if mv:
            k = _norm(mv.name)
            if k and k not in seen:
                found.append(mv)
                seen.add(k)
            continue

        # token pode vir colado (ex: "Withdraw Dark Pulse")
        sub = _viab_find_moves_in_text(db, tok)
        if sub:
            for mv2 in sub:
                k = _norm(mv2.name)
                if k and k not in seen:
                    found.append(mv2)
                    seen.add(k)
        else:
            missing.append(tok)

    # fallback final: varrer no texto completo (apenas no modo completo)
    if include_fulltext:
        full_hits = _viab_find_moves_in_text(db, archetype.get("raw", ""))
        for mv in full_hits:
            k = _norm(mv.name)
            if k and k not in seen:
                found.append(mv)
                seen.add(k)

    return found, missing


def _viab_apply_to_session(
    archetype: dict,
    rank_default: int,
    excel_path: str,
    mode: str = "full",
    core_n: int = 3,
) -> tuple[list[str], list[str], str | None]:
    """Aplica um arquétipo: adiciona golpes encontrados e tenta definir habilidade sugerida.

    mode:
      - "full": adiciona o kit completo (lista inteira + fallback no texto).
      - "core": adiciona só o núcleo (primeiros core_n golpes explicitamente listados; sem fallback).
    """
    try:
        db = load_move_db(excel_path)
    except Exception:
        db = None

    if db is None:
        return [], ["(Não consegui carregar o banco de golpes)"], None

    mode = (mode or "full").strip().lower()
    core_n = int(core_n or 0)

    if mode == "core":
        arch2 = dict(archetype or {})
        toks = (arch2.get("moves_tokens") or [])
        if core_n > 0:
            toks = toks[:core_n]
        arch2["moves_tokens"] = toks
        moves_found, missing = _viab_suggest_moves(db, arch2, include_fulltext=False)
    else:
        moves_found, missing = _viab_suggest_moves(db, archetype, include_fulltext=True)

    existing = {_norm(m.get("name")) for m in (st.session_state.get("cg_moves") or []) if isinstance(m, dict)}
    added_names: list[str] = []

    for mv in moves_found:
        k = _norm(mv.name)
        if not k or k in existing:
            continue
        st.session_state["cg_moves"].append(_cg_confirm_move(mv, int(rank_default)))
        existing.add(k)
        added_names.append(mv.name)

    # habilidade sugerida
    ability_suggested = archetype.get("ability")
    if isinstance(ability_suggested, str):
        ability_suggested = ability_suggested.strip()
    else:
        ability_suggested = None

    return added_names, missing, ability_suggested

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

# --- navegação programática (antes do radio key="page") ---
if "nav_to" in st.session_state:
    st.session_state.update({"page": st.session_state.pop("nav_to")})

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Ir para:",
    [
        "Pokédex (Busca)",
        "Trainer Hub (Meus Pokémons)",
        "Criação Guiada de Fichas",
        "Minhas Fichas",
        "PvP – Arena Tática",
        "Mochila",
        "Compendium de Ga'Al",
    ],
    key="page",
)

if page != "PvP – Arena Tática":
    stop_pvp_sync_listener()


# ==============================================================================
# COMPENDIUM DE GA'AL — Aba completa (UPGRADED)
# - Split view (lista à esquerda / dossiê à direita)
# - Breadcrumbs + navegação Próximo/Anterior
# - Favoritos + Recentes (Locais e NPCs)
# - Busca global full-text (Cidades, Sublocais, NPCs, Regiões)
# - Cross-links (cidades/NPCs mencionados)
# - Sprites de Pokémon (cache local offline)
# - Tags automáticas + overrides em JSON (exportável)
# ==============================================================================

from pathlib import Path
import hashlib

try:
    from docx import Document
except Exception:
    Document = None


# ----------------------------
# CONFIG
# ----------------------------
# (Preferência atual) Compendium em JSON
COMP_JSON_LOCAIS = "gaal_locais.json"
COMP_JSON_GINASIOS = "gaal_ginasios.json"          # opcional (detalhes/staff via NPCs)
COMP_JSON_NPCS_VIVOS = "gaal_npcs_vivos.json"
COMP_JSON_NPCS_MORTOS = "gaal_npcs_mortos.json"    # opcional

# (Fallback legado) Compendium em DOCX
COMP_DOC_LOCAIS = "GAAL_Banco_de_Dados_Locais_Unificado_v2.docx"
COMP_DOC_NPCS_VIVOS = "Npcs_Pokemon_vivos_profundo_v2.docx"
COMP_DOC_NPCS_MORTOS = "Npcs_Pokemon_mortos.docx"  # opcional
COMP_DOC_GINASIOS = "ginasios.docx"                # opcional (detalhes extras de ginásios)
COMP_DEFAULT_MAP = "GaAl_2.jpg"  # mapa geral (fallback)


COMP_REGIOES_PRINCIPAIS = [
    "Baía Morta",
    "Baixo Deserto",
    "Campos Longos",
    "Chifre de Ga'Al",
    "Deserto Irrigado",
    "Terras Fluviais",
]

# Overrides opcionais via arquivo (sem mexer no código):
# Estrutura:
# {
#   "cities": {"Obsidian": ["Industrial","Portuária"]},
#   "npcs": {"Dra. Tulsi": ["Cientista","Ranger"]}
# }
COMP_TAGS_JSON = "compendium_tags.json"


# ----------------------------
# THEME
# ----------------------------
def apply_compendium_theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"]{
            background:
                radial-gradient(900px 500px at 12% 0%, rgba(250, 204, 21, 0.10), transparent 60%),
                radial-gradient(1200px 700px at 92% 10%, rgba(244, 114, 182, 0.10), transparent 60%),
                linear-gradient(180deg, #0c0a0a 0%, #0f172a 55%, #09080b 100%);
            color: #f8fafc;
        }
        [data-testid="stHeader"]{ background: transparent !important; }
        h1,h2,h3,h4{ color:#f8fafc !important; letter-spacing:-0.02em; }
        .stMarkdown, .stMarkdown p, label { color: #e2e8f0 !important; }

        .comp-shell{
            border: 1px solid rgba(148,163,184,0.18);
            background: rgba(15,23,42,0.55);
            border-radius: 22px;
            padding: 18px 18px;
            box-shadow: 0 24px 70px rgba(0,0,0,0.45);
            backdrop-filter: blur(12px);
        }
        .comp-hero{
            border: 1px solid rgba(203,213,225,0.18);
            background: linear-gradient(135deg, rgba(17,24,39,0.72), rgba(2,6,23,0.55));
            border-radius: 20px;
            padding: 18px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.35);
        }
        .comp-hero-title{
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
        }
        .comp-hero-kicker{
            color: rgba(248,250,252,0.70);
            font-size: 14px;
        }
        .comp-muted{ color: rgba(226,232,240,0.72); }
        .comp-cta{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid rgba(250,204,21,0.35);
            background: rgba(250,204,21,0.12);
            color: #fde68a;
            font-weight: 600;
        }
        .comp-panel{
            border: 1px solid rgba(148,163,184,0.16);
            background: rgba(15,23,42,0.45);
            border-radius: 18px;
            padding: 14px;
        }
        .comp-card{
            border: 1px solid rgba(148,163,184,0.18);
            background: rgba(2,6,23,0.55);
            border-radius: 16px;
            padding: 12px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.30);
        }
        .comp-divider{
            height: 1px;
            background: rgba(148,163,184,0.18);
            margin: 12px 0;
        }
        .comp-block{
            border: 1px solid rgba(148,163,184,0.12);
            background: rgba(15,23,42,0.45);
            border-radius: 16px;
            padding: 12px;
            margin: 12px 0;
        }
        .comp-row{
            border: 1px solid rgba(148,163,184,0.12);
            background: rgba(2,6,23,0.35);
            border-radius: 16px;
            padding: 10px 12px;
            margin: 10px 0;
        }
        .comp-mini{
            font-size: 12px;
            color: rgba(226,232,240,0.70);
        }
        .comp-map{
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,0.2);
            box-shadow: 0 12px 30px rgba(0,0,0,0.35);
        }
        .comp-stat{
            border: 1px solid rgba(148,163,184,0.2);
            background: rgba(15,23,42,0.65);
            border-radius: 14px;
            padding: 10px 12px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_top_menu_compendium(selected: str):
    st.markdown("<div class='ds-home'>", unsafe_allow_html=True)

    opts = [("menu", "Menu"),
            ("npcs", "NPCs"),
            ("ginasios", "Ginásios"),
            ("locais", "Locais"),
            ("sair", "Sair")]

    for key, label in opts:
        if st.button(label, key=f"top_{key}"):
            st.session_state["comp_view"] = key
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# ASSETS (index rápido)
# ----------------------------
def _comp_base_dirs() -> list[str]:
    try:
        base = os.path.dirname(__file__)
    except Exception:
        base = os.getcwd()

    roots = [
        os.getcwd(),
        base,
        os.path.join(base, "assets"),
        os.path.join(base, "Assets"),
        os.path.join(os.getcwd(), "assets"),
        os.path.join(os.getcwd(), "Assets"),
        os.path.join(base, "data"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(base, "assets", "compendium"),
        os.path.join(base, "assets", "locais"),
        os.path.join(base, "assets", "cities"),
        os.path.join(base, "assets", "npcs"),
        os.path.join(base, "assets", "portraits"),
        os.path.join(base, "cidades"),
        os.path.join(base, "treinadores"),
        os.path.join(os.getcwd(), "assets", "compendium"),
        os.path.join(os.getcwd(), "assets", "locais"),
        os.path.join(os.getcwd(), "assets", "cities"),
        os.path.join(os.getcwd(), "assets", "npcs"),
        os.path.join(os.getcwd(), "assets", "portraits"),
        os.path.join(os.getcwd(), "cidades"),
        os.path.join(os.getcwd(), "treinadores"),
    ]
    uniq = []
    for r in roots:
        if r and os.path.exists(r) and r not in uniq:
            uniq.append(r)
    return uniq


def _stem_key(s: str) -> str:
    x = _norm(s)
    x = x.replace("vila de ", "")
    x = x.replace("—", "-")
    x = re.sub(r"[^\w\s-]+", "", x)
    x = x.replace("-", " ")
    x = re.sub(r"\s+", " ", x).strip()
    x = x.replace(" ", "_")
    return x


@st.cache_data(show_spinner=False)
def _build_image_index(roots: tuple[str, ...]) -> dict[str, dict]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    by_key: dict[str, str] = {}
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in exts:
                    continue
                stem = os.path.splitext(fn)[0]
                k = _stem_key(stem)
                p = os.path.join(dirpath, fn)
                if k and (k not in by_key or len(p) < len(by_key[k])):
                    by_key[k] = p

                # index adicional sem sufixos numéricos
                base = re.sub(r"[_-]?\d+$", "", stem).strip()
                if base:
                    kb = _stem_key(base)
                    if kb and kb not in by_key:
                        by_key[kb] = p
    return {"by_key": by_key, "keys": sorted(by_key.keys())}


def comp_find_image(name: str) -> str | None:
    roots = tuple(_comp_base_dirs())
    idx = _build_image_index(roots)
    by_key = idx.get("by_key", {})
    all_keys = idx.get("keys", [])

    variants = [
        _stem_key(name),
        _stem_key(name).replace("_", ""),
        _stem_key(name).replace("_", "-"),
    ]
    for k in variants:
        if k in by_key and os.path.exists(by_key[k]):
            return by_key[k]

    # tenta aproximação por similaridade
    for target in variants:
        if not target:
            continue
        close = difflib.get_close_matches(target, all_keys if all_keys else [], n=1, cutoff=0.78)
        if close:
            p = by_key.get(close[0])
            if p and os.path.exists(p):
                return p
        if all_keys:
            for k in all_keys:
                if target in k or k in target:
                    p = by_key.get(k)
                    if p and os.path.exists(p):
                        return p

    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for ext in exts:
        fname = f"{_stem_key(name)}{ext}"
        try:
            p = _resolve_asset_path(fname)
        except Exception:
            p = fname
        if p and os.path.exists(p):
            return p

    return None


# ----------------------------
# TAG OVERRIDES (JSON + sessão)
# ----------------------------
def _load_comp_tags_overrides() -> dict:
    roots = _comp_base_dirs()
    # tenta localizar o json nos roots
    candidates = []
    for r in roots:
        candidates.append(os.path.join(r, COMP_TAGS_JSON))
    # também tenta via resolve
    try:
        candidates.insert(0, _resolve_asset_path(COMP_TAGS_JSON))
    except Exception:
        pass

    for p in candidates:
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    obj.setdefault("cities", {})
                    obj.setdefault("npcs", {})
                    return obj
            except Exception:
                return {"cities": {}, "npcs": {}}
    return {"cities": {}, "npcs": {}}


def _get_live_overrides() -> dict:
    if "comp_live_overrides" not in st.session_state:
        st.session_state["comp_live_overrides"] = {"cities": {}, "npcs": {}}
    return st.session_state["comp_live_overrides"]


def _merge_tags(entity_type: str, name: str, inferred: list[str]) -> list[str]:
    base = _load_comp_tags_overrides()
    live = _get_live_overrides()
    manual = (base.get(entity_type, {}) or {}).get(name, [])
    manual2 = (live.get(entity_type, {}) or {}).get(name, [])
    tags = []
    for t in (manual + manual2 + inferred):
        if t and t not in tags:
            tags.append(t)
    return tags[:6]


# ----------------------------
# DOCX PARSERS
# ----------------------------
def _heading_level(p) -> int:
    try:
        style = (p.style.name or "")
    except Exception:
        style = ""
    m = re.match(r"Heading\s+(\d+)", style)
    return int(m.group(1)) if m else 0


def _clean_title(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_name_desc(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in (text or "").split("\n") if p.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], "\n\n".join(parts[1:])


def _is_fake_city_h2(h2: str) -> bool:
    t = _norm(h2)
    if t.startswith("visao geral"):
        return True
    if t.startswith("anexo"):
        return True
    if t.startswith("como ler"):
        return True
    if t.startswith("ginasios "):
        return True
    if t.startswith("locais em "):
        return True
    if t.startswith("ginasio de "):
        return True
    return False


def _h2_city_from_pattern(h2: str) -> str | None:
    s = _clean_title(h2)
    m = re.match(r"^(.+?)\s*-\s*vis[aã]o geral$", _norm(s))
    if m:
        left = re.split(r"[—-]", s, maxsplit=1)[0].strip()
        return left
    return None

def _is_city_line(line: str, known_cities: set[str]) -> bool:
    t = (line or "").strip()
    if not t:
        return False
    if ":" in t:
        return False
    if len(t) > 40:
        return False
    if _norm(t) in {_norm(x) for x in known_cities}:
        return True
    if re.fullmatch(r"[A-Za-zÀ-ÿ'’\- ]+", t) and t[0].isupper():
        return True
    return False


@st.cache_data(show_spinner=False)
def parse_ginasios_docx(doc_path: str, known_cities: list[str]) -> dict:
    gyms: dict[str, dict] = {}
    if Document is None or not os.path.exists(doc_path):
        return gyms

    known = set([c for c in known_cities if c])
    doc = Document(doc_path)

    cur_city: str | None = None
    cur_person: dict | None = None
    mode = "normal"  # "history"
    history_item: dict | None = None

    def ensure_city(city: str):
        gyms.setdefault(city, {"leaders": [], "vices": [], "people": [], "history": []})

    def flush_person():
        nonlocal cur_person
        if not cur_city or not cur_person:
            cur_person = None
            return
        ensure_city(cur_city)

        occ = _norm(cur_person.get("fields", {}).get("Ocupação atual", ""))

        if "líder" in occ or "lider" in occ:
            gyms[cur_city]["leaders"].append(cur_person)
        elif "vice" in occ:
            gyms[cur_city]["vices"].append(cur_person)
        else:
            gyms[cur_city]["people"].append(cur_person)

        cur_person = None

    def flush_history():
        nonlocal history_item
        if not cur_city or not history_item:
            history_item = None
            return
        ensure_city(cur_city)
        gyms[cur_city]["history"].append(history_item)
        history_item = None

    def new_person(name: str):
        return {"name": name.strip(), "fields": {}, "pokemons": [], "text": ""}

    def add_text(obj: dict, txt: str):
        txt = (txt or "").strip()
        if not txt:
            return
        obj["text"] = (obj["text"] + "\n\n" + txt).strip()

    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue

        if "registro histórico" in _norm(t):
            mode = "history"
            flush_person()
            continue

        if mode == "normal" and _is_city_line(t, known):
            flush_person()
            cur_city = _clean_title(t)
            ensure_city(cur_city)
            continue

        if mode == "history" and cur_city is None:
            cur_city = "__REGISTRO_HISTORICO__"
            ensure_city(cur_city)

        if mode == "history" and re.match(r"^\d+\.\s", t):
            flush_history()
            history_item = {"title": t.strip(), "fields": {}, "pokemons": [], "text": ""}
            continue

        if mode == "normal":
            if cur_city and cur_person is None and ":" not in t and len(t) <= 40:
                cur_person = new_person(t)
                continue

        if ":" in t:
            k, v = t.split(":", 1)
            kN = _norm(k)
            v = v.strip()

            if "pokemon" in kN:
                pokes = [x.strip() for x in re.split(r",|;|/|\|", v) if x.strip()]
                if mode == "history" and history_item is not None:
                    history_item["pokemons"].extend(pokes)
                elif cur_person is not None:
                    cur_person["pokemons"].extend(pokes)
                continue

            if mode == "history" and history_item is not None:
                history_item["fields"][k.strip()] = v
            elif cur_person is not None:
                cur_person["fields"][k.strip()] = v
            else:
                if cur_city:
                    ensure_city(cur_city)
                    gyms[cur_city].setdefault("notes", [])
                    gyms[cur_city]["notes"].append({k.strip(): v})
            continue

        if mode == "history":
            if history_item is not None:
                add_text(history_item, t)
            else:
                ensure_city(cur_city or "__REGISTRO_HISTORICO__")
                gyms[cur_city or "__REGISTRO_HISTORICO__"].setdefault("history_text", "")
                gyms[cur_city or "__REGISTRO_HISTORICO__"]["history_text"] = (
                    gyms[cur_city or "__REGISTRO_HISTORICO__"]["history_text"] + "\n\n" + t
                ).strip()
        else:
            if cur_person is not None:
                add_text(cur_person, t)
            else:
                ensure_city(cur_city or "__SEM_CIDADE__")
                gyms[cur_city or "__SEM_CIDADE__"].setdefault("city_text", "")
                gyms[cur_city or "__SEM_CIDADE__"]["city_text"] = (
                    gyms[cur_city or "__SEM_CIDADE__"]["city_text"] + "\n\n" + t
                ).strip()

    flush_person()
    flush_history()
    return gyms



@st.cache_data(show_spinner=False)
def load_compendium_data(
    locais_path: str,
    locais_mtime: float,
    npcs_vivos_path: str,
    npcs_vivos_mtime: float,
    npcs_mortos_path: str,
    npcs_mortos_mtime: float,
) -> dict:
    if Document is None:
        raise RuntimeError("python-docx não está disponível (instale python-docx).")

    data = {
        "regions": {},   # reg -> {"intro": str, "sections": {title: text}, "cities": [city]}
        "cities": {},    # city -> {"region": reg, "sections": {title: text}, "sublocais": [{"name","text"}]}
        "npcs": {},      # name -> {...}
    }

    # ---------- LOCAIS ----------
    if os.path.exists(locais_path):
        doc = Document(locais_path)

        cur_reg: str | None = None
        cur_city: str | None = None
        cur_h3: str | None = None
        inside_sublocais = False
        chifre_city_mode: str | None = None

        def ensure_region(r: str):
            data["regions"].setdefault(r, {"intro": "", "sections": {}, "cities": []})

        def ensure_city(c: str, r: str):
            data["cities"].setdefault(c, {"region": r, "sections": {}, "sublocais": []})
            if c not in data["regions"][r]["cities"]:
                data["regions"][r]["cities"].append(c)

        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if not t:
                continue

            lvl = _heading_level(p)
            t_clean = _clean_title(t)

            if lvl == 1:
                cur_reg = t_clean
                ensure_region(cur_reg)
                cur_city = None
                cur_h3 = None
                inside_sublocais = False
                chifre_city_mode = None
                continue

            if cur_reg is None:
                continue

            if lvl == 2:
                maybe_city = _h2_city_from_pattern(t_clean)
                if maybe_city:
                    cur_city = maybe_city
                    chifre_city_mode = maybe_city
                    ensure_city(cur_city, cur_reg)
                    cur_h3 = "Visão geral"
                    inside_sublocais = False
                    continue

                tn = _norm(t_clean)
                m_gym = re.match(r"^ginasio de (.+)$", tn)
                m_loc = re.match(r"^locais em (.+)$", tn)
                if m_gym and chifre_city_mode and _norm(m_gym.group(1)) in {_norm(chifre_city_mode), _norm(chifre_city_mode).replace("vila de ", "")}:
                    cur_city = chifre_city_mode
                    cur_h3 = f"Ginásio de {chifre_city_mode}"
                    inside_sublocais = False
                    continue
                if m_loc and chifre_city_mode and _norm(m_loc.group(1)) in {_norm(chifre_city_mode), _norm(chifre_city_mode).replace("vila de ", "")}:
                    cur_city = chifre_city_mode
                    cur_h3 = "Sublocais e pontos de interesse"
                    inside_sublocais = True
                    continue

                if _is_fake_city_h2(t_clean):
                    cur_city = None
                    cur_h3 = t_clean
                    inside_sublocais = False
                    chifre_city_mode = None
                    data["regions"][cur_reg]["sections"].setdefault(cur_h3, "")
                    continue

                cur_city = t_clean
                if _norm(cur_city).startswith("vila de "):
                    cur_city = cur_city.split(" ", 2)[-1].strip()

                ensure_city(cur_city, cur_reg)
                cur_h3 = None
                inside_sublocais = False
                chifre_city_mode = cur_city
                continue

            if lvl == 3:
                if cur_city:
                    cur_h3 = t_clean
                    inside_sublocais = ("sublocais" in _norm(cur_h3) or "pontos de interesse" in _norm(cur_h3))
                    data["cities"][cur_city]["sections"].setdefault(cur_h3, "")
                else:
                    cur_h3 = t_clean
                    inside_sublocais = False
                    data["regions"][cur_reg]["sections"].setdefault(cur_h3, "")
                continue

            if lvl == 4:
                if cur_city and inside_sublocais:
                    nm, desc = _split_name_desc(t)
                    data["cities"][cur_city]["sublocais"].append({"name": nm, "text": desc})
                else:
                    if cur_city and cur_h3:
                        k = f"{cur_h3} • {t_clean}"
                        data["cities"][cur_city]["sections"].setdefault(k, "")
                    elif (not cur_city) and cur_h3:
                        k = f"{cur_h3} • {t_clean}"
                        data["regions"][cur_reg]["sections"].setdefault(k, "")
                continue

            # texto normal
            if cur_city:
                ensure_city(cur_city, cur_reg)

                if inside_sublocais:
                    nm, desc = _split_name_desc(t)
                    if nm and desc:
                        data["cities"][cur_city]["sublocais"].append({"name": nm, "text": desc})
                    else:
                        if data["cities"][cur_city]["sublocais"]:
                            last = data["cities"][cur_city]["sublocais"][-1]
                            last["text"] = (last["text"] + "\n\n" + t).strip()
                        else:
                            data["cities"][cur_city]["sections"].setdefault("Notas", "")
                            data["cities"][cur_city]["sections"]["Notas"] = (data["cities"][cur_city]["sections"]["Notas"] + "\n\n" + t).strip()
                else:
                    if cur_h3:
                        data["cities"][cur_city]["sections"].setdefault(cur_h3, "")
                        data["cities"][cur_city]["sections"][cur_h3] = (data["cities"][cur_city]["sections"][cur_h3] + "\n\n" + t).strip()
                    else:
                        data["cities"][cur_city]["sections"].setdefault("Visão geral", "")
                        data["cities"][cur_city]["sections"]["Visão geral"] = (data["cities"][cur_city]["sections"]["Visão geral"] + "\n\n" + t).strip()
            else:
                ensure_region(cur_reg)
                if cur_h3:
                    data["regions"][cur_reg]["sections"].setdefault(cur_h3, "")
                    data["regions"][cur_reg]["sections"][cur_h3] = (data["regions"][cur_reg]["sections"][cur_h3] + "\n\n" + t).strip()
                else:
                    data["regions"][cur_reg]["intro"] = (data["regions"][cur_reg]["intro"] + "\n\n" + t).strip()

        for r, obj in data["regions"].items():
            if not obj["intro"]:
                for k in list(obj["sections"].keys()):
                    if "visão geral" in _norm(k) and obj["sections"].get(k):
                        obj["intro"] = obj["sections"][k]
                        break

    # ---------- NPCs ----------
    def parse_npcs(doc_path: str, fallback_status: str | None = None) -> dict[str, dict]:
        if not os.path.exists(doc_path):
            return {}
        d = Document(doc_path)
        npcs: dict[str, dict] = {}
        cur: str | None = None
        cur_section: str | None = None

        ID_KEYS = {
            "status": "status",
            "idade": "idade",
            "local de origem": "origem",
            "ocupação atual": "ocupacao",
            "ocupacao atual": "ocupacao",
            "pokémons conhecidos": "pokemons",
            "pokemons conhecidos": "pokemons",
        }

        def ensure(nm: str):
            npcs.setdefault(nm, {"name": nm, "status": fallback_status or "", "idade": "", "origem": "", "ocupacao": "", "pokemons": [], "sections": {}})

        for p in d.paragraphs:
            t = (p.text or "").strip()
            if not t:
                continue
            lvl = _heading_level(p)

            if lvl == 2:
                cur = _clean_title(t)
                ensure(cur)
                cur_section = None
                continue

            if cur is None:
                continue

            if re.fullmatch(r"[—\-]{5,}", t):
                continue

            if t.endswith(":") and len(t) <= 60 and ":" not in t[:-1]:
                cur_section = t[:-1].strip()
                npcs[cur]["sections"].setdefault(cur_section, "")
                continue

            if ":" in t:
                k, v = t.split(":", 1)
                k2 = _norm(k)
                v = v.strip()
                if k2 in ID_KEYS:
                    field = ID_KEYS[k2]
                    if field == "pokemons":
                        pokes = [x.strip() for x in re.split(r",|;|\|", v) if x.strip()]
                        npcs[cur]["pokemons"] = pokes
                    else:
                        npcs[cur][field] = v
                    if not npcs[cur].get("status") and fallback_status:
                        npcs[cur]["status"] = fallback_status
                    continue

            sec = cur_section or "Lore"
            npcs[cur]["sections"].setdefault(sec, "")
            npcs[cur]["sections"][sec] = (npcs[cur]["sections"][sec] + "\n\n" + t).strip()

        for nm, obj in npcs.items():
            if not obj.get("status"):
                obj["status"] = fallback_status or "Vivo(a)"
        return npcs

    vivos = parse_npcs(npcs_vivos_path, fallback_status="Vivo(a)")
    mortos = parse_npcs(npcs_mortos_path, fallback_status="Morto(a)") if os.path.exists(npcs_mortos_path) else {}
    data["npcs"] = {**vivos, **mortos}

    
    return data


# ----------------------------
# JSON loaders (NOVO)
# ----------------------------
def _json_read(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_compendium_json_data(
    locais_path: str,
    locais_mtime: float,
    npcs_vivos_path: str,
    npcs_vivos_mtime: float,
    npcs_mortos_path: str,
    npcs_mortos_mtime: float,
) -> dict:
    """
    Carrega Compendium em JSON mantendo o MESMO contrato do loader legado (DOCX):
    {
      "regions": {...},
      "cities": {...},
      "npcs": {...},
    }
    """
    data = {"regions": {}, "cities": {}, "npcs": {}}

    # LOCAIS (obrigatório para a aba de Locais)
    if locais_path and os.path.exists(locais_path):
        j = _json_read(locais_path)
        data["regions"] = (j.get("regions") or {})
        data["cities"] = (j.get("cities") or {})

    # NPCs (vivos + mortos)
    npcs_all: dict = {}
    if npcs_vivos_path and os.path.exists(npcs_vivos_path):
        jv = _json_read(npcs_vivos_path)
        npcs_all.update(jv.get("npcs") or {})
    if npcs_mortos_path and os.path.exists(npcs_mortos_path):
        jm = _json_read(npcs_mortos_path)
        # se tiver colisão de nome, mortos não sobrescrevem vivos
        for k, v in (jm.get("npcs") or {}).items():
            npcs_all.setdefault(k, v)
    data["npcs"] = npcs_all
    return data


def _extract_gym_meta_from_text(t: str) -> dict:
    """
    Extrai metadados do ginásio a partir do texto do 'Ginásio de <Cidade>' no JSON de locais.
    Aceita variações do padrão: 'Líder: X | Tipo: Y | Localização: Z'
    """
    out: dict = {}
    if not t:
        return out
    # Linha principal com pipes
    m = re.search(r"L[íi]der\s*:\s*([^|\n]+)\|\s*Tipo\s*:\s*([^|\n]+)(?:\|\s*Localiza[cç][aã]o\s*:\s*([^\n]+))?", t, flags=re.IGNORECASE)
    if m:
        out["lider"] = (m.group(1) or "").strip()
        out["tipo"] = (m.group(2) or "").strip()
        if m.group(3):
            out["localizacao"] = (m.group(3) or "").strip()
        return out

    # Alternativo: linhas soltas
    m1 = re.search(r"L[íi]der\s*:\s*(.+)", t, flags=re.IGNORECASE)
    if m1:
        out["lider"] = (m1.group(1) or "").strip()
    m2 = re.search(r"Tipo\s*:\s*(.+)", t, flags=re.IGNORECASE)
    if m2:
        out["tipo"] = (m2.group(1) or "").strip()
    m3 = re.search(r"Localiza[cç][aã]o\s*:\s*(.+)", t, flags=re.IGNORECASE)
    if m3:
        out["localizacao"] = (m3.group(1) or "").strip()
    return out


@st.cache_data(show_spinner=False)
def load_compendium_gym_data_json(
    locais_path: str,
    locais_mtime: float,
    ginasios_path: str,
    ginasios_mtime: float,
) -> dict:
    """
    Constrói bundle do ginásio no mesmo formato do legado:
      {"gyms": {city: {"meta":{}, "staff":{}, "staff_npcs":{}, "narrative": ""}},
       "npcs_extra": {name: npc_obj}}
    - 'meta' vem do texto 'Ginásio de <Cidade>' no JSON de locais.
    - 'staff_npcs' e 'npcs_extra' vêm do JSON de ginasios (que é uma coleção de NPCs).
    """
    gyms: dict[str, dict] = {}
    npcs_extra: dict[str, dict] = {}

    # 1) Meta + narrativa via locais.json
    if locais_path and os.path.exists(locais_path):
        jl = _json_read(locais_path)
        cities = (jl.get("cities") or {})
        for city, cobj in cities.items():
            sec = (cobj.get("sections") or {})
            # tenta chave exata "Ginásio de {city}" (normal)
            key = None
            for k in sec.keys():
                if _norm(k).startswith(_norm("Ginásio de " + str(city))):
                    key = k
                    break
            txt_gym = (sec.get(key) if key else "") or ""
            meta = _extract_gym_meta_from_text(txt_gym)
            narrative = txt_gym.strip()
            if meta or narrative:
                gyms[city] = {"meta": meta, "staff": {}, "staff_npcs": {}, "narrative": narrative}

    # 2) Staff via ginasios.json (NPCs)
    if ginasios_path and os.path.exists(ginasios_path):
        jg = _json_read(ginasios_path)
        for nm, npc in (jg.get("npcs") or {}).items():
            if not isinstance(npc, dict):
                continue
            npcs_extra[nm] = npc

            # tenta vincular a um ginásio/cidade pelo texto de ocupação
            occ = _norm(npc.get("ocupacao") or "")
            # procura "ginasio de X"
            m = re.search(r"ginasio\s+de\s+([a-z0-9\s\'\-]+)", occ)
            if not m:
                continue
            city_guess = (m.group(1) or "").strip()
            # best match com nomes de cidades
            city_hit = None
            if gyms:
                for c in gyms.keys():
                    if _norm(c) == _norm(city_guess):
                        city_hit = c
                        break
            # se ainda não achou, tenta comparar com cidades do locais
            if not city_hit and locais_path and os.path.exists(locais_path):
                cities = _json_read(locais_path).get("cities") or {}
                for c in cities.keys():
                    if _norm(c) == _norm(city_guess):
                        city_hit = c
                        break

            if not city_hit:
                continue

            g = gyms.setdefault(city_hit, {"meta": {}, "staff": {}, "staff_npcs": {}, "narrative": ""})
            role = "staff"
            if "vice" in occ:
                role = "vice"
            elif "lider" in occ or "líder" in occ:
                role = "lider"
            g.setdefault("staff", {})
            g["staff"][role] = nm
            g.setdefault("staff_npcs", {})
            g["staff_npcs"][nm] = npc

    return {"gyms": gyms, "npcs_extra": npcs_extra}


def _comp_mtime(p: str) -> float:
    try:
        return os.path.getmtime(p) if p and os.path.exists(p) else 0.0
    except Exception:
        return 0.0



# ----------------------------
# GINÁSIOS (integração opcional via ginasios.docx + fichas no DOCX de locais)
# ----------------------------
def _parse_ficha_block(lines: list[str]) -> dict:
    out: dict[str, str] = {}
    for ln in lines:
        ln = (ln or "").strip()
        if not ln or ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = _norm(k)
        v = v.strip()
        if not k:
            continue
        out[k] = v
    return out


def _canon_city_key(s: str) -> str:
    # mantém acentos no display, mas cria chave normalizada para match
    return _norm(s or "")


@st.cache_data(show_spinner=False)
def load_compendium_gym_data(
    locais_path: str,
    locais_mtime: float,
    ginasios_path: str,
    ginasios_mtime: float,
) -> dict:
    """Carrega dados de ginásios e (se necessário) cria/enriquece NPCs a partir do arquivo ginasios.docx.
    - Meta estruturada (tipo/status/localização/arena) vem do DOCX de locais via blocos [FICHA]...[/FICHA].
    - Detalhes de líder/vice e Pokémons podem vir de ginasios.docx.
    Retorna: {"gyms": {city: {...}}, "npcs_extra": {name: npc_obj}}
    """
    if Document is None:
        return {"gyms": {}, "npcs_extra": {}}

    gyms: dict[str, dict] = {}
    npcs_extra: dict[str, dict] = {}

    # -------- 1) Meta via [FICHA] no DOCX de locais --------
    if locais_path and os.path.exists(locais_path):
        try:
            d = Document(locais_path)
            cur_city: str | None = None
            in_ficha = False
            ficha_lines: list[str] = []
            narrative_lines: list[str] = []

            def flush():
                nonlocal cur_city, ficha_lines, narrative_lines
                if not cur_city:
                    return
                meta = _parse_ficha_block(ficha_lines)
                # normaliza campos esperados
                city = meta.get("cidade") or cur_city
                city_disp = (city or cur_city).strip()
                g = gyms.setdefault(city_disp, {"meta": {}, "staff": {}, "staff_npcs": {}, "narrative": ""})
                g["meta"] = {**g.get("meta", {}), **meta}
                if narrative_lines:
                    txt = "\n\n".join([x.strip() for x in narrative_lines if x.strip()]).strip()
                    if txt:
                        g["narrative"] = (g.get("narrative","") + "\n\n" + txt).strip()
                ficha_lines = []
                narrative_lines = []

            for p in d.paragraphs:
                t = (p.text or "").strip()
                if not t:
                    continue

                # detecta nome do ginásio "Ginásio de X" (muito comum no seu doc unificado)
                m = re.match(r"^Gin[aá]sio\s+de\s+(.+)$", t.strip(), flags=re.IGNORECASE)
                if m and not in_ficha:
                    # muda de ginásio
                    flush()
                    cur_city = m.group(1).strip()
                    continue

                if t.strip() == "[FICHA]":
                    in_ficha = True
                    ficha_lines = []
                    continue
                if t.strip() == "[/FICHA]":
                    in_ficha = False
                    # não flush aqui; vem narrativa depois
                    continue

                if in_ficha:
                    ficha_lines.append(t)
                else:
                    # narrativa do ginásio (logo após a ficha, até o próximo "Ginásio de ...")
                    if cur_city:
                        narrative_lines.append(t)

            flush()
        except Exception:
            pass

    # -------- 2) Staff via ginasios.docx (opcional) --------
    # Formato do ginasios.docx: blocos por cidade, com nomes e linhas "Campo: valor" + seção "Pokémons"
    if ginasios_path and os.path.exists(ginasios_path):
        try:
            d2 = Document(ginasios_path)
            cur_city: str | None = None
            cur_person: str | None = None
            cur_person_lines: list[str] = []
            in_pokes = False

            def ensure_person(nm: str) -> dict:
                npcs_extra.setdefault(nm, {"name": nm, "status": "Vivo(a)", "idade": "", "origem": "", "ocupacao": "", "pokemons": [], "sections": {}})
                return npcs_extra[nm]

            def commit_person():
                nonlocal cur_person, cur_person_lines, cur_city, in_pokes
                if not (cur_city and cur_person):
                    cur_person = None
                    cur_person_lines = []
                    in_pokes = False
                    return

                npc = ensure_person(cur_person)
                # parse linhas "Campo: valor" e narrativa
                section = "Ginásio (importado)"
                npc["sections"].setdefault(section, "")

                # tenta inferir papel
                role = None
                blob_role = _norm(" ".join(cur_person_lines))
                if "vice" in blob_role:
                    role = "vice_lider"
                if "lider" in blob_role or "líder" in blob_role:
                    # se tiver vice e líder, o primeiro costuma ser líder
                    role = role or "lider"

                # campos
                pokes: list[str] = []
                narrative: list[str] = []
                for ln in cur_person_lines:
                    ln = (ln or "").strip()
                    if not ln:
                        continue
                    if ":" in ln:
                        k, v = ln.split(":", 1)
                        k2 = _norm(k)
                        v = v.strip()
                        if k2 == "idade":
                            npc["idade"] = npc.get("idade") or v
                            continue
                        if k2 in {"local de nascimento", "local de origem", "origem"}:
                            npc["origem"] = npc.get("origem") or v
                            continue
                        if k2 in {"ocupacao atual", "ocupação atual", "ocupacao", "ocupação"}:
                            npc["ocupacao"] = npc.get("ocupacao") or v
                            continue
                        if k2 in {"pokemons", "pokémons", "pokemons conhecidos", "pokémons conhecidos"}:
                            pokes = [x.strip() for x in re.split(r",|;|\|", v) if x.strip()]
                            continue
                        # outros campos viram narrativa curta
                        narrative.append(ln)
                    else:
                        narrative.append(ln)

                if pokes:
                    # não sobrescreve se já existir
                    if not npc.get("pokemons"):
                        npc["pokemons"] = pokes

                if narrative:
                    txt = "\n\n".join(narrative).strip()
                    if txt:
                        npc["sections"][section] = (npc["sections"][section] + "\n\n" + txt).strip()

                # pluga no ginásio da cidade
                g = gyms.setdefault(cur_city, {"meta": {}, "staff": {}, "staff_npcs": {}, "narrative": ""})
                g["staff_npcs"][cur_person] = npc

                # se meta já indicar líder/vice, respeita. senão usa heurística.
                meta = g.get("meta") or {}
                lider = (meta.get("lider") or "").strip()
                vice = (meta.get("vice_lider") or meta.get("vice-lider") or "").strip()

                if role == "vice_lider" and not vice:
                    g["staff"]["vice_lider"] = cur_person
                elif role == "lider" and not lider:
                    g["staff"]["lider"] = cur_person
                else:
                    # fallback: se ainda não tem líder, assume líder; senão vice
                    if not (g["staff"].get("lider") or lider):
                        g["staff"]["lider"] = cur_person
                    elif not (g["staff"].get("vice_lider") or vice):
                        g["staff"]["vice_lider"] = cur_person

                # reset
                cur_person = None
                cur_person_lines = []
                in_pokes = False

            # cidades candidatas: nomes que aparecem também no compendium (heurística: palavra só / curto)
            for p in d2.paragraphs:
                t = (p.text or "").strip()
                if not t:
                    continue

                # detecta cidade: linha curta sem ":" e sem muitos espaços
                if ":" not in t and len(t) <= 40 and len(t.split()) <= 3 and t.lower() not in {"pokémons", "pokemons"}:
                    # se já estávamos em uma pessoa, fecha
                    if cur_person:
                        commit_person()
                    cur_city = t.strip()
                    gyms.setdefault(cur_city, {"meta": {}, "staff": {}, "staff_npcs": {}, "narrative": ""})
                    continue

                # detecta nome de pessoa: linha sem ":" e com 1-4 palavras (evita títulos)
                if ":" not in t and len(t) <= 60 and 1 <= len(t.split()) <= 5 and t.lower() not in {"pokémons", "pokemons"}:
                    # evita capturar descrições como "Líder:" na mesma linha
                    if t.lower().startswith("líder") or t.lower().startswith("lider") or t.lower().startswith("vice"):
                        # trata como narrativa
                        if cur_person:
                            cur_person_lines.append(t)
                        else:
                            if cur_city:
                                gyms[cur_city]["narrative"] = (gyms[cur_city].get("narrative","") + "\n\n" + t).strip()
                        continue
                    # troca pessoa
                    if cur_person:
                        commit_person()
                    cur_person = t.strip()
                    cur_person_lines = []
                    continue

                # conteúdo da pessoa/cidade
                if cur_person:
                    cur_person_lines.append(t)
                elif cur_city:
                    gyms[cur_city]["narrative"] = (gyms[cur_city].get("narrative","") + "\n\n" + t).strip()

            if cur_person:
                commit_person()

        except Exception:
            pass

    # -------- 3) Normalização final / merge de leader/vice --------
    for city, g in list(gyms.items()):
        meta = g.get("meta") or {}
        # se meta define nomes, usa para staff
        lider = (meta.get("lider") or "").strip()
        vice = (meta.get("vice_lider") or meta.get("vice-lider") or "").strip()
        if lider and "lider" not in (g.get("staff") or {}):
            g.setdefault("staff", {})["lider"] = lider
        if vice and "vice_lider" not in (g.get("staff") or {}):
            g.setdefault("staff", {})["vice_lider"] = vice

        # garante objetos para leader/vice no staff_npcs se existirem em npcs_extra
        for role_key in ["lider", "vice_lider"]:
            nm = (g.get("staff") or {}).get(role_key)
            if nm and nm in npcs_extra and nm not in (g.get("staff_npcs") or {}):
                g.setdefault("staff_npcs", {})[nm] = npcs_extra[nm]

    return {"gyms": gyms, "npcs_extra": npcs_extra}




def comp_load() -> dict:
    # ----------------------------
    # Preferência: JSON
    # ----------------------------
    loc_j = _resolve_asset_path(COMP_JSON_LOCAIS)
    npc_v_j = _resolve_asset_path(COMP_JSON_NPCS_VIVOS)
    npc_m_j = _resolve_asset_path(COMP_JSON_NPCS_MORTOS)
    gin_j = _resolve_asset_path(COMP_JSON_GINASIOS)

    use_json = bool(loc_j and os.path.exists(loc_j))

    if use_json:
        data = load_compendium_json_data(
            loc_j, _comp_mtime(loc_j),
            npc_v_j, _comp_mtime(npc_v_j),
            npc_m_j, _comp_mtime(npc_m_j),
        )
    
        bundle = load_compendium_gym_data_json(
            loc_j, _comp_mtime(loc_j),
            gin_j, _comp_mtime(gin_j),
        ) if (loc_j or gin_j) else {"gyms": {}, "npcs_extra": {}}
    
        # =========================
        # NPCs gerais (Vivos + Mortos + Extras)
        # =========================
        npcs_gerais = {}
        npcs_gerais.update(data.get("npcs", {}) or {})
        npcs_gerais.update(bundle.get("npcs_extra", {}) or {})

    else:
        # ----------------------------
        # Fallback legado: DOCX
        # ----------------------------
        loc_p = _resolve_asset_path(COMP_DOC_LOCAIS)
        npc_v_p = _resolve_asset_path(COMP_DOC_NPCS_VIVOS)
        npc_m_p = _resolve_asset_path(COMP_DOC_NPCS_MORTOS)
        gin_p = _resolve_asset_path(COMP_DOC_GINASIOS)

        data = load_compendium_data(
            loc_p, _comp_mtime(loc_p),
            npc_v_p, _comp_mtime(npc_v_p),
            npc_m_p, _comp_mtime(npc_m_p),
        )

        # integra ginásios (meta + staff), sem quebrar se o arquivo não existir
        bundle = load_compendium_gym_data(
            loc_p, _comp_mtime(loc_p),
            gin_p, _comp_mtime(gin_p),
        ) if (loc_p or gin_p) else {"gyms": {}, "npcs_extra": {}}

    # ----------------------------
    # Merge: gyms + NPCs extras
    # ----------------------------
    gyms = (bundle.get("gyms") or {})
    if gyms:
        data["gyms"] = gyms
    else:
        data.setdefault("gyms", {})

    # adiciona NPCs extras (líder/vice que não existam no JSON/DOCX de NPCs)
    extra = bundle.get("npcs_extra") or {}
    for nm, obj in extra.items():
        if nm not in data.get("npcs", {}):
            data.setdefault("npcs", {})
            data["npcs"][nm] = obj
        else:
            # preenche só o que está vazio
            dst = data["npcs"][nm]
            for k in ["idade", "origem", "ocupacao"]:
                if not (dst.get(k) or "").strip() and (obj.get(k) or "").strip():
                    dst[k] = obj.get(k)
            if (not dst.get("pokemons")) and obj.get("pokemons"):
                dst["pokemons"] = obj.get("pokemons")
            # injeta seção importada sem sobrescrever
            sec = "Ginásio (importado)"
            if obj.get("sections", {}).get(sec):
                dst.setdefault("sections", {})
                if not dst["sections"].get(sec):
                    dst["sections"][sec] = obj["sections"][sec]

    return data



# ----------------------------
# INFERÊNCIA DE TAGS + MENCÕES
# ----------------------------
def infer_city_tags(city_obj: dict) -> list[str]:
    blob = _norm(" ".join([city_obj.get("region","")] + list((city_obj.get("sections") or {}).values())))
    tags = []
    def add(tag, cond):
        nonlocal tags
        if cond and tag not in tags:
            tags.append(tag)

    add("Industrial", any(w in blob for w in ["industr", "usina", "refin", "fabrica", "minas", "petroleo", "petróleo"]))
    add("Arruinada", any(w in blob for w in ["ruina", "ruína", "abandon", "decad", "colaps"]))
    add("Portuária", any(w in blob for w in ["porto", "baia", "baía", "mar", "oceano", "cais"]))
    add("Desértica", any(w in blob for w in ["deserto", "areia", "duna", "arido", "árido"]))
    add("Florestal", any(w in blob for w in ["floresta", "bosque", "selva", "ranger", "copas"]))
    add("Montanhosa", any(w in blob for w in ["monte", "montanha", "serra", "pedra", "vulcao", "vulcão"]))
    add("Urbana", any(w in blob for w in ["avenida", "estadio", "estádio", "shopping", "metro", "aeroporto", "centro"]))
    return tags[:4]


def infer_npc_tags(npc_obj: dict) -> list[str]:
    blob = _norm(" ".join([npc_obj.get("ocupacao","")] + list(((npc_obj.get("sections") or {}) or {}).values())))
    tags = []
    def add(tag, cond):
        nonlocal tags
        if cond and tag not in tags:
            tags.append(tag)

    add("Líder de Ginásio", ("lider" in blob or "líder" in blob) and "ginas" in blob)
    add("Cientista", any(w in blob for w in ["cient", "laborat", "pesquis", "dra.", "dr."]))
    add("Ranger", "ranger" in blob)
    add("Treinador", "treinador" in blob)
    add("Mercenário", any(w in blob for w in ["mercen", "caca", "caça", "cacador", "caçador"]))
    add("Político", any(w in blob for w in ["prefeit", "govern", "senador", "movimento civil", "liga"]))
    return tags[:4]


def city_mentions_npcs(city_obj: dict, npc_names: list[str]) -> list[str]:
    text = _norm(" ".join(list((city_obj.get("sections") or {}).values())))
    hits = []
    for nm in npc_names:
        if len(nm) < 3:
            continue
        if _norm(nm) in text:
            hits.append(nm)
    return hits[:16]


def npc_mentions_cities(npc_obj: dict, city_names: list[str]) -> list[str]:
    text = _norm(" ".join(list((npc_obj.get("sections") or {}).values())))
    hits = []
    for cn in city_names:
        if len(cn) < 3:
            continue
        if _norm(cn) in text:
            hits.append(cn)
    return hits[:16]


def npc_mentions_npcs(npc_obj: dict, npc_names: list[str], self_name: str) -> list[str]:
    text = _norm(" ".join(list((npc_obj.get("sections") or {}).values())))
    hits = []
    for nm in npc_names:
        if nm == self_name:
            continue
        if len(nm) < 3:
            continue
        if _norm(nm) in text:
            hits.append(nm)
    return hits[:16]


# ----------------------------
# SPRITES (cache local offline)
# ----------------------------
def _sprite_cache_dir() -> Path:
    p = Path(os.getcwd()) / "assets" / "sprites_cache"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # fallback
        p = Path(os.getcwd()) / "sprites_cache"
        p.mkdir(parents=True, exist_ok=True)
    return p


def _to_pokeapi_simple(name: str) -> str:
    x = _norm(name)
    x = x.replace("♀", "-f").replace("♂", "-m")
    x = x.replace(".", "").replace("'", "").replace("’", "")
    x = x.replace(" ", "-")
    x = re.sub(r"[^a-z0-9\-]+", "", x)
    return x


def _extract_pokemon_name(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.split(r"[|/;,]", text)[0]
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\b(?:lvl|lv|level)\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-zÀ-ÿ0-9'\-\. ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _closest_pokeapi_name(raw: str) -> str:
    base = to_pokeapi_name(raw)
    if not api_name_map:
        return base
    if base in api_name_map:
        return base
    base_simple = base.split("-")[0]
    if base_simple in api_name_map:
        return base_simple
    keys = list(api_name_map.keys())
    match = difflib.get_close_matches(base, keys, n=1, cutoff=0.78)
    if not match and base_simple != base:
        match = difflib.get_close_matches(base_simple, keys, n=1, cutoff=0.76)
    return match[0] if match else base


@st.cache_data(ttl=60*60*24, show_spinner=False)
def _poke_sprite_cached(poke_name: str) -> str | None:
    """
    Retorna um caminho local (cache) se possível; fallback para URL.
    """
    raw = _extract_pokemon_name(poke_name)
    if not raw:
        return None

    # resolve query
    try:
        q = _closest_pokeapi_name(raw)
    except Exception:
        q = _to_pokeapi_simple(raw)

    # nome de arquivo robusto
    key = hashlib.sha1(q.encode("utf-8")).hexdigest()[:12]
    out = _sprite_cache_dir() / f"{key}_{q}.png"

    if out.exists():
        return str(out)

    # tenta baixar da PokeAPI
    try:
        url = f"{POKEAPI_BASE}/pokemon/{q}"
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            # sem internet? retorna None (não quebra)
            return str(out) if out.exists() else None
        pjson = r.json()
        sprites = (pjson.get("sprites") or {})
        sprite_url = sprites.get("front_default") or sprites.get("front_shiny")
        if not sprite_url:
            # tenta gen8 icons
            vers = (sprites.get("versions") or {})
            gen8 = (vers.get("generation-viii") or {})
            icons = (gen8.get("icons") or {})
            sprite_url = icons.get("front_default")

        if not sprite_url:
            return None

        img = requests.get(sprite_url, timeout=12)
        if img.status_code != 200:
            return None

        out.write_bytes(img.content)
        return str(out)

    except Exception:
        return str(out) if out.exists() else None


# ----------------------------
# BUSCA GLOBAL (full-text)
# ----------------------------
def _norm_tokens(q: str) -> list[str]:
    qn = _norm(q)
    toks = [t for t in re.split(r"\s+", qn) if t]
    # remove tokens muito curtos
    toks = [t for t in toks if len(t) >= 2]
    return toks


def _snippet(text: str, toks: list[str], n: int = 220) -> str:
    if not text:
        return ""
    tnorm = _norm(text)
    idx = None
    for tk in toks:
        j = tnorm.find(tk)
        if j != -1:
            idx = j
            break
    if idx is None:
        return (text[:n] + "…") if len(text) > n else text
    # tenta aproximar no texto original (sem normalizar)
    start = max(0, idx - 80)
    end = min(len(text), idx + (n - 80))
    chunk = text[start:end].strip()
    if start > 0:
        chunk = "…" + chunk
    if end < len(text):
        chunk = chunk + "…"
    return chunk


@st.cache_data(show_spinner=False)

def build_comp_search_index(data: dict) -> list[dict]:
    idx: list[dict] = []
    regions = data.get("regions", {}) or {}
    cities = data.get("cities", {}) or {}
    npcs = data.get("npcs", {}) or {}
    gyms = data.get("gyms", {}) or {}

    # regiões
    for r, robj in regions.items():
        blob = " ".join([robj.get("intro","")] + list((robj.get("sections") or {}).values()))
        idx.append({"type":"region", "key": r, "title": r, "text": blob, "region": r})

    # cidades e sublocais
    for c, cobj in cities.items():
        blob = " ".join(list((cobj.get("sections") or {}).values()))
        idx.append({"type":"city", "key": c, "title": c, "text": blob, "region": cobj.get("region","")})
        for sl in (cobj.get("sublocais") or []):
            nm = (sl.get("name") or "").strip()
            tx = (sl.get("text") or "").strip()
            if not nm and not tx:
                continue
            idx.append({"type":"sublocal", "key": f"{c}::{nm}", "title": nm or "(Sublocal)", "text": tx, "city": c, "region": cobj.get("region","")})

    # ginásios
    for city, g in gyms.items():
        meta = g.get("meta") or {}
        staff = g.get("staff") or {}
        # tenta região via cidades conhecidas
        reg = (cities.get(city, {}) or {}).get("region", "")
        title = f"Ginásio — {city}"
        blob = " ".join([
            city,
            str(reg),
            str(meta.get("status","")),
            str(meta.get("tipo","")),
            str(meta.get("localizacao","")),
            str(meta.get("arena_extra","")),
            str(meta.get("observacao","")),
            str(meta.get("lider","")),
            str(meta.get("vice_lider","")),
            str(staff.get("lider","")),
            str(staff.get("vice_lider","")),
            str(g.get("narrative","")),
        ])
        idx.append({"type":"gym", "key": city, "title": title, "text": blob, "city": city, "region": reg})

    # npcs
    for n, nobj in npcs.items():
        blob = " ".join([nobj.get("ocupacao",""), nobj.get("origem","")] + list(((nobj.get("sections") or {}) or {}).values()))
        idx.append({"type":"npc", "key": n, "title": n, "text": blob, "status": nobj.get("status","")})

    return idx

def comp_search(index: list[dict], query: str, types: set[str] | None = None, limit: int = 40) -> list[dict]:
    toks = _norm_tokens(query)
    if not toks:
        return []
    res = []
    for d in index:
        if types and d.get("type") not in types:
            continue
        t = _norm(d.get("text","") + " " + d.get("title",""))
        score = 0
        for tk in toks:
            if tk in t:
                score += 3
                score += t.count(tk)
        if score > 0:
            res.append((score, d))
    res.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, d in res[:limit]:
        dd = dict(d)
        dd["score"] = score
        dd["snippet"] = _snippet(d.get("text",""), toks)
        out.append(dd)
    return out


# ----------------------------
# Favoritos/Recentes
# ----------------------------
def _init_comp_state():
    st.session_state.setdefault("comp_axis", "🌍 Locais")
    st.session_state.setdefault("comp_fav_cities", [])
    st.session_state.setdefault("comp_fav_npcs", [])
    st.session_state.setdefault("comp_recent_cities", [])
    st.session_state.setdefault("comp_recent_npcs", [])
    st.session_state.setdefault("comp_region", None)
    st.session_state.setdefault("comp_city", None)
    st.session_state.setdefault("comp_npc_selected", None)


def _touch_recent(key: str, value: str, limit: int = 12):
    # key = "comp_recent_cities" ou "comp_recent_npcs"
    arr = st.session_state.get(key, []) or []
    arr = [x for x in arr if x != value]
    arr.insert(0, value)
    st.session_state[key] = arr[:limit]


def _toggle_fav(key: str, value: str):
    # key = "comp_fav_cities" ou "comp_fav_npcs"
    arr = st.session_state.get(key, []) or []
    if value in arr:
        arr = [x for x in arr if x != value]
    else:
        arr = [value] + arr
    st.session_state[key] = arr


def _is_fav(key: str, value: str) -> bool:
    return value in (st.session_state.get(key, []) or [])


# ----------------------------
# UI helpers
# ----------------------------
def _chip_row(tags: list[str]) -> None:
    if not tags:
        return
    chips = "".join([f'<span class="comp-chip">{t}</span>' for t in tags])
    st.markdown(chips, unsafe_allow_html=True)


def _breadcrumb(items: list[str]) -> None:
    st.markdown("**" + " › ".join(items) + "**")


def _nav_prev_next(current: str, ordered: list[str], prefix: str):
    if not current or current not in ordered or len(ordered) <= 1:
        return
    i = ordered.index(current)
    c1, c2, c3 = st.columns([0.55, 0.55, 1.3], gap="small")
    with c1:
        if st.button("⬅ Anterior", key=f"{prefix}_prev_{_stem_key(current)}"):
            st.session_state[prefix] = ordered[(i - 1) % len(ordered)]
            st.rerun()
    with c2:
        if st.button("Próximo ➡", key=f"{prefix}_next_{_stem_key(current)}"):
            st.session_state[prefix] = ordered[(i + 1) % len(ordered)]
            st.rerun()
    with c3:
        st.caption(f"{i+1}/{len(ordered)}")


def _render_region_panel(region_name: str, region_obj: dict) -> None:
    img_region = comp_find_image(region_name)
    img_map = comp_find_image(COMP_DEFAULT_MAP)
    st.markdown('<div class="comp-hero">', unsafe_allow_html=True)
    cA, cB = st.columns([1, 1.7], gap="large")
    with cA:
        if img_map:
            st.image(img_map, use_container_width=True)
            st.caption("🗺️ Mapa geral")
        elif img_region:
            st.image(img_region, use_container_width=True)
        else:
            st.caption("🖼️ (sem imagem encontrada para esta região)")
    with cB:
        st.markdown(f"## 🌍 {region_name}")
        if img_region and img_map != img_region:
            st.image(img_region, use_container_width=True)
        intro = (region_obj.get("intro") or "").strip()
        if intro:
            paras = [p.strip() for p in intro.split("\n\n") if p.strip()]
            st.markdown("\n\n".join(paras[:2]))
            if len(paras) > 2:
                with st.expander("Ler mais"):
                    st.markdown("\n\n".join(paras[2:]))
        else:
            st.caption("(sem texto de identidade detectado)")

        cities = region_obj.get("cities") or []
        if cities:
            st.markdown("**Cidades para começar a leitura:**")
            for cname in cities[:4]:
                if st.button(f"➡ {cname}", key=f"region_city_pick_{_stem_key(region_name)}_{_stem_key(cname)}"):
                    st.session_state["comp_city"] = cname
                    _touch_recent("comp_recent_cities", cname)
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_city_dossier(city_name: str, city_obj: dict, npcs: dict[str, dict], cities: dict[str, dict], gyms: dict[str, dict] | None = None, region_obj: dict | None = None) -> None:
    img_city = comp_find_image(city_name)
    img_map = comp_find_image(city_obj.get("region","")) or comp_find_image(COMP_DEFAULT_MAP)

    # header + ações
    st.markdown('<div class="comp-hero">', unsafe_allow_html=True)
    hA, hB, hC = st.columns([1, 1, 1.2], gap="large")
    with hA:
        if img_city:
            st.image(img_city, use_container_width=True)
            st.caption("📸 Paisagem da cidade")
        else:
            st.caption("🖼️ (sem imagem da cidade)")
    with hB:
        if img_map:
            st.image(img_map, use_container_width=True)
            st.caption("🗺️ Mapa da região")
        else:
            st.caption("🗺️ (sem mapa)")
    with hC:
        st.markdown(f"## 🏙️ {city_name}")
        st.markdown(f'<div class="comp-muted">Região: <b>{city_obj.get("region","")}</b></div>', unsafe_allow_html=True)

        fav = _is_fav("comp_fav_cities", city_name)
        if st.button("⭐ Remover favorito" if fav else "⭐ Favoritar", key=f"fav_city_{_stem_key(city_name)}"):
            _toggle_fav("comp_fav_cities", city_name)
            st.rerun()

        npc_names = list(npcs.keys())
        mentions = city_mentions_npcs(city_obj, npc_names)
        if mentions:
            st.markdown("**NPCs citados:**")
            cols = st.columns(3)
            for i, nm in enumerate(mentions[:9]):
                with cols[i % 3]:
                    if st.button(f"🧑 {nm}", key=f"comp_city_npc_{_stem_key(city_name)}_{_stem_key(nm)}"):
                        st.session_state["comp_axis"] = "🧑‍🤝‍🧑 NPCs"
                        st.session_state["comp_npc_selected"] = nm
                        _touch_recent("comp_recent_npcs", nm)
                        st.rerun()

        # cidades relacionadas (mesma região)
        same_region = [c for c, obj in cities.items() if obj.get("region") == city_obj.get("region") and c != city_name]
        if same_region:
            st.markdown("**Continue lendo nesta região:**")
            for cname in same_region[:4]:
                if st.button(f"➡ {cname}", key=f"rel_city_{_stem_key(city_name)}_{_stem_key(cname)}"):
                    st.session_state["comp_city"] = cname
                    _touch_recent("comp_recent_cities", cname)
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # trilha de leitura
    if region_obj:
        st.markdown('<div class="comp-panel">', unsafe_allow_html=True)
        st.markdown("### 📖 Próximo capítulo")
        region_cities = [c for c in (region_obj.get("cities") or []) if c in cities and c != city_name]
        if region_cities:
            next_city = region_cities[0]
            if st.button(f"🚪 Ir para {next_city}", key=f"next_city_{_stem_key(city_name)}"):
                st.session_state["comp_city"] = next_city
                _touch_recent("comp_recent_cities", next_city)
                st.rerun()
        mentions = city_mentions_npcs(city_obj, list(npcs.keys()))
        if mentions:
            st.markdown("**NPCs para conhecer:**")
            cols = st.columns(min(3, len(mentions)))
            for i, nm in enumerate(mentions[:3]):
                with cols[i % len(cols)]:
                    if st.button(f"🧑 {nm}", key=f"city_npc_next_{_stem_key(city_name)}_{_stem_key(nm)}"):
                        st.session_state["comp_axis"] = "🧑‍🤝‍🧑 NPCs"
                        st.session_state["comp_npc_selected"] = nm
                        _touch_recent("comp_recent_npcs", nm)
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    
    # ----------------------------
    # 🏅 GINÁSIO (estruturado)
    # ----------------------------
    gyms = gyms or {}
    g = gyms.get(city_name)
    if g:
        meta = g.get("meta") or {}
        staff = g.get("staff") or {}
        staff_npcs = g.get("staff_npcs") or {}

        st.markdown("## 🏅 Ginásio")
        chips = []
        if meta.get("tipo"): chips.append(f"Tipo: {meta.get('tipo')}")
        if meta.get("status"): chips.append(f"Status: {meta.get('status')}")
        if chips:
            _chip_row(chips)

        # bloco meta
        st.markdown('<div class="comp-block">', unsafe_allow_html=True)
        if meta.get("localizacao"):
            st.markdown(f"**Localização:** {meta.get('localizacao')}")
        if meta.get("arena_extra"):
            st.markdown(f"**Arena/Extra:** {meta.get('arena_extra')}")
        if meta.get("observacao"):
            st.markdown(f"**Observação:** {meta.get('observacao')}")
        st.markdown("</div>", unsafe_allow_html=True)

        lider = (meta.get("lider") or staff.get("lider") or "").strip()
        vice = (meta.get("vice_lider") or meta.get("vice-lider") or staff.get("vice_lider") or "").strip()

        def person_card(role_label: str, name: str):
            if not name:
                return
            npc_obj = npcs.get(name) or staff_npcs.get(name) or {}
            st.markdown('<div class="comp-card">', unsafe_allow_html=True)
            cA, cB = st.columns([0.65, 1.35], gap="large")
            with cA:
                imgp = comp_find_image(name)
                if imgp:
                    st.image(imgp, use_container_width=True)
                else:
                    st.caption("🖼️ (sem retrato)")
            with cB:
                st.markdown(f"### {role_label}: {name}")
                if npc_obj.get("ocupacao"):
                    st.caption(npc_obj.get("ocupacao"))
                pokes = npc_obj.get("pokemons") or []
                if pokes:
                    st.markdown("**Pokémons (sprites):**")
                    cols = st.columns(min(6, max(1, len(pokes))))
                    for i, pnm in enumerate(pokes[:12]):
                        with cols[i % len(cols)]:
                            spr = _poke_sprite_cached(pnm)
                            if spr:
                                st.image(spr, width=56)
                            st.caption(pnm)
                if name in npcs:
                    if st.button("Abrir NPC", key=f"open_gym_npc_{_stem_key(city_name)}_{_stem_key(name)}"):
                        st.session_state["comp_axis"] = "🧑‍🤝‍🧑 NPCs"
                        st.session_state["comp_npc_selected"] = name
                        _touch_recent("comp_recent_npcs", name)
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        if lider or vice:
            colL, colV = st.columns(2, gap="large")
            with colL:
                person_card("🏅 Líder", lider)
            with colV:
                person_card("🥈 Vice", vice)

        # narrativa/observações do ginásio
        if (g.get("narrative") or "").strip():
            with st.expander("📜 Detalhes do Ginásio (texto)", expanded=False):
                st.markdown((g.get("narrative") or "").strip())


    # Seções (modo operação: seletor)
    sections = city_obj.get("sections", {}) or {}
    sublocs = city_obj.get("sublocais", []) or []

    ordered_keys = list(sections.keys())

    # ordem preferencial
    pref = ["Visão geral", "Como é viver", "Ginásio", "Sublocais", "Treinadores", "Economia", "Riscos", "Rumores", "Missões"]
    def kscore(k: str):
        nk = _norm(k)
        for i, p in enumerate(pref):
            if _norm(p) in nk:
                return (i, k)
        return (999, k)

    ordered_keys = [k for k in sorted(ordered_keys, key=kscore) if (sections.get(k) or "").strip()]

    opts = ["Tudo"] + ordered_keys + (["📍 Locais Importantes"] if sublocs else [])
    sel = st.selectbox("Seção do dossiê", opts, index=0, key=f"city_section_{_stem_key(city_name)}")

    if sel == "Tudo":
        for title in ordered_keys:
            body = (sections.get(title) or "").strip()
            if not body:
                continue
            st.markdown(f"### {title}")
            st.markdown(body)
    elif sel == "📍 Locais Importantes":
        st.markdown("### 📍 Locais Importantes")
    else:
        st.markdown(f"### {sel}")
        st.markdown((sections.get(sel) or "").strip())

    if sublocs and (sel in ["Tudo", "📍 Locais Importantes"]):
        st.markdown("### 📍 Locais Importantes")
        for it in sublocs:
            nm = (it.get("name") or "").strip()
            tx = (it.get("text") or "").strip()
            if not nm and not tx:
                continue
            with st.expander(f"📍 {nm}" if nm else "📍 Local"):
                if tx:
                    st.markdown(tx)
                else:
                    st.caption("(sem descrição)")



def _render_npc_dossier(nm: str, npc: dict, cities: dict[str, dict], npcs: dict[str, dict], gyms: dict[str, dict] | None = None) -> None:
    img = comp_find_image(nm)

    st.markdown('<div class="comp-hero">', unsafe_allow_html=True)
    hA, hB = st.columns([1, 1.6], gap="large")
    with hA:
        if img:
            st.image(img, use_container_width=True)
        else:
            st.caption("🖼️ (sem retrato encontrado)")
        fav = _is_fav("comp_fav_npcs", nm)
        if st.button("⭐ Remover favorito" if fav else "⭐ Favoritar", key=f"fav_npc_{_stem_key(nm)}"):
            _toggle_fav("comp_fav_npcs", nm)
            st.rerun()

    with hB:
        st.markdown(f"## 🧑 {nm}")

        status = npc.get("status","")
        idade = npc.get("idade","")
        origem = npc.get("origem","")
        ocup = npc.get("ocupacao","")

        st.markdown('<div class="comp-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 1️⃣ Identidade")
        c1, c2 = st.columns(2)
        with c1:
            if status: st.markdown(f"**Status:** {status}")
            if idade: st.markdown(f"**Idade:** {idade}")
            if origem: st.markdown(f"**Origem:** {origem}")
        with c2:
            if ocup: st.markdown(f"**Ocupação:** {ocup}")

        pokes = npc.get("pokemons") or []
        if pokes:
            st.markdown("**Pokémons conhecidos:**")
            spr_cols = st.columns(min(6, max(1, len(pokes))))
            for i, pnm in enumerate(pokes[:12]):
                with spr_cols[i % len(spr_cols)]:
                    spr = _poke_sprite_cached(pnm)
                    if spr:
                        st.image(spr, width=56)
                    st.caption(pnm)

        st.markdown('<div class="comp-panel">', unsafe_allow_html=True)
        st.markdown("### 🔮 Continue lendo")
        origem = (npc.get("origem") or "").strip()
        status = (npc.get("status") or "").strip()
        related = []
        for other, obj in npcs.items():
            if other == nm:
                continue
            if origem and _norm(obj.get("origem","")) == _norm(origem):
                related.append(other)
            elif status and _norm(obj.get("status","")) == _norm(status):
                related.append(other)
        if related:
            for other in related[:3]:
                if st.button(f"🧑 Conhecer {other}", key=f"npc_related_{_stem_key(nm)}_{_stem_key(other)}"):
                    st.session_state["comp_npc_selected"] = other
                    _touch_recent("comp_recent_npcs", other)
                    st.rerun()
        else:
            st.caption("Abra outro NPC da lista ao lado para continuar a história.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------
    # 🏅 Referências de Ginásio
    # ----------------------------
    gyms = gyms or {}
    gym_refs = []
    for city, g in gyms.items():
        meta = g.get("meta") or {}
        lider = (meta.get("lider") or (g.get("staff") or {}).get("lider") or "").strip()
        vice = (meta.get("vice_lider") or meta.get("vice-lider") or (g.get("staff") or {}).get("vice_lider") or "").strip()
        if _norm(lider) == _norm(nm):
            gym_refs.append(("🏅 Líder", city, meta.get("tipo",""), meta.get("status","")))
        if _norm(vice) == _norm(nm):
            gym_refs.append(("🥈 Vice", city, meta.get("tipo",""), meta.get("status","")))

    if gym_refs:
        st.markdown("### 🏅 Ginásio")
        for role, city, tipo, status in gym_refs:
            line = f"**{role}** em **{city}**"
            meta_bits = []
            if tipo:
                meta_bits.append(f"Tipo: {tipo}")
            if status:
                meta_bits.append(f"Status: {status}")
            if meta_bits:
                line += " — " + " • ".join(meta_bits)
            st.markdown(line)
            if city in cities:
                if st.button(f"Abrir cidade: {city}", key=f"npc_open_city_{_stem_key(nm)}_{_stem_key(city)}"):
                    st.session_state["comp_axis"] = "🌍 Locais"
                    st.session_state["comp_region"] = cities[city].get("region")
                    st.session_state["comp_city"] = city
                    _touch_recent("comp_recent_cities", city)
                    st.rerun()

    # Cross-links: cidades e NPCs citados
    city_names = list(cities.keys())
    npc_names = list(npcs.keys())
    cm = npc_mentions_cities(npc, city_names)
    nm_hits = npc_mentions_npcs(npc, npc_names, nm)
    if cm:
        st.markdown("**Cidades citadas:**")
        cols = st.columns(3)
        for i, cn in enumerate(cm[:9]):
            with cols[i % 3]:
                if st.button(f"🏙️ {cn}", key=f"npc_city_{_stem_key(nm)}_{_stem_key(cn)}"):
                    st.session_state["comp_axis"] = "🌍 Locais"
                    # tenta inferir região
                    st.session_state["comp_region"] = cities.get(cn, {}).get("region")
                    st.session_state["comp_city"] = cn
                    _touch_recent("comp_recent_cities", cn)
                    st.rerun()

    if nm_hits:
        st.markdown("**Outros NPCs citados:**")
        cols = st.columns(3)
        for i, other in enumerate(nm_hits[:9]):
            with cols[i % 3]:
                if st.button(f"🧑 {other}", key=f"npc_npc_{_stem_key(nm)}_{_stem_key(other)}"):
                    st.session_state["comp_npc_selected"] = other
                    _touch_recent("comp_recent_npcs", other)
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 2️⃣ Lore completa")
    sections = npc.get("sections") or {}
    pref = ["Histórico", "Rumores", "Objetivos", "Estado Atual", "Lore"]
    def score(k):
        nk = _norm(k)
        for i, p in enumerate(pref):
            if _norm(p) in nk:
                return (i, k)
        return (999, k)

    for k in sorted(sections.keys(), key=score):
        v = (sections.get(k) or "").strip()
        if not v:
            continue
        if k != "Lore":
            st.markdown(f"#### {k}")
        st.markdown(v)


@st.cache_data
def get_font_base64(font_path):
    """Lê o arquivo de fonte e converte para base64 para uso no CSS."""
    try:
        with open(font_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None
# ==============================================================================
# 📚 COMPENDIUM NOVO (JSON + DARK SOULS) - CORRIGIDO
# ==============================================================================

def render_compendium_page() -> None:
    # garante default
    if "comp_view" not in st.session_state:
        st.session_state["comp_view"] = "home"
    
    # --- INÍCIO DA INSERÇÃO ---
    font_b64 = get_font_base64("fonts/DarkSouls.ttf")
    font_css = f"@font-face {{ font-family: 'DarkSouls'; src: url('data:font/ttf;base64,{font_b64}') format('truetype'); }}" if font_b64 else ""

    st.markdown(f"""
    <style>
        {font_css}
        :root {{
            --ds-font: 'DarkSouls', serif;
            --ds-gold-dim: rgba(255,215,0,0.55);
        }}
        /* Fundo preto no compendium */
        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background: #000 !important;
            color: #f8fafc;
        }}
        /* Aplica a fonte em tudo na aba compendium */
        [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {{
            font-family: var(--ds-font) !important;
        }}
        /* ESTILO DOS BOTÕES (O que você gostou) */
        /* Remove a caixa/borda padrão e deixa só o texto */
        .stButton > button {{
            background: transparent !important;
            border: none !important;
            color: #888 !important; /* Cor normal (cinza escuro) */
            font-size: 24px !important;
            text-shadow: 0px 0px 5px rgba(0,0,0,0.8);
            font-family: 'DarkSouls', serif !important;
            text-transform: uppercase;
            transition: transform 0.2s, color 0.2s;
        }}
        .stButton > button:hover {{
            color: #FFD700 !important; text-shadow: 0 0 10px #FFD700; transform: scale(1.1);
        }}
        .stButton > button:active, .stButton > button:focus {{
            color: #FFD700 !important; outline: none !important; border: none !important; box-shadow: none !important;
        }}
        
    /* Esconde elementos padrão do Streamlit para imersão */
    [data-testid="stHeader"] {{ visibility: hidden; }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    
    /* Esconde o resto da UI padrão do Streamlit (menu, toolbar, rodapé) */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ visibility: hidden !important; height: 0px !important; }}
    [data-testid="stStatusWidget"] {{ visibility: hidden !important; }}
    [data-testid="stDeployButton"] {{ display: none !important; }}

    
/* HOME (igual app 35) */
.ds-home {{
    min-height: 74vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 22px;
    padding: 10px 0 0 0;
}}
.ds-title {{
    text-align: center;
    color: var(--ds-white);
    font-size: 56px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin: 0;
}}
.ds-press {{
    text-align: center;
    color: var(--ds-faint);
    font-size: 14px;
    letter-spacing: 0.34em;
    text-transform: uppercase;
    margin: 0;
}}
@keyframes dsBlink {{
    0%, 48% {{ opacity: 0.10; }}
    60%, 100% {{ opacity: 0.88; }}
}}
.ds-blink {{ animation: dsBlink 1.05s ease-in-out infinite; }}

/* Tabs (radio horizontal) no rodapé */
div[data-testid="stRadio"] {{
    position: fixed !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    bottom: 48px !important;
    z-index: 10000 !important;
    padding: 10px 18px !important;
    background: rgba(0,0,0,0.0) !important;
}}
/* Tabs (radio horizontal) no rodapé — SOMENTE NA HOME */
.ds-home div[data-testid="stRadio"] {{
    position: fixed !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    bottom: 48px !important;
    z-index: 10000 !important;
    padding: 10px 18px !important;
    background: rgba(0,0,0,0.0) !important;
}}
.ds-home div[data-testid="stRadio"] > label {{ display: none !important; }}

/* Linha dourada fina acima dos tabs */
.ds-home div[data-testid="stRadio"]::before {{
    content: "";
    display: block;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--ds-gold-dim), transparent);
    margin-bottom: 10px;
}}

/* Estilo das opções */
.ds-home div[role="radiogroup"] {{
    display: flex !important;
    gap: 58px !important;
    justify-content: center !important;
    align-items: center !important;
}}
.ds-home div[role="radiogroup"] > label {{
    position: relative !important;
    padding: 6px 10px !important;
    cursor: pointer !important;
    color: rgba(255,255,255,0.70) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.24em !important;
    font-size: 12px !important;
    user-select: none !important;
    transition: all 120ms ease !important;
}}
.ds-home div[role="radiogroup"] > label:hover {{
    color: #FFD700 !important;
    text-shadow: 0 0 10px rgba(255,215,0,0.65) !important;
}}
.ds-home div[role="radiogroup"] > label[data-checked="true"] {{
    color: #FFD700 !important;
    text-shadow: 0 0 10px rgba(255,215,0,0.65) !important;
}}
/* Esconde o bolinha padrão do radio */
.ds-home div[role="radiogroup"] input {{ display: none !important; }}



.ds-frame {{
            background: rgba(0,0,0,0.55);
            border: 2px solid rgba(176,143,60,0.55);
            box-shadow: 0 0 45px rgba(0,0,0,0.9);
            border-radius: 12px;
            padding: 26px 26px 18px 26px;
            position: relative;
        }}
        .ds-frame::after {{
            content: "";
            position: absolute;
            top: 10px; left: 10px; right: 10px; bottom: 10px;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            pointer-events: none;
        }}
        .ds-name {{
            font-size: 52px;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            text-align: center;
            margin: 0 0 10px 0;
            padding-bottom: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.10);
        }}
        .ds-meta {{
            text-align: center;
            color: var(--ds-faint);
            letter-spacing: 0.20em;
            text-transform: uppercase;
            font-size: 13px;
            margin-bottom: 18px;
        }}
        .ds-portrait {{
            display:flex; justify-content:center; margin: 12px 0 10px 0;
        }}
        .ds-portrait img {{
            max-width: 320px; width:100%;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 0 26px rgba(0,0,0,0.75);
        }}
        .ds-section-title {{
            margin-top: 14px; margin-bottom: 10px;
            color: var(--ds-faint);
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 8px;
            text-align:center;
        }}
        .ds-poke-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(0,0,0,0.4);
            border: 1px solid #443311;
            border-radius: 8px;
            padding: 8px;
            transition: all 0.3s ease;
        }}
        .ds-poke-container:hover {{
            border-color: #FFD700;
            background: rgba(20,20,10,0.8);
            transform: translateY(-3px);
        }}
        .ds-poke-img {{
            width: 60px;
            height: 60px;
            object-fit: contain;
            filter: sepia(1) brightness(0.8) contrast(1.2);
            opacity: 0.85;
            transition: all 0.4s ease;
        }}
        .ds-poke-container:hover .ds-poke-img {{
            filter: sepia(0) brightness(1) contrast(1);
            opacity: 1;
        }}
        .ds-poke-name {{
            font-size: 10px;
            text-transform: uppercase;
            color: #887766;
            margin-top: 5px;
            text-align: center;
            letter-spacing: 1px;
        }}
        .ds-poke-container:hover .ds-poke-name {{
            color: #FFD700;
        }}
        .ds-history p {{
            color: rgba(255,255,255,0.88);
            font-size: 18px;
            line-height: 1.68;
            text-align: justify;
            margin: 0 0 14px 0;
        }}
        /* TOP NAV — texto puro */
        .ds-tab div[data-testid="stButton"] > button {{
          background: transparent !important;
          border: none !important;
          box-shadow: none !important;
          color: rgba(255,255,255,0.72) !important;
        }}
        
        /* Hover dourado */
        .ds-tab div[data-testid="stButton"] > button:hover {{
          color: rgba(255,215,0,0.95) !important;
          text-shadow: 0 0 14px rgba(255,215,0,0.35) !important;
        }}
        
        /* Selecionado = dourado sempre */
        .ds-tab.selected div[data-testid="stButton"] > button {{
          color: rgba(255,215,0,0.98) !important;
          text-shadow: 0 0 18px rgba(255,215,0,0.45) !important;
        }}
        .ds-gold-top{{
          height: 1px;
          width: 100%;
          margin: 12px 0 18px 0;
          background: linear-gradient(90deg, transparent, var(--ds-gold-dim), transparent);
        }}
        .ds-nav-item,
        .ds-nav-item:visited {{
          color: rgba(255,255,255,0.72) !important;
          text-decoration: none !important;
        }}
        
        .ds-nav-item:hover {{
          color: rgba(255,215,0,0.95) !important;
          text-shadow: 0 0 14px rgba(255,215,0,0.35) !important;
          text-decoration: none !important;
        }}
        
        .ds-nav-item.selected {{
          color: rgba(255,215,0,0.98) !important;
          text-shadow: 0 0 16px rgba(255,215,0,0.45) !important;
          text-decoration: none !important;
        }}
        
        /* Remove outline padrão */
        .ds-tab div[data-testid="stButton"] > button:focus {{
          outline: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
        
    components.html("<div class='ds-gold-top'></div>", height=10)

    # 4️⃣ Menu
   
    # ----------------------------
    # Estado do Compendium — NÃO cai direto em NPCs
    # ----------------------------
    st.session_state.setdefault("comp_view", "home")
    st.session_state.setdefault("comp_selected_npc", None)
    try:
        comp_data = comp_load()  # retorna {"regions","cities","npcs","gyms",...}
    except Exception as e:
        st.error(f"Falha ao carregar Compendium: {e}")
        return
    
    npcs_gerais = comp_data.get("npcs", {}) or {}
    # ----------------------------
    # Navegação via query params (?cv=...) — evita click-detector na HOME e no TOP NAV
    # ----------------------------
    def _qp_get(key: str):
        try:
            qp = st.query_params
            v = qp.get(key)
            if v is None:
                return None
            if isinstance(v, list):
                return v[0] if v else None
            return str(v)
        except Exception:
            qp = st.experimental_get_query_params()
            v = qp.get(key)
            if not v:
                return None
            return v[0]
    
    def _clear_qp():
        try:
            st.query_params.clear()
        except Exception:
            st.experimental_set_query_params()
    
    
    

    
    def _consume_comp_qp():
        cv = _qp_get("cv")  # home / npcs / ginasios / locais / sair
        if not cv:
            return
    
        if cv == "sair":
            st.session_state["nav_to"] = "Pokédex (Busca)"
        else:
            st.session_state["comp_view"] = cv
    
            # se voltou pro HOME, reseta o radio do rodapé
            if cv == "home":
                st.session_state["ds_home_tabs"] = "__home__"
                st.session_state["ds_home_tabs_prev"] = "__home__"
    
            if cv != "npcs":
                st.session_state["comp_selected_npc"] = None
    
        _clear_qp()
        st.rerun()
    



            
    # ----------------------------
    # Navegação (sempre no topo)
    # ----------------------------
    # ----------------------------
    # Navegação (HOME embaixo / outras páginas em cima e sticky)
    # ----------------------------
    def _go(view: str):
        st.session_state["comp_view"] = view
        if view != "npcs":
            st.session_state["comp_selected_npc"] = None
        st.rerun()
    
    
        st.markdown("</div>", unsafe_allow_html=True)



    _consume_comp_qp()
    

    # =====================================================================

    # NPCs (VERSÃO CORRIGIDA - SAFE IDs)
    # =====================================================================
    if st.session_state["comp_view"] == "npcs":
        try:
            from st_click_detector import click_detector
        except ImportError:
            st.error("Biblioteca não instalada. Adicione 'st-click-detector' ao requirements.txt e reinicie o app.")
            return

        render_ds_tools_nav(st.session_state["comp_view"])
        st.markdown("""
        <style>
        /* Fundo global do app (mata rgb(14,17,23)) */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        section.main {
          background: #000 !important;
        }
        
        /* Wrappers internos */
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="stBlock"]{
          background: transparent !important;
        }
        
        /* pega exatamente o div com style inline rgb(14, 17, 23) */
        section.main div[style*="background: rgb(14, 17, 23)"]{
          background: #000 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    

        
        # CSS das molduras (não interfere no click)
        css = """
        <style>
          .ds-npc-banner{
            position: sticky;
            top: 0;
            left: 0;
            right: 0;
            z-index: 12000;
            text-align: center;
            background: rgba(0,0,0,0.85);
            border-bottom: 1px solid rgba(176,143,60,0.6);
            box-shadow: 0 6px 18px rgba(0,0,0,0.6);
            padding: 14px 12px 12px 12px;
            text-transform: uppercase;
            letter-spacing: 0.25em;
          }
          .ds-npc-banner-title{
            font-size: 20px;
            color: rgba(255,255,255,0.95);
            margin: 0 0 6px 0;
          }
          .ds-npc-banner-sub{
            font-size: 11px;
            color: rgba(255,255,255,0.65);
            margin: 0;
          }
          .ds-npc-banner-spacer{
            height: 0px;
          }
          .ds-npc-panel{
            background-repeat:no-repeat;
            background-position:center;
            background-size:100% 100%;
            padding: 28px 28px 26px 28px;
            min-height: 0px;     /* deixa crescer pelo conteúdo */
          }
          .ds-npc-panel.left{
          background: transparent !important;
          background-image: none !important;
          border: none !important;
          box-shadow: none !important;
            padding-top: 6px !important;   /* bem pequeno */

        }
        

        
        iframe[title^="st_click_detector"]{
          background:  #000 !important;
          border: none !important;
          box-shadow: none !important;
        }
        /* 1) pinta o container do componente (o mais importante) */
        div[data-testid="stComponentFrame"]{
          background: #000 !important;
          border: none !important;
          box-shadow: none !important;
          padding: 0 !important;
          margin: 0 !important;
        }
        
        /* 2) pinta o iframe também */
        div[data-testid="stComponentFrame"] iframe,
        iframe[title^="st_click_detector"],
        iframe[title*="click_detector"]{
          background: #000 !important;
          border: none !important;
          box-shadow: none !important;
        }
        
        /* 3) remove padding extra do wrapper do Streamlit */
        div[data-testid="stElementContainer"]{
          padding: 0 !important;
          margin: 0 !important;
          background: transparent !important;
        }        
        /* remove padding/margem que às vezes vira “caixa” */
        .ds-npc-panel.left div[data-testid="stElementContainer"]{
          padding: 0 !important;
          margin: 0 !important;
        }
        
               
        
          /* grid automático */
          .ds-grid{
            display:grid;
            grid-template-columns:repeat(auto-fill, minmax(140px, 1fr));
            gap: 10px;
            width:100%;
          }
        
          /* evita qualquer camada bloquear cliques */
          .ds-npc-panel, .ds-npc-panel * { pointer-events:auto; }
        </style>
        """

        
        st.markdown(css, unsafe_allow_html=True)
        st.markdown("""
        <style>
          .ds-npc-panel{ background: transparent !important; }
          .ds-npc-panel.left{
            background: transparent !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
          }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(
        st.markdown(
            "<div class='ds-npc-banner'>"
            "<div class='ds-npc-banner-title'>Selecione um NPC</div>"
            "<div class='ds-npc-banner-sub'>Clique em um retrato à esquerda</div>"
            "</div>",
            unsafe_allow_html=True,
        )


        # --- LAYOUT PRINCIPAL (SEM INDENTAÇÃO ERRADA) ---
        left, right = st.columns([1.25, 2.15], gap="large")
    
        # --- COLUNA ESQUERDA ---
        with left:
            st.markdown("<div class='ds-npc-panel left'>", unsafe_allow_html=True)
            search = st.text_input(
                "Buscar personagem",
                key="ds_npc_search",
                placeholder="Nome ou história...",
            ).strip()
    
            def _norm(s: str) -> str:
                if not isinstance(s, str):
                    return ""
                return re.sub(r"\s+", " ", s).strip().lower()
    
            q = _norm(search)
    
            # Preparar lista
            items = []
            for nome, obj in (npcs_gerais or {}).items():
                if not isinstance(obj, dict):
                    continue
    
                historia = ""
                secs = obj.get("sections") or {}
                if isinstance(secs, dict):
                    historia = secs.get("História") or secs.get("Historia") or ""
    
                hay = _norm(nome) + " " + _norm(historia)
                if not q or q in hay:
                    items.append((nome, obj))
    
            items.sort(key=lambda x: x[0])
    
            # Cache de imagens
            @st.cache_data(show_spinner=False)
            def _thumb_data_uri(path: str, max_w: int = 360, max_h: int = 520) -> str:
                try:
                    from PIL import Image
                    import io, base64, os
    
                    if not path or not os.path.exists(path):
                        return ""
                    img = Image.open(path).convert("RGB")
                    img.thumbnail((max_w, max_h))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    return f"data:image/jpeg;base64,{b64}"
                except:
                    return ""
    
            if not items:
                st.info("Nenhum NPC encontrado.")
            else:
                content_html = """
                <style>
                  html, body{
                    background: #000 !important;
                    margin: 0 !important;
                    padding: 6px !important;  /* pinta também a “folga” */

                  }
                :root{ --background-color:#000 !important; --text-color:#fff !important; }

                  body, div { background:#000 !important; }
                
                  div[style*="background: rgb(14, 17, 23)"]{
                    background:#000 !important;
                  }
                                
                  .ds-npc-grid{
                    display:grid;
                    grid-template-columns:repeat(4, 1fr);
                    gap: 10px;
                    width:100%;
                    row-gap:12px;

                    background: #000 !important;
                  }
                                
                  .ds-card {
                    position: relative;
                    aspect-ratio: 3/4;
                    border: 2px solid #554422;
                    background: #000 !important;   /* <- ISSO remove o “cinza” */

                    border-radius: 8px;
                    overflow: hidden;
                    cursor: pointer;
                    transition: transform 0.1s;
                  }
                
                  .ds-card:hover { border-color: #FFD700; transform: scale(1.02); }
                
                  .ds-card img {
                    width: 100%;
                    height:100%;
                    object-fit: cover;
                    filter: brightness(0.8);
                    display: block;
                    background: #000 !important;   /* reforço (se o browser/iframe insistir) */

                  }
                
                  .ds-name-tag {
                    position: absolute;
                    bottom: 0; left: 0; right: 0;
                    background: rgba(0,0,0,0.85);
                    color: #ddd;
                    font-size: 10px;
                    text-align: center;
                    padding: 4px;
                    font-weight: bold;
                    text-transform: uppercase;
                  }
                
                  a { text-decoration: none; display: block; }
                </style>
                
                <div class="ds-npc-grid">

                """
   
                id_map = {}
    
                for idx, (nome, obj) in enumerate(items):
                    safe_id = str(idx)
                    id_map[safe_id] = nome
    
                    img_path = None
                    try:
                        img_path = comp_find_image(nome)
                    except:
                        pass
    
                    src = _thumb_data_uri(img_path) if img_path else ""
                    img_html = f"<img src='{src}' />" if src else "<div style='width:100%;height:100%;background:#222;'></div>"
    
                    content_html += f"""
                    <a href="javascript:void(0)" id="{safe_id}">
                        <div class="ds-card">
                            {img_html}
                            <div class="ds-name-tag">{nome}</div>
                        </div>
                    </a>
                    """
    
                content_html += "</div>"
                clicked_safe_id = click_detector(content_html)
    
                if clicked_safe_id is not None:
                    nome_selecionado = id_map.get(str(clicked_safe_id))
                    if nome_selecionado and nome_selecionado != st.session_state.get("comp_selected_npc"):
                        st.session_state["comp_selected_npc"] = nome_selecionado
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)  # NÃO remove: fecha <div class='ds-npc-panel left'>

            
        # --- COLUNA DIREITA ---
        # --- COLUNA DIREITA ---
        with right:
            st.markdown("<div class='ds-npc-panel right'>", unsafe_allow_html=True)
        
            sel = st.session_state.get("comp_selected_npc")
            if not sel:
                st.markdown(
                    "<div class='ds-frame'><div class='ds-name' style='font-size:30px;'>NENHUM NPC SELECIONADO</div>"
                    unsafe_allow_html=True,
                )
            else:
                npc = npcs_gerais.get(sel, {}) or {}
                ocupacao = npc.get("ocupacao", "")
                idade = npc.get("idade", "")
                status = npc.get("status", "")
        
                # retrato grande (base64 p/ garantir)
                portrait_b64 = ""
                portrait_path = None
                try:
                    portrait_path = comp_find_image(sel)
                except Exception:
                    portrait_path = None
        
                ext = "png"
                if portrait_path and os.path.exists(portrait_path):
                    try:
                        with open(portrait_path, "rb") as f:
                            portrait_b64 = base64.b64encode(f.read()).decode("utf-8")
                        ext = os.path.splitext(portrait_path)[1].lower().replace(".", "")
                        if ext not in ("png", "jpg", "jpeg", "webp"):
                            ext = "png"
                    except Exception:
                        portrait_b64 = ""
                        ext = "png"
        
                # sprites dos pokemons (usa mapa oficial)
                pokemons = npc.get("pokemons") or npc.get("pokemons_conhecidos") or []
                if not isinstance(pokemons, list):
                    pokemons = []
        
                try:
                    name_map = get_official_pokemon_map() or {}
                except Exception:
                    name_map = {}
        
                sprite_imgs = []
                for pkm in pokemons:
                    try:
                        url = get_pokemon_image_url(str(pkm), name_map, mode="sprite", shiny=False)
                    except Exception:
                        url = ""
                    if url:
                        sprite_imgs.append(url)
        
                # história
                historia = ""
                secs = npc.get("sections") or {}
                if isinstance(secs, dict):
                    historia = secs.get("História") or secs.get("Historia") or ""
        
                # highlight busca na história
                q2 = st.session_state.get("ds_npc_search", "").strip()
                h_html = ""
                if isinstance(historia, str) and historia.strip():
                    paragraphs = [p.strip() for p in historia.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        safe = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        if q2:
                            pattern = re.compile(re.escape(q2), re.IGNORECASE)
                            safe = pattern.sub(lambda m: f"<span class='ds-mark'>{m.group(0)}</span>", safe)
                        h_html += f"<p>{safe}</p>"
                else:
                    h_html = "<p>(Sem história cadastrada)</p>"
        
                meta_parts = []
                if ocupacao:
                    meta_parts.append(str(ocupacao))
                if status:
                    meta_parts.append(f"STATUS: {status}")
                if idade:
                    meta_parts.append(f"IDADE: {idade}")
                meta_line = " | ".join(meta_parts) if meta_parts else ""
        
                portrait_html = ""
                if portrait_b64:
                    portrait_html = f"<div class='ds-portrait'><img src='data:image/{ext};base64,{portrait_b64}' /></div>"
        
                sprites_html = ""
                if sprite_imgs:
                    sprites_html = "<div class='ds-sprites'>" + "".join(
                        f"<img src='{u}' alt='sprite'/>" for u in sprite_imgs
                    ) + "</div>"
        
                st.markdown(
                    f"""
                    <div class="ds-frame">
                        <div class="ds-name">{sel}</div>
                        <div class="ds-meta">{meta_line}</div>
                        {portrait_html}
                        {sprites_html}
                        <div class="ds-section-title">História</div>
                        <div class="ds-history">{h_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
            st.markdown("</div>", unsafe_allow_html=True)
        
        return

    
    # =====================================================================
    # Ginásios / Locais (placeholder)
    # =====================================================================
    if st.session_state["comp_view"] == "ginasios":
        render_ds_tools_nav(st.session_state["comp_view"])
        st.markdown(
            "<div class='ds-frame'><div class='ds-name'>GINÁSIOS</div><div class='ds-meta'>EM CONSTRUÇÃO</div></div>",
            unsafe_allow_html=True,
        )
        return
    
    if st.session_state["comp_view"] == "locais":
        render_ds_tools_nav(st.session_state["comp_view"])
        st.markdown(
            "<div class='ds-frame'><div class='ds-name'>LOCAIS</div><div class='ds-meta'>EM CONSTRUÇÃO</div></div>",
            unsafe_allow_html=True,
        )
        return
    
    # ... (código existente acima: if st.session_state["comp_view"] == "locais": ...)

    # =========================================================
    # VIEW: HOME (estilo do app 35 — sem clicker e sem ENTER)
    # =========================================================
    if st.session_state["comp_view"] == "home":
        st.markdown(
            """
            <div class="ds-home">
                <div class="ds-title">BEM VINDO A GA'AL</div>
                <div class="ds-press ds-blink">PRESS ANY BUTTON</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_key = st.radio(
            "Compendium Tabs",
            ["__home__", "npcs", "ginasios", "locais", "sair"],  # <-- placeholder
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="ds_home_tabs",
            format_func=lambda v: {
                "__home__": "",      # não mostra texto
                "npcs": "NPCs",
                "ginasios": "Ginásios",
                "locais": "Locais",
                "sair": "Sair",
            }[v],
        )
        
        # esconde visualmente o primeiro item (placeholder)
        st.markdown(
            """
            <style>
              /* some o 1º item do radio (placeholder) */
              div[data-testid="stRadio"] label:first-child { display:none !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # Evita rerun em loop na primeira renderização
        if "ds_home_tabs_prev" not in st.session_state:
            st.session_state["ds_home_tabs_prev"] = tab_key
            return
        
        if st.session_state["ds_home_tabs_prev"] != tab_key:
            st.session_state["ds_home_tabs_prev"] = tab_key
        
            if tab_key == "__home__":
                return
        
            if tab_key == "sair":
                st.session_state["nav_to"] = "Pokédex (Busca)"
            else:
                st.session_state["comp_view"] = tab_key  # <-- já vem sem acento
        
            st.rerun()
        
        return


    
    
    def _tentar_achar_imagem_compendium(nome):
        if not nome:
            return None
        import os
        for tentativa in [nome, nome.replace(" ", "_"), nome.replace(" ", ""), nome.lower()]:
            for ext in [".png", ".jpg", ".jpeg"]:
                if os.path.exists(tentativa + ext):
                    return tentativa + ext
                if os.path.exists("assets/" + tentativa + ext):
                    return "assets/" + tentativa + ext
        return None
# ==============================================================================
# PÁGINA 1: POKEDEX (VISÃO DE FOCO + CARROSSEL INFERIOR)
# (As demais abas usam o tema global, sem sobrescritas claras.)
# ==============================================================================
if page == "Pokédex (Busca)":
    st.markdown("""
    <style>
    /* ============================================================
       1. ESTILO GLOBAL E GERAL (MANTIDO)
       ============================================================ */
    [data-testid="stAppViewContainer"] {
        animation: pageFade 0.35s ease-in;
    }
    [data-testid="stMainBlockContainer"] {
        animation: contentSlide 0.35s ease-in;
    }
    h1, h2, h3 {
        color: #f8fafc;
        text-shadow: 0 1px 2px rgba(15, 23, 42, 0.6);
    }
    .pokedex-shell {
        border-radius: 18px;
        padding: 18px 18px 8px 18px;
        border: 2px solid rgba(148, 163, 184, 0.55);
        box-shadow: inset 0 0 18px rgba(15, 23, 42, 0.55);
        background: rgba(15, 23, 42, 0.65);
    }
    
    /* Fontes P2 */
    .pokedex-shell, .pokedex-card, .pokedex-info-value, .pokedex-info-title, .pokedex-header {
        font-family: "Press Start 2P", cursive !important;
    }
    
    .pokedex-header {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        background: rgba(226, 232, 240, 0.9);
        padding: 6px 16px;
        border-radius: 16px;
        font-size: 12px;
        color: #0f172a;
    }
    .pokedex-grid-note {
        font-size: 12px;
        color: #e2e8f0;
        text-align: center;
        margin: 6px 0 10px 0;
    }
    .pokedex-card {
        background: rgba(15, 23, 42, 0.82);
        color: #f8fafc;
        padding: 18px;
        border-radius: 16px;
        border: 2px solid rgba(255,255,255,0.35);
        margin-top: 18px;
    }
    .pokedex-detail-grid { display: grid; gap: 12px; }
    
    .pokedex-info-card {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        color: #f8fafc !important;
        padding: 10px;
        border-radius: 8px;
    }
    .pokedex-info-card--dark { background: #e2e8f0; }
    
    .pokedex-info-title {
        font-size: 12px;
        color: #38bdf8 !important;
        margin-bottom: 4px;
    }
    .pokedex-info-value {
        font-size: 14px;
        color: #0f172a;
        line-height: 1.6;
    }
    .pokedex-info-card--wide { padding: 12px 14px; }
    .pokedex-info-card--wide .pokedex-info-value { font-size: 12px; }
    .pokedex-info-card--wide .section-title { margin-top: 0; }
    
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
        display: flex; gap: 12px; overflow-x: auto; padding: 10px 4px;
    }
    .pokedex-carousel img {
        width: 72px; height: 72px; image-rendering: pixelated;
        background: rgba(255,255,255,0.25); border-radius: 10px; padding: 6px;
    }

    /* CARROSSEL INFERIOR */
    .pokedex-footer-carousel {
        display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 12px; padding: 14px;
        background: rgba(0, 0, 0, 0.30); border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.18); scroll-behavior: smooth;
    }
    .pokedex-footer-carousel::-webkit-scrollbar { height: 8px; }
    .pokedex-footer-carousel::-webkit-scrollbar-thumb { background: #FFCC00; border-radius: 10px; }

    .carousel-item {
        flex: 0 0 auto; width: 70px; height: 70px; border-radius: 12px;
        display: grid; place-items: center; cursor: pointer;
        background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
        transition: transform 0.15s;
    }
    .carousel-item:hover { transform: scale(1.12); }
    .carousel-item img { width: 54px; height: 54px; image-rendering: pixelated; }
    .carousel-item-active { border: 2px solid #FFCC00; background: rgba(255, 204, 0, 0.10); }

    .info-label { color: #ffd166; font-weight: 800; }
    .section-title { color: #80ed99; font-weight: 900; margin-top: 10px; }
    .hi-red { color: #ff5c5c; font-weight: 900; }
    .hi-cyan { color: #4dd6ff; font-weight: 900; }
    .hi-purple { color: #b197ff; font-weight: 900; }
    .power-badge {
        display: block; width: fit-content; margin: 10px auto 0 auto;
        padding: 6px 12px; border-radius: 999px;
        background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.25);
        color: #ffd166; font-weight: 900; text-align: center;
    }
    @keyframes pageFade { from { opacity: 0.92; } to { opacity: 1; } }
    @keyframes contentSlide { from { transform: translateY(8px); opacity: 0.92; } to { transform: translateY(0); opacity: 1; } }


    /* ============================================================
       2. O NOVO CSS DOS TILES (CORRIGIDO PARA BATER COM O PYTHON)
       ============================================================ */

    /* MOLDURA DO CARD */
    .dex-card-frame {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 110px;        
        width: 100%;
        border-radius: 12px;
        margin-bottom: 8px;
        transition: transform 0.2s ease;
        background: rgba(15, 23, 42, 0.6);
        position: relative;
    }
    
    .dex-card-frame:hover {
        transform: scale(1.02);
    }

    /* A IMAGEM */
    .dex-sprite-img {
        max-width: 80px;
        max-height: 80px;
        width: auto;
        height: auto;
        object-fit: contain;
        image-rendering: pixelated;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.5));
        z-index: 1;
    }

    /* --- AQUI ESTAVA O ERRO: NOMES DAS CLASSES --- */
    /* Antes estava .frame-caught, agora é .dex-frame--caught */

    /* 🟢 CAPTURADO (Verde) */
    .dex-frame--caught {
        border: 2px solid #00ff41;
        box-shadow: 0 0 12px rgba(0, 255, 65, 0.25), inset 0 0 15px rgba(0, 60, 20, 0.6);
        background: rgba(0, 60, 20, 0.4);
    }

    /* 🔵 VISTO (Azul) */
    .dex-frame--seen {
        border: 2px solid #00d0ff;
        box-shadow: 0 0 12px rgba(0, 208, 255, 0.25), inset 0 0 15px rgba(0, 40, 60, 0.6);
        background: rgba(0, 40, 60, 0.4);
    }

    /* ⭐ WISHLIST (Dourado) */
    .dex-frame--wish {
        border: 2px solid #ffd700;
        box-shadow: 0 0 12px rgba(255, 215, 0, 0.25), inset 0 0 15px rgba(60, 50, 0, 0.6);
        background: rgba(60, 50, 0, 0.4);
    }

    /* ⚪ PADRÃO (Cinza) */
    .dex-frame--default {
        border: 2px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.03);
    }
    </style>
    """, unsafe_allow_html=True)
if page == "Pokédex (Busca)":
    dex_param = st.query_params.get("dex", None)
    if dex_param:
        st.session_state["pokedex_selected"] = str(dex_param)
        st.query_params.clear() # Limpa a URL após capturar
        st.rerun()
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
            st.markdown("<div class='pokedex-detail-grid'>", unsafe_allow_html=True)
            for label, value in entries:
                # Título: Descrição
                if label == "Descrição da Pokedex":
                    value_html = str(value).replace("\n", "<br>")
                    st.markdown(
                        f"""
                        <div class='pokedex-info-card pokedex-info-card--wide'>
                            <div class='section-title'>📘 Descrição da Pokédex</div>
                            <div class='pokedex-info-value'>{value_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    continue

                # Título: Viabilidade (texto grande)
                if label == "Viabilidade":
                    viab = (
                        str(value)
                        .replace("PARCEIROS:", "<br><br><strong>👥 PARCEIROS:</strong>")
                        .replace("Explicação:", "<br><br><strong>💡 EXPLICAÇÃO:</strong>")
                        .replace("Habilidade:", "<strong>✨ Habilidade:</strong>")
                    )

                    viab = viab.replace("\n", "<br>")

                    # pinta o FIR (e outros códigos se quiser)
                    viab = viab.replace("FIR", "<span class='hi-red'>FIR</span>")

                    # destaca os códigos de estratégia no texto (mantém o que você já tinha)
                    for code in codes:
                        viab = re.sub(rf"\b{re.escape(code)}\b", f"<span class='hi-purple'>{code}</span>", viab)

                    st.markdown(
                        f"""
                        <div class='pokedex-info-card pokedex-info-card--wide pokedex-info-card--dark'>
                            <div class='section-title'>🧠 Viabilidade</div>
                            <div class='pokedex-info-value'>{viab}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    continue

                # Campos normais (Tipo, Raridade, Biomas, Região etc.)
                value_html = str(value).replace("\n", "<br>")
                st.markdown(
                    f"""
                    <div class='pokedex-info-card'>
                        <div class='pokedex-info-title'>{label}</div>
                        <div class='pokedex-info-value'>{value_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)


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
            # --- LÓGICA DE FORMAS VISUAIS (Dex) ---
            # Por padrão, usa o nome do Excel
            target_visual_name = p_name 
            
            # Se for Lycanroc, exibe o seletor
            if "lycanroc" in p_name.lower().strip():
                lyc_dex_form = st.radio(
                    "Visualizar Forma:",
                    ["Midday", "Midnight", "Dusk"],
                    horizontal=True,
                    key="dex_lyc_visual_selector"
                )
                
                # Mapeia a escolha para o nome da API
                if lyc_dex_form == "Midnight":
                    target_visual_name = "lycanroc-midnight"
                elif lyc_dex_form == "Dusk":
                    target_visual_name = "lycanroc-dusk"
                else:
                    target_visual_name = "lycanroc-midday"

            # --- RENDERIZA A IMAGEM PRINCIPAL ---
            # Usa o nome específico da forma para buscar a arte oficial
            final_img_url = get_pokemon_image_url(target_visual_name, api_name_map, mode="artwork", shiny=False)
            st.image(final_img_url, use_container_width=True)
        
            # Nível de Poder (mantido)
            np = row.get("Nivel_Poder", row.get("Nível de Poder", ""))
            if str(np).strip() != "" and str(np).lower() != "nan":
                st.markdown(
                    f"<div class='power-badge'>⚡ Nível de Poder: {np}</div>",
                    unsafe_allow_html=True
                )

        with top_right:
            render_info_columns(info_entries[midpoint:])

        render_status_controls()
        render_info_tags()

        # --- CARROSSEL DE SPRITES (Atualizado para seguir a forma) ---
        st.markdown("#### 🎞️ Variações (Sprites)")
        
        # Gera URLs baseadas no nome da forma (target_visual_name) em vez do número da Dex
        sprite_normal = get_pokemon_image_url(target_visual_name, api_name_map, mode="sprite", shiny=False)
        sprite_shiny  = get_pokemon_image_url(target_visual_name, api_name_map, mode="sprite", shiny=True)
        
        sprites_html = f"""
            <div class='pokedex-carousel'>
                <div style="text-align:center; font-size:10px;">
                    <img src='{sprite_normal}' style='width:70px; image-rendering: pixelated;'><br>Normal
                </div>
                <div style="text-align:center; font-size:10px;">
                    <img src='{sprite_shiny}' style='width:70px; image-rendering: pixelated;'><br>Shiny
                </div>
            </div>
        """
        st.markdown(sprites_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # --- CARROSSEL INFERIOR (navegação) ---
      
        dex_param = st.query_params.get("dex", None)
        if dex_param:
            st.session_state["pokedex_selected"] = str(dex_param)
            st.query_params.clear() 
            st.rerun()
        
        # Monta os itens do carrossel usando strings simples para evitar conflitos de chaves
        items_html_list = []
        for _, r_car in filtered_df.iterrows():
            pid = str(r_car["Nº"])
            sprite = pokemon_pid_to_image(pid, mode="sprite", shiny=False)
            # Define se o item está ativo para aplicar o estilo de borda amarela
            is_active = "carousel-item-active" if pid == str(st.session_state.get("pokedex_selected")) else ""
            
            # Criamos o HTML de cada item individualmente
            item_node = '''
            <div class="carousel-item {ACTIVE_CLASS}" onclick="selectDex('{PID}')">
                <img src="{SPRITE}" alt="{PID}">
            </div>
            '''.replace("{ACTIVE_CLASS}", is_active).replace("{PID}", pid).replace("{SPRITE}", sprite)
            
            items_html_list.append(item_node)
        
        all_items_string = "".join(items_html_list)
        
        # Template HTML/CSS/JS (String tripla sem o 'f' para não quebrar o código) 
        html_template = '''
        <style>
            .pokedex-footer-carousel {
                display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 12px; padding: 14px;
                background: rgba(0, 0, 0, 0.30); border-radius: 15px;
                border: 1px solid rgba(255,255,255,0.18); scroll-behavior: smooth;
            }
            .pokedex-footer-carousel::-webkit-scrollbar { height: 8px; }
            .pokedex-footer-carousel::-webkit-scrollbar-thumb { background: #FFCC00; border-radius: 10px; }
            
            .carousel-item {
                flex: 0 0 auto; width: 70px; height: 70px; border-radius: 12px;
                display: grid; place-items: center; cursor: pointer;
                background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
                transition: transform 0.15s;
            }
            .carousel-item:hover { transform: scale(1.12); border-color: #FFCC00; }
            .carousel-item-active { border: 2px solid #FFCC00; background: rgba(255, 204, 0, 0.10); }
            .carousel-item img { width: 54px; height: 54px; image-rendering: pixelated; }
        </style>
        
        <div class="pokedex-footer-carousel">
            REPLACE_ME
        </div>
        
        <script>
            function selectDex(pid) {
                // window.parent permite que o clique saia do iframe e atualize o site principal 
                const url = new URL(window.parent.location.href);
                url.searchParams.set("dex", pid);
                window.parent.location.assign(url.toString());
            }
        </script>
        '''
        
        # Injeta os itens e renderiza o componente
        final_carousel_html = html_template.replace("REPLACE_ME", all_items_string)
        components.html(final_carousel_html, height=120)
                
               
        

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
            st.markdown("<div class='pokedex-grid-note'>Clique em um Pokémon para ver os detalhes.</div>", unsafe_allow_html=True)

            grid_cols = 6 # Reduzi para 6 para as bordas não ficarem espremidas
            rows = list(filtered_df.iterrows())

            st.markdown("<div class='pokedex-grid'>", unsafe_allow_html=True)
            for start in range(0, len(rows), grid_cols):
                cols = st.columns(grid_cols)
                for col, (index, row_g) in zip(cols, rows[start : start + grid_cols]):
                    dex_num = str(row_g["Nº"])
                    p_name = row_g["Nome"]

                    # 1. Gera o link da imagem
                    sprite_url = pokemon_pid_to_image(dex_num, mode="sprite", shiny=False)

                    # 2. Define o status
                    is_caught = dex_num in user_data.get("caught", [])
                    is_seen = dex_num in user_data.get("seen", [])
                    is_wished = dex_num in user_data.get("wishlist", [])

                    if is_caught:
                        status_class = "dex-frame--caught"
                        icon = "✅"
                    elif is_wished:
                        status_class = "dex-frame--wish"
                        icon = "⭐"
                    elif is_seen:
                        status_class = "dex-frame--seen"
                        icon = "👁️"
                    else:
                        status_class = "dex-frame--default"
                        icon = ""

                    display_name = f"{icon} {p_name}".strip()

                    with col:
                        # 3. Renderiza a MOLDURA + IMAGEM usando HTML puro
                        # Isso garante que a imagem fique DENTRO da borda colorida
                        html_card = (
                            f'<div class="dex-card-frame {status_class}">\n'
                            f'    <img src="{sprite_url}" class="dex-sprite-img" alt="{p_name}">\n'
                            "</div>"
                        )
                        st.markdown(html_card, unsafe_allow_html=True)

                        # 4. Botão de interação fica logo abaixo
                        st.button(
                            display_name,
                            key=f"dex_btn_{dex_num}_{index}",
                            help=f"#{dex_num} • {p_name}",
                            on_click=select_pokedex_entry,
                            args=(dex_num,),
                            use_container_width=True, # Faz o botão alinhar com a moldura
                        )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 2: TRAINER HUB
# ==============================================================================
if page == "Trainer Hub (Meus Pokémons)":
    st.title("🏕️ Central do Treinador")

    # ----------------------------
    # GBA HUB CSS (visual)
    # ----------------------------

    # ----------------------------
    # Init de dados persistidos
    # ----------------------------
    user_data.setdefault("stats", {})          # agora guarda também stgr/int/notes
    user_data.setdefault("wishlist", [])
    user_data.setdefault("shinies", [])
    user_data.setdefault("favorite_moves", {}) # {pid: [move_name,...]}
    user_data.setdefault("forms", {})

    # estados de UI
    st.session_state.setdefault("hub_selected_pid", None)      # abre ficha
    st.session_state.setdefault("hub_context_pid", None)       # menu contexto BOX
    st.session_state.setdefault("hub_context_action", None)    # "move" ou "view"
    st.session_state.setdefault("hub_box_page", 1)             # paginação box

    # ----------------------------
    # Helpers
    # ----------------------------
    def _pid_key(pid: str) -> str:
        return str(pid).strip()

    def _get_pokemon_row(pid: str):
        try:
            hit = df[df["Nº"].astype(str) == str(pid)]
            return hit.iloc[0] if not hit.empty else None
        except Exception:
            return None

    def _get_pokemon_name(pid: str) -> str:
        if str(pid).startswith("EXT:"):
            return str(pid).replace("EXT:", "").strip() or "Visitante"
        row = _get_pokemon_row(pid)
        return str(row["Nome"]) if row is not None and "Nome" in row else f"ID {pid}"

    def _get_pokemon_type(pid: str) -> str:
        if str(pid).startswith("EXT:"):
            return "Visitante"
        row = _get_pokemon_row(pid)
        return str(row["Tipo"]) if row is not None and "Tipo" in row else "—"

    def _get_pokemon_np(pid: str):
        if str(pid).startswith("EXT:"):
            return "—"
        row = _get_pokemon_row(pid)
        return row.get("Nivel_Poder") if row is not None else "—"

    def _get_sprite(pid: str) -> str:
        if str(pid).startswith("EXT:"):
            return get_pokemon_image_url(_get_pokemon_name(pid), api_name_map, mode="sprite", shiny=False)
        
        is_shiny = str(pid) in user_data.get("shinies", [])
        
        # ✅ NOVO: Verifica se tem forma salva
        saved_form = user_data.get("forms", {}).get(str(pid))
        if saved_form:
            # Usa o prefixo EXT: para forçar a busca pelo nome da forma (ex: lycanroc-midnight)
            return pokemon_pid_to_image(f"EXT:{saved_form}", mode="sprite", shiny=is_shiny)
            
        return pokemon_pid_to_image(str(pid), mode="sprite", shiny=is_shiny)

    def _get_artwork(pid: str) -> str:
        if str(pid).startswith("EXT:"):
            return get_pokemon_image_url(_get_pokemon_name(pid), api_name_map, mode="artwork", shiny=False)
            
        is_shiny = str(pid) in user_data.get("shinies", [])
        
        # ✅ NOVO: Verifica se tem forma salva na Artwork também
        saved_form = user_data.get("forms", {}).get(str(pid))
        if saved_form:
            return pokemon_pid_to_image(f"EXT:{saved_form}", mode="artwork", shiny=is_shiny)
            
        return pokemon_pid_to_image(str(pid), mode="artwork", shiny=is_shiny)

    def _ensure_stats_slot(pid: str) -> dict:
        pid = _pid_key(pid)
        user_data["stats"].setdefault(pid, {})
        # campos mínimos (se não existir, cria)
        for k in ["dodge", "parry", "thg", "stgr", "int", "will", "fortitude", "notes"]:
            user_data["stats"][pid].setdefault(k, 0 if k != "notes" else "")
        return user_data["stats"][pid]

    def _select_pid(pid: str):
        st.session_state["hub_selected_pid"] = _pid_key(pid)
        st.session_state["hub_context_pid"] = None
        st.session_state["hub_context_action"] = None

    def _open_box_context(pid: str):
        st.session_state["hub_context_pid"] = _pid_key(pid)
        st.session_state["hub_context_action"] = None

    def _move_box_to_party(pid: str):
        pid = _pid_key(pid)
        if pid in [str(x) for x in user_data.get("party", [])]:
            return
        # limite 8 (visitantes também contam)
        if len(user_data.get("party", [])) >= 8:
            st.warning("Sua equipe ativa já está cheia (máximo 8). Remova alguém antes.")
            return
        user_data.setdefault("party", [])
        user_data["party"].append(pid)
        save_data_cloud(trainer_name, user_data)
        st.success("Movido para a equipe!")
        st.rerun()

    def _remove_from_party(pid: str):
        pid = _pid_key(pid)
        user_data["party"] = [p for p in user_data.get("party", []) if str(p) != pid]
        save_data_cloud(trainer_name, user_data)
        st.rerun()

    # ----------------------------
    # Carrega fichas do Firebase (mapa pid -> sheet)
    # ----------------------------
    sheets_map = {}
    try:
        db, bucket = init_firebase()
        for sh in list_sheets(db, trainer_name) or []:
            p = (sh.get("pokemon") or {})
            pid = p.get("id")
            if pid is None:
                continue
            sheets_map[str(pid)] = sh
    except Exception:
        sheets_map = {}

    # ----------------------------
    # Tabs do Hub
    # ----------------------------
    t_main, t_wish, t_seen = st.tabs(["🎮 Equipe & BOX", "🌟 Lista de Interesses", "👁️ Pokédex (Vistos)"])

    # ==========================
    # TAB PRINCIPAL: BOX + PARTY
    # ==========================
    with t_main:

        # ==========================
        # FICHA (painel superior)
        # ==========================
        sel = st.session_state.get("hub_selected_pid")
        if sel:
            pid = str(sel)
            is_ext = pid.startswith("EXT:")
            pname = _get_pokemon_name(pid)

            st.markdown('<div class="gba-window summary">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="gba-header">
              <div class="left">
                <span class="gba-chip">SUMMARY</span>
                <span class="gba-title" style="font-size:0.75rem;">{pname}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            cA, cB = st.columns([1, 1.6], gap="large")

            # painel esquerdo: imagem + notas + ações rápidas
            with cA:
                # --- LÓGICA DE FORMA DO LYCANROC NO HUB ---
                # Verifica se é Lycanroc para mostrar o seletor
                final_hub_image = _get_artwork(pid) # Imagem padrão (ou shiny se já estiver marcado)
                
                if "lycanroc" in pname.lower().strip():
                    st.caption("Visualizar Forma:")
                    
                    # 1. Determina qual está salvo atualmente (ou padrão Midday)
                    current_saved = user_data.get("forms", {}).get(pid, "lycanroc-midday")
                    
                    # Mapeamento: Nome API <-> Nome Bonito
                    form_map = {
                        "lycanroc-midday": "Midday",
                        "lycanroc-midnight": "Midnight", 
                        "lycanroc-dusk": "Dusk"
                    }
                    reverse_map = {v: k for k, v in form_map.items()}
                    
                    # Define o index do radio baseado no salvo
                    options = list(form_map.values())
                    try:
                        default_idx = options.index(form_map.get(current_saved, "Midday"))
                    except:
                        default_idx = 0

                    lyc_choice = st.radio(
                        "Forma",
                        options,
                        index=default_idx,
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"hub_lyc_selector_{pid}"
                    )
                    
                    # 2. Se mudou, salva e recarrega
                    selected_api_name = reverse_map[lyc_choice]
                    
                    if selected_api_name != current_saved:
                        user_data["forms"][pid] = selected_api_name
                        save_data_cloud(trainer_name, user_data)
                        st.rerun()

                # Renderiza a imagem final (Os helpers _get_artwork agora leem user_data['forms'] automaticamente)
                final_hub_image = _get_artwork(pid) 
                st.image(final_hub_image, use_container_width=True)
                
                # --- Resto do código original da coluna esquerda ---
                stats_slot = _ensure_stats_slot(pid)

                st.markdown('<div class="gba-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### Ações rápidas")
                if st.button("Fechar ficha", key="hub_close_sheet", use_container_width=True):
                    st.session_state["hub_selected_pid"] = None
                    st.rerun()

                st.markdown('<div class="gba-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="gba-notes"><div class="gba-caption">NOTAS</div></div>', unsafe_allow_html=True)

                notes = st.text_area(
                    "Notas",
                    value=str(stats_slot.get("notes", "")),
                    height=140,
                    key=f"hub_notes_{pid}",
                    label_visibility="collapsed",
                )
                if notes != stats_slot.get("notes", ""):
                    stats_slot["notes"] = notes
                    user_data["stats"][pid] = stats_slot
                    save_data_cloud(trainer_name, user_data)

            # painel direito: stats + resumo + golpes
            with cB:
                sheet = sheets_map.get(pid) if (not is_ext) else None

                if sheet is None:
                    st.warning("Este Pokémon não tem ficha salva. Preencha os atributos mínimos para usar no Hub.")
                    stats_slot = _ensure_stats_slot(pid)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        dodge = st.number_input("Dodge", min_value=0, max_value=99, value=int(stats_slot.get("dodge", 0)), key=f"hub_dodge_{pid}")
                        parry = st.number_input("Parry", min_value=0, max_value=99, value=int(stats_slot.get("parry", 0)), key=f"hub_parry_{pid}")
                        thg = st.number_input("Thg", min_value=0, max_value=99, value=int(stats_slot.get("thg", 0)), key=f"hub_thg_{pid}")
                    with c2:
                        stgr = st.number_input("Stgr", min_value=0, max_value=99, value=int(stats_slot.get("stgr", 0)), key=f"hub_stgr_{pid}")
                        intel = st.number_input("Int", min_value=0, max_value=99, value=int(stats_slot.get("int", 0)), key=f"hub_int_{pid}")
                        will = st.number_input("Will", min_value=0, max_value=99, value=int(stats_slot.get("will", 0)), key=f"hub_will_{pid}")
                    with c3:
                        fort = st.number_input("Fortitude", min_value=0, max_value=99, value=int(stats_slot.get("fortitude", 0)), key=f"hub_fort_{pid}")

                    new_stats = {
                        "dodge": int(dodge),
                        "parry": int(parry),
                        "thg": int(thg),
                        "stgr": int(stgr),
                        "int": int(intel),
                        "will": int(will),
                        "fortitude": int(fort),
                        "notes": str(stats_slot.get("notes", "")),
                    }
                    if new_stats != {k: stats_slot.get(k) for k in new_stats.keys()}:
                        user_data["stats"][pid] = new_stats
                        save_data_cloud(trainer_name, user_data)

                    st.info("Sem ficha salva: golpes, advantages e skills não disponíveis aqui ainda.")
                else:
                    stats = sheet.get("stats") or {}
                    moves = sheet.get("moves") or []
                    advantages = sheet.get("advantages") or []
                    skills_raw = sheet.get("skills") or []

                    stats_slot = _ensure_stats_slot(pid)

                    st.markdown(
                        f"""
                        <div class="gba-chips">
                            <span class="gba-chip">Dodge {int(stats.get("dodge", 0))}</span>
                            <span class="gba-chip">Parry {int(stats.get("parry", 0))}</span>
                            <span class="gba-chip">Thg {int(stats.get("thg", 0))}</span>
                            <span class="gba-chip">Fort {int(stats.get("fortitude", 0))}</span>
                            <span class="gba-chip">Will {int(stats.get("will", 0))}</span>
                            <span class="gba-chip">Stgr {int(stats.get("stgr", 0))}</span>
                            <span class="gba-chip">Int {int(stats.get("int", 0))}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                    def _skills_list(raw) -> list[str]:
                        items = []
                        if isinstance(raw, list):
                            for row in raw:
                                if isinstance(row, dict):
                                    name = str(row.get("name", "")).strip()
                                    ranks = int(row.get("ranks", 0) or 0)
                                    if name and ranks > 0:
                                        items.append(f"{name} ({ranks})")
                                elif isinstance(row, str):
                                    name = row.strip()
                                    if name:
                                        items.append(name)
                        elif isinstance(raw, dict):
                            for name, ranks in raw.items():
                                try:
                                    ranks = int(ranks)
                                except Exception:
                                    ranks = 0
                                if name and ranks > 0:
                                    items.append(f"{name} ({ranks})")
                        elif isinstance(raw, str):
                            for line in raw.splitlines():
                                line = line.strip().lstrip("-•").strip()
                                if line:
                                    items.append(line)
                        return items

                    skills_list = _skills_list(skills_raw)

                    adv_html = "<div class='hub-summary-block'><strong>⭐ Advantages</strong>"
                    if advantages:
                        adv_html += "<ul>" + "".join(f"<li>{a}</li>" for a in advantages) + "</ul>"
                    else:
                        adv_html += "<div class='hub-muted'>Sem advantages registradas.</div>"
                    adv_html += "</div>"

                    skills_html = "<div class='hub-summary-block'><strong>🎯 Skills</strong>"
                    if skills_list:
                        skills_html += "<ul>" + "".join(f"<li>{s}</li>" for s in skills_list) + "</ul>"
                    else:
                        skills_html += "<div class='hub-muted'>Sem skills registradas.</div>"
                    skills_html += "</div>"

                    summary_html = f"""
                    <div class="hub-summary-grid">
                        {adv_html}
                        {skills_html}
                    </div>
                    """
                    st.markdown(summary_html, unsafe_allow_html=True)

                    try:
                        mvdb = load_move_db(file_name)
                    except Exception:
                        mvdb = None

                    def _based_stat(move_name: str, move_meta: dict | None = None) -> str:
                        move_meta = move_meta or {}
                        cat_meta = str(move_meta.get("category", "") or "").strip().lower()
                    
                        # 1) Se veio do golpe “criado do zero” (meta)
                        if move_meta.get("is_special") is True:
                            return "Int"
                        if move_meta.get("is_special") is False:
                            return "Stgr"
                    
                        # 2) Pela categoria salva no meta
                        if "status" in cat_meta:
                            return "—"
                        if "especial" in cat_meta or "special" in cat_meta:
                            return "Int"
                        if "físico" in cat_meta or "fisico" in cat_meta or "physical" in cat_meta:
                            return "Stgr"
                    
                        # 3) Fallback: buscar no banco (Excel) se disponível
                        if mvdb is None:
                            return "Stgr"
                        mv = mvdb.get_by_name(move_name)
                        if mv is None:
                            return "Stgr"
                    
                        cat = (mv.categoria or "").strip().lower()
                        if "status" in cat:
                            return "—"
                        if "especial" in cat or "special" in cat:
                            return "Int"
                        if "físico" in cat or "fisico" in cat or "physical" in cat:
                            return "Stgr"
                    
                        return "Stgr"
                    
                    
                    def _final_rank(m: dict) -> tuple[int, str]:
                        base = int(m.get("rank", 0) or 0)
                        bstat = _based_stat(m.get("name", ""), m.get("meta") or {})
                    
                        if bstat == "Stgr":
                            bonus = int(stats.get("stgr", 0) or 0)
                        elif bstat == "Int":
                            bonus = int(stats.get("int", 0) or 0)
                        else:  # "—" (Status)
                            bonus = 0
                    
                        return base + bonus, bstat


                    fav = user_data.get("favorite_moves", {}).get(pid, [])
                    if not isinstance(fav, list):
                        fav = []

                    st.markdown("### ⭐ Golpes Favoritos (até 4)")
                    all_names = [m.get("name", "Golpe") for m in moves]
                    fav = [n for n in fav if n in all_names]

                    shown = 0
                    for m in moves:
                        name = m.get("name", "Golpe")
                        if name in fav and shown < 4:
                            base = int(m.get("rank", 0) or 0)
                            fr, bstat = _final_rank(m)
    
                            if bstat == "—":  # Status
                                st.write(f"**{name}** — Rank base {base} → **{fr}** (Status)")
                            else:             # Physical/Special
                                st.write(f"**{name}** — Rank base {base} + {bstat} → **{fr}**")
                            build_txt = (m.get("build") or "").strip()
                            if build_txt:
                                st.code(build_txt, language="text")
                            shown += 1
                    if shown == 0:
                        st.caption("Nenhum favorito definido.")

                    st.markdown('<div class="gba-divider"></div>', unsafe_allow_html=True)

                    st.markdown("### 📜 Lista completa de golpes")
                    for idx, m in enumerate(moves):
                        name = m.get("name", "Golpe")
                        fr, bstat = _final_rank(m)
                        checked = name in fav
                        c1, c2 = st.columns([0.15, 0.85])
                        with c1:
                            star = st.checkbox("⭐", value=checked, key=f"hub_star_{pid}_{idx}")
                        with c2:
                            base = int(m.get("rank", 0) or 0)

                            if bstat == "—":  # Golpe de Status
                                st.write(f"**{name}** — base {base} → **{fr}** (Status)")
                            else:             # Physical / Special
                                st.write(f"**{name}** — base {base} + {bstat} → **{fr}**")

                            build_txt = (m.get("build") or "").strip()
                            if build_txt:
                                with st.expander("Ingredientes do golpe"):
                                    st.code(build_txt, language="text")
                        if star and name not in fav:
                            fav.append(name)
                        if (not star) and name in fav:
                            fav.remove(name)

                    if len(fav) > 4:
                        fav = fav[:4]
                        st.warning("Favoritos limitados a 4. Mantive os 4 primeiros que você marcou.")

                    if user_data.get("favorite_moves", {}).get(pid, []) != fav:
                        user_data.setdefault("favorite_moves", {})
                        user_data["favorite_moves"][pid] = fav
                        save_data_cloud(trainer_name, user_data)

        

        
        st.markdown('<div class="hub-split">', unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 1])

        # ---------
        # BOX (capturados)
        # ---------
        with col_left:
            # Container principal da BOX
            st.markdown('<div class="grass-box">', unsafe_allow_html=True)
        
            st.markdown("""
            <div class="gba-header">
              <div class="left">
                <span class="gba-chip">BOX</span>
                <span class="gba-title" style="font-size:0.75rem;">Capturados</span>
              </div>
              <div class="right">Pokémons</div>
            </div>
            """, unsafe_allow_html=True)
        
            # ✅ NOVO FILTRO: Identifica quem está na party ativa
            party_ids = [str(p) for p in (user_data.get("party") or [])]

            # ✅ FILTRO ATUALIZADO: Pega capturados, remove quem está na party e ignora EXT
            caught_all = [
                str(c) for c in user_data.get("caught", []) 
                if str(c) not in party_ids
            ]
            # Remove duplicados mantendo a ordem
            caught_all = list(dict.fromkeys(caught_all))

            # Lógica de Paginação (Mantém igual)
            PAGE_SIZE = 36  # Grid 6x6
            total_pages = max(1, ((len(caught_all) + PAGE_SIZE - 1) // PAGE_SIZE)) if caught_all else 1
            page_now = int(st.session_state.get("hub_box_page", 1))
            page_now = max(1, min(total_pages, page_now))
            st.session_state["hub_box_page"] = page_now

            # Controles de Navegação da BOX
            cpg1, cpg2, cpg3 = st.columns([1, 2, 1])
            with cpg1:
                if st.button("⬅️", key="hub_box_prev", disabled=(page_now <= 1)):
                    st.session_state["hub_box_page"] = page_now - 1
                    st.rerun()
            with cpg2:
                st.markdown(f"<div class='hub-sub' style='text-align:center'>Página <b>{page_now}</b> / {total_pages}</div>", unsafe_allow_html=True)
            with cpg3:
                if st.button("➡️", key="hub_box_next", disabled=(page_now >= total_pages)):
                    st.session_state["hub_box_page"] = page_now + 1
                    st.rerun()

            if not caught_all:
                st.info("Você ainda não marcou nenhum Pokémon como capturado.")
            else:
                # Delimitação dos Pokémons da página atual
                start = (page_now - 1) * PAGE_SIZE
                end = start + PAGE_SIZE
                page_ids = caught_all[start:end]

                # Renderização do Grid 6 colunas
                grid_cols = 6
                for r in range(0, len(page_ids), grid_cols):
                    cols = st.columns(grid_cols)
                    for col, pid in zip(cols, page_ids[r : r + grid_cols]):
                        with col:
                            sprite = _get_sprite(pid)
                            name = _get_pokemon_name(pid)
                            
                            # Usamos HTML puro para o Slot para evitar que o Streamlit quebre o layout
                            st.markdown(f'''
                                <div class="box-slot-grass">
                                    <img src="{sprite}" style="width: 55px; image-rendering: pixelated;">
                                </div>
                            ''', unsafe_allow_html=True)
                            
                            if st.button(name, key=f"hub_box_{pid}_{r}", use_container_width=True):
                                _open_box_context(pid)
                                st.rerun()

            # 
            # Menu de contexto (Aparece ao clicar em um Pokémon da BOX)
            ctx_pid = st.session_state.get("hub_context_pid")
            if ctx_pid:
                st.markdown("<div class='hub-divider'></div>", unsafe_allow_html=True)
                pname_ctx = _get_pokemon_name(ctx_pid)
                st.markdown(f"**Menu:** {pname_ctx}")
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("➡️ Mover p/ equipe", key="hub_ctx_move"):
                        _move_box_to_party(ctx_pid)
                with c2:
                    if st.button("📄 Ver ficha", key="hub_ctx_view"):
                        _select_pid(ctx_pid)
                        st.rerun()
                with c3:
                    if st.button("✖️ Fechar", key="hub_ctx_close"):
                        st.session_state["hub_context_pid"] = None
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True) # Fecha grass-box

            # --- 👇 NOVO BLOCO: ADICIONAR EXTERNO ---
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            with st.expander("➕ Adicionar Visitante (Fora da Dex)"):
                st.caption("Adicione um Pokémon que não está na Pokédex do sistema.")
                ext_name = st.text_input("Nome (ex: MissingNo)", key="hub_add_ext")
                
                if st.button("📥 Adicionar à Box"):
                    if ext_name.strip():
                        # Cria ID único com prefixo EXT:
                        new_ext_id = f"EXT:{ext_name.strip()}"
                        
                        # Garante a lista
                        user_data.setdefault("caught", [])
                        
                        if new_ext_id not in user_data["caught"]:
                            user_data["caught"].append(new_ext_id)
                            save_data_cloud(trainer_name, user_data)
                            st.success(f"**{ext_name}** adicionado com sucesso!")
                            st.rerun()
                        else:
                            st.warning("Este Pokémon já está na sua lista.")
            # ----------------------------------------

        # ---------
        # PARTY (equipe ativa)
        # ---------
        # ---------
        # PARTY (equipe ativa)
        # ---------
        with col_right:
            # Container principal da Equipe
            st.markdown('<div class="team-box">', unsafe_allow_html=True)

            # Busca a lista de IDs na party
            party = [str(p) for p in (user_data.get("party") or [])]
            
            if not party:
                st.info("Sua equipe está vazia. Use a BOX para mover um Pokémon.")
            else:
                n = len(party)
        
                # Layout adaptativo conforme a quantidade de Pokémon
                if n <= 2:
                    cols_n = 1
                elif n <= 4:
                    cols_n = 2
                else:
                    cols_n = 2  # 2x4 compacto
        
                # Renderização da grade da equipe
                for r in range(0, n, cols_n):
                    cols = st.columns(cols_n)
                    for col, pid in zip(cols, party[r:r+cols_n]):
                        with col:
                            sprite = _get_sprite(pid)
                            name = _get_pokemon_name(pid)
                            typ = _get_pokemon_type(pid)
                            npv = _get_pokemon_np(pid)
                
                            st.markdown(f'''
                                <div class="gba-party-slot">
                                    <img src="{sprite}" style="width: 85px; image-rendering: pixelated; margin-bottom: 8px;">
                                    <div style="text-align: center; width: 100%;">
                                        <div style="color: white; font-family: 'Press Start 2P'; font-size: 10px; margin-bottom: 5px; text-shadow: 2px 2px 0px #000;">{name}</div>
                                        <div style="color: #ffd166; font-size: 9px; font-weight: bold;">{typ} • NP {npv}</div>
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)
                
                            b1, b2 = st.columns(2)
                            with b1:
                                if st.button("Abrir", key=f"hub_party_open_{pid}", use_container_width=True):
                                    _select_pid(pid)
                                    st.rerun()
                            with b2:
                                if st.button("Remover", key=f"hub_party_rm_{pid}", use_container_width=True):
                                    _remove_from_party(pid)
        
            st.markdown('</div>', unsafe_allow_html=True)  # Fecha team-box
        

    # ==========================
    # TAB: Lista de interesses
    # ==========================
    with t_wish:
        st.subheader("🌟 Lista de Interesses")
        st.caption("Marque aqui o que você quer caçar / o que quer registrar como shiny.")
        colw1, colw2 = st.columns(2)

        with colw1:
            st.markdown("#### ✅ Wishlist")
            wish = user_data.get("wishlist", [])
            new_item = st.text_input("Adicionar Pokémon (nome)", key="hub_wish_add")
            if st.button("Adicionar", key="hub_wish_add_btn"):
                if new_item:
                    wish.append(new_item.strip())
                    user_data["wishlist"] = list(dict.fromkeys(wish))
                    save_data_cloud(trainer_name, user_data)
                    st.rerun()
            if wish:
                for i, it in enumerate(wish):
                    c1, c2 = st.columns([0.8, 0.2])
                    with c1:
                        st.write(f"- {it}")
                    with c2:
                        if st.button("Remover", key=f"hub_wish_rm_{i}"):
                            user_data["wishlist"] = [x for x in wish if x != it]
                            save_data_cloud(trainer_name, user_data)
                            st.rerun()
            else:
                st.info("Sua wishlist está vazia.")

        with colw2:
            st.markdown("#### ✨ Shinies")
            st.caption("IDs (da sua Pokédex) marcados como shiny para imagem.")
            caught_all = [str(c) for c in user_data.get("caught", []) if not str(c).startswith("EXT:")]
            if caught_all:
                shiny_sel = st.multiselect(
                    "Marcar como shiny (capturados)",
                    options=caught_all,
                    default=[str(x) for x in user_data.get("shinies", []) if str(x) in caught_all],
                    key="hub_shiny_multi",
                )
                if set(shiny_sel) != set([str(x) for x in user_data.get("shinies", [])]):
                    user_data["shinies"] = shiny_sel
                    save_data_cloud(trainer_name, user_data)
                    st.rerun()
            else:
                st.info("Você precisa capturar Pokémons para marcar shinies.")

    # ==========================
    # TAB: vistos
    # ==========================
    with t_seen:
        total = len(df) if df is not None else 0
        vistos = len(user_data.get("seen", []))
        st.markdown("### Progresso da Pokédex")
        st.progress(min(vistos / total, 1.0) if total else 0.0)
        st.write(f"**{vistos}** de **{total}** Pokémons registrados.")

    # ==========================
    # CRIAÇÃO DE 
    # ==========================

elif page == "Criação Guiada de Fichas":
    st.title("🧩 Criação Guiada de Fichas")
    
    # CORREÇÃO: Inicializa o last_page se não existir
    if "last_page" not in st.session_state:
        st.session_state["last_page"] = ""

    # Só reseta o menu se estivermos REALMENTE vindo de outra página
    if st.session_state["last_page"] != "Criação Guiada de Fichas":
        if st.session_state.get("cg_force_guided"):
            st.session_state["cg_view"] = "guided"
            st.session_state["cg_force_guided"] = False
        else:
            st.session_state["cg_view"] = "menu"
        
        # IMPORTANTE: Atualiza o last_page para evitar o loop no próximo rerun
        st.session_state["last_page"] = "Criação Guiada de Fichas"

    if "cg_view" not in st.session_state:
        st.session_state["cg_view"] = "menu"

    # menu interno
    if st.session_state["cg_view"] == "menu":
        choice = st.radio(
            "Escolha o que você quer fazer ao abrir a criação de ficha:",
            ["Criação Guiada", "Criação de Golpes"],
            horizontal=True
        )
        if choice == "Criação Guiada":
            st.session_state["cg_view"] = "guided"
            st.rerun()
        else:
            st.session_state["cg_view"] = "moves"
            st.rerun()

    # ==========================
    # A) CRIAÇÃO DE GOLPES (REAL)
    # ==========================
    if st.session_state["cg_view"] == "moves":
        st.subheader("⚔️ Criação de Golpes")

        # ✅ CHAMA O COMPONENTE QUE VOCÊ JÁ CRIOU (move_creator_ui.py)
        return_to_view = st.session_state.get("cg_return_to")
        if return_to_view != "guided":
            return_to_view = None
        render_move_creator(
            excel_path="golpes_pokemon_MM_reescritos.xlsx",
            state_key_prefix="cg_moves_ui",
            return_to_view=return_to_view,
        )
        # botão voltar
        if st.button("⬅️ Voltar para a ficha", key="btn_back_to_sheet"):
            st.session_state["cg_view"] = st.session_state.get("cg_return_to", "menu")
            st.rerun()


    # ==========================
    # B) CRIAÇÃO GUIADA (FICHA) - OTIMIZADA
    # ==========================
    if st.session_state["cg_view"] == "guided":
        st.subheader("🧬 Criação Guiada")
    
        # 1. Inicialização segura: Só executa se o draft não existir
        if "cg_draft" not in st.session_state:
            cg_init()
            
        nome_salvo = st.session_state.get("cg_draft", {}).get("pname", "")
    
        # 2. Input de Nome ÚNICO (Evita DuplicateElementKey)
        pname = st.text_input(
            "Digite o nome do Pokémon (ex: Blastoise)", 
            value=nome_salvo,  # <--- Aqui está a mágica. Se for a 1ª vez, entra ""
            placeholder="Ex: Blastoise", 
            key="cg_pname"
        )
        # Tenta pegar o NP salvo. Se não existir, assume 0.
        np_salvo = st.session_state.get("cg_np", 0)
        # Garante que é um inteiro (caso tenha vindo None ou string por algum erro estranho)
        if np_salvo is None: 
            np_salvo = 0
    
        # 3. Processamento e Criação da Ficha (TUDO deve estar dentro deste IF)
        if pname:
            # ✅ LAYOUT: Sugestões da Pokédex logo abaixo da busca
            if len(pname) >= 2:
                matches = df[df["Nome"].str.lower().str.contains(pname.lower(), na=False)].head(10)
                if not matches.empty:
                    st.caption("Sugestões encontradas na sua Pokédex:")
                    st.write(matches[["Nº", "Nome"]])
                    
            raw_name = pname.strip().lower()
            
            is_nidoran_generic = raw_name in ["nidoran", "nidoran♀", "nidoran♂", "nidoran-f", "nidoran-m"]
            is_lycanroc_generic = raw_name == "lycanroc"

            if is_nidoran_generic:
                choice = st.radio(
                    "Qual Nidoran?",
                    ["Nidoran ♀", "Nidoran ♂"],
                    horizontal=True,
                    key="nidoran_choice"
                )
                poke_query = "nidoran-f" if "♀" in choice else "nidoran-m"
                pname = "Nidoran ♀" if "♀" in choice else "Nidoran ♂"

            elif is_lycanroc_generic:
                lyc_choice = st.radio(
                    "Qual forma do Lycanroc?",
                    ["Midday (Dia)", "Midnight (Noite)", "Dusk (Crepúsculo)"],
                    horizontal=True,
                    key="lycanroc_choice"
                )
                
                if "Noite" in lyc_choice:
                    poke_query = "lycanroc-midnight"
                    pname = "Lycanroc (Midnight)"
                elif "Crepúsculo" in lyc_choice:
                    poke_query = "lycanroc-dusk"
                    pname = "Lycanroc (Dusk)"
                else:
                    poke_query = "lycanroc-midday"
                    pname = "Lycanroc (Midday)"
                    
            else:
                # Usa a função auxiliar apenas aqui
                poke_query = to_pokeapi_name(pname)

            # Atualiza o rascunho com o nome específico da forma
            st.session_state["cg_draft"]["pname"] = pname
    
            # Busca ID no Excel
            row = df[df["Nome"].str.lower() == pname.lower()]
            pid = str(int(row.iloc[0]["Nº"])) if not row.empty else "0"
    
            # Busca dados na API
            with st.spinner("Buscando dados do Pokémon online (stats + ability + tipos)..."):
                pjson = pokeapi_get_pokemon(poke_query)
                base_stats = pokeapi_parse_stats(pjson)
                types = pokeapi_parse_types(pjson)
                abilities = pokeapi_parse_abilities(pjson)

                imported_name = st.session_state.get("cg_imported_name")
                imported_types = (
                    st.session_state.get("cg_imported_types")
                    if imported_name and _norm(imported_name) == _norm(pname)
                    else None
                )
                imported_abilities = (
                    st.session_state.get("cg_imported_abilities")
                    if imported_name and _norm(imported_name) == _norm(pname)
                    else None
                )
                if imported_types:
                    types = list(imported_types)
                if imported_abilities:
                    abilities = sorted(set(abilities + list(imported_abilities)))
                
                # Escolha de Habilidades
                saved_abilities = st.session_state.get("cg_abilities")
                if not isinstance(saved_abilities, list):
                    saved_abilities = None
                chosen_abilities = st.multiselect(
                    "Escolha a(s) habilidade(s) (pode mais de uma):",
                    options=abilities,
                    default=[
                        a
                        for a in (
                            saved_abilities
                            or (imported_abilities if imported_abilities else (abilities[:1] if abilities else []))
                        )
                        if a in abilities
                    ],
                )
                
                if not chosen_abilities:
                    chosen_abilities = abilities
                st.session_state["cg_abilities"] = chosen_abilities
    
            # 3) NP / PP
            np_sugerido = get_np_for_pokemon(df, pid, fallback_np=6)
            # --- LÓGICA DE RECUPERAÇÃO DO NP (CORREÇÃO) ---
            # Tenta pegar do rascunho salvo (draft) e do widget atual
            val_draft = st.session_state.get("cg_draft", {}).get("np", 0)
            val_widget = st.session_state.get("cg_np", 0)
            np_salvo_final = max(int(val_draft or 0), int(val_widget or 0))
            
            np_ = st.number_input(
                "NP do seu Pokémon (o jogador informa)", 
                min_value=0, 
                value=int(np_salvo_final), # <--- Carrega o valor salvo ou 0
                step=1, 
                key="cg_np", 
                on_change=_cg_sync_from_np
            )
            # --- SALVAMENTO FORÇADO ---
            # Salva no rascunho imediatamente para não perder ao trocar de aba
            if "cg_draft" in st.session_state:
                st.session_state["cg_draft"]["np"] = np_
            pp_total = calc_pp_budget(np_)
    
            pp_spent_moves = sum((m.get("pp_cost") or 0) for m in st.session_state.get("cg_moves", []))
            pp_spent_moves += sum(_move_accuracy_pp(m) for m in st.session_state.get("cg_moves", []))
    
            tabs = st.tabs(
                [
                    "1️⃣ Visão Geral",
                    "2️⃣ Abilities e Defesas",
                    "3️⃣ Skills e Advantages",
                    "4️⃣ Golpes",
                    "5️⃣ Revisão e Exportação",
                ]
            )
            
            pp_abilities = 0
            pp_defenses = 0
            pp_skills = 0
            pp_advantages = 0
            pp_moves = pp_spent_moves
    
            # 4) Atributos base
            PL = int(np_)
            cap = 2 * PL
            
            atk = int(base_stats.get("attack", 10))
            spatk = int(base_stats.get("special-attack", 10))
            spe = int(base_stats.get("speed", 10))
            def_ = int(base_stats.get("defense", 10))
            spdef = int(base_stats.get("special-defense", 10))
            
            int_base = max(0, (spatk - 10) // 10)
            stgr_base = max(0, (atk - 10) // 10)
            
            den_td = max(1, def_ + spe)
            thg_base = round((def_ / den_td) * cap)
            dodge_base = cap - thg_base
            
            den_wf = max(1, spdef + def_)
            will_base = round((spdef / den_wf) * cap)
            fort_base = cap - will_base
    
            with tabs[0]:
                st.markdown(
                    f"""
                    <div class="cg-card">
                        <div class="cg-title">Visão Geral</div>
                        <span class="cg-pill">NP {np_}</span>
                        <span class="cg-pill">PP Total {pp_total}</span>
                        <span class="cg-pill">Tipos: {', '.join(types)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Pokémon:** {pname}  \n"
                    f"**Abilities escolhidas:** {', '.join(chosen_abilities)}"
                )
                st.info("Use as abas para preencher cada etapa. O total de PP gastos é somado automaticamente no final.")

                # --------------------------
                # Preset por Viabilidade (Arquétipos)
                # --------------------------
                viab_text = ""
                try:
                    if isinstance(row, pd.DataFrame) and (not row.empty) and ("Viabilidade" in row.columns):
                        viab_text = str(row.iloc[0].get("Viabilidade", "") or "").strip()
                except Exception:
                    viab_text = ""

                if viab_text and viab_text.strip() and viab_text.strip().lower() not in {"nan", "sem dados.", "sem dados"}:
                    with st.expander("📌 Arquétipo de Viabilidade (1 clique)", expanded=True):
                        st.caption("Isso vira um *preset* para facilitar a criação: sugere **golpes + habilidade** e você ajusta depois. Não altera suas fórmulas nem regras de PP.")
                        archetypes = _viab_parse_archetypes(viab_text)
                        if not archetypes:
                            st.write(viab_text)
                        else:
                            labels = [a.get("label", "Arquétipo") for a in archetypes]
                            sel_idx = st.selectbox(
                                "Escolha um arquétipo (da sua Pokédex):",
                                options=list(range(len(labels))),
                                format_func=lambda i: labels[int(i)],
                                key="cg_viab_sel_idx",
                            )
                            arch = archetypes[int(sel_idx)]

                            st.markdown(f"**Estratégia:** `{arch.get('code') or '—'}`")
                            if arch.get("ability"):
                                st.markdown(f"**Habilidade sugerida:** `{arch['ability']}`")
                            if arch.get("partners"):
                                st.markdown("**Parceiros sugeridos:** " + ", ".join(arch["partners"]))

                            st.text_area(
                                "Texto da Viabilidade (referência)",
                                value=str(arch.get("raw", "") or ""),
                                height=120,
                                key="cg_viab_raw_view",
                                disabled=True,
                            )

                            rank_default = st.slider(
                                "Rank padrão para adicionar golpes desse arquétipo",
                                1, 20,
                                min(10, max(1, int(np_ or 1))),
                                key="cg_viab_rank_default",
                            )

                            excel_path_viab = _resolve_asset_path("golpes_pokemon_MM_reescritos.xlsx")

                            colA, colB = st.columns([1, 1])
                            with colA:
                                core_n = 3
                                b1, b2 = st.columns(2)
                                with b1:
                                    apply_full = st.button(
                                        "✅ Adicionar kit completo",
                                        key="cg_viab_apply_full_btn",
                                        use_container_width=True,
                                    )
                                with b2:
                                    apply_core = st.button(
                                        f"⚡ Só núcleo ({core_n} golpes)",
                                        key="cg_viab_apply_core_btn",
                                        use_container_width=True,
                                    )

                                mode = "full" if apply_full else ("core" if apply_core else None)
                                if mode:
                                    added, missing, abil = _viab_apply_to_session(
                                        arch,
                                        int(rank_default),
                                        excel_path_viab,
                                        mode=mode,
                                        core_n=core_n,
                                    )

                                    # tenta setar habilidade sugerida (se ela existir no Pokémon)
                                    if abil:
                                        abil_norm = _norm(abil)
                                        matched_abil = None
                                        for a in (abilities or []):
                                            if _norm(a) == abil_norm:
                                                matched_abil = a
                                                break
                                        if matched_abil:
                                            cur = st.session_state.get("cg_abilities") or []
                                            if matched_abil not in cur:
                                                cur = [matched_abil] + cur
                                            st.session_state["cg_abilities"] = cur

                                    # guarda para revisão
                                    st.session_state["cg_viab_selected"] = arch
                                    st.session_state["cg_viab_mode"] = (
                                        "Kit completo" if mode == "full" else f"Núcleo ({core_n})"
                                    )

                                    msg = f"Golpes adicionados: {len(added)}"
                                    if added:
                                        msg += " — " + ", ".join(added[:8]) + ("…" if len(added) > 8 else "")
                                    st.success(msg)

                                    if missing:
                                        st.warning(
                                            "Não encontrei no banco: "
                                            + ", ".join(missing[:12])
                                            + ("…" if len(missing) > 12 else "")
                                            + ". Você pode criar esses golpes na aba **Criação de Golpes**."
                                        )
                                    st.rerun()
                            with colB:
                                st.caption("Depois, vá na aba **4️⃣ Golpes** para ajustar Rank/PP/Acerto em **Lista & ajustes**.")

    
            with tabs[1]:
                st.markdown("### 📊 Atributos (auto + editável)")
                cap = 2 * int(st.session_state.get("cg_np", 0) or 0)
                _cg_init_defenses_if_missing(dodge_base, fort_base)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    stgr = st.number_input(
                        "Stgr (Força)",
                        value=int(st.session_state.get("cg_stgr", stgr_base)),
                        min_value=0, max_value=99, key="cg_stgr",
                    )
                    intellect = st.number_input(
                        "Int (Intelecto)",
                        value=int(st.session_state.get("cg_int", int_base)),
                        min_value=0, max_value=99, key="cg_int",
                    )
                
                with col2:
                    dodge_value = int(st.session_state.get("cg_dodge", dodge_base))
                    dodge_min = min(dodge_value, max(0, int(dodge_base) - 2))
                    dodge_max = max(dodge_value, min(99, int(dodge_base) + 2))
                    dodge = st.number_input(
                        "Dodge", key="cg_dodge",
                        value=dodge_value, min_value=dodge_min, max_value=dodge_max,
                        on_change=_cg_sync_from_dodge,
                    )
                    parry = st.number_input(
                        "Parry", key="cg_parry",
                        value=int(st.session_state.get("cg_parry", st.session_state.get("cg_dodge", dodge_base))),
                        min_value=0, max_value=99, disabled=True,
                    )
            
                with col3:
                    thg = st.number_input(
                        "Thg (Toughness)", key="cg_thg",
                        value=int(st.session_state.get("cg_thg", max(0, cap - int(st.session_state.get("cg_dodge", dodge_base))))),
                        min_value=0, max_value=99, disabled=True,
                    )
                    fortitude = st.number_input(
                        "Fortitude", key="cg_fortitude",
                        value=int(st.session_state.get("cg_fortitude", fort_base)),
                        min_value=min(int(st.session_state.get("cg_fortitude", fort_base)), max(0, int(fort_base) - 2)),
                        max_value=max(int(st.session_state.get("cg_fortitude", fort_base)), min(99, int(fort_base) + 2)),
                        on_change=_cg_sync_from_fortitude,
                    )
                    will = st.number_input(
                        "Will", key="cg_will",
                        value=int(st.session_state.get("cg_will", max(0, cap - int(st.session_state.get("cg_fortitude", fort_base))))),
                        min_value=0, max_value=99, disabled=True,
                    )
    
                st.markdown("### 💰 PP automático")
                pp_stgr = int(stgr) * 2
                pp_int  = int(intellect) * 2
                pp_dodge = int(dodge) * 1
                pp_parry = int(parry) * 1
                pp_thg   = int(thg) * 1
                pp_will  = int(will) * 1
                pp_fort  = int(fortitude) * 1
                
                pp_abilities_auto = pp_stgr + pp_int
                pp_defenses_auto  = pp_dodge + pp_parry + pp_thg + pp_will + pp_fort
                
                st.write(f"**Abilities:** {pp_abilities_auto} PP (STGR {pp_stgr} + INT {pp_int})")
                st.write(f"**Defesas:** {pp_defenses_auto} PP (Dodge {pp_dodge} + Parry {pp_parry} + Thg {pp_thg} + Will {pp_will} + Fort {pp_fort})")
                
                pp_abilities = pp_abilities_auto
                pp_defenses  = pp_defenses_auto
    
            with tabs[2]:
                st.markdown("### 🧠 Skills (M&M 3e)")
                if "cg_skills" not in st.session_state:
                    st.session_state["cg_skills"] = {k: 0 for k in SKILLS_MM3}
                    st.session_state["cg_skill_custom"] = []
    
                cols_sk = st.columns(3)
                total_skill_ranks = 0
                for i, sk in enumerate(SKILLS_MM3):
                    with cols_sk[i % 3]:
                        v_sk = st.number_input(sk, min_value=0, max_value=40, value=int(st.session_state["cg_skills"].get(sk, 0)), step=1, key=f"cg_skill_{sk}")
                        st.session_state["cg_skills"][sk] = int(v_sk)
                        total_skill_ranks += int(v_sk)
    
                st.divider()
                st.markdown("### Skills extras")
                add_name = st.text_input("Nome da skill extra", key="cg_skill_add_name")
                if st.button("➕ Adicionar skill extra", key="cg_skill_add_btn"):
                    if add_name.strip():
                        st.session_state["cg_skill_custom"].append({"name": add_name.strip(), "ranks": 0})
    
                for idx, row_sk in enumerate(list(st.session_state["cg_skill_custom"])):
                    c1, c2, c3 = st.columns([6, 2, 2])
                    with c1: st.write(row_sk["name"])
                    with c2:
                        rv = st.number_input("Ranks", min_value=0, max_value=40, value=int(row_sk.get("ranks", 0)), key=f"cg_skill_custom_rank_{idx}")
                        row_sk["ranks"] = int(rv)
                        total_skill_ranks += int(rv)
                    with c3:
                        if st.button("❌", key=f"cg_skill_custom_del_{idx}"):
                            st.session_state["cg_skill_custom"].pop(idx)
                            st.rerun()
    
                pp_skills = total_skill_ranks / 2
                st.info(f"Total de ranks: **{total_skill_ranks}** → PP em Skills: **{pp_skills}**")
    
                st.markdown("### ⭐ Advantages (sugestões)")
                st.session_state.setdefault("cg_advantages_custom", [])
                adv_suggestions = suggest_advantages(pjson=pjson, base_stats=base_stats, types=types, abilities=abilities)
                if not adv_suggestions:
                    st.info("Nenhuma vantagem sugerida automaticamente.")

                st.markdown("### Advantages manuais")
                adv_manual_name = st.text_input("Digite a advantage manual", key="cg_advantage_add_name")
                if st.button("➕ Adicionar advantage manual", key="cg_advantage_add_btn"):
                    adv_manual_name = adv_manual_name.strip()
                    if adv_manual_name and adv_manual_name not in st.session_state["cg_advantages_custom"]:
                        st.session_state["cg_advantages_custom"].append(adv_manual_name)

                for idx, adv_name in enumerate(list(st.session_state["cg_advantages_custom"])):
                    c1, c2 = st.columns([8, 2])
                    with c1:
                        st.write(adv_name)
                    with c2:
                        if st.button("❌", key=f"cg_advantage_custom_del_{idx}"):
                            st.session_state["cg_advantages_custom"].pop(idx)
                            st.rerun()

                labels = [a.label() for a in adv_suggestions]
                notes_map = {a.label(): (a.note or "") for a in adv_suggestions}
                saved_adv = st.session_state.get("cg_advantages") or []
                all_advantages = sorted(set(labels + list(saved_adv) + list(st.session_state["cg_advantages_custom"])))
                chosen_labels = st.multiselect(
                    "Selecione advantages:",
                    options=all_advantages,
                    default=[lab for lab in saved_adv if lab in all_advantages],
                )
                chosen_adv = chosen_labels
                for lab in chosen_labels:
                    if notes_map.get(lab):
                        st.caption(f"• {lab}: {notes_map[lab]}")
                st.session_state["cg_advantages"] = chosen_adv
    
                pp_advantages = len(chosen_adv)
                st.info(f"Advantages escolhidas: **{pp_advantages} PP**")
    
            with tabs[3]:
                st.markdown("### ⚔️ Golpes & Poderes (fluxo guiado)")
                st.caption(
                    "Adicione golpes prontos, monte um poder **passo a passo** (sem precisar ler o texto do M&M) "
                    "ou use o criador completo. Suas fórmulas e regras de PP não mudam — apenas organizamos o processo."
                )

                excel_path = _resolve_asset_path("golpes_pokemon_MM_reescritos.xlsx")
                try:
                    db_moves_guided = load_move_db(excel_path)
                except Exception as e:
                    st.error(f"Não consegui carregar o Excel de golpes: {e}")
                    db_moves_guided = None

                # garante lista de golpes na sessão
                st.session_state.setdefault("cg_moves", st.session_state.get("cg_draft", {}).get("moves", []))
                if "cg_draft" in st.session_state:
                    st.session_state["cg_draft"]["moves"] = st.session_state["cg_moves"]

                # orçamento (mantém sua regra: PP_total + 20 para golpes)
                pp_cap_moves = int(pp_total) + 20
                acerto_pp_total = sum(_move_accuracy_pp(m) for m in st.session_state.get("cg_moves", []))
                pp_spent_moves_live = sum((m.get("pp_cost") or 0) for m in st.session_state.get("cg_moves", [])) + acerto_pp_total

                cA, cB, cC = st.columns(3)
                cA.metric("PP gastos (Golpes + Acerto)", int(pp_spent_moves_live))
                cB.metric("Limite para Golpes", pp_cap_moves)
                cC.metric("Slots de golpes", len(st.session_state.get("cg_moves", [])))
                st.progress(min(1.0, float(pp_spent_moves_live) / float(max(1, pp_cap_moves))))

                if pp_spent_moves_live > pp_cap_moves:
                    st.error("Você estourou o limite de PP para golpes (PP_total + 20). Remova golpes, reduza ranks ou acerto.")

                sub_tabs = st.tabs([
                    "⚡ Adicionar rápido",
                    "🧪 Assistente (passo a passo)",
                    "🛠️ Criador completo (avançado)",
                    "📦 Lista & ajustes",
                ])

                # --------------------------
                # (A) Adicionar rápido
                # --------------------------
                with sub_tabs[0]:
                    st.markdown("#### 1) Sugestões pelo Pokémon (Bulbapedia)")
                    st.caption("Lista de golpes que o Pokémon pode aprender. Eu comparo com o banco e deixo 1 clique para adicionar.")

                    rank_default = st.slider(
                        "Rank padrão para adicionar",
                        1, 20,
                        min(10, max(1, int(np_ or 1))),
                        key="cg_quick_rank_default"
                    )

                    if db_moves_guided is not None and isinstance(pjson, dict):
                        api_moves = _pokeapi_parse_move_names(pjson)
                        matched = []
                        for nm in api_moves[:120]:  # limita para não ficar pesado
                            mv = _try_match_move_in_db(db_moves_guided, nm)
                            if mv:
                                matched.append(mv)

                        # remove duplicados por nome
                        seen = set()
                        uniq = []
                        for mv in matched:
                            k = _norm(mv.name)
                            if k not in seen:
                                seen.add(k)
                                uniq.append(mv)

                        if uniq:
                            # tenta priorizar STAB (mesmo tipo do Pokémon)
                            stab = [m for m in uniq if _norm(getattr(m, "tipo", "")) in {_norm(t) for t in (types or [])}]
                            other = [m for m in uniq if m not in stab]
                            uniq = stab + other

                            default_pick = [m.name for m in uniq[:8]]
                            pick = st.multiselect(
                                "Selecione e adicione (sugestões)",
                                options=[m.name for m in uniq[:60]],
                                default=default_pick[:4],
                                key="cg_quick_pick"
                            )

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("➕ Adicionar selecionados", key="cg_quick_add_selected"):
                                    existing = {_norm(m.get("name", "")) for m in st.session_state.get("cg_moves", [])}
                                    added = 0
                                    for nm in pick:
                                        mv = db_moves_guided.get_by_name(nm)
                                        if not mv:
                                            continue
                                        if _norm(mv.name) in existing:
                                            continue

                                        pp_auto = None
                                        try:
                                            tmp = mv.pp_cost(int(rank_default))
                                            if isinstance(tmp, tuple):
                                                pp_auto = tmp[0]
                                        except Exception:
                                            pp_auto = None

                                        st.session_state["cg_moves"].append(
                                            _cg_confirm_move(
                                                mv,
                                                int(rank_default),
                                                pp_override=(int(pp_auto) if pp_auto is not None else None),
                                                accuracy=_default_accuracy_from_raw(mv),
                                            )
                                        )
                                        added += 1

                                    st.success(f"Adicionados: {added} golpe(s).")
                                    st.rerun()
                            with col2:
                                st.caption("Dica: se algum golpe ficar sem PP ou com Rank muito alto, preencha em **Lista & ajustes**.")
                        else:
                            st.info("Não encontrei golpes do learnset no banco atual. Use a busca abaixo ou o assistente.")

                    st.divider()
                    st.markdown("#### 2) Buscar no banco (rápido)")
                    q = st.text_input("Digite parte do nome (ex.: thunder, punch, protect)", key="cg_quick_search")
                    if db_moves_guided is None:
                        st.stop()

                    if q and len(q.strip()) >= 2:
                        results = db_moves_guided.search_by_name_prefix(q.strip())[:20]
                        if not results:
                            sugg = db_moves_guided.suggest_by_description(q.strip(), top_k=8)
                            results = [mv for (mv, _s) in sugg]

                        if results:
                            existing = {_norm(m.get("name", "")) for m in st.session_state.get("cg_moves", [])}
                            for mv in results:
                                with st.container(border=True):
                                    st.write(f"**{mv.name}** — {getattr(mv,'tipo','—')} / {getattr(mv,'categoria','—')}")
                                    bullets = _summarize_build(getattr(mv, "build", ""))
                                    if bullets:
                                        st.caption(" • ".join(bullets))

                                    pp_auto = None
                                    try:
                                        tmp = mv.pp_cost(int(rank_default))
                                        if isinstance(tmp, tuple):
                                            pp_auto = tmp[0]
                                    except Exception:
                                        pp_auto = None

                                    c1, c2, c3 = st.columns([2, 2, 2])
                                    with c1:
                                        st.caption(f"Rank: {rank_default}")
                                    with c2:
                                        if pp_auto is not None:
                                            st.caption(f"PP sugerido: **{int(pp_auto)}**")
                                        else:
                                            st.caption("PP: **manual**")
                                    with c3:
                                        key_base = f"{_norm(mv.name)}_{rank_default}"
                                        if _norm(mv.name) in existing:
                                            st.button("✅ Já na ficha", disabled=True, key=f"cg_quick_already_{key_base}")
                                        else:
                                            if pp_auto is None:
                                                pp_manual = st.number_input("PP", min_value=1, value=1, step=1, key=f"cg_quick_pp_{key_base}")
                                                if st.button("➕ Adicionar", key=f"cg_quick_add_{key_base}"):
                                                    st.session_state["cg_moves"].append(_cg_confirm_move(mv, int(rank_default), pp_override=int(pp_manual)))
                                                    st.rerun()
                                            else:
                                                if st.button("➕ Adicionar", key=f"cg_quick_add_{key_base}"):
                                                    st.session_state["cg_moves"].append(_cg_confirm_move(mv, int(rank_default), pp_override=int(pp_auto)))
                                                    st.rerun()
                        else:
                            st.info("Nada encontrado. Tente outra palavra ou use o Assistente para montar o efeito.")
                    else:
                        st.caption("Informe pelo menos 2 caracteres para buscar.")

                # --------------------------
                                # (B) Assistente (passo a passo)
                # --------------------------
                with sub_tabs[1]:
                    st.caption(
                        "Crie um poder customizado com orientação. "
                        "Use o modo **Rápido** para builds simples, ou o modo **Universal** para "
                        "recriar **qualquer golpe existente** (copiando/colando a build ou carregando do banco)."
                    )
                
                    mode = st.radio(
                        "Modo do assistente",
                        ["Rápido (perguntas)", "Universal (recriar qualquer golpe)"],
                        horizontal=True,
                        key="cg_pa_mode",
                    )
                
                    # ==========================
                    # MODO RÁPIDO (atual)
                    # ==========================
                    if mode.startswith("Rápido"):
                        if interpret_effects_to_build is None:
                            st.error("O interpretador de efeitos não carregou. Verifique o arquivo move_interpreter.py.")
                        else:
                            with st.form("cg_power_assistant_form", clear_on_submit=False):
                                name = st.text_input("Nome do golpe/poder", key="cg_pa_name")
                                cat = st.radio("Categoria", ["Físico (Stgr)", "Especial (Int)", "Status"], horizontal=True, key="cg_pa_cat")
                                is_special = bool(cat.startswith("Especial"))
                                if cat.startswith("Status"):
                                    is_special = False
                
                                rank = st.number_input("Rank do golpe", min_value=0, max_value=30, value=int(rank_default), step=1, key="cg_pa_rank")
                
                                st.markdown("**Qual é o efeito principal?**")
                                cE1, cE2, cE3 = st.columns(3)
                                with cE1:
                                    e_damage = st.checkbox("Dano", key="cg_pa_damage")
                                    e_aff = st.checkbox("Status (Affliction)", key="cg_pa_aff")
                                with cE2:
                                    e_weaken = st.checkbox("Debuff (Weaken)", key="cg_pa_weaken")
                                    e_heal = st.checkbox("Cura (Healing)", key="cg_pa_heal")
                                with cE3:
                                    e_create = st.checkbox("Criação/Barreira (Create)", key="cg_pa_create")
                                    e_env = st.checkbox("Ambiente/Clima (Environment)", key="cg_pa_env")
                
                                weaken_targets = {"weaken_stgr": False, "weaken_int": False, "weaken_dodge": False, "weaken_will": False}
                                if e_weaken:
                                    st.markdown("**Weaken afeta o quê?**")
                                    w1, w2, w3, w4 = st.columns(4)
                                    weaken_targets["weaken_stgr"] = w1.checkbox("Stgr", key="cg_pa_w_stgr")
                                    weaken_targets["weaken_int"] = w2.checkbox("Int", key="cg_pa_w_int")
                                    weaken_targets["weaken_dodge"] = w3.checkbox("Dodge", key="cg_pa_w_dodge")
                                    weaken_targets["weaken_will"] = w4.checkbox("Will", key="cg_pa_w_will")
                
                                st.markdown("**Como ele atinge o alvo?**")
                                cA, cB, cC = st.columns(3)
                                with cA:
                                    ranged = st.checkbox("Ranged", key="cg_pa_ranged")
                                with cB:
                                    perception = st.checkbox("Perception", key="cg_pa_perception")
                                with cC:
                                    area = st.selectbox("Área (opcional)", ["—", "Burst", "Cone", "Line", "Cloud"], index=0, key="cg_pa_area")
                                    area_val = None if area == "—" else area
                
                                submit = st.form_submit_button("🧪 Gerar build")
                
                            if submit:
                                effects = {
                                    "damage": bool(e_damage),
                                    "affliction": bool(e_aff),
                                    "weaken": bool(e_weaken),
                                    "healing": bool(e_heal),
                                    "create": bool(e_create),
                                    "environment": bool(e_env),
                                }
                                effects.update(weaken_targets)
                
                                build = interpret_effects_to_build(
                                    rank=int(rank),
                                    is_special=bool(is_special),
                                    effects=effects,
                                    area=area_val,
                                    perception=bool(perception),
                                    ranged=bool(ranged),
                                )
                
                                tmp_mv = Move(
                                    name=(name.strip() or "Poder Personalizado"),
                                    tipo=(types[0] if types else "—"),
                                    categoria=("Status" if cat.startswith("Status") else ("Especial" if is_special else "Físico")),
                                    descricao="Criado via Assistente (Rápido)",
                                    build=build,
                                    how_it_works="",
                                    resist_stat="",
                                    ranged=bool(ranged),
                                    perception_area=bool(perception),
                                    tags=[],
                                    raw={},
                                )
                                pp_auto, why = tmp_mv.pp_cost(int(rank))
                
                                st.markdown("#### Resultado")
                                st.code(build, language="text")
                                if pp_auto is not None:
                                    st.info(f"PP sugerido: **{int(pp_auto)}** — {why}")
                                else:
                                    st.warning(f"PP não definido automaticamente — {why}. Informe manualmente.")
                
                                pp_final = int(st.number_input("PP total do golpe", min_value=1, value=int(pp_auto) if pp_auto else 1, step=1, key="cg_pa_pp_final"))
                
                                if st.button("➕ Adicionar este poder na ficha", key="cg_pa_add"):
                                    st.session_state["cg_moves"].append({
                                        "name": tmp_mv.name,
                                        "rank": int(rank),
                                        "build": build,
                                        "pp_cost": int(pp_final),
                                        "accuracy": 0,
                                        "meta": {
                                            "custom": True,
                                            "ranged": bool(ranged),
                                            "perception_area": bool(perception),
                                            "category": tmp_mv.categoria,
                                        },
                                    })
                                    st.success("Golpe adicionado! Vá em **Lista & ajustes** para revisar acerto/PP.")
                                    st.rerun()
                
                    # ==========================
                    # MODO UNIVERSAL
                    # ==========================
                    else:
                        st.markdown("### 🧩 Universal — recriar qualquer golpe do sistema")
                        st.caption(
                            "Você pode **carregar um golpe existente** do banco e editar a build, "
                            "ou **colar uma build** e trabalhar em cima dela. "
                            "Esse modo é **texto fiel**: se você colar a build do Excel, você consegue recriar 100%."
                        )
                
                        # ---- Fonte do template
                        src_kind = st.radio(
                            "Começar de:",
                            ["Golpe existente (modelo)", "Colar build", "Em branco"],
                            horizontal=True,
                            key="cg_u_src_kind",
                        )
                
                        # Estado base
                        if "cg_u_loaded_raw" not in st.session_state:
                            st.session_state["cg_u_loaded_raw"] = None
                
                        if src_kind.startswith("Golpe existente"):
                            q = st.text_input("Buscar modelo pelo nome (prefixo)", key="cg_u_q")
                            hits = db.search_by_name_prefix(q) if q else []
                            if hits:
                                opt = st.selectbox("Selecione um golpe", [m.name for m in hits[:50]], key="cg_u_pick")
                                if st.button("⬇️ Carregar modelo", key="cg_u_load_model", type="primary"):
                                    mv = db.get_by_name(opt)
                                    if mv:
                                        st.session_state["cg_u_loaded_raw"] = getattr(mv, "raw", {})
                                        st.session_state["cg_u_name"] = mv.name
                                        # tenta mapear categoria -> radio
                                        cat_map = {
                                            "fisico": "Físico (Stgr)",
                                            "físico": "Físico (Stgr)",
                                            "especial": "Especial (Int)",
                                            "status": "Status",
                                        }
                                        st.session_state["cg_u_cat"] = cat_map.get((mv.categoria or "").strip().lower(), "Status")
                                        st.session_state["cg_u_build"] = (mv.build or "").strip()
                                        st.session_state["cg_u_build_pending"] = st.session_state.get("cg_u_build", "")
                                        st.success("Modelo carregado. Você pode editar a build abaixo.")
                                        st.rerun()
                            else:
                                st.info("Digite o início do nome para listar modelos.")
                
                        elif src_kind.startswith("Colar"):
                            pasted = st.text_area("Cole a build aqui", height=140, key="cg_u_paste")
                            if st.button("⬇️ Carregar build colada", key="cg_u_load_paste", type="primary"):
                                st.session_state["cg_u_loaded_raw"] = None
                                st.session_state["cg_u_build"] = (pasted or "").strip()
                                st.session_state["cg_u_build_pending"] = st.session_state.get("cg_u_build", "")
                                st.success("Build carregada no editor.")
                                st.rerun()
                
                        else:
                            st.caption("Em branco: você vai montar a build a partir dos atalhos abaixo.")
                
                        st.markdown("---")
                        cL, cR = st.columns([1.15, 0.85])
                
                        with cL:
                            name_u = st.text_input("Nome do golpe/poder", key="cg_u_name")
                            cat_u = st.radio("Categoria", ["Físico (Stgr)", "Especial (Int)", "Status"], horizontal=True, key="cg_u_cat")
                            is_special_u = bool(str(cat_u).startswith("Especial"))
                            if str(cat_u).startswith("Status"):
                                is_special_u = False
                            rank_u = st.number_input("Rank do golpe", min_value=0, max_value=30, value=int(rank_default), step=1, key="cg_u_rank")
                
                            # --- sync de build (evita erro de session_state com key de widget) ---

                
                            if "cg_u_build_pending" in st.session_state:

                
                                st.session_state["cg_u_build_widget"] = st.session_state.pop("cg_u_build_pending")

                
                            elif "cg_u_build_widget" not in st.session_state:

                
                                st.session_state["cg_u_build_widget"] = (st.session_state.get("cg_u_build") or "")

                
                            

                
                            build_u = st.text_area(
                                "Build M&M (adaptado) — você pode editar livremente",
                                height=220,
                                key="cg_u_build_widget",
                                placeholder="Ex.: Damage (Rank = ) [Ranged] [Extra: Increased Range 1]; Linked Affliction 1 (...) ...",
                            )

                            st.session_state["cg_u_build"] = (build_u or "").strip()
                
                        with cR:
                            st.markdown('**Atalhos (opcionais):**')
                            st.caption('Eles só **inserem texto** na build (não mudam suas fórmulas).')
                            linked = st.checkbox("Prefixar com 'Linked' no próximo efeito-base", value=False, key='cg_u_next_linked')
                            base = st.selectbox('Inserir efeito-base', ['Damage','Affliction','Weaken','Healing','Nullify','Create','Environment','Teleport','Dazzle','Concealment','Deflect','Immunity','Enhanced Trait','Feature','Movement'], index=0, key='cg_u_base')
                            rank_b = st.number_input('Rank do efeito-base', min_value=0, max_value=30, value=int(st.session_state.get('cg_u_rank', 1)), step=1, key='cg_u_base_rank')
                            area = st.selectbox('Área (opcional)', ['—','Burst','Cone','Line','Cloud','Perception Area','Shapeable','Selective'], index=0, key='cg_u_area')
                            extras_txt = st.text_input('Extras (texto, opcional)', key='cg_u_extras_txt', placeholder='Ex.: [Extra: Increased Range 1] [Extra: Improved Critical 1]')
                            flaws_txt = st.text_input('Flaws/Limites (texto, opcional)', key='cg_u_flaws_txt', placeholder='Ex.: [Flaw: Limited - Objects] [Flaw: Distracting]')
                            if st.button('➕ Inserir efeito na build', key='cg_u_insert_base'):
                                piece = f"{base} {int(rank_b)}"
                                if area != '—':
                                    piece += f" [Area: {area}]"
                                if extras_txt.strip():
                                    piece += ' ' + extras_txt.strip()
                                if flaws_txt.strip():
                                    piece += ' ' + flaws_txt.strip()
                                if linked:
                                    piece = 'Linked ' + piece
                                cur = (st.session_state.get('cg_u_build') or '').strip()
                                if cur:
                                    if not cur.endswith(';'):
                                        cur += ';'
                                    cur += ' ' + piece
                                else:
                                    cur = piece
                                st.session_state['cg_u_build'] = cur
                                st.session_state['cg_u_build_pending'] = cur
                                st.rerun()
                            st.markdown('**Inserir modificadores comuns:**')
                            m1, m2 = st.columns(2)
                            if m1.button('[Ranged]', key='cg_u_m_ranged'):
                                cur = (st.session_state.get('cg_u_build') or '').rstrip()
                                cur = (cur + ' [Ranged]').strip()
                                st.session_state['cg_u_build'] = cur
                                st.session_state['cg_u_build_pending'] = cur
                                st.rerun()
                            if m1.button('[Perception]', key='cg_u_m_perception'):
                                cur = (st.session_state.get('cg_u_build') or '').rstrip()
                                cur = (cur + ' [Perception]').strip()
                                st.session_state['cg_u_build'] = cur
                                st.session_state['cg_u_build_pending'] = cur
                                st.rerun()
                            if m1.button('[Extra: Increased Range 1]', key='cg_u_m_inc_range'):
                                cur = (st.session_state.get('cg_u_build') or '').rstrip()
                                cur = (cur + ' [Extra: Increased Range 1]').strip()
                                st.session_state['cg_u_build'] = cur
                                st.session_state['cg_u_build_pending'] = cur
                                st.rerun()
                            if m2.button('[Extra: Improved Critical 1]', key='cg_u_m_imp_crit'):
                                cur = (st.session_state.get('cg_u_build') or '').rstrip()
                                cur = (cur + ' [Extra: Improved Critical 1]').strip()
                                st.session_state['cg_u_build'] = cur
                                st.session_state['cg_u_build_pending'] = cur
                                st.rerun()
                            if m2.button('[Flaw: Limited - Objects]', key='cg_u_m_lim_obj'):
                                cur = (st.session_state.get('cg_u_build') or '').rstrip()
                                cur = (cur + ' [Flaw: Limited - Objects]').strip()
                                st.session_state['cg_u_build'] = cur
                                st.session_state['cg_u_build_pending'] = cur
                                st.rerun()
                            if m2.button('[Affect Others]', key='cg_u_m_affect_others'):
                                cur = (st.session_state.get('cg_u_build') or '').rstrip()
                                cur = (cur + ' [Affect Others]').strip()
                                st.session_state['cg_u_build'] = cur
                                st.session_state['cg_u_build_pending'] = cur
                                st.rerun()
                            st.markdown('**Ferramentas rápidas:**')
                            t1, t2 = st.columns(2)
                            if t1.button('🧹 Limpar build', key='cg_u_clear'):
                                st.session_state['cg_u_build'] = ''
                                st.session_state['cg_u_build_pending'] = ''
                                st.rerun()
                            if t2.button('↩️ Desfazer última peça', key='cg_u_undo'):
                                cur = (st.session_state.get('cg_u_build') or '').strip()
                                if ';' in cur:
                                    cur = cur.rsplit(';', 1)[0].strip()
                                else:
                                    cur = ''
                                st.session_state['cg_u_build'] = cur
                                st.session_state['cg_u_build_pending'] = cur
                                st.rerun()
                        st.markdown("---")
                        st.markdown("#### Resultado")
                        final_build = (st.session_state.get("cg_u_build") or "").strip()
                        if not final_build:
                            st.info("Sem build ainda. Carregue um modelo, cole uma build, ou use os atalhos.")
                        else:
                            st.code(final_build, language="text")
                
                            # PP: se veio do Excel (raw), usa o mesmo pp_cost do Move (sem alterar regra).
                            raw = st.session_state.get("cg_u_loaded_raw") or {}
                            tmp_mv = Move(
                                name=(name_u.strip() or "Poder Personalizado"),
                                tipo=(types[0] if types else "—"),
                                categoria=("Status" if str(cat_u).startswith("Status") else ("Especial" if is_special_u else "Físico")),
                                descricao="Criado via Assistente (Universal)",
                                build=final_build,
                                how_it_works="",
                                resist_stat=_safe_str(raw.get("Resist Stat") or ""),
                                ranged=("[Ranged]" in final_build) or (" ranged" in final_build.lower()),
                                perception_area=("Perception Area" in final_build) or ("perception area" in final_build.lower()),
                                tags=[],
                                raw=raw,
                            )
                            pp_auto, why = tmp_mv.pp_cost(int(rank_u))
                            if pp_auto is not None:
                                st.info(f"PP sugerido: **{int(pp_auto)}** — {why}")
                            else:
                                st.warning(f"PP não definido automaticamente — {why}. Informe manualmente.")
                
                            pp_final = int(st.number_input("PP total do golpe", min_value=1, value=int(pp_auto) if pp_auto else 1, step=1, key="cg_u_pp_final"))
                
                            if st.button("➕ Adicionar este poder na ficha", key="cg_u_add", type="primary"):
                                st.session_state["cg_moves"].append({
                                    "name": tmp_mv.name,
                                    "rank": int(rank_u),
                                    "build": final_build,
                                    "pp_cost": int(pp_final),
                                    "accuracy": 0,
                                    "meta": {
                                        "custom": True,
                                        "ranged": bool(tmp_mv.ranged),
                                        "perception_area": bool(tmp_mv.perception_area),
                                        "category": tmp_mv.categoria,
                                    },
                                })
                                st.success("Golpe adicionado! Vá em **Lista & ajustes** para revisar acerto/PP.")
                                st.rerun()
# (C) Criador completo (sua tela existente)
                # --------------------------
                with sub_tabs[2]:
                    st.caption("Modo avançado: use a tela completa de criação e edição de golpes.")
                    disabled_add = pp_spent_moves_live >= pp_cap_moves
                    if disabled_add:
                        st.error("Limite atingido (PP_total + 20).")

                    show_full = st.checkbox("Abrir o criador completo aqui (avançado)", value=False, key="cg_show_full_creator_inline")
                    if show_full:
                        render_move_creator(
                            excel_path=_resolve_asset_path("golpes_pokemon_MM_reescritos.xlsx"),
                            state_key_prefix="cg_moves_inline",
                            return_to_view=None,
                        )
                    else:
                        st.info("Marque a caixa acima para abrir o criador completo nesta aba. Se preferir, abra pelo menu **Criação de Golpes**.")

                    if st.button("Abrir pelo menu Criação de Golpes", key="cg_go_creator_menu"):
                        st.session_state["cg_view"] = "moves"
                        st.session_state["cg_return_to"] = "guided"
                        st.rerun()

                # --------------------------
                # (D) Lista & ajustes
                # --------------------------
                with sub_tabs[3]:
                    st.markdown("#### 📦 Golpes confirmados nesta ficha")
                    if st.session_state.get("cg_moves"):
                        stats_now = st.session_state.get("cg_draft", {}).get("stats", {})
                        np_value = int(st.session_state.get("cg_np", 0) or 0)

                        for i, m_gv in enumerate(list(st.session_state["cg_moves"]), start=1):
                            c1, c2, c3 = st.columns([6, 2, 1], vertical_alignment="center")
                            with c1:
                                accuracy = int(m_gv.get("accuracy", 0) or 0)
                                pp_here = m_gv.get("pp_cost")
                                st.write(f"{i}. **{m_gv.get('name','Golpe')}** (Rank {m_gv.get('rank','—')}) — PP: {pp_here} | Acerto {accuracy}")
                                if m_gv.get("build"):
                                    bullets = _summarize_build(m_gv.get("build", ""))
                                    if bullets:
                                        st.caption(" • ".join(bullets[:6]))
                                    with st.expander("Ingredientes do golpe", expanded=False):
                                        st.code(m_gv["build"], language="text")

                            with c2:
                                # editar Rank do golpe
                                cur_rank = int(m_gv.get("rank", 1) or 1)
                                new_rank = st.number_input(
                                    "Rank",
                                    min_value=1,
                                    max_value=50,
                                    value=int(cur_rank),
                                    step=1,
                                    key=f"cg_guided_move_rank_{i}",
                                )
                                if st.button("Definir rank", key=f"cg_guided_move_set_rank_{i}"):
                                    m_gv["rank"] = int(new_rank)
                                    pp_recalc, _why = _cg_recalculate_pp(m_gv, int(new_rank), db_moves_guided)
                                    if pp_recalc is not None:
                                        m_gv["pp_cost"] = int(pp_recalc)
                                    if db_moves_guided:
                                        try:
                                            mv_db = db_moves_guided.get_by_name(str(m_gv.get("name") or ""))
                                        except Exception:
                                            mv_db = None
                                        if mv_db:
                                            try:
                                                m_gv["build"] = mv_db.render_build(int(new_rank))
                                            except Exception:
                                                pass
                                    st.rerun()

                                acc_limit = _move_accuracy_limit(m_gv, np_value, stats_now)
                                current_acc = int(m_gv.get("accuracy", 0) or 0)
                                safe_max = max(int(acc_limit), int(current_acc))
                                new_acc = st.number_input(
                                    "Acerto",
                                    min_value=0,
                                    max_value=int(safe_max),
                                    value=int(current_acc),
                                    step=1,
                                    key=f"cg_guided_move_acc_{i}",
                                )
                                st.caption(f"Limite sugerido: {acc_limit}")
                                if st.button("Definir acerto", key=f"cg_guided_move_set_{i}"):
                                    m_gv["accuracy"] = int(new_acc)
                                    st.rerun()

                                # se o golpe está sem PP, pede preenchimento aqui
                                if m_gv.get("pp_cost") is None:
                                    pp_fix = st.number_input("PP do golpe", min_value=1, value=1, step=1, key=f"cg_fix_pp_{i}")
                                    if st.button("Definir PP", key=f"cg_fix_pp_btn_{i}"):
                                        m_gv["pp_cost"] = int(pp_fix)
                                        st.rerun()

                            with c3:
                                if st.button("🗑️ Remover", key=f"cg_guided_move_rm_{i}", use_container_width=True):
                                    st.session_state["cg_moves"].pop(i - 1)
                                    st.rerun()
                    else:
                        st.info("Nenhum golpe confirmado ainda. Use as abas acima para adicionar e ajustar.")

                    # Totais
                    acerto_pp_total = sum(_move_accuracy_pp(m) for m in st.session_state.get("cg_moves", []))
                    pp_spent_moves_live = sum((m.get("pp_cost") or 0) for m in st.session_state.get("cg_moves", [])) + acerto_pp_total
                    st.info(f"PP de Acerto (soma): {acerto_pp_total}")
                    st.info(f"PP gastos em Golpes (incluindo acerto): **{int(pp_spent_moves_live)}** / **{pp_cap_moves}**")

                # atualiza para a aba de revisão
                pp_moves = pp_spent_moves_live
            with tabs[4]:
                st.markdown("### 🧾 Revisão de PP")
                # Arquétipo aplicado (opcional)
                arch = st.session_state.get("cg_viab_selected")
                if isinstance(arch, dict):
                    st.markdown("### 🧩 Arquétipo aplicado (Viabilidade)")
                    st.write(f"**Estratégia:** {arch.get('code') or '—'}")
                    if arch.get("ability"):
                        st.write(f"**Habilidade sugerida:** {arch.get('ability')}")
                    if arch.get("partners"):
                        st.write("**Parceiros sugeridos:** " + ", ".join(arch.get("partners") or []))
                
                pp_spent_total = float(pp_abilities) + float(pp_defenses) + float(pp_skills) + float(pp_advantages) + float(pp_moves)
                st.markdown(f"""
                    <div class="cg-card">
                        <div class="cg-title">Resumo de PP</div>
                        <div>Abilities: {pp_abilities} | Defesas: {pp_defenses} | Skills: {pp_skills} | Adv: {pp_advantages} | Golpes: {pp_moves}</div>
                        <hr/><strong>Total gasto:</strong> {pp_spent_total} / {pp_total}
                    </div>
                """, unsafe_allow_html=True)
    
                if pp_spent_total > pp_total: st.warning("PP total ultrapassado.")
                else: st.success("PP total dentro do limite. ✅")
    
                pdf_bytes = build_sheet_pdf(
                    pname=pname, np_=np_, types=types, abilities=chosen_abilities,
                    stats={"stgr": int(stgr), "intellect": int(intellect), "dodge": int(dodge), "parry": int(parry), "fortitude": int(fortitude), "will": int(will)},
                    chosen_adv=chosen_adv, moves=st.session_state.get("cg_moves", [])
                )
    
                st.download_button("⬇️ Exportar PDF", data=pdf_bytes, file_name=f"ficha_{pname}_{np_}.pdf", mime="application/pdf")
    
                if st.button("☁️ Salvar ficha na Nuvem", key="btn_save_sheet_cloud"):
                    db_fs, bkt_fs = init_firebase()
                    skills_payload = []
                    for sk_n, sk_r in st.session_state.get("cg_skills", {}).items():
                        if int(sk_r) > 0: skills_payload.append({"name": sk_n, "ranks": int(sk_r)})
                    for row_sk in st.session_state.get("cg_skill_custom", []):
                        if row_sk.get("name") and int(row_sk.get("ranks", 0)) > 0:
                            skills_payload.append({"name": row_sk["name"], "ranks": int(row_sk["ranks"])})
    
                    payload_fs = {
                        "pokemon": {"id": int(pid), "name": pname, "types": types, "abilities": chosen_abilities},
                        "np": int(np_), "pp_budget_total": int(pp_total), "pp_spent_total": float(pp_spent_total),
                        "stats": {"stgr": int(stgr), "int": int(intellect), "dodge": int(dodge), "parry": int(parry), "fortitude": int(fortitude), "will": int(will)},
                        "advantages": chosen_adv, "skills": skills_payload, "moves": st.session_state.get("cg_moves", [])
                    }
                    sid_fs, _ = save_sheet_with_pdf(db=db_fs, bucket=bkt_fs, trainer_name=trainer_name, sheet_payload=payload_fs, pdf_bytes=pdf_bytes, sheet_id=st.session_state.get("cg_edit_sheet_id"))
                    st.success(f"✅ Salva! ID: {sid_fs}")
                    st.session_state["cg_edit_sheet_id"] = None
    
    
            if st.button("⬅️ Voltar"):
                st.session_state["cg_view"] = "menu"
                st.rerun()
    
        else:
            # Caso não tenha nome digitado
            st.info("💡 Digite o nome do Pokémon acima para começar a gerar a ficha.")
            if st.button("⬅️ Voltar ao Menu", key="btn_back_empty"):
                st.session_state["cg_view"] = "menu"
                st.rerun()


# =================
# MINHAS FICHAS
# =================
elif page == "Minhas Fichas":
    st.title("📚 Minhas Fichas")
    st.caption("Veja e gerencie as fichas salvas na nuvem.")

    with st.expander("📥 Importar ficha via PDF", expanded=False):
        st.caption("Envie um PDF exportado aqui para recriar a ficha automaticamente.")
        pdf_upload = st.file_uploader("Selecionar PDF da ficha", type=["pdf"], key="sheet_pdf_import")
        if pdf_upload and st.button("📥 Importar PDF", key="sheet_pdf_import_btn"):
            try:
                imported_sheet = parse_sheet_pdf(pdf_upload.getvalue())
                apply_imported_sheet_to_session(imported_sheet)
                st.session_state["cg_force_guided"] = True
                st.session_state["nav_to"] = "Criação Guiada de Fichas"
                st.success("Ficha importada! Abrindo na criação guiada...")
                st.rerun()
            except Exception as exc:
                st.error(f"Não consegui importar o PDF: {exc}")

    db, bucket = init_firebase()
    sheets = list_sheets(db, trainer_name)

    if not sheets:
        st.info("Você ainda não tem fichas salvas.")
    else:
        for sheet in sheets:
            pokemon = sheet.get("pokemon", {})
            pname = pokemon.get("name", "Pokémon")
            pid = pokemon.get("id", "—")
            np_ = sheet.get("np", "—")
            updated_at = sheet.get("updated_at", "—")
            created_at = sheet.get("created_at", "—")
            sheet_id = sheet.get("_sheet_id")
            pdf_meta = sheet.get("pdf") or {}
            storage_path = pdf_meta.get("storage_path")

            with st.expander(f"🧾 {pname} (ID {pid}) — NP {np_}"):
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("✏️ Editar ficha", key=f"edit_sheet_{sheet_id}"):
                        apply_sheet_to_session(sheet, sheet_id=sheet_id)
                        st.session_state["cg_force_guided"] = True
                        st.session_state["nav_to"] = "Criação Guiada de Fichas"
                        st.rerun()
                with action_col2:
                    if st.button("🗑️ Excluir ficha", key=f"delete_sheet_{sheet_id}"):
                        delete_sheet(db, bucket, trainer_name, sheet_id, storage_path=storage_path)
                        st.success("Ficha excluída com sucesso.")
                        st.rerun()
                st.write(f"**Atualizada em:** {updated_at}")
                st.write(f"**Criada em:** {created_at}")
                st.write(f"**PP Total:** {sheet.get('pp_budget_total', '—')}")

                moves = sheet.get("moves") or []
                if moves:
                    st.markdown("**Golpes:**")
                    for i, m in enumerate(moves, start=1):
                        st.write(f"{i}. {m.get('name', 'Golpe')} (Rank {m.get('rank', '—')})")
                else:
                    st.caption("Sem golpes registrados.")


                if storage_path:
                    if st.button("📄 Baixar PDF", key=f"download_pdf_{sheet.get('_sheet_id')}"):
                        blob = bucket.blob(storage_path)
                        pdf_bytes = blob.download_as_bytes()
                        st.download_button(
                            "Clique para baixar o PDF",
                            data=pdf_bytes,
                            file_name=f"ficha_{pname}_{pid}.pdf",
                            mime="application/pdf",
                            key=f"download_pdf_btn_{sheet.get('_sheet_id')}",
                        )


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
    battle_sheets_map = {}
    if is_player:
        try:
            for sh in list_sheets(db, trainer_name) or []:
                p = (sh.get("pokemon") or {})
                pid = p.get("id")
                if pid is None:
                    continue
                battle_sheets_map[str(pid)] = sh
        except Exception:
            battle_sheets_map = {}



# =========================
    # VIEW: BATTLE (CÓDIGO CONSOLIDADO E CORRIGIDO)
    # =========================
    if view == "battle":
        sync_watchdog(db, rid)
        if not rid or not room:
            st.session_state["pvp_view"] = "lobby"
            st.rerun()
        
        # --- AQUI: INICIA O SISTEMA DE SYNC AUTOMÁTICO ---
        # Isso cria a thread que fica "dormindo" até o Firebase avisar de uma mudança.
        # -------------------------------------------------

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
                saved_form = user_data.get("forms", {}).get(str(pid))
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
                        "form": saved_form, # ✅ ENVIANDO PARA O BANCO
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
            saved_form = p_data.get("form", None)  # ✅ LER DO BANCO
            
            if t_name == trainer_name:
                # CORREÇÃO: Converte para string, remove espaços e trata vazio como '0'
                # Além disso, ignora a chave 'notes' que é texto puro.
                stats_is_bad = not stats or all(
                    int(str(v).strip() or 0) == 0 
                    for k, v in stats.items() if k != "notes"
                )
                
                if stats_is_bad:
                    if "stats" in user_data:
                        local_s = user_data["stats"].get(str(p_id)) or user_data["stats"].get(p_id)
                        if local_s:
                            stats = local_s
            
            return hp, cond, stats, shiny_status, saved_form
        
        def get_poke_display_name(pid):
            row = df[df['Nº'].astype(str) == str(pid)]
            if not row.empty: return row.iloc[0]['Nome']
            return str(pid)

        # Definição da Função de Renderização da Coluna (DEFINIDA ANTES DE USAR)
        # Definição da Função de Renderização da Coluna (MELHORADA)
        def render_player_column(p_name, p_label, is_me):
            st.markdown(f"### {p_label}")
            
            # Busca party e estado público
            p_doc_data = db.collection("rooms").document(rid).collection("public_state").document("players").get().to_dict() or {}
            party_list = p_doc_data.get(p_name, [])[:8] 
            
            # Variáveis de Estado de Ação
            moving_piece_id = st.session_state.get("moving_piece_id")
            placing_pid = st.session_state.get("placing_pid")
            
            state = get_state(db, rid)
            all_pieces = state.get("pieces") or []
            seen_pids = state.get("seen") or []
            
            if not party_list:
                st.caption("Aguardando...")
                return
        
            p_pieces_on_board = [p for p in all_pieces if p.get("owner") == p_name]
        
            for i, pid in enumerate(party_list):
                # 1. Agora recuperamos 5 valores (incluindo p_form)
                cur_hp, cur_cond, cur_stats, is_shiny, p_form = get_poke_data(p_name, pid)
                
                is_on_map = any(str(p["pid"]) == str(pid) for p in p_pieces_on_board)
                p_obj = next((p for p in p_pieces_on_board if str(p["pid"]) == str(pid)), None)
                already_seen = str(pid) in seen_pids
                
                # Ícone de HP
                if cur_hp >= 5: hpi = "💚"
                elif cur_hp >= 3: hpi = "🟡"
                elif cur_hp >= 1: hpi = "🔴"
                else: hpi = "💀"
                
                # 2. Lógica da imagem com prioridade para a Forma
                if p_form:
                     sprite_url = pokemon_pid_to_image(f"EXT:{p_form}", mode="sprite", shiny=is_shiny)
                else:
                     sprite_url = pokemon_pid_to_image(pid, mode="sprite", shiny=is_shiny)
        
                # Checa se ESTE Pokémon específico está realizando uma ação
                is_moving_this = (p_obj and moving_piece_id == p_obj.get("id"))
                is_placing_this = (placing_pid == pid)

                # Estilo da Borda: Amarelo se movendo, Azul se colocando, Padrão caso contrário
                border_color = "#FFCC00" if is_moving_this else ("#38bdf8" if is_placing_this else None)
                
                # Container Visual
                with st.container(border=True):
                    if is_me:
                        # Se estiver movendo ou colocando, destaca visualmente
                        if is_moving_this or is_placing_this:
                             st.markdown(f"""
                                <div style="
                                    background: rgba(0,0,0,0.3); 
                                    border: 2px solid {border_color}; 
                                    border-radius: 8px; 
                                    padding: 5px; 
                                    text-align: center; 
                                    margin-bottom: 5px;">
                                    <div style="font-weight:bold; color:{border_color}; font-size:12px;">
                                        {'📍 SELECIONE O DESTINO NO MAPA' if is_moving_this else '📍 CLIQUE NO MAPA PARA POSICIONAR'}
                                    </div>
                                </div>
                             """, unsafe_allow_html=True)

                        c_img, c_ctrl = st.columns([1, 2.5])
                        
                        with c_img:
                            # Imagem (Cinza se fainted)
                            if cur_hp == 0:
                                st.markdown(f'<img src="{sprite_url}" style="width:100%; filter:grayscale(100%); opacity:0.6;">', unsafe_allow_html=True)
                                st.caption("**FAINTED**")
                            else:
                                st.image(sprite_url, width="stretch")
        
                            # Botões de Controle da Peça (Abaixo da imagem)
                            if is_on_map and p_obj:
                                is_rev = p_obj.get("revealed", True)
                                c_vis, c_del = st.columns(2)
                                with c_vis:
                                    if st.button("👁️" if is_rev else "✅", key=f"v_{p_name}_{pid}_{i}", help="Revelar/Esconder"):
                                        p_obj["revealed"] = not is_rev
                                        upsert_piece(db, rid, p_obj)
                                        if p_obj["revealed"]: mark_pid_seen(db, rid, pid)
                                        st.rerun()
                                with c_del:
                                    if st.button("❌", key=f"r_{p_name}_{pid}_{i}", help="Remover do Mapa"):
                                        delete_piece(db, rid, p_obj["id"])
                                        add_public_event(db, rid, "pokemon_removed", p_name, {"pid": pid})
                                        st.session_state["moving_piece_id"] = None # Reseta move se deletar
                                        st.rerun()

                        with c_ctrl:
                            # --- LÓGICA DE INTERFACE DE AÇÃO ---
                            if is_moving_this:
                                st.info("Clique em um quadrado vazio.")
                                if st.button("🔙 Cancelar Mover", key=f"cncl_move_{pid}"):
                                    st.session_state["moving_piece_id"] = None
                                    st.rerun()
                            
                            elif is_placing_this:
                                st.info("Clique onde quer invocar.")
                                if st.button("🔙 Cancelar", key=f"cncl_place_{pid}"):
                                    st.session_state["placing_pid"] = None
                                    st.rerun()
                                    
                            else:
                                # Interface Padrão (HP e Status)
                                st.markdown(f"**{hpi} HP: {cur_hp}/6**")
                                
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

                                # Botões de Ação Principal
                                if cur_hp > 0:
                                    # Bloqueia botões se outra ação estiver ocorrendo
                                    is_busy = (moving_piece_id is not None) or (placing_pid is not None)
                                    
                                    if is_on_map:
                                        if st.button("🚶 Mover", key=f"m_{p_name}_{pid}_{i}", disabled=is_busy, use_container_width=True):
                                            st.session_state["moving_piece_id"] = p_obj["id"]
                                            st.rerun()
                                    else:
                                        if st.button("📍 Colocar no Campo", key=f"p_{p_name}_{pid}_{i}", disabled=is_busy, use_container_width=True):
                                            st.session_state["placing_pid"] = pid
                                            st.session_state["placing_effect"] = None
                                            st.rerun()
        
                    
                    else:
                        # ==========================
                        # VISÃO DO OPONENTE (Corrigida)
                        # ==========================
                        piece_obj = next((p for p in p_pieces_on_board if str(p["pid"]) == str(pid)), None)
                        is_revealed = piece_obj.get("revealed", True) if piece_obj else False
                        
                        # Se estiver no campo e revelado, OU se já foi visto antes (está na bench mas conhecido)
                        show_full = (piece_obj and is_revealed) or already_seen
                        
                        status_txt = "(Mochila)" if not piece_obj else ("(Escondido)" if not is_revealed else "")
        
                        if show_full:
                            p_real_name = get_poke_display_name(pid)
                            c1, c2 = st.columns([1, 2])
                            
                            with c1: 
                                # ✅ CORREÇÃO 1: Imagem em escala de cinza se HP for 0
                                # Antes usava st.image direto, que sempre é colorido.
                                if cur_hp == 0:
                                    st.markdown(f'<img src="{sprite_url}" style="width:50px; filter:grayscale(100%); opacity:0.6;">', unsafe_allow_html=True)
                                else:
                                    st.image(sprite_url, width=50)
        
                            with c2:
                                st.markdown(f"**{p_real_name}**")
                                st.caption(f"{hpi} HP: {cur_hp}/6 {status_txt}")
                                
                                # ✅ CORREÇÃO 2: Exibir ícones de status para o oponente
                                if cur_cond:
                                    # Junta os ícones em uma string (ex: "🔥 ☠️")
                                    cond_str = " ".join(cur_cond)
                                    st.markdown(f"Status: {cond_str}")
        
                                if cur_hp == 0: 
                                    st.caption("**FAINTED**")
                        else:
                            # Pokémon desconhecido / escondido
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
            # 1. Captura a forma (5ª variável) do banco de dados
            hp_check, _, _, _, p_form = get_poke_data(p.get("owner"), p.get("pid"))
            
            p["status"] = "fainted" if hp_check == 0 else "active"
            
            # ✅ O TRUQUE: Se tiver forma salva, substituímos o PID temporariamente
            # Isso força o renderizador do mapa a baixar o sprite da forma correta (ex: EXT:lycanroc-midnight)
            if p_form:
                p["pid"] = f"EXT:{p_form}"

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
        last_events = list_public_events(db, rid, limit=1)
        last_dice = next((e for e in last_events if e.get("type") == "dice"), None)

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
            if last_dice:
                pl = last_dice.get("payload", {})
                dice_line = f"🎲 {last_dice.get('by')}: <strong>{pl.get('result')}</strong> (d{pl.get('sides')})"
                dice_bg = "rgba(34, 197, 94, 0.18)"
                dice_border = "rgba(34, 197, 94, 0.5)"
            else:
                dice_line = "🎲 Aguardando rolagem..."
                dice_bg = "rgba(148, 163, 184, 0.15)"
                dice_border = "rgba(148, 163, 184, 0.35)"
            st.markdown(
                f"""
                <div style="
                    min-height: 42px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 8px;
                    padding: 6px 10px;
                    border-radius: 10px;
                    border: 1px solid {dice_border};
                    background: {dice_bg};
                    font-weight: 600;">
                    {dice_line}
                </div>
                """,
                unsafe_allow_html=True,
            )
            # ✅ PEDIDO: Mostrar Código da Sala aqui em cima
            st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 10px;'>
        <span style='font-family: "Press Start 2P"; font-size: 18px;'>🏟️ Arena:</span>
        <span style='font-family: "Press Start 2P"; font-size: 24px; color: #FFCC00; background: #333; padding: 5px 10px; border-radius: 5px;'>{rid}</span>
    </div>
    """, unsafe_allow_html=True) 
            
            col_me, col_map, col_opps = st.columns([1.5, 3, 2])
           

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
                        battle_ref.set({"status": "setup", "attacker": trainer_name, "attack_move": None, "logs": []})
                        st.rerun()
                else:
                    st.caption("Aguardando combate...")
            
            # [FASE 1] CONFIGURAR ATAQUE
            elif b_data["status"] == "setup":
                # CORREÇÃO: Parêntese fechado corretamente aqui
                st.caption(f"**Atacante:** {b_data.get('attacker')}")
                
                if b_data.get("attacker") == trainer_name:
                    attacker_pid = None
                    attacker_sheet = None
                    attacker_stats = {}
                    if current_party:
                        attacker_pid = st.selectbox(
                            "Seu Pokémon (Atacante)",
                            options=current_party,
                            format_func=get_poke_display_name,
                            key=f"atk_self_{rid}",
                        )
                        attacker_sheet = battle_sheets_map.get(str(attacker_pid))
                        if attacker_sheet:
                            attacker_stats = attacker_sheet.get("stats") or {}
                        else:
                            _, _, attacker_stats, _, _ = get_poke_data(trainer_name, attacker_pid)
                    else:
                        st.info("Sua party está vazia para selecionar o atacante.")
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
                        

                        move_payload = None
                        selected_accuracy = None
                        selected_damage = None
                        if attacker_sheet:
                            moves = attacker_sheet.get("moves") or []
                        else:
                            moves = []

                        if moves:
                            def _move_label(idx: int) -> str:
                                mv = moves[idx]
                                name = mv.get("name", "Golpe")
                                rank = int(mv.get("rank", 0) or 0)
                                based, stat_val = _move_stat_value(mv.get("meta") or {}, attacker_stats)
                                damage = rank + stat_val
                                acc = int(mv.get("accuracy", 0) or 0)
                                return f"{name}. Acerto: {acc}. Dano {damage}"

                            move_options = ["manual"] + list(range(len(moves)))
                            move_choice = st.selectbox(
                                "Golpe (opcional)",
                                options=move_options,
                                format_func=lambda x: "Manual (sem golpe)" if x == "manual" else _move_label(x),
                                key=f"atk_move_{rid}",
                            )

                            if move_choice != "manual":
                                selected_move = moves[int(move_choice)]
                                selected_accuracy = int(selected_move.get("accuracy", 0) or 0)
                                rank = int(selected_move.get("rank", 0) or 0)
                                based, stat_val = _move_stat_value(selected_move.get("meta") or {}, attacker_stats)
                                selected_damage = rank + stat_val
                                move_payload = {
                                    "name": selected_move.get("name", "Golpe"),
                                    "accuracy": selected_accuracy,
                                    "damage": selected_damage,
                                    "rank": rank,
                                    "based_stat": based,
                                    "stat_value": stat_val,
                                }
                        else:
                            st.caption("Sem golpes disponíveis para este Pokémon.")

                        ac1, ac2 = st.columns([2, 1])
                        with ac1:
                            atk_mod = st.number_input("Acerto (Modificador)", value=0, step=1)
                        with ac2:
                            if selected_accuracy is not None:
                                st.markdown(f"**Acerto sugerido:** {selected_accuracy}")
                            else:
                                st.caption("Sem acerto sugerido.")
                        
                        if st.button("⚔️ Rolar Ataque"):
                            if target_id:
                                d20 = random.randint(1, 20)
                                t_p = next((p for p in all_pieces if p['id'] == target_id), None)
                                
                                # Pega stats do alvo
                                _, _, t_stats, _, _ = get_poke_data(t_p['owner'], t_p['pid'])
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
                                    "attack_move": move_payload,
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
                        _, _, t_stats, _, _ = get_poke_data(trainer_name, b_data.get('target_pid'))
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
                    move_payload = b_data.get("attack_move") or {}
                    suggested_damage = move_payload.get("damage")

                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        dmg_input = st.number_input("Dano Base / Rank", min_value=0, value=0)
                    with c2:
                        if suggested_damage is not None:
                            st.markdown(f"**Dano sugerido:** {suggested_damage}")
                        else:
                            st.caption("Sem dano sugerido.")
                    with c3:
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
                        _, _, t_stats, _, _ = get_poke_data(trainer_name, b_data.get('target_pid'))
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
        # 7. LAYOUT DAS COLUNAS (EQUILIBRADO PARA 4 JOGADORES)
        # =========================
        if not tiles:
            st.warning("Sem mapa.")
            st.stop()

        # Proporção [1.8, 5, 1.8] ou [2, 6, 2] é o "ponto doce"
        # Mantém o mapa grande (~60-65% da tela) mas dá fôlego para as laterais
        c_me, c_map, c_opps = st.columns([1.8, 5, 1.8])

        with c_me:
            render_player_column(trainer_name, "🎒 Sua Equipe", is_me=True)

        with c_map:
            st.markdown(f"### 🗺️ Arena (Sala {rid})")
            
            # --- ALERTA VISUAL DE AÇÃO NO MAPA ---
            #if st.session_state.get("moving_piece_id"):
            #    st.warning("🏃 MODO MOVIMENTO: Clique em um quadrado vazio para mover o Pokémon.", icon="📍")
            #elif st.session_state.get("placing_pid"):
            #    st.info("📍 MODO POSICIONAMENTO: Clique no mapa para colocar o Pokémon.", icon="⬇️")
            #elif st.session_state.get("placing_effect"):
            #    eff_icon = st.session_state.get("placing_effect")
            #    st.info(f"✨ MODO TERRENO: Clique para adicionar {eff_icon}.", icon="✨")
            # -------------------------------------

            # Ferramentas de Campo
            with st.expander("🛠️ Itens e Terrenos", expanded=False):
                if is_player:
                    effects_map = {"Fogo":"🔥", "Gelo":"🧊", "Água":"💧", "Rocha":"🪨", "Nuvem":"☁️", "Sol":"☀️", "Grama":"🍃", "Raio":"⚡"}
                    curr = st.session_state.get("placing_effect")
                    cols = st.columns(8)
                    for i, (k, v) in enumerate(effects_map.items()):
                        # Destaca o botão se estiver selecionado
                        btn_type = "primary" if curr == v else "secondary"
                        if cols[i].button(v, key=f"ef_{k}", type=btn_type):
                            st.session_state["placing_effect"] = v if curr != v else None
                            # Se ativar efeito, cancela outras ações para evitar bugs
                            st.session_state["moving_piece_id"] = None
                            st.session_state["placing_pid"] = None
                            st.rerun()
                    if st.button("Limpar Tudo"):
                        db.collection("rooms").document(rid).collection("public_state").document("state").update({
                            "effects": [],
                            "updatedAt": firestore.SERVER_TIMESTAMP,
                        })
                        st.rerun()

            show_grid = st.checkbox("Grade Tática", value=True, key=f"grid_{rid}")
            
            # ... (Restante do código de renderização do mapa permanece igual) ...
            state_updated_at = state.get("updatedAt")
            map_signature = json.dumps({
                "seed": seed,
                "tiles": tiles_packed,
                "theme": theme_key,
                "pieces": pieces_to_draw,
                "effects": field_effects,
                "grid": show_grid,
                "updatedAt": state_updated_at,
            }, sort_keys=True, default=str)
            if st.session_state.get("map_cache_sig") != map_signature:
                st.session_state["map_cache_sig"] = map_signature
                st.session_state["map_cache_img"] = render_map_with_pieces(
                    tiles,
                    theme_key,
                    seed,
                    pieces_to_draw,
                    trainer_name,
                    room,
                    effects=field_effects,
                    show_grid=show_grid,
                )
            img = st.session_state.get("map_cache_img")
            
            # Use o cursor crosshair se estiver em modo de edição
            cursor_style = "default"
            if st.session_state.get("moving_piece_id") or st.session_state.get("placing_pid") or st.session_state.get("placing_effect"):
                # Nota: streamlit_image_coordinates não suporta CSS direto cursor:pointer facilmente, 
                # mas o feedback visual acima ajuda.
                pass 

            click = streamlit_image_coordinates(img, key=f"map_{rid}")
            

        with c_opps:
            st.markdown("### 🆚 Oponentes")
            opponents = sorted(list(set([p for p in all_players if p != trainer_name])))
            
            if not opponents:
                st.caption("Aguardando...")
            else:
                for idx, opp_name in enumerate(opponents):
                    icons = ["🔴", "🟡", "🌸"]
                    icon = icons[idx] if idx < len(icons) else "⚪"
                    
                    # ✅ MELHORIA: Para 3 oponentes, usamos Expanders para não esticar a tela infinitamente
                    with st.expander(f"{icon} {opp_name}", expanded=(idx == 0)):
                        render_player_column(opp_name, "", is_me=False)

        # =========================
        # 8. LÓGICA DE CLIQUE
        # =========================
        if click and "x" in click and "y" in click:
            col = int(click["x"] // TILE_SIZE)
            row = int(click["y"] // TILE_SIZE)
            if 0 <= row < grid and 0 <= col < grid:
                ppid = st.session_state.get("placing_pid")
                peff = st.session_state.get("placing_effect")
                moving_piece_id = st.session_state.get("moving_piece_id")

                if peff:
                    curr = state.get("effects") or []
                    new = [e for e in curr if not (int(e["row"])==row and int(e["col"])==col)]
                    new.append({"icon": peff, "row": row, "col": col, "id": str(uuid.uuid4())[:8]})
                    db.collection("rooms").document(rid).collection("public_state").document("state").update({
                        "effects": new,
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                    })
                    add_public_event(db, rid, "effect", trainer_name, {"icon": peff, "to": [row, col]})
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
                    add_public_event(db, rid, "piece_placed", trainer_name, {"pid": ppid, "to": [row, col]})
                    st.session_state.pop("placing_pid", None)
                    st.rerun()
                elif moving_piece_id and is_player:
                    s_now = get_state(db, rid)
                    all_p = s_now.get("pieces") or []
                    mover = next((p for p in all_p if p["id"] == moving_piece_id), None)
                    if mover:
                        # 1. Guarda a posição antiga para o Log
                        old_pos = [mover["row"], mover["col"]]

                        # 2. Atualiza para a nova posição
                        mover["row"] = row
                        mover["col"] = col

                        # 3. Salva a peça no Firebase
                        upsert_piece(db, rid, mover)

                        # 4. Registra o movimento publicamente NO LOG
                        add_public_event(db, rid, "move", trainer_name, {
                            "pid": mover["pid"],
                            "from": old_pos,
                            "to": [row, col]
                        })



                        # 5. Limpa a seleção e recarrega
                        st.session_state["moving_piece_id"] = None
                        st.rerun()

        # Fora da lógica de clique, mas no final da View Battle
        render_public_log_fragment(db, rid)
        st.stop()
        
    elif view == "lobby":
            stop_pvp_sync_listener()
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
    
            
    
    

    

elif page == "Compendium de Ga'Al":
    render_compendium_page()

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
            st.image("mochila.png", width=150, caption="MINHA MOCHILA")
        except:
            st.image("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/main-stats/adventure-guide.png", width=150)
        
        # Ajuste de Saldo
        new_money = st.number_input("Ajustar saldo", value=int(user_data["backpack"]["money"]), step=100)
        if new_money != user_data["backpack"]["money"]:
            user_data["backpack"]["money"] = new_money
            save_data_cloud(trainer_name, user_data) 
        
        if st.button("🧹 Limpar itens vazios"):
            for k in ["medicine", "pokeballs", "tms", "key_items"]:
                user_data["backpack"][k] = [i for i in user_data["backpack"][k] if i["name"] and i.get("qty", 0) > 0]
            save_data_cloud(trainer_name, user_data) 
            st.rerun()
    
    with col_content:
            tabs = st.tabs(["💊 Medicamentos", "🔴 Poké Bolas", "💿 TMs", "🔑 Itens-chave"])
            cfg = [(tabs[0], "medicine", True), (tabs[1], "pokeballs", True), 
                   (tabs[2], "tms", True), (tabs[3], "key_items", False)]
    
            for tab, key, use_img in cfg:
                with tab:
                    current_list = user_data["backpack"].get(key, [])
                    
                    # Interface de edição 
                    updated_items = []
                    # Exibimos a lista atual + 1 linha sempre vazia no final
                    display_count = len(current_list) + 1
                    
                    for i in range(display_count):
                        item_data = current_list[i] if i < len(current_list) else {"name": "", "qty": 0}
                        res = render_item_row(key, i, item_data, show_image=use_img)
                        
                        if res["name"]: # Se o usuário digitou algo 
                            updated_items.append(res)
    
                    # Se a lista atualizada for maior que a original, salvamos 
                    # automaticamente para gerar a nova linha vazia 
                    if len(updated_items) > len(current_list):
                        user_data["backpack"][key] = updated_items
                        save_data_cloud(trainer_name, user_data)
                        st.rerun()
    
                    if st.button(f"💾 Confirmar {key.title()}", key=f"sv_{key}"):
                        user_data["backpack"][key] = updated_items
                        save_data_cloud(trainer_name, user_data)
                        st.success("Mochila sincronizada com sucesso.")    
    
    
    
    
    
    
