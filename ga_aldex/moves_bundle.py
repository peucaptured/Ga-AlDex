# ga_aldex/moves_bundle.py
# Extraído do app.py original (MoveDB + Move Creator + utilitários próximos)

import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import re
import json
import time
import base64
import gzip
import io
import os
import html
import unicodedata
import difflib
import uuid
import random
import mimetypes
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageEnhance

# Dependências opcionais usadas pelo seu projeto
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# move_db.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Utils
# ----------------------------
def _norm(s: str) -> str:
    import unicodedata

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


def _draw_tactical_grid(img, grid, tile_size: int):
    """
    Desenha a grade por cima da imagem.

    Aceita:
      - grid como matriz (H x W): list[list[...]]
      - grid como int: tamanho (NxN)
      - grid como tuple/list (h, w)
    """
    draw = ImageDraw.Draw(img)

    # 1) Descobre dimensões (h, w)
    h = w = 0

    if isinstance(grid, int):
        h = w = int(grid)
    elif isinstance(grid, (tuple, list)) and len(grid) == 2 and all(isinstance(x, int) for x in grid):
        h, w = int(grid[0]), int(grid[1])
    else:
        # assume matriz
        try:
            h = len(grid)
            w = len(grid[0]) if h else 0
        except Exception:
            return  # fail-safe: não desenha grade se vier algo inesperado

    if h <= 0 or w <= 0:
        return

    # 2) linhas verticais
    for x in range(w + 1):
        px = x * tile_size
        draw.line([(px, 0), (px, h * tile_size)], fill=(0, 0, 0, 90), width=1)

    # 3) linhas horizontais
    for y in range(h + 1):
        py = y * tile_size
        draw.line([(0, py), (w * tile_size, py)], fill=(0, 0, 0, 90), width=1)


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

@st.cache_data(show_spinner=False)
def _audio_data_uri(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    data = p.read_bytes()
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "audio/mpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"
    
@st.cache_data(show_spinner=False)
def comp_img_data_uri(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""

    data = p.read_bytes()
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def apply_npc_overrides(base_npcs: dict, overrides: dict) -> dict:
    base_npcs = base_npcs or {}
    overrides = overrides or {}

    for npc_name, over in overrides.items():
        if not isinstance(over, dict):
            continue

        base = base_npcs.get(npc_name) or {"name": npc_name, "sections": {}}
        if not isinstance(base, dict):
            base = {"name": npc_name, "sections": {}}

        merged = dict(base)  # shallow copy

        # nunca deixar perder sections (lore)
        merged_sections = {}
        if isinstance(base.get("sections"), dict):
            merged_sections.update(base["sections"])
        if isinstance(over.get("sections"), dict):
            # override só adiciona/atualiza; não apaga o que existe
            for k, v in over["sections"].items():
                if (str(v).strip() if v is not None else ""):
                    merged_sections[k] = v
        merged["sections"] = merged_sections

        # campos "seguros" que você pode sobrescrever
        for k in ["idade", "origem", "ocupacao", "status", "img", "name"]:
            if k in over and over[k] not in (None, ""):
                merged[k] = over[k]

        # pokémons: dedupe preservando ordem
        def _dedupe_list(xs):
            out, seen = [], set()
            for x in xs or []:
                s = str(x).strip()
                if not s:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
            return out

        if "pokemons" in over:
            merged["pokemons"] = _dedupe_list(over.get("pokemons"))
        if "pokemons_conhecidos" in over:
            merged["pokemons_conhecidos"] = _dedupe_list(over.get("pokemons_conhecidos"))

        base_npcs[npc_name] = merged

    return base_npcs


def render_bgm(track_path: str, volume: float = 0.35) -> None:
    src = _audio_data_uri(track_path)
    vol = max(0.0, min(1.0, float(volume)))

    components.html(
        """
<div style="height:0; overflow:hidden;">
  <audio id="ds-bgm" preload="auto" playsinline></audio>
</div>

<script>
(function () {
  const SRC = __SRC__;
  const VOL = __VOL__;

  const a = document.getElementById("ds-bgm");
  if (!a) return;

  a.loop = true;
  a.volume = VOL;

  if (a.dataset.src !== SRC) {
      // PARA a faixa anterior
      try { a.pause(); } catch(e) {}
      try { a.currentTime = 0; } catch(e) {}
    
      // limpa fontes antigas e descarrega
      a.innerHTML = "";
      a.removeAttribute("src");
      a.load();
    
      // carrega a nova faixa
      a.dataset.src = SRC;
      const s = document.createElement("source");
      s.src = SRC;
      a.appendChild(s);
      a.load();
    }

  async function tryPlay() {
    try {
      await a.play();
      localStorage.setItem("ds_bgm_unlocked", "1");
    } catch (e) {
      if (a.dataset.clickHooked === "1") return;
      a.dataset.clickHooked = "1";
      const parentDoc = window.parent && window.parent.document ? window.parent.document : document;
      const handler = async () => {
        parentDoc.removeEventListener("click", handler, true);
        try { await a.play(); } catch (e2) {}
        localStorage.setItem("ds_bgm_unlocked", "1");
      };
      parentDoc.addEventListener("click", handler, true);
    }
  }

  tryPlay();
})();
</script>
        """.replace("__SRC__", json.dumps(src)).replace("__VOL__", str(vol)),
        height=0,
        width=0,
        scrolling=False,
    )

# --- PLANO B: VIGIA DE SINCRONIZAÇÃO ---
@st.fragment(run_every=2) # Roda esta função sozinha a cada 2 segundos
def sync_watchdog(db, rid):
    if not rid:
        return

    # Evita "tremedeira" e perda de clique durante ações (mover/colocar/terreno).
    def _pvp_in_action() -> bool:
        return bool(
            st.session_state.get("moving_piece_id")
            or st.session_state.get("placing_pid")
            or st.session_state.get("placing_trainer")
            or st.session_state.get("placing_effect")
        )

    try:
        doc_ref = db.collection("rooms").document(rid).collection("public_state").document("state")
        snapshot = doc_ref.get()
        if not snapshot.exists:
            return

        server_data = snapshot.to_dict() or {}
        server_time = server_data.get("updatedAt")

        if "last_map_update" not in st.session_state:
            st.session_state["last_map_update"] = server_time
            st.session_state["pvp_sync_pending"] = False
            return

        if server_time != st.session_state.get("last_map_update"):
            st.session_state["last_map_update"] = server_time

            # Se o usuário está no meio de uma ação, marcamos como pendente e não rerunamos agora.
            if _pvp_in_action() or float(st.session_state.get("arena_pause_until", 0) or 0) > time.time():
                st.session_state["pvp_sync_pending"] = True
                return

            st.session_state["pvp_sync_pending"] = False
            st.rerun()

    except Exception:
        # Se der erro de conexão, ignora e tenta na próxima
        return

try:
    from move_interpreter import interpret_effects_to_build
except Exception:
    interpret_effects_to_build = None

@st.cache_resource
def load_move_db(excel_path: str) -> "MoveDB":
    return MoveDB.from_excel(excel_path, sheet_name="Golpes_MM")

def _comp_mtime(p: str) -> float:
    try:
        return os.path.getmtime(p) if p and os.path.exists(p) else 0.0
    except Exception:
        return 0.0


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






