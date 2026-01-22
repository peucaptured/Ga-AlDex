# move_db.py
from __future__ import annotations

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

    def render_build(self, rank: int) -> str:
        b = (self.build or "").strip()
        if not b:
            return ""
    
        # 1) Trocar Rank = PL por Rank = <rank>
        b = re.sub(r"Rank\s*=\s*PL", f"Rank = {rank}", b, flags=re.IGNORECASE)
        b = re.sub(r"Rank\s*=\s*X", f"Rank = {rank}", b, flags=re.IGNORECASE)
    
        # 2) Escalar efeitos numéricos: Damage/Weaken/Affliction/etc. para o mesmo rank
        # Ex.: "Weaken ... 1" -> "Weaken ... 10"
        def _scale(m):
            effect = m.group(1)
            return f"{effect} {rank}"
    
        b = re.sub(r"\b(Damage|Weaken|Affliction|Healing|Nullify|Create)\s+\d+\b", _scale, b, flags=re.IGNORECASE)
    
        # 3) Deduplicar segmentos "Linked ..." idênticos (exatos)
        # separa por ';' (seu excel costuma usar isso)
        parts = [p.strip() for p in b.split(";") if p.strip()]
        seen = set()
        uniq = []
        for p in parts:
            key = re.sub(r"\s+", " ", p.lower()).strip()
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        # ==============================
        # ==========================================
        # DEFINIÇÃO DE RESISTÊNCIA DO DANO (FINAL)
        # ==========================================
        name_desc = f"{self.name} {self.descricao or ''}".lower()
        categoria = (self.categoria or "").lower()
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
        
        # 3) Aplica no primeiro Damage
        b = re.sub(
            r"(Damage\s+\d+)(?![^\[]*\])",
            rf"\1 (Resisted by {damage_resist})",
            b,
            count=1,
            flags=re.IGNORECASE
        )

    
        return "; ".join(uniq)


    def pp_cost(self, rank: int) -> Tuple[Optional[float], str]:
        """
        Retorna (pp_cost, explicacao). Aqui "PP" = Power Points do M&M.

        - Se existir coluna PP_Custo no Excel, usa direto.
        - Caso contrário, usa um estimador simples (você pode trocar depois pela sua tabela oficial).
        Regra base do M&M: custo final por graduação = custo básico + extras - falhas. :contentReference[oaicite:3]{index=3}
        """
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
            return None, "Sem PP_Custo no Excel e não foi possível estimar por falta de palavras-chave na build."

        # custo aproximado = (custo_por_rank * rank)
        est = round(base_per_rank * rank, 2)
        return est, "Estimativa simples (troque pela sua tabela oficial quando quiser)."


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

    def get_by_name(self, name: str) -> Optional[Move]:
        key = _norm(name)
        hit = self.df[self.df["__name_norm"] == key]
        if hit.empty:
            return None
        row = hit.iloc[0].to_dict()

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
            mv = Move(
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
