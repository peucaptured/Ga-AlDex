# -*- coding: utf-8 -*-
"""
Helpers de Session State (modularização final).

Este arquivo agora também centraliza o *gate* de rerun para evitar:
- "tremedeira" do mapa / perda de clique
- rerun automático atropelando ações do usuário
"""
from __future__ import annotations

import time
import streamlit as st

# ==========================
# Básicos
# ==========================
def ss_get(key: str, default=None):
    return st.session_state.get(key, default)

def ss_set(key: str, value):
    st.session_state[key] = value

def clear_keys(*keys: str):
    for k in keys:
        st.session_state.pop(k, None)

# ==========================
# Busy gate (ação do usuário)
# ==========================
def set_busy(seconds: float = 0.6, *, key: str = "arena_pause_until"):
    """Bloqueia reruns automáticos por alguns segundos (ex: após clicar em mover/colocar)."""
    st.session_state[key] = time.time() + float(seconds)

def is_busy(*, key: str = "arena_pause_until") -> bool:
    return time.time() < float(st.session_state.get(key, 0) or 0)

def pvp_in_action() -> bool:
    """Detecta se o usuário está no meio de uma ação no PvP (mover/colocar/terreno/avatar)."""
    return bool(
        st.session_state.get("moving_piece_id")
        or st.session_state.get("placing_pid")
        or st.session_state.get("placing_trainer")
        or st.session_state.get("placing_effect")
        or is_busy()
    )

def clear_pvp_action_flags():
    """Limpa flags de ação do PvP (fail-safe)."""
    clear_keys("moving_piece_id", "placing_pid", "placing_trainer", "placing_effect")

# ==========================
# Rerun gate (cooldown global)
# ==========================
_RERUN_LAST_TS_KEY = "_rerun_last_ts"
_RERUN_PENDING_KEY = "_rerun_pending"
_RERUN_PENDING_REASON_KEY = "_rerun_pending_reason"
_RERUN_COOLDOWN_SEC = 0.45  # evita rerun em rajada (watchdog + cliques)

def can_rerun_now(*, cooldown_sec: float = _RERUN_COOLDOWN_SEC) -> bool:
    now = time.time()
    last = float(st.session_state.get(_RERUN_LAST_TS_KEY, 0) or 0)
    return (now - last) >= float(cooldown_sec)

def mark_last_rerun():
    st.session_state[_RERUN_LAST_TS_KEY] = time.time()

def request_rerun(reason: str = "rerun", *, force: bool = False, cooldown_sec: float = _RERUN_COOLDOWN_SEC):
    """Pede rerun respeitando cooldown. Se não puder agora, marca como pendente."""
    st.session_state[_RERUN_PENDING_KEY] = True
    st.session_state[_RERUN_PENDING_REASON_KEY] = str(reason or "rerun")

    if force:
        mark_last_rerun()
        st.session_state[_RERUN_PENDING_KEY] = False
        st.rerun()

    # Sem force: respeita cooldown e evita atropelar ação do usuário
    if pvp_in_action():
        return

    if not can_rerun_now(cooldown_sec=cooldown_sec):
        return

    mark_last_rerun()
    st.session_state[_RERUN_PENDING_KEY] = False
    st.rerun()

def flush_pending_rerun(*, cooldown_sec: float = _RERUN_COOLDOWN_SEC):
    """Se existir rerun pendente e já é seguro, executa."""
    if not st.session_state.get(_RERUN_PENDING_KEY):
        return
    if pvp_in_action():
        return
    if not can_rerun_now(cooldown_sec=cooldown_sec):
        return
    # chama sem force, pra registrar last_ts
    request_rerun(st.session_state.get(_RERUN_PENDING_REASON_KEY, "pending"), force=False, cooldown_sec=cooldown_sec)
