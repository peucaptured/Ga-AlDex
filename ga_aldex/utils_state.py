# -*- coding: utf-8 -*-
"""
Helpers de Session State (opcional, modularização final).
NÃO altera lógica do app: são apenas utilitários reutilizáveis.
"""
from __future__ import annotations
import time
import streamlit as st

def ss_get(path: str, default=None):
    """Lê st.session_state usando chave simples (sem nested)."""
    return st.session_state.get(path, default)

def ss_set(key: str, value):
    st.session_state[key] = value

def set_busy(seconds: float = 0.6, *, key: str = "arena_pause_until"):
    st.session_state[key] = time.time() + float(seconds)

def is_busy(*, key: str = "arena_pause_until") -> bool:
    return time.time() < float(st.session_state.get(key, 0) or 0)

def clear_keys(*keys: str):
    for k in keys:
        st.session_state.pop(k, None)
