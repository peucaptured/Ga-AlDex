# move_creator_ui.py
import streamlit as st
from move_db import MoveDB
from move_interpreter import interpret_effects_to_build

@st.cache_resource
def load_move_db(excel_path: str) -> MoveDB:
    return MoveDB.from_excel(excel_path, sheet_name="Golpes_MM")

def render_move_creator(
    excel_path: str,
    state_key_prefix: str = "mc",
    return_to_view: str | None = None,
):
    """
    Tela do criador de golpes:
    - busca por nome
    - rank
    - sugestão por descrição (fallback)
    - botão "Confirmar golpe" que adiciona em st.session_state["cg_moves"]
    """
    db = load_move_db(excel_path)

    if "cg_moves" not in st.session_state:
        st.session_state["cg_moves"] = []

    st.subheader("⚔️ Criação de Golpes (M&M)")
    tab1, tab2, tab3 = st.tabs(["🔎 Buscar por nome", "🧩 Criar por descrição", "🛠️ Criar do zero"])
    def _confirm_move(mv, rank: int, build: str, pp):
        st.session_state["cg_moves"].append({
            "name": mv.name,
            "rank": int(rank),
            "build": build,
            "pp_cost": pp,
            "meta": {
                "ranged": bool(mv.ranged),
                "perception_area": bool(mv.perception_area),
            }
        })


    def _render_move_card(mv, rank: int):
        st.markdown(f"### 🌀 {mv.name}  ({mv.tipo} / {mv.categoria})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ranged", "SIM" if mv.ranged else "NÃO")
        c2.metric("Perception Area", "SIM" if mv.perception_area else "NÃO")
        c3.metric("Resist Stat", mv.resist_stat or "—")

        st.write("**Descrição:**")
        st.write(mv.descricao or "—")

        st.write("**Build M&M (rank escolhido):**")

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
                r_damage = st.number_input("Rank Damage", min_value=0, max_value=30, value=int(rank),
                                           key=f"{state_key_prefix}_r_damage_{mv.name}_{rank}")
                r_aff = st.number_input("Rank Affliction", min_value=0, max_value=30, value=int(rank),
                                        key=f"{state_key_prefix}_r_aff_{mv.name}_{rank}")

            with cB:
                r_weaken = st.number_input("Rank Weaken", min_value=0, max_value=30, value=int(rank),
                                           key=f"{state_key_prefix}_r_weaken_{mv.name}_{rank}")
                r_heal = st.number_input("Rank Healing", min_value=0, max_value=30, value=0,
                                         key=f"{state_key_prefix}_r_heal_{mv.name}_{rank}")

            with cC:
                r_create = st.number_input("Rank Create", min_value=0, max_value=30, value=0,
                                           key=f"{state_key_prefix}_r_create_{mv.name}_{rank}")
                r_env = st.number_input("Rank Environment", min_value=0, max_value=30, value=0,
                                        key=f"{state_key_prefix}_r_env_{mv.name}_{rank}")

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

        # ✅ PP: se customizou, usa o manual; se não, calcula pelo Excel (PP por Rank × rank)
        if sub_ranks:
            pp = int(manual_pp or 0)
            why = "PP informado manualmente (porque você escolheu ranks por sub-efeito)."
        else:
            pp, why = mv.pp_cost(rank)

        if pp is not None:
            st.info(f"PP: **{pp}** — {why}")
        else:
            st.warning(f"PP: não definido — {why}")

    with tab1:
        name = st.text_input("Nome do golpe", key=f"{state_key_prefix}_name")
        rank = st.slider("Rank", 1, 20, 10, key=f"{state_key_prefix}_rank")

        # ✅ persistência do último resultado (senão os botões "Confirmar" não funcionam)
        sel_name_key = f"{state_key_prefix}_selected_name"
        sel_rank_key = f"{state_key_prefix}_selected_rank"

        if st.button("Buscar", key=f"{state_key_prefix}_search", type="primary"):
            mv = db.get_by_name(name)
            if not mv:
                st.session_state.pop(sel_name_key, None)
                st.error("Não achei pelo nome. Use a aba 'Criar por descrição'.")
            else:
                st.session_state[sel_name_key] = mv.name
                st.session_state[sel_rank_key] = int(rank)

        # renderiza sempre o último resultado buscado (para os botões existirem no rerun)
        if st.session_state.get(sel_name_key):
            mv = db.get_by_name(st.session_state[sel_name_key])
            if mv:
                _render_move_card(mv, int(st.session_state.get(sel_rank_key, rank)))

with tab2:
        desc = st.text_area("Descrição do golpe", height=120, key=f"{state_key_prefix}_desc")
        rank2 = st.slider("Rank para renderizar sugestões", 1, 20, 10, key=f"{state_key_prefix}_rank2")
        top_k = st.slider("Sugestões", 3, 10, 5, key=f"{state_key_prefix}_topk")

        # ✅ persistência das sugestões (mesmo motivo: botões precisam existir no rerun)
        sugg_key = f"{state_key_prefix}_sugg_names"

        if st.button("Sugerir", key=f"{state_key_prefix}_suggest"):
            sugg = db.suggest_by_description(desc, top_k=top_k)
            if not sugg:
                st.session_state.pop(sugg_key, None)
                st.warning("Digite uma descrição.")
            else:
                st.session_state[sugg_key] = [mv.name for (mv, _score) in sugg]

        names = st.session_state.get(sugg_key) or []
        if names:
            for i, nm in enumerate(names, start=1):
                mv = db.get_by_name(nm)
                if not mv:
                    continue
                with st.expander(f"{i}) {mv.name}", expanded=(i == 1)):
                    _render_move_card(mv, int(rank2))


 
