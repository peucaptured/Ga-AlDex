# move_creator_ui.py
import streamlit as st
from move_db import MoveDB
from move_interpreter import interpret_effects_to_build

@st.cache_resource
def load_move_db(excel_path: str) -> MoveDB:
    return MoveDB.from_excel(excel_path, sheet_name="Golpes_MM")

def render_move_creator(excel_path: str, state_key_prefix: str = "mc"):
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

    def _render_move_card(mv, rank: int):
        st.markdown(f"### 🌀 {mv.name}  ({mv.tipo} / {mv.categoria})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ranged", "SIM" if mv.ranged else "NÃO")
        c2.metric("Perception Area", "SIM" if mv.perception_area else "NÃO")
        c3.metric("Resist Stat", mv.resist_stat or "—")

        st.write("**Descrição:**")
        st.write(mv.descricao or "—")

        st.write("**Build M&M (rank escolhido):**")
        build = mv.render_build(rank)
        st.code(build, language="text")

        pp, why = mv.pp_cost(rank)
        if pp is not None:
            st.info(f"PP (estimado ou do Excel): **{pp}** — {why}")
        else:
            st.warning(f"PP: não definido — {why}")

        if mv.how_it_works:
            st.write("**Como funciona:**")
            st.write(mv.how_it_works)

        # ✅ confirmar golpe
        if st.button(f"✅ Confirmar {mv.name}", key=f"{state_key_prefix}_confirm_{mv.name}_{rank}"):
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
            st.success(f"Adicionado: {mv.name} (Rank {rank})")

    with tab1:
        name = st.text_input("Nome do golpe", key=f"{state_key_prefix}_name")
        rank = st.slider("Rank", 1, 20, 10, key=f"{state_key_prefix}_rank")
        if st.button("Buscar", key=f"{state_key_prefix}_search", type="primary"):
            mv = db.get_by_name(name)
            if not mv:
                st.error("Não achei pelo nome. Use a aba 'Criar por descrição'.")
            else:
                _render_move_card(mv, rank)

    with tab2:
        desc = st.text_area("Descrição do golpe", height=120, key=f"{state_key_prefix}_desc")
        rank2 = st.slider("Rank para renderizar sugestões", 1, 20, 10, key=f"{state_key_prefix}_rank2")
        top_k = st.slider("Sugestões", 3, 10, 5, key=f"{state_key_prefix}_topk")
        if st.button("Sugerir", key=f"{state_key_prefix}_suggest"):
            sugg = db.suggest_by_description(desc, top_k=top_k)
            if not sugg:
                st.warning("Digite uma descrição.")
            else:
                for i, (mv, score) in enumerate(sugg, start=1):
                    with st.expander(f"{i}) {mv.name} — similaridade {score:.3f}", expanded=(i == 1)):
                        _render_move_card(mv, rank2)

    st.divider()
    st.subheader("📦 Golpes confirmados nesta ficha")

    if not st.session_state["cg_moves"]:
        st.info("Nenhum golpe confirmado ainda.")
    else:
        for i, m in enumerate(st.session_state["cg_moves"]):
            c1, c2, c3 = st.columns([6, 2, 2])
    
            with c1:
                st.write(f"**{m['name']}** (Rank {m['rank']}) — PP: {m.get('pp_cost')}")
    
            with c2:
                st.caption(" ")
    
            with c3:
                if st.button("❌ Remover", key=f"{state_key_prefix}_remove_{i}"):
                    st.session_state["cg_moves"].pop(i)
                    st.rerun()
                    
        with tab3:
            st.subheader("🛠️ Criar Golpe do Zero")
        
            rank = st.slider("Rank do golpe", 1, 20, 10)
            is_special = st.checkbox("Golpe Especial (Intelect Based)", value=True)
        
            st.markdown("### Efeitos")
            effects = {
                "damage": st.checkbox("Causar Dano"),
                "affliction": st.checkbox("Causar Affliction"),
                "weaken": st.checkbox("Causar Weaken"),
                "healing": st.checkbox("Cura"),
                "create": st.checkbox("Create"),
                "environment": st.checkbox("Environment"),
            }
        
            st.markdown("### Detalhes do Weaken")
            effects["weaken_stgr"] = st.checkbox("Weaken Strength")
            effects["weaken_int"] = st.checkbox("Weaken Intellect")
            effects["weaken_dodge"] = st.checkbox("Weaken Dodge")
            effects["weaken_will"] = st.checkbox("Weaken Will")
        
            st.markdown("### Modificadores")
            area = st.selectbox("Área", ["Nenhuma", "Burst", "Cone", "Line"])
            perception = st.checkbox("Perception Area")
            ranged = st.checkbox("Ranged")
        
            build = interpret_effects_to_build(
                rank=rank,
                is_special=is_special,
                effects=effects,
                area=None if area == "Nenhuma" else area,
                perception=perception,
                ranged=ranged,
            )
        
            st.markdown("### Build Gerada")
            st.code(build, language="text")
        
            if st.button("✅ Confirmar golpe criado do zero"):
                st.session_state["cg_moves"].append({
                    "name": "Golpe Customizado",
                    "rank": rank,
                    "build": build,
                    "pp_cost": None,
                    "meta": {"custom": True}
                })
                st.success("Golpe customizado adicionado à ficha.")

