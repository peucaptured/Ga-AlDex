# ui_styles.py
# Estilos visuais globais do app

import streamlit as st

def apply_global_styles():
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
       IFRAME / COMPONENTES (deixar fundo transparente)
       ========================================================= */
    div[data-testid="stIFrame"],
    div[data-testid="stIFrame"] > iframe {
      background: transparent !important;
      border: none !important;
    }
    
    /* Alguns builds do Streamlit usam wrappers adicionais */
    iframe {
      background: transparent !important;
    }
    
    /* =========================================================
       IFRAME DO st_click_detector (Custom Component)
       ========================================================= */
    div[data-testid="stCustomComponentV1"],
    div[data-testid="stCustomComponentV1"] > iframe,
    div[data-testid="stCustomComponentV1"] iframe {
      background: transparent !important;
      border: none !important;
    }
    
    /* fallback: se Streamlit usar outro testid */
    div[data-testid="stCustomComponent"],
    div[data-testid="stCustomComponent"] > iframe,
    div[data-testid="stCustomComponent"] iframe {
      background: transparent !important;
      border: none !important;
    }
    
    /* fallback geral */
    iframe {
      background: transparent !important;
    }
    
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
