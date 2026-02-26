# powers_data.py
# Catalogo completo do sistema M&M 3e adaptado para Pokemon
# Usado pelo Golpe Builder para montar builds customizadas

# =============================================
# EFEITOS PRIMARIOS
# =============================================
EFFECTS = {
    # ── Ataque ──
    "damage": {
        "base_cost": 2, "type": "attack",
        "default_resist": "Toughness",
        "label_pt": "Causar Dano", "label_en": "Damage",
        "desc_pt": "Causa dano direto ao alvo. Resistido por Toughness por padrao.",
        "action": "standard", "default_range": "close",
    },
    "affliction": {
        "base_cost": 1, "type": "attack",
        "default_resist": "Fortitude",
        "label_pt": "Causar Aflicao (Condicoes)", "label_en": "Affliction",
        "desc_pt": "Impoe condicoes debilitantes em 3 graus: leve, moderado e severo.",
        "action": "standard", "has_conditions": True, "default_range": "close",
    },
    "weaken": {
        "base_cost": 1, "type": "attack",
        "default_resist": "Fortitude",
        "label_pt": "Reduzir Atributo (Weaken)", "label_en": "Weaken",
        "desc_pt": "Reduz um ou mais atributos do alvo (Strength, Dodge, Will, etc.).",
        "action": "standard", "has_targets": True, "target_source": "STAT_TARGETS",
        "default_range": "close",
    },
    "nullify": {
        "base_cost": 1, "type": "attack",
        "default_resist": "Will",
        "label_pt": "Anular Poder (Nullify)", "label_en": "Nullify",
        "desc_pt": "Cancela ou suprime poderes ativos do alvo.",
        "action": "standard", "default_range": "ranged",
    },
    "move_object": {
        "base_cost": 2, "type": "attack",
        "default_resist": None,
        "label_pt": "Mover Objeto (Empurrar/Puxar)", "label_en": "Move Object",
        "desc_pt": "Move alvos ou objetos a distancia (empurrar, puxar, telecinese).",
        "action": "standard", "default_range": "ranged",
    },
    "deflect": {
        "base_cost": 1, "type": "attack",
        "default_resist": None,
        "label_pt": "Deflexao (Deflect)", "label_en": "Deflect",
        "desc_pt": "Desvia ataques a distancia direcionados ao usuario ou aliados.",
        "action": "standard", "default_range": "ranged",
    },
    "mind_reading": {
        "base_cost": 2, "type": "attack",
        "default_resist": "Will",
        "label_pt": "Leitura Mental (Mind Reading)", "label_en": "Mind Reading",
        "desc_pt": "Le os pensamentos do alvo. Resistido por Will.",
        "action": "standard", "default_range": "close",
    },
    "transform": {
        "base_cost": 5, "type": "attack",
        "default_resist": "Fortitude",
        "label_pt": "Transformar (Transform)", "label_en": "Transform",
        "desc_pt": "Transforma o alvo em algo diferente (pedra, gelo, etc.).",
        "action": "standard", "default_range": "close",
    },

    # ── Suporte ──
    "healing": {
        "base_cost": 2, "type": "support",
        "default_resist": None,
        "label_pt": "Curar (Healing)", "label_en": "Healing",
        "desc_pt": "Restaura condicoes de dano do alvo.",
        "action": "standard", "default_range": "close",
    },
    "enhanced_trait": {
        "base_cost": 1, "type": "support",
        "default_resist": None,
        "label_pt": "Aumentar Atributo Proprio", "label_en": "Enhanced",
        "desc_pt": "Aumenta temporariamente um atributo do proprio Pokemon (max +5).",
        "action": "free", "has_targets": True, "target_source": "STAT_TARGETS",
        "default_range": "personal",
    },
    "protection": {
        "base_cost": 1, "type": "support",
        "default_resist": None,
        "label_pt": "Protecao (Toughness+)", "label_en": "Protection",
        "desc_pt": "Aumenta a Toughness do Pokemon.",
        "action": "free", "default_range": "personal",
    },
    "immunity": {
        "base_cost": 1, "type": "support",
        "default_resist": None,
        "label_pt": "Imunidade", "label_en": "Immunity",
        "desc_pt": "Confere imunidade a um tipo de efeito especifico.",
        "action": "free", "default_range": "personal",
    },
    "regeneration": {
        "base_cost": 1, "type": "support",
        "default_resist": None,
        "label_pt": "Regeneracao", "label_en": "Regeneration",
        "desc_pt": "Recuperacao continua de dano ao longo dos turnos.",
        "action": "free", "default_range": "personal",
    },
    "feature": {
        "base_cost": 1, "type": "support",
        "default_resist": None,
        "label_pt": "Caracteristica (Feature)", "label_en": "Feature",
        "desc_pt": "Habilidade menor com utilidade narrativa ou mecanica simples.",
        "action": "free", "default_range": "personal", "flat_cost": True,
    },
    "absorption": {
        "base_cost": 1, "type": "support",
        "default_resist": None,
        "label_pt": "Absorcao (Absorption)", "label_en": "Absorption",
        "desc_pt": "Absorve um tipo de energia/dano e converte em cura ou poder.",
        "action": "free", "default_range": "personal",
    },

    # ── Movimento ──
    "teleport": {
        "base_cost": 2, "type": "movement",
        "default_resist": None,
        "label_pt": "Teletransporte", "label_en": "Teleport",
        "desc_pt": "Move o Pokemon instantaneamente para outro local.",
        "action": "move", "default_range": "personal",
    },
    "flight": {
        "base_cost": 2, "type": "movement",
        "default_resist": None,
        "label_pt": "Voo (Flight)", "label_en": "Flight",
        "desc_pt": "Permite ao Pokemon voar.",
        "action": "free", "default_range": "personal",
    },
    "speed": {
        "base_cost": 1, "type": "movement",
        "default_resist": None,
        "label_pt": "Velocidade (Speed)", "label_en": "Speed",
        "desc_pt": "Aumenta a velocidade de movimento terrestre.",
        "action": "free", "default_range": "personal",
    },
    "swimming": {
        "base_cost": 1, "type": "movement",
        "default_resist": None,
        "label_pt": "Natacao (Swimming)", "label_en": "Swimming",
        "desc_pt": "Permite nadar com velocidade igual ao rank.",
        "action": "free", "default_range": "personal",
    },
    "burrowing": {
        "base_cost": 1, "type": "movement",
        "default_resist": None,
        "label_pt": "Escavar (Burrowing)", "label_en": "Burrowing",
        "desc_pt": "Move-se pelo subsolo.",
        "action": "free", "default_range": "personal",
    },
    "leaping": {
        "base_cost": 1, "type": "movement",
        "default_resist": None,
        "label_pt": "Salto (Leaping)", "label_en": "Leaping",
        "desc_pt": "Aumenta a distancia de salto.",
        "action": "free", "default_range": "personal",
    },
    "movement": {
        "base_cost": 2, "type": "movement",
        "default_resist": None,
        "label_pt": "Movimento Especial (Movement)", "label_en": "Movement",
        "desc_pt": "Formas especiais de movimento (escalar paredes, caminhar na agua, etc.).",
        "action": "free", "default_range": "personal",
    },
    "quickness": {
        "base_cost": 1, "type": "movement",
        "default_resist": None,
        "label_pt": "Rapidez (Quickness)", "label_en": "Quickness",
        "desc_pt": "Realiza tarefas rotineiras muito mais rapido.",
        "action": "free", "default_range": "personal",
    },
    "elongation": {
        "base_cost": 1, "type": "movement",
        "default_resist": None,
        "label_pt": "Alongamento (Elongation)", "label_en": "Elongation",
        "desc_pt": "Estende partes do corpo, aumentando alcance de Close.",
        "action": "free", "default_range": "personal",
    },

    # ── Sensorial ──
    "senses": {
        "base_cost": 1, "type": "sensory",
        "default_resist": None,
        "label_pt": "Sentidos (Senses)", "label_en": "Senses",
        "desc_pt": "Sentidos aprimorados ou extras (visao noturna, tremorsense, etc.).",
        "action": "free", "default_range": "personal", "flat_cost": True,
    },
    "remote_sensing": {
        "base_cost": 1, "type": "sensory",
        "default_resist": None,
        "label_pt": "Sensoriamento Remoto", "label_en": "Remote Sensing",
        "desc_pt": "Projeta sentidos para um local distante.",
        "action": "standard", "default_range": "ranged",
    },
    "communication": {
        "base_cost": 4, "type": "sensory",
        "default_resist": None,
        "label_pt": "Comunicacao (Communication)", "label_en": "Communication",
        "desc_pt": "Comunicacao mental ou a distancia.",
        "action": "free", "default_range": "ranged",
    },

    # ── Utilidade ──
    "create": {
        "base_cost": 2, "type": "utility",
        "default_resist": None,
        "label_pt": "Criar Objeto/Terreno", "label_en": "Create",
        "desc_pt": "Manifesta objetos solidos ou altera o terreno de batalha.",
        "action": "standard", "default_range": "close",
    },
    "environment": {
        "base_cost": 1, "type": "utility",
        "default_resist": None,
        "label_pt": "Alterar Ambiente (Clima/Terreno)", "label_en": "Environment",
        "desc_pt": "Muda condicoes climaticas ou ambientais numa area.",
        "action": "standard", "default_range": "ranged",
    },
    "concealment": {
        "base_cost": 2, "type": "utility",
        "default_resist": None,
        "label_pt": "Ocultacao (Invisibilidade)", "label_en": "Concealment",
        "desc_pt": "Torna o Pokemon invisivel ou parcialmente oculto.",
        "action": "standard", "default_range": "personal",
    },
    "insubstantial": {
        "base_cost": 5, "type": "utility",
        "default_resist": None,
        "label_pt": "Intangibilidade", "label_en": "Insubstantial",
        "desc_pt": "Torna o Pokemon imaterial, atravessando objetos solidos.",
        "action": "standard", "default_range": "personal",
    },
    "illusion": {
        "base_cost": 1, "type": "utility",
        "default_resist": "Will",
        "label_pt": "Ilusao (Illusion)", "label_en": "Illusion",
        "desc_pt": "Cria imagens/sons ilusorios. Custo varia por sentido afetado (1/rank por sentido).",
        "action": "standard", "default_range": "ranged",
    },
    "morph": {
        "base_cost": 5, "type": "utility",
        "default_resist": None,
        "label_pt": "Metamorfose (Morph)", "label_en": "Morph",
        "desc_pt": "Muda a aparencia do Pokemon para outra forma.",
        "action": "standard", "default_range": "personal",
    },
    "growth": {
        "base_cost": 2, "type": "utility",
        "default_resist": None,
        "label_pt": "Crescimento (Growth)", "label_en": "Growth",
        "desc_pt": "Aumenta o tamanho do Pokemon (+1 Stgr, -1 Dodge/Stealth por rank).",
        "action": "free", "default_range": "personal",
    },
    "shrinking": {
        "base_cost": 2, "type": "utility",
        "default_resist": None,
        "label_pt": "Reducao (Shrinking)", "label_en": "Shrinking",
        "desc_pt": "Reduz o tamanho do Pokemon (+1 Dodge/Stealth, -1 Stgr por rank).",
        "action": "free", "default_range": "personal",
    },
    "summon": {
        "base_cost": 2, "type": "utility",
        "default_resist": None,
        "label_pt": "Invocar (Summon)", "label_en": "Summon",
        "desc_pt": "Invoca uma criatura ou copia para ajudar em combate.",
        "action": "standard", "default_range": "close",
    },
    "variable": {
        "base_cost": 7, "type": "utility",
        "default_resist": None,
        "label_pt": "Variavel (Variable)", "label_en": "Variable",
        "desc_pt": "Pool de pontos que pode ser redistribuido em efeitos diferentes.",
        "action": "standard", "default_range": "personal",
    },
}

# =============================================
# EXTRAS (aumentam o custo)
# =============================================
# cost_type:
#   "per_rank"      = +cost por rank do efeito
#   "flat"          = +cost fixo (uma vez)
#   "flat_per_rank" = +cost fixo por rank DESTE extra
#   "custom"        = custo definido pelo usuario
#
# compatible_effects: lista de effect_keys onde este extra pode ser usado.
#   Se ausente, o extra e GENERICO (disponivel para todos os efeitos).
EXTRAS = {
    # ==================================================================
    # CUSTOM / ESPECIAL POKEMON
    # ==================================================================
    "custom_stgr_based": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "[Custom 0/r: Stgr Based]",
        "label_en": "Custom 0/r: Stgr Based",
        "desc_pt": "Dano baseado em Forca (Stgr). Golpes fisicos.",
        "build_fragment": "[Custom 0/r: Stgr Based]",
        "group": "custom",
    },
    "custom_int_based": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "[Custom 0/r: Intelect Based]",
        "label_en": "Custom 0/r: Intelect Based",
        "desc_pt": "Dano baseado em Intelecto (Int). Golpes especiais.",
        "build_fragment": "[Custom 0/r: Intelect Based]",
        "group": "custom",
    },
    "custom_free": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "[Custom 0/r: (texto livre)]",
        "label_en": "Custom",
        "desc_pt": "Modificador customizado sem custo por rank. Descreva o efeito.",
        "has_description": True,
        "group": "custom",
    },
    "custom_plus1": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "[Custom +1/r: (texto livre)]",
        "label_en": "Custom +1/r",
        "desc_pt": "Modificador customizado que custa +1 por rank. Descreva o efeito.",
        "has_description": True,
        "group": "custom",
    },
    "custom_minus1": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "[Custom -1/r: (texto livre)]",
        "label_en": "Custom -1/r",
        "desc_pt": "Modificador customizado que reduz -1 por rank. Descreva o efeito.",
        "has_description": True,
        "group": "custom",
    },

    # ==================================================================
    # GENERICOS (disponiveis para qualquer efeito)
    # ==================================================================
    # ── Alcance ──
    "ranged": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "A Distancia (Ranged)", "label_en": "Ranged",
        "desc_pt": "O ataque pode ser feito a distancia em vez de corpo-a-corpo.",
        "group": "range",
    },
    "increased_range": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Alcance Aumentado (Increased Range)", "label_en": "Increased Range",
        "desc_pt": "Dobra o alcance do efeito por rank deste extra.",
        "group": "range",
    },
    "extended_range": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Alcance Estendido (Extended Range)", "label_en": "Extended Range",
        "desc_pt": "Alcance extra alem do padrao.",
        "group": "range",
    },

    # ── Area ──
    "area_burst": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Area: Explosao (Burst)", "label_en": "Area: Burst",
        "desc_pt": "Afeta todos numa esfera de 30 pes por rank.",
        "group": "area", "build_fragment": "[Area: Burst]",
    },
    "area_cone": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Area: Cone", "label_en": "Area: Cone",
        "desc_pt": "Afeta uma area em formato de cone a frente.",
        "group": "area", "build_fragment": "[Area: Cone]",
    },
    "area_line": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Area: Linha (Line)", "label_en": "Area: Line",
        "desc_pt": "Afeta todos numa linha reta.",
        "group": "area", "build_fragment": "[Area: Line]",
    },
    "area_cloud": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Area: Nuvem (Cloud)", "label_en": "Area: Cloud",
        "desc_pt": "Cria uma nuvem persistente que afeta quem entra na area.",
        "group": "area", "build_fragment": "[Area: Cloud]",
    },
    "area_perception": {
        "cost": 2, "cost_type": "per_rank",
        "label_pt": "Area: Percepcao (Perception)", "label_en": "Perception Area",
        "desc_pt": "Afeta todos que o Pokemon pode perceber. Sem teste de ataque.",
        "group": "area", "build_fragment": "[Perception Area]",
    },
    "selective": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Seletivo (Selective)", "label_en": "Selective",
        "desc_pt": "Permite escolher quem e atingido dentro da area.",
        "requires_group": "area",
    },

    # ── Combate (generico) ──
    "accurate": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Preciso (Accurate)", "label_en": "Accurate",
        "desc_pt": "+2 bonus de ataque por rank deste extra.",
    },
    "improved_critical": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Critico Aprimorado (Improved Critical)", "label_en": "Improved Critical",
        "desc_pt": "Reduz o numero necessario para acerto critico em 1 por rank.",
        "max_ranks": 4,
    },
    "multi_attack": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Multi-Ataque (Multi-Attack)", "label_en": "Multi-Attack",
        "desc_pt": "Pode atacar multiplos alvos em Close ou concentrar no mesmo alvo.",
    },
    "armor_piercing": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Perfurar Armadura (Armor-Piercing)", "label_en": "Armor-Piercing",
        "desc_pt": "Ignora ranks de Impervious do alvo.",
    },
    "penetrating": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Penetrante (Penetrating)", "label_en": "Penetrating",
        "desc_pt": "Ranks de Penetrating ignoram Impervious Toughness.",
    },

    # ── Acao/Reacao (generico) ──
    "counter": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "Contra-Ataque (Counter)", "label_en": "Counter",
        "desc_pt": "Pode ser usado como reacao para anular efeito similar do oponente.",
    },
    "reaction": {
        "cost": 3, "cost_type": "per_rank",
        "label_pt": "Reacao (Reaction)", "label_en": "Reaction",
        "desc_pt": "Ativa automaticamente ao gatilho (ex: ao ser atingido), sem usar acao.",
    },
    "triggered": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Gatilho (Triggered)", "label_en": "Triggered",
        "desc_pt": "Ativa quando uma condicao especifica e satisfeita.",
    },

    # ── Duracao/Persistencia (generico) ──
    "secondary_effect": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Efeito Secundario (Secondary Effect)", "label_en": "Secondary Effect",
        "desc_pt": "O efeito se aplica novamente no turno seguinte automaticamente.",
    },
    "continuous": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Continuo (Continuous)", "label_en": "Continuous",
        "desc_pt": "O efeito persiste sem necessidade de concentracao.",
    },
    "increased_duration": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Duracao Aumentada (Increased Duration)", "label_en": "Increased Duration",
        "desc_pt": "Aumenta a duracao do efeito em 1 nivel (instantaneo->concentracao->sustentado->continuo).",
    },
    "contagious": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Contagioso (Contagious)", "label_en": "Contagious",
        "desc_pt": "O efeito se espalha para outros que tocam o alvo afetado.",
    },

    # ── Alvo (generico) ──
    "affects_others": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Afeta Outros (Affects Others)", "label_en": "Affects Others",
        "desc_pt": "Permite aplicar efeitos pessoais em outros alvos.",
    },
    "affects_insubstantial": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Afeta Intangiveis", "label_en": "Affects Insubstantial",
        "desc_pt": "O efeito funciona contra alvos intangiveis.",
        "max_ranks": 2,
    },
    "affects_objects": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Afeta Objetos (Affects Objects)", "label_en": "Affects Objects",
        "desc_pt": "O efeito funciona em objetos alem de criaturas.",
    },

    # ── Outros (generico) ──
    "incurable": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Incuravel (Incurable)", "label_en": "Incurable",
        "desc_pt": "O efeito nao pode ser curado por Healing normal.",
    },
    "reversible": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Reversivel (Reversible)", "label_en": "Reversible",
        "desc_pt": "Permite desfazer o efeito aplicado voluntariamente.",
    },
    "subtle": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Sutil (Subtle)", "label_en": "Subtle",
        "desc_pt": "Dificil de perceber. 1 rank = sutil, 2 ranks = indetectavel.",
        "max_ranks": 2,
    },
    "variable_descriptor": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Descritor Variavel (Variable Descriptor)", "label_en": "Variable Descriptor",
        "desc_pt": "Pode mudar o tipo de dano/efeito (fogo, gelo, eletrico, etc.).",
        "max_ranks": 2,
    },
    "alternate_resistance": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Resistencia Alternativa (Alternate Resistance)", "label_en": "Alternate Resistance",
        "desc_pt": "O alvo resiste com um atributo diferente do padrao.",
    },
    "alternate_effect": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Efeito Alternativo (Alternate Effect)", "label_en": "Alternate Effect",
        "desc_pt": "Compartilha custo com outro poder; so um ativo por vez.",
        "max_ranks": 2,
    },
    "linked": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "Vinculado (Linked)", "label_en": "Linked",
        "desc_pt": "Encadeia este efeito com outro, ambos ativam juntos.",
    },
    "innate": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Inato (Innate)", "label_en": "Innate",
        "desc_pt": "Parte da natureza do Pokemon. Nao pode ser anulado.",
    },
    "precise": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Preciso (Precise)", "label_en": "Precise",
        "desc_pt": "Controle fino dos detalhes do efeito.",
    },
    "ricochet": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Ricochete (Ricochet)", "label_en": "Ricochet",
        "desc_pt": "O ataque quica em superficies, permite atacar de angulos diferentes.",
    },
    "homing": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Rastreador (Homing)", "label_en": "Homing",
        "desc_pt": "Tenta atingir novamente no proximo turno se errar.",
    },
    "split": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Dividir (Split)", "label_en": "Split",
        "desc_pt": "Divide ranks do efeito entre multiplos alvos.",
    },

    # ==================================================================
    # EXTRAS EXCLUSIVOS POR PODER
    # ==================================================================

    # ── Damage ──
    "strength_based": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "Baseado em Forca (Strength-based)", "label_en": "Strength-based",
        "desc_pt": "Adiciona Stgr ao dano. Reduz custo base pois depende do atributo.",
        "compatible_effects": ["damage"],
    },

    # ── Affliction ──
    "extra_condition": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Condicao Extra (Extra Condition)", "label_en": "Extra Condition",
        "desc_pt": "Impoe uma condicao adicional em cada grau de falha (2 condicoes por grau).",
        "compatible_effects": ["affliction"],
    },
    "cumulative": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Cumulativo (Cumulative)", "label_en": "Cumulative",
        "desc_pt": "Usos repetidos acumulam o efeito em vez de substitui-lo.",
        "compatible_effects": ["affliction", "weaken", "damage", "mind_reading"],
    },
    "progressive": {
        "cost": 2, "cost_type": "per_rank",
        "label_pt": "Progressivo (Progressive)", "label_en": "Progressive",
        "desc_pt": "O efeito avanca automaticamente de grau a cada turno sem novo ataque.",
        "compatible_effects": ["affliction", "weaken"],
    },

    # ── Weaken ──
    "broad": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Amplo (Broad)", "label_en": "Broad",
        "desc_pt": "Afeta qualquer atributo de um grupo amplo (ex: todas Abilities), um por vez.",
        "compatible_effects": ["weaken", "nullify"],
    },
    "simultaneous": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Simultaneo (Simultaneous)", "label_en": "Simultaneous",
        "desc_pt": "Combinado com Broad, afeta todos os atributos do grupo ao mesmo tempo.",
        "compatible_effects": ["weaken", "nullify", "remote_sensing"],
    },

    # ── Nullify ──
    "effortless": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Sem Esforco (Effortless)", "label_en": "Effortless",
        "desc_pt": "Remove a necessidade de esforco extra apos falhar por 2+ graus.",
        "compatible_effects": ["nullify", "mind_reading"],
    },
    "randomize": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "Aleatorizar (Randomize)", "label_en": "Randomize",
        "desc_pt": "Efeitos anulados ficam descontrolados em vez de cancelados.",
        "compatible_effects": ["nullify"],
    },

    # ── Move Object ──
    "damaging": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Danificante (Damaging)", "label_en": "Damaging",
        "desc_pt": "Pode causar dano igual ao rank diretamente (esmagar, arremessar).",
        "compatible_effects": ["move_object"],
    },

    # ── Deflect ──
    "reflect": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Refletir (Reflect)", "label_en": "Reflect",
        "desc_pt": "Reflete ataques desviados de volta ao atacante.",
        "compatible_effects": ["deflect", "immunity"],
    },
    "redirect": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Redirecionar (Redirect)", "label_en": "Redirect",
        "desc_pt": "Redireciona ataques desviados para qualquer alvo no alcance. Requer Reflect.",
        "compatible_effects": ["deflect", "immunity"],
    },

    # ── Mind Reading ──
    "sensory_link": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Vinculo Sensorial (Sensory Link)", "label_en": "Sensory Link",
        "desc_pt": "Percebe atraves dos sentidos do alvo (seus proprios sentidos ficam inativos).",
        "compatible_effects": ["mind_reading"],
    },

    # ── Healing ──
    "energizing": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Energizante (Energizing)", "label_en": "Energizing",
        "desc_pt": "Pode curar condicoes de Fatigado e Exausto alem de dano.",
        "compatible_effects": ["healing"],
    },
    "persistent_healing": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Persistente (Persistent)", "label_en": "Persistent",
        "desc_pt": "Cura persiste; condicoes curadas nao retornam.",
        "compatible_effects": ["healing", "regeneration"],
    },
    "restorative": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Restaurador (Restorative)", "label_en": "Restorative",
        "desc_pt": "Pode restaurar ranks de atributos removidos por Weaken.",
        "compatible_effects": ["healing"],
    },
    "resurrection": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Ressurreicao (Resurrection)", "label_en": "Resurrection",
        "desc_pt": "Pode trazer de volta a vida (morto ha menos de [rank] minutos).",
        "compatible_effects": ["healing"],
    },
    "stabilize": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Estabilizar (Stabilize)", "label_en": "Stabilize",
        "desc_pt": "Pode estabilizar personagens morrendo.",
        "compatible_effects": ["healing"],
    },

    # ── Protection ──
    "impervious": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Impervio (Impervious)", "label_en": "Impervious",
        "desc_pt": "Ignora efeitos com rank menor que o rank de Impervious.",
        "compatible_effects": ["protection", "immunity"],
    },
    "sustained_protection": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "Sustentado (Sustained)", "label_en": "Sustained",
        "desc_pt": "Muda de permanente para sustentado (pode ser ligado/desligado).",
        "compatible_effects": ["protection"],
    },

    # ── Regeneration ──
    # persistent_healing ja cobre regeneration (compatible_effects inclui regeneration)

    # ── Teleport ──
    "tp_accurate": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Preciso (Accurate - Teleport)", "label_en": "Accurate (Teleport)",
        "desc_pt": "Nao precisa sentir o destino com precisao; descricao geral basta.",
        "compatible_effects": ["teleport"],
    },
    "change_direction": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Mudar Direcao (Change Direction)", "label_en": "Change Direction",
        "desc_pt": "Pode mudar a orientacao apos teleportar.",
        "compatible_effects": ["teleport"],
    },
    "change_velocity": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Mudar Velocidade (Change Velocity)", "label_en": "Change Velocity",
        "desc_pt": "Chega parado (sem dano de queda ao teleportar no ar).",
        "compatible_effects": ["teleport"],
    },
    "easy": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Facil (Easy)", "label_en": "Easy",
        "desc_pt": "Evita ficar atordoado/vulneravel durante teleportes estendidos.",
        "compatible_effects": ["teleport"],
    },
    "extended_teleport": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Estendido (Extended - Teleport)", "label_en": "Extended (Teleport)",
        "desc_pt": "Duas acoes de movimento para teleporte de rank+8 distancia (causa atordoamento).",
        "compatible_effects": ["teleport"],
    },
    "increased_mass": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Massa Aumentada (Increased Mass)", "label_en": "Increased Mass",
        "desc_pt": "Carrega massa adicional igual ao rank do extra.",
        "compatible_effects": ["teleport"],
    },
    "portal": {
        "cost": 2, "cost_type": "per_rank",
        "label_pt": "Portal (Portal)", "label_en": "Portal",
        "desc_pt": "Cria um portal de 1,5m mantido por concentracao.",
        "compatible_effects": ["teleport"],
    },
    "turnabout": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Ida e Volta (Turnabout)", "label_en": "Turnabout",
        "desc_pt": "Teleporta, faz acao padrao, teleporta de volta no mesmo turno.",
        "compatible_effects": ["teleport"],
    },

    # ── Flight ──
    "aquatic_flight": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Aquatico (Aquatic Flight)", "label_en": "Aquatic",
        "desc_pt": "Permite movimento subaquatico com velocidade de Flight rank -2.",
        "compatible_effects": ["flight"],
    },

    # ── Burrowing ──
    "burrow_penetrating": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Penetrante (Penetrating - Burrowing)", "label_en": "Penetrating (Burrowing)",
        "desc_pt": "Permite escavar materiais super-duros normalmente impenetraveis.",
        "compatible_effects": ["burrowing"],
    },

    # ── Mind Reading (extras adicionais) ──
    # cumulative e effortless ja incluem mind_reading em compatible_effects

    # ── Nullify (extras adicionais) ──
    # broad, simultaneous e effortless ja incluem nullify

    # ── Communication ──
    "rapid": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Rapido (Rapid)", "label_en": "Rapid",
        "desc_pt": "Comunicacao 10x mais rapida por rank (10x, 100x, 1000x...).",
        "compatible_effects": ["communication"],
    },

    # ── Illusion ──
    "independent_illusion": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Independente (Independent)", "label_en": "Independent",
        "desc_pt": "Ilusoes ativas requerem apenas acao livre para manter em vez de padrao.",
        "compatible_effects": ["illusion"],
    },

    # ── Concealment ──
    "concealment_attack": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "Ataque (Attack - Concealment)", "label_en": "Attack (Concealment)",
        "desc_pt": "Impoe ocultacao em alvos involuntarios (raio de invisibilidade, campo de trevas).",
        "compatible_effects": ["concealment"],
    },

    # ── Insubstantial ──
    "affects_corporeal": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Afeta Corporeos (Affects Corporeal)", "label_en": "Affects Corporeal",
        "desc_pt": "Efeitos funcionam em alvos corporeos enquanto incorporeo.",
        "compatible_effects": ["insubstantial"],
    },

    # ── Create ──
    "movable": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Movivel (Movable)", "label_en": "Movable",
        "desc_pt": "Move objetos criados via Move Object com rank igual ao Create.",
        "compatible_effects": ["create"],
    },
    "stationary": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "Estacionario (Stationary)", "label_en": "Stationary",
        "desc_pt": "Objetos criados flutuam imoveis no ar, resistem movimento com Strength.",
        "compatible_effects": ["create"],
    },
    "tether": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Amarra (Tether)", "label_en": "Tether",
        "desc_pt": "Conexao permite usar sua propria Strength para mover objetos criados.",
        "compatible_effects": ["create"],
    },

    # ── Remote Sensing ──
    "no_conduit": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Sem Condutor (No Conduit)", "label_en": "No Conduit",
        "desc_pt": "Efeitos sensoriais no local remoto nao afetam voce.",
        "compatible_effects": ["remote_sensing"],
    },

    # ── Morph ──
    "metamorph": {
        "cost": 1, "cost_type": "flat_per_rank",
        "label_pt": "Metamorfo (Metamorph)", "label_en": "Metamorph",
        "desc_pt": "Mantem formas alternativas completas com conjuntos de atributos diferentes.",
        "compatible_effects": ["morph"],
    },

    # ── Shrinking ──
    "atomic": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Atomico (Atomic)", "label_en": "Atomic",
        "desc_pt": "Em Shrinking 20, encolhe a nivel molecular/atomico, passa por objetos solidos.",
        "compatible_effects": ["shrinking"],
    },
    "normal_strength": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Forca Normal (Normal Strength)", "label_en": "Normal Strength",
        "desc_pt": "Mantem Strength, Speed e Intimidation normais enquanto encolhido.",
        "compatible_effects": ["shrinking"],
    },

    # ── Summon ──
    "active_summon": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Ativo (Active)", "label_en": "Active",
        "desc_pt": "Lacaios sao independentes; nao ficam atordoados, tem acoes completas.",
        "compatible_effects": ["summon"],
    },
    "controlled_summon": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Controlado (Controlled)", "label_en": "Controlled",
        "desc_pt": "Lacaios tem condicao controlada com direcao completa.",
        "compatible_effects": ["summon"],
    },
    "heroic_summon": {
        "cost": 2, "cost_type": "per_rank",
        "label_pt": "Heroico (Heroic)", "label_en": "Heroic",
        "desc_pt": "Criaturas invocadas funcionam como NPCs completos, nao lacaios.",
        "compatible_effects": ["summon"],
    },
    "horde": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Horda (Horde)", "label_en": "Horde",
        "desc_pt": "Invoca multiplos lacaios simultaneamente como acao padrao.",
        "compatible_effects": ["summon"],
    },
    "mental_link": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Vinculo Mental (Mental Link)", "label_en": "Mental Link",
        "desc_pt": "Comunicacao telepatica e comando com lacaios.",
        "compatible_effects": ["summon"],
    },
    "multiple_minions": {
        "cost": 2, "cost_type": "per_rank",
        "label_pt": "Multiplos Lacaios (Multiple Minions)", "label_en": "Multiple Minions",
        "desc_pt": "Cada aplicacao dobra a contagem maxima de lacaios.",
        "compatible_effects": ["summon"],
    },
    "sacrifice": {
        "cost": 1, "cost_type": "flat",
        "label_pt": "Sacrificio (Sacrifice)", "label_en": "Sacrifice",
        "desc_pt": "Redireciona efeitos de resistencia recebidos para um lacaio no alcance.",
        "compatible_effects": ["summon"],
    },
    "variable_type": {
        "cost": 1, "cost_type": "per_rank",
        "label_pt": "Tipo Variavel (Variable Type)", "label_en": "Variable Type",
        "desc_pt": "Invoca variedades diferentes de lacaios (+1 tipo geral, +2 amplo).",
        "compatible_effects": ["summon"],
    },
}

# =============================================
# FALHAS (reduzem o custo)
# =============================================
FLAWS = {
    # ==================================================================
    # GENERICAS (disponiveis para qualquer efeito)
    # ==================================================================
    "limited": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado (Limited)", "label_en": "Limited",
        "desc_pt": "Uso restrito a certas condicoes (descreva a limitacao).",
        "has_description": True,
    },
    "grab_based": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Baseado em Agarrar (Grab-Based)", "label_en": "Grab-Based",
        "desc_pt": "Deve estar agarrando o alvo para usar o efeito.",
    },
    "tiring": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Cansativo (Tiring)", "label_en": "Tiring",
        "desc_pt": "Causa 1 nivel de fadiga ao Pokemon cada vez que usa.",
    },
    "unreliable": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Nao Confiavel (Unreliable)", "label_en": "Unreliable",
        "desc_pt": "Chance de falhar (50%, ou rolagem necessaria).",
    },
    "concentration": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Concentracao (Concentration)", "label_en": "Concentration",
        "desc_pt": "Deve manter concentracao cada turno para sustentar o efeito.",
    },
    "reduced_range": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Alcance Reduzido (Reduced Range)", "label_en": "Reduced Range",
        "desc_pt": "Alcance menor que o padrao. Ranged para Close ou Close para Pessoal.",
    },
    "side_effect": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Efeito Colateral (Side Effect)", "label_en": "Side Effect",
        "desc_pt": "Causa uma consequencia negativa ao proprio Pokemon.",
        "has_description": True,
    },
    "sense_dependent": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Dependente de Sentido (Sense Dependent)", "label_en": "Sense Dependent",
        "desc_pt": "So funciona se o alvo puder perceber o ataque (visao, audicao, etc.).",
    },
    "activation_move": {
        "cost": -1, "cost_type": "flat",
        "label_pt": "Ativacao: Acao de Movimento", "label_en": "Activation (Move)",
        "desc_pt": "Requer uma acao de movimento para ativar antes de usar.",
    },
    "activation_standard": {
        "cost": -2, "cost_type": "flat",
        "label_pt": "Ativacao: Acao Padrao", "label_en": "Activation (Standard)",
        "desc_pt": "Requer uma acao padrao completa para ativar antes de usar.",
    },
    "feedback": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Retroalimentacao (Feedback)", "label_en": "Feedback",
        "desc_pt": "Se o efeito for anulado/destruido, causa dano ao Pokemon.",
    },
    "uncontrollable": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Incontrolavel (Uncontrollable)", "label_en": "Uncontrollable",
        "desc_pt": "O Pokemon nao consegue impedir o efeito de ativar.",
    },
    "removable": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Removivel (Removable)", "label_en": "Removable",
        "desc_pt": "O efeito pode ser retirado ou desativado por outros.",
    },
    "distracting": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Distrativo (Distracting)", "label_en": "Distracting",
        "desc_pt": "O Pokemon fica vulneravel enquanto usa o poder.",
    },
    "check_required": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Teste Necessario (Check Required)", "label_en": "Check Required",
        "desc_pt": "Deve passar num teste antes de usar o poder.",
        "has_description": True,
    },
    "diminished_range": {
        "cost": -1, "cost_type": "flat_per_rank",
        "label_pt": "Alcance Diminuido (Diminished Range)", "label_en": "Diminished Range",
        "desc_pt": "Reduz incrementos de alcance por rank.",
    },
    "increased_action": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Acao Aumentada (Increased Action)", "label_en": "Increased Action",
        "desc_pt": "Requer uma acao maior para usar (free para move, move para standard, etc.).",
    },
    "fades": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Enfraquece (Fades)", "label_en": "Fades",
        "desc_pt": "O efeito perde 1 rank por turno ate desaparecer.",
    },
    "resistible": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Resistivel (Resistible)", "label_en": "Resistible",
        "desc_pt": "O efeito pode ser resistido com um teste adicional.",
    },

    # ── Custom (generico) ──
    "custom_flaw_free": {
        "cost": 0, "cost_type": "per_rank",
        "label_pt": "[Flaw Custom 0/r: (texto livre)]", "label_en": "Custom Flaw",
        "desc_pt": "Falha customizada sem reducao. Descreva a restricao.",
        "has_description": True, "group": "custom",
    },
    "custom_flaw_minus1": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "[Flaw Custom -1/r: (texto livre)]", "label_en": "Custom Flaw -1/r",
        "desc_pt": "Falha customizada que reduz -1 por rank. Descreva a restricao.",
        "has_description": True, "group": "custom",
    },

    # ==================================================================
    # FALHAS EXCLUSIVAS POR PODER
    # ==================================================================

    # ── Affliction ──
    "limited_degree": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Grau Limitado (Limited Degree)", "label_en": "Limited Degree",
        "desc_pt": "O efeito so alcanca ate um certo grau de severidade.",
        "compatible_effects": ["affliction"],
    },
    "instant_recovery": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Recuperacao Instantanea (Instant Recovery)", "label_en": "Instant Recovery",
        "desc_pt": "O alvo se recupera automaticamente no final do turno, sem teste.",
        "compatible_effects": ["affliction"],
    },

    # ── Move Object ──
    "limited_direction": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Direcao Limitada (Limited Direction)", "label_en": "Limited Direction",
        "desc_pt": "Pode mover objetos apenas numa direcao especifica (cima/baixo, atrair/repelir).",
        "compatible_effects": ["move_object"],
    },
    "limited_material": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Material Limitado (Limited Material)", "label_en": "Limited Material",
        "desc_pt": "Pode mover apenas um tipo de material (metais, plantas, rochas, etc.).",
        "has_description": True,
        "compatible_effects": ["move_object"],
    },

    # ── Mind Reading ──
    "mr_limited_language": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado por Idioma (Limited by Language)", "label_en": "Limited by Language",
        "desc_pt": "So funciona se compartilhar um idioma comum com o alvo.",
        "compatible_effects": ["mind_reading"],
    },
    "mr_limited_emotions": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado a Emocoes (Limited to Emotions)", "label_en": "Limited to Emotions",
        "desc_pt": "Pode ler apenas conteudo emocional, nao pensamentos coerentes.",
        "compatible_effects": ["mind_reading"],
    },
    "mr_limited_sensory": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado a Vinculo Sensorial (Limited to Sensory Link)", "label_en": "Limited to Sensory Link",
        "desc_pt": "Permite apenas percepcao sensorial, nao leitura de pensamentos/memorias.",
        "compatible_effects": ["mind_reading"],
    },
    "mr_limited_surface": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado a Pensamentos Superficiais", "label_en": "Limited to Surface Thoughts",
        "desc_pt": "Restrito a pensamentos de nivel superficial apenas.",
        "compatible_effects": ["mind_reading"],
    },
    "mr_feedback": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Retroalimentacao Mental (Feedback - Mind Reading)", "label_en": "Feedback (Mind Reading)",
        "desc_pt": "Sofre dano se o alvo for ferido ou tiver pensamentos traumaticos.",
        "compatible_effects": ["mind_reading"],
    },

    # ── Healing ──
    "empathic": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Empatico (Empathic)", "label_en": "Empathic",
        "desc_pt": "Voce absorve as condicoes que cura do alvo para si mesmo.",
        "compatible_effects": ["healing"],
    },
    "temporary": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Temporario (Temporary)", "label_en": "Temporary",
        "desc_pt": "Beneficios duram apenas 1 hora, depois as condicoes curadas retornam.",
        "compatible_effects": ["healing"],
    },

    # ── Immunity ──
    "limited_half_effect": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado a Metade (Limited - Half Effect)", "label_en": "Limited (Half Effect)",
        "desc_pt": "Sofre metade do efeito normal em vez de imunidade total.",
        "compatible_effects": ["immunity"],
    },

    # ── Regeneration ──
    "source_dependent": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Dependente de Fonte (Source)", "label_en": "Source",
        "desc_pt": "So funciona com acesso a uma fonte especifica (sangue, luz solar, etc.).",
        "has_description": True,
        "compatible_effects": ["regeneration"],
    },

    # ── Teleport ──
    "tp_limited_extended": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado a Estendido (Limited to Extended)", "label_en": "Limited to Extended",
        "desc_pt": "Pode apenas fazer teleportes estendidos (requer Extra Extended).",
        "compatible_effects": ["teleport"],
    },
    "tp_medium": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Medio (Medium - Teleport)", "label_en": "Medium",
        "desc_pt": "Requer um meio especifico (fios, sombras, chamas, espelhos, etc.).",
        "has_description": True,
        "compatible_effects": ["teleport"],
    },

    # ── Flight ──
    "gliding": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Planar (Gliding)", "label_en": "Gliding",
        "desc_pt": "Distancia limitada a altura + rank de velocidade; precisa de correntes.",
        "compatible_effects": ["flight"],
    },
    "levitation": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Levitacao (Levitation)", "label_en": "Levitation",
        "desc_pt": "Apenas movimento vertical; deslocamento horizontal requer vento.",
        "compatible_effects": ["flight"],
    },
    "platform": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Plataforma (Platform)", "label_en": "Platform",
        "desc_pt": "Voo depende de plataforma em pe; pode ser interrompido.",
        "compatible_effects": ["flight"],
    },
    "wings": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Asas (Wings)", "label_en": "Wings",
        "desc_pt": "Asas fisicas que podem ser presas ou restringidas.",
        "compatible_effects": ["flight"],
    },

    # ── Leaping ──
    "full_power_only": {
        "cost": -1, "cost_type": "flat",
        "label_pt": "Apenas Potencia Maxima (Full Power)", "label_en": "Full Power",
        "desc_pt": "Restrito a saltos de distancia maxima apenas; sem saltos curtos.",
        "compatible_effects": ["leaping"],
    },

    # ── Quickness ──
    "quick_limited_type": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado a Tipo (Limited - Physical/Mental)", "label_en": "Limited (One Type)",
        "desc_pt": "Aplica-se apenas a tarefas fisicas ou mentais, nao ambas.",
        "compatible_effects": ["quickness"],
    },
    "quick_limited_task": {
        "cost": -2, "cost_type": "per_rank",
        "label_pt": "Limitado a Tarefa (Limited - One Task)", "label_en": "Limited (One Task)",
        "desc_pt": "Aplica-se apenas a uma tarefa especifica (leitura, calculo, etc.).",
        "has_description": True,
        "compatible_effects": ["quickness"],
    },

    # ── Senses ──
    "senses_noticeable": {
        "cost": -1, "cost_type": "flat",
        "label_pt": "Perceptivel (Noticeable - Senses)", "label_en": "Noticeable",
        "desc_pt": "Sentidos sao particularmente visiveis (olhos brilhantes, som emitido, etc.).",
        "compatible_effects": ["senses"],
    },

    # ── Remote Sensing ──
    "rs_feedback": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Retroalimentacao (Feedback - Remote Sensing)", "label_en": "Feedback (Remote Sensing)",
        "desc_pt": "Ataques no ponto sensorial remoto podem afeta-lo.",
        "compatible_effects": ["remote_sensing"],
    },
    "rs_medium": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Medio (Medium - Remote Sensing)", "label_en": "Medium (Remote Sensing)",
        "desc_pt": "Requer um meio especifico (sombras, chamas, espelhos, agua).",
        "has_description": True,
        "compatible_effects": ["remote_sensing"],
    },

    # ── Illusion ──
    "ill_limited_subject": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Limitado a Um Sujeito (Limited to One Subject)", "label_en": "Limited to One Subject",
        "desc_pt": "Apenas um unico alvo por vez pode perceber sua ilusao.",
        "compatible_effects": ["illusion"],
    },

    # ── Concealment ──
    "blending": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Camuflagem (Blending)", "label_en": "Blending",
        "desc_pt": "So funciona enquanto se move a velocidade terrestre rank-1 ou menor.",
        "compatible_effects": ["concealment"],
    },
    "partial_concealment": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Parcial (Partial)", "label_en": "Partial",
        "desc_pt": "Fornece ocultacao parcial em vez de total.",
        "compatible_effects": ["concealment"],
    },
    "passive_concealment": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Passivo (Passive)", "label_en": "Passive",
        "desc_pt": "Ocultacao termina ao realizar um ataque ou teste de efeito.",
        "compatible_effects": ["concealment"],
    },

    # ── Insubstantial ──
    "absent_strength": {
        "cost": -1, "cost_type": "flat",
        "label_pt": "Forca Ausente (Absent Strength)", "label_en": "Absent Strength",
        "desc_pt": "Remove Strength efetiva na forma Insubstantial rank 1.",
        "compatible_effects": ["insubstantial"],
    },

    # ── Create ──
    "create_feedback": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Retroalimentacao (Feedback - Create)", "label_en": "Feedback (Create)",
        "desc_pt": "Sofre dano quando objetos criados recebem dano.",
        "compatible_effects": ["create"],
    },
    "proportional": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Proporcional (Proportional)", "label_en": "Proportional",
        "desc_pt": "Volume rank + Toughness rank deve ser igual ao rank de Create (troca).",
        "compatible_effects": ["create"],
    },

    # ── Summon ──
    "attitude": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Atitude (Attitude)", "label_en": "Attitude",
        "desc_pt": "Lacaios tem disposicao indiferente ou hostil, requerem persuasao. -1 indiferente, -2 hostil.",
        "compatible_effects": ["summon"],
    },

    # ── Variable ──
    "slow_variable": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Lento (Slow)", "label_en": "Slow",
        "desc_pt": "Pode reconfigurar apenas fora de acao (lab, livro, equipamento).",
        "compatible_effects": ["variable"],
    },

    # ── Enhanced Trait ──
    "reduced_trait": {
        "cost": -1, "cost_type": "per_rank",
        "label_pt": "Atributo Reduzido (Reduced Trait)", "label_en": "Reduced Trait",
        "desc_pt": "Um ou mais atributos diminuem enquanto outros sao aumentados.",
        "has_description": True,
        "compatible_effects": ["enhanced_trait"],
    },
}

# =============================================
# CONDICOES DE AFLICAO (3 graus)
# =============================================
AFFLICTION_CONDITIONS = {
    "degree_1": {
        "label_pt": "Grau 1 (Leve)",
        "options": [
            {"key": "dazed",      "label_en": "Dazed",      "label_pt": "Atordoado (Dazed)",      "desc_pt": "So pode fazer uma acao free por turno."},
            {"key": "fatigued",   "label_en": "Fatigued",   "label_pt": "Fatigado (Fatigued)",    "desc_pt": "-1 em todos os checks."},
            {"key": "hindered",   "label_en": "Hindered",   "label_pt": "Impedido (Hindered)",    "desc_pt": "Velocidade reduzida pela metade."},
            {"key": "impaired",   "label_en": "Impaired",   "label_pt": "Prejudicado (Impaired)", "desc_pt": "-2 em checks com o atributo afetado."},
            {"key": "vulnerable", "label_en": "Vulnerable", "label_pt": "Vulneravel (Vulnerable)", "desc_pt": "Defesas ativas reduzidas pela metade."},
        ],
    },
    "degree_2": {
        "label_pt": "Grau 2 (Moderado)",
        "options": [
            {"key": "stunned",     "label_en": "Stunned",     "label_pt": "Aturdido (Stunned)",       "desc_pt": "Nao pode fazer acoes."},
            {"key": "exhausted",   "label_en": "Exhausted",   "label_pt": "Exausto (Exhausted)",      "desc_pt": "-5 em checks; velocidade reduzida."},
            {"key": "immobile",    "label_en": "Immobile",    "label_pt": "Imovel (Immobile)",        "desc_pt": "Nao pode se mover."},
            {"key": "disabled",    "label_en": "Disabled",    "label_pt": "Desabilitado (Disabled)",  "desc_pt": "-5 em checks com atributo afetado."},
            {"key": "defenseless", "label_en": "Defenseless", "label_pt": "Indefeso (Defenseless)",   "desc_pt": "Defesas ativas caem para 0."},
            {"key": "compelled",   "label_en": "Compelled",   "label_pt": "Compelido (Compelled)",    "desc_pt": "Deve obedecer comandos simples."},
        ],
    },
    "degree_3": {
        "label_pt": "Grau 3 (Severo)",
        "options": [
            {"key": "incapacitated", "label_en": "Incapacitated", "label_pt": "Incapacitado",           "desc_pt": "Fora de combate totalmente."},
            {"key": "asleep",        "label_en": "Asleep",        "label_pt": "Adormecido (Asleep)",     "desc_pt": "Indefeso e inconsciente."},
            {"key": "paralyzed",     "label_en": "Paralyzed",     "label_pt": "Paralisado (Paralyzed)",  "desc_pt": "Imovel e Indefeso, mas consciente."},
            {"key": "transformed",   "label_en": "Transformed",   "label_pt": "Transformado",            "desc_pt": "Alteracao fisica fundamental."},
            {"key": "controlled",    "label_en": "Controlled",    "label_pt": "Controlado (Controlled)", "desc_pt": "Sob controle total do atacante."},
            {"key": "unaware",       "label_en": "Unaware",       "label_pt": "Inconsciente (Unaware)",  "desc_pt": "Nao tem consciencia do ambiente."},
        ],
    },
}

# =============================================
# ALVOS DE ATRIBUTO (Weaken / Enhanced Trait)
# =============================================
STAT_TARGETS = [
    {"key": "stgr",      "label_en": "Stgr",      "label_pt": "Forca (Stgr)"},
    {"key": "int",       "label_en": "Int",        "label_pt": "Intelecto (Int)"},
    {"key": "dodge",     "label_en": "Dodge",      "label_pt": "Esquiva (Dodge)"},
    {"key": "parry",     "label_en": "Parry",      "label_pt": "Aparar (Parry)"},
    {"key": "will",      "label_en": "Will",       "label_pt": "Vontade (Will)"},
    {"key": "fortitude", "label_en": "Fortitude",  "label_pt": "Fortitude"},
    {"key": "toughness", "label_en": "Toughness",  "label_pt": "Resistencia (Toughness)"},
    {"key": "initiative","label_en": "Initiative",  "label_pt": "Iniciativa (Initiative)"},
    {"key": "speed",     "label_en": "Speed",      "label_pt": "Velocidade (Speed)"},
]

RESIST_STATS = ["Toughness", "Fortitude", "Will", "Dodge", "Parry"]

# =============================================
# IMMUNITY OPTIONS  (chave, label, custo em PP)
# rank de Immunity = soma dos custos dos itens escolhidos
# =============================================
IMMUNITY_OPTIONS = [
    # ── 1 PP cada ──
    {"key": "sleep",           "label_pt": "Sono (Sleep)",                  "cost": 1},
    {"key": "poison",          "label_pt": "Veneno (Poison)",               "cost": 1},
    {"key": "disease",         "label_pt": "Doença (Disease)",              "cost": 1},
    {"key": "aging",           "label_pt": "Envelhecimento (Aging)",        "cost": 1},
    {"key": "starvation",      "label_pt": "Fome e Sede (Starvation)",      "cost": 1},
    {"key": "cold_env",        "label_pt": "Frio Ambiental",                "cost": 1},
    {"key": "heat_env",        "label_pt": "Calor Ambiental",               "cost": 1},
    {"key": "pressure",        "label_pt": "Alta Pressão",                  "cost": 1},
    {"key": "radiation_env",   "label_pt": "Radiação Ambiental",            "cost": 1},
    {"key": "vacuum",          "label_pt": "Vácuo",                         "cost": 1},
    {"key": "suffocation_one", "label_pt": "Um Tipo de Asfixia",            "cost": 1},
    # ── 2 PP cada ──
    {"key": "critical_hits",   "label_pt": "Acertos Críticos",              "cost": 2},
    {"key": "suffocation_all", "label_pt": "Asfixia Total",                 "cost": 2},
    {"key": "uncommon_desc",   "label_pt": "Descritor Incomum (especificar)","cost": 2},
    # ── 5 PP cada ──
    {"key": "alteration",      "label_pt": "Efeitos de Alteração",          "cost": 5},
    {"key": "sense_affliction","label_pt": "Aflições Sensoriais",           "cost": 5},
    {"key": "emotion",         "label_pt": "Efeitos Emocionais",            "cost": 5},
    {"key": "traps",           "label_pt": "Armadilhas / Agarrar",          "cost": 5},
    {"key": "fatigue",         "label_pt": "Efeitos de Fadiga",             "cost": 5},
    {"key": "interaction",     "label_pt": "Perícias de Interação",         "cost": 5},
    {"key": "fire_dmg",        "label_pt": "Dano de Fogo (Fire Damage)",    "cost": 5},
    {"key": "cold_dmg",        "label_pt": "Dano de Frio (Cold Damage)",    "cost": 5},
    {"key": "electric_dmg",    "label_pt": "Dano Elétrico",                 "cost": 5},
    {"key": "sonic_dmg",       "label_pt": "Dano Sônico (Sonic Damage)",    "cost": 5},
    {"key": "magic_dmg",       "label_pt": "Dano Mágico (Magic Damage)",    "cost": 5},
    {"key": "falling_dmg",     "label_pt": "Dano de Queda (Falling Damage)","cost": 5},
    # ── 10 PP cada ──
    {"key": "life_support",    "label_pt": "Suporte de Vida (Life Support)","cost": 10},
    {"key": "common_fire",     "label_pt": "Fogo – Descritor Completo",     "cost": 10},
    {"key": "common_cold",     "label_pt": "Frio – Descritor Completo",     "cost": 10},
    {"key": "common_electric", "label_pt": "Eletricidade – Descritor Completo","cost": 10},
    {"key": "common_radiation","label_pt": "Radiação – Descritor Completo", "cost": 10},
    # ── 20 PP cada ──
    {"key": "bludgeoning",     "label_pt": "Impacto Físico (Bludgeoning)",  "cost": 20},
    {"key": "energy",          "label_pt": "Energia (Energy)",              "cost": 20},
    # ── 30 PP cada ──
    {"key": "all_fortitude",   "label_pt": "Todos os Efeitos Fortitude",    "cost": 30},
    {"key": "all_will",        "label_pt": "Todos os Efeitos Vontade",      "cost": 30},
    # ── Personalizado (custo definido pelo usuário) ──
    {"key": "custom",          "label_pt": "Personalizado…",                "cost": 0},
]

# =============================================
# SENSES OPTIONS  (chave, label, custo em PP)
# rank de Senses = soma dos custos dos sentidos escolhidos
# Sentidos com per_rank=True permitem adicionar ranks extras
# =============================================
SENSES_OPTIONS = [
    # ── 1 PP cada (fixo) ──
    {"key": "danger_sense",    "label_pt": "Sentido de Perigo (Danger Sense)",     "cost": 1, "per_rank": False},
    {"key": "direction_sense", "label_pt": "Senso de Direção (Direction Sense)",   "cost": 1, "per_rank": False},
    {"key": "distance_sense",  "label_pt": "Senso de Distância (Distance Sense)",  "cost": 1, "per_rank": False},
    {"key": "low_light",       "label_pt": "Visão com Pouca Luz (Low-light Vision)","cost": 1, "per_rank": False},
    {"key": "infravision",     "label_pt": "Infravermelho (Infravision)",           "cost": 1, "per_rank": False},
    {"key": "ultra_hearing",   "label_pt": "Ultra-Audição (Ultra-hearing)",         "cost": 1, "per_rank": False},
    {"key": "ultra_vision",    "label_pt": "Ultra-Visão (Ultra-vision)",            "cost": 1, "per_rank": False},
    {"key": "tremorsense",     "label_pt": "Tremorsense / Vibração",                "cost": 1, "per_rank": False},
    {"key": "radius",          "label_pt": "Sentido Radial (Radius)",               "cost": 1, "per_rank": False},
    {"key": "ranged_touch",    "label_pt": "Toque à Distância (Ranged Touch)",      "cost": 1, "per_rank": False},
    {"key": "radio",           "label_pt": "Recepção de Rádio (Radio)",             "cost": 1, "per_rank": False},
    # ── 1 PP por rank (per_rank=True) ──
    {"key": "awareness",       "label_pt": "Percepção Especial (Awareness)",        "cost": 1, "per_rank": True},
    {"key": "acute",           "label_pt": "Sentido Aguçado (Acute)",               "cost": 1, "per_rank": True},
    {"key": "analytical",      "label_pt": "Sentido Analítico (Analytical)",        "cost": 1, "per_rank": True},
    {"key": "extended",        "label_pt": "Sentido Estendido (Extended)",           "cost": 1, "per_rank": True},
    {"key": "tracking",        "label_pt": "Rastreamento (Tracking)",               "cost": 1, "per_rank": True},
    {"key": "rapid",           "label_pt": "Sentido Rápido (Rapid)",                "cost": 1, "per_rank": True},
    {"key": "microscopic",     "label_pt": "Visão Microscópica (Microscopic Vision)","cost": 1, "per_rank": True},
    # ── 2 PP fixo ──
    {"key": "darkvision",           "label_pt": "Visão no Escuro (Darkvision)",             "cost": 2, "per_rank": False},
    {"key": "counters_concealment", "label_pt": "Ignora Ocultação (Counters Concealment)",  "cost": 2, "per_rank": False},
    {"key": "counters_illusions",   "label_pt": "Ignora Ilusões (Counters Illusions)",      "cost": 2, "per_rank": False},
    # ── 4 PP fixo ──
    {"key": "penetrates_concealment","label_pt": "Penetra Ocultação Total",                 "cost": 4, "per_rank": False},
    {"key": "postcognition",        "label_pt": "Pós-cognição (Postcognition)",             "cost": 4, "per_rank": False},
    {"key": "precognition",         "label_pt": "Pré-cognição (Precognition)",              "cost": 4, "per_rank": False},
    # ── Personalizado ──
    {"key": "custom",               "label_pt": "Personalizado…",                           "cost": 0, "per_rank": False},
]

# =============================================
# COMMUNICATION TYPES
# =============================================
COMMUNICATION_TYPES = [
    {"key": "mental",    "label_pt": "Mental (Telepatia)"},
    {"key": "auditory",  "label_pt": "Auditivo (Som / Ultrassom)"},
    {"key": "radio",     "label_pt": "Rádio (AM/FM/Ondas)"},
    {"key": "visual",    "label_pt": "Visual (Laser / Luz)"},
    {"key": "olfactory", "label_pt": "Olfativo (Feromônios)"},
    {"key": "tactile",   "label_pt": "Tátil (Vibração)"},
    {"key": "custom",    "label_pt": "Personalizado (Tipo de Sentido Customizado)"},
]

# =============================================
# MOVEMENT OPTIONS  (cada opção = 1 rank = 2 PP)
# =============================================
MOVEMENT_OPTIONS = [
    {"key": "env_adaptation",   "label_pt": "Adaptação Ambiental (subaquático, gravidade-zero, etc.)"},
    {"key": "wall_crawling",    "label_pt": "Escalar Paredes e Tetos (Wall Crawling)"},
    {"key": "swing",            "label_pt": "Balanço / Teia (Swing)"},
    {"key": "slide",            "label_pt": "Deslizamento Prono (Slide)"},
    {"key": "walk_water_stand", "label_pt": "Caminhar na Água – em pé (Walk on Water, 1 rank)"},
    {"key": "walk_water_prone", "label_pt": "Caminhar na Água – também deitado (Walk on Water, 2 ranks)"},
    {"key": "sure_footed",      "label_pt": "Terreno Difícil sem Penalidade (Sure-Footed)"},
    {"key": "trackless",        "label_pt": "Sem Rastros (Trackless)"},
    {"key": "space_travel_1",   "label_pt": "Viagem Espacial – outros planetas do sistema solar"},
    {"key": "space_travel_2",   "label_pt": "Viagem Espacial – outros sistemas estelares"},
    {"key": "space_travel_3",   "label_pt": "Viagem Espacial – galáxias distantes"},
    {"key": "time_travel_1",    "label_pt": "Viagem no Tempo – ponto fixo"},
    {"key": "time_travel_2",    "label_pt": "Viagem no Tempo – passado OU futuro"},
    {"key": "time_travel_3",    "label_pt": "Viagem no Tempo – qualquer época"},
]

# =============================================
# VANTAGENS DE COMBATE
# =============================================
COMBAT_ADVANTAGES = [
    {"key": "power_attack",       "label_en": "Power Attack",       "label_pt": "Ataque Poderoso",       "desc_pt": "Troca precisao (-2 a -5) por dano (+2 a +5).",               "has_ranks": False},
    {"key": "all_out_attack",     "label_en": "All-Out Attack",     "label_pt": "Ataque Total",          "desc_pt": "Troca defesa (-2 a -5) por bonus de ataque (+2 a +5).",     "has_ranks": False},
    {"key": "accurate_attack",    "label_en": "Accurate Attack",    "label_pt": "Ataque Preciso",        "desc_pt": "Troca dano (-2 a -5) por precisao (+2 a +5).",              "has_ranks": False},
    {"key": "defensive_attack",   "label_en": "Defensive Attack",   "label_pt": "Ataque Defensivo",      "desc_pt": "Troca ataque (-2 a -5) por defesa (+2 a +5).",              "has_ranks": False},
    {"key": "improved_critical",  "label_en": "Improved Critical",  "label_pt": "Critico Aprimorado",    "desc_pt": "Reduz o limiar de acerto critico em 1 por rank.",           "has_ranks": True, "max_ranks": 4},
    {"key": "improved_grab",      "label_en": "Improved Grab",      "label_pt": "Agarrar Aprimorado",    "desc_pt": "Pode agarrar com ataque corpo-a-corpo normal.",             "has_ranks": False},
    {"key": "improved_hold",      "label_en": "Improved Hold",      "label_pt": "Imobilizar Aprimorado", "desc_pt": "+5 para manter agarramento.",                               "has_ranks": False},
    {"key": "fast_grab",          "label_en": "Fast Grab",          "label_pt": "Agarrar Rapido",        "desc_pt": "Agarra como acao free apos ataque bem-sucedido.",           "has_ranks": False},
    {"key": "move_by_action",     "label_en": "Move-by Action",     "label_pt": "Acao em Movimento",     "desc_pt": "Pode se mover antes e depois de atacar no mesmo turno.",    "has_ranks": False},
    {"key": "takedown",           "label_en": "Takedown Attack",    "label_pt": "Abater (Takedown)",     "desc_pt": "Ataque livre ao derrubar alvo adjacente.",                  "has_ranks": True, "max_ranks": 2},
    {"key": "evasion",            "label_en": "Evasion",            "label_pt": "Evasao",                "desc_pt": "+2 Dodge contra efeitos de Area por rank.",                 "has_ranks": True, "max_ranks": 2},
    {"key": "uncanny_dodge",      "label_en": "Uncanny Dodge",      "label_pt": "Esquiva Instintiva",    "desc_pt": "Nao fica vulneravel quando surpreendido.",                  "has_ranks": False},
    {"key": "interpose",          "label_en": "Interpose",          "label_pt": "Interpor",              "desc_pt": "Recebe ataque no lugar de aliado adjacente.",               "has_ranks": False},
    {"key": "redirect",           "label_en": "Redirect",           "label_pt": "Redirecionar",          "desc_pt": "Redireciona ataque que errou para outro alvo adjacente.",   "has_ranks": False},
    {"key": "fearless",           "label_en": "Fearless",           "label_pt": "Destemido",             "desc_pt": "Imune a efeitos de medo e intimidacao.",                   "has_ranks": False},
    {"key": "taunt",              "label_en": "Taunt",              "label_pt": "Provocar (Taunt)",      "desc_pt": "Usa interacao para abaixar defesas do alvo.",               "has_ranks": False},
    {"key": "startle",            "label_en": "Startle",            "label_pt": "Sobressaltar",          "desc_pt": "Usa intimidacao para deixar o alvo vulneravel.",            "has_ranks": False},
    {"key": "improved_initiative", "label_en": "Improved Initiative", "label_pt": "Iniciativa Aprimorada", "desc_pt": "+4 em iniciativa por rank.",                              "has_ranks": True, "max_ranks": 4},
    {"key": "ranged_attack",      "label_en": "Ranged Attack",      "label_pt": "Ataque a Distancia",    "desc_pt": "+1 bonus de ataque ranged por rank.",                       "has_ranks": True, "max_ranks": 20},
    {"key": "close_attack",       "label_en": "Close Attack",       "label_pt": "Ataque Corpo-a-Corpo",  "desc_pt": "+1 bonus de ataque close por rank.",                        "has_ranks": True, "max_ranks": 20},
    {"key": "great_endurance",    "label_en": "Great Endurance",    "label_pt": "Grande Resistencia",    "desc_pt": "Vantagem em checks de fadiga e resistencia.",               "has_ranks": False},
    {"key": "diehard",            "label_en": "Diehard",            "label_pt": "Obstinado (Diehard)",   "desc_pt": "Estabiliza automaticamente quando moribundo.",              "has_ranks": False},
    {"key": "luck",               "label_en": "Luck",               "label_pt": "Sorte (Luck)",          "desc_pt": "Pontos de sorte para re-rolar ou forcar re-rolagem.",       "has_ranks": True, "max_ranks": 20},
]

# =============================================
# REGRAS DE ADAPTACAO POKEMON
# =============================================
ADAPTATION_RULES = {
    "area_max_rank": 4,
    "stat_boost_cap": 5,
    "hp_stages": 6,
    "linked_no_repeat": True,
    "min_cost_per_rank": 1,
    "category_map": {
        "fisico":  {"custom": "[Custom 0/r: Stgr Based]",    "label_pt": "Fisico",   "label_en": "Physical"},
        "especial": {"custom": "[Custom 0/r: Intelect Based]", "label_pt": "Especial", "label_en": "Special"},
        "status":  {"custom": "",                              "label_pt": "Status",   "label_en": "Status"},
    },
}

# =============================================
# DESCRITORES DE TIPO
# =============================================
TYPE_DESCRIPTORS = {
    "normal": "energia normal", "fire": "fogo", "water": "agua",
    "grass": "natureza/planta", "electric": "eletricidade", "ice": "gelo",
    "fighting": "forca de combate", "poison": "veneno", "ground": "terra",
    "flying": "vento/ar", "psychic": "energia psiquica", "bug": "poder inseto",
    "rock": "rocha", "ghost": "energia fantasma", "dragon": "energia draconica",
    "dark": "energia sombria", "steel": "metal/aco", "fairy": "energia feerica",
}

# =============================================
# KITS PRE-CONSTRUIDOS (Status / Clima)
# Cada kit e uma lista de componentes prontos
# para inserir no golpe com um clique
# =============================================
PRESET_KITS = {
    # ── Status Pokemon ──
    "poison": {
        "label_pt": "Veneno (Poison)",
        "label_en": "Poison",
        "desc_pt": "Dano progressivo: comeca com 2, dobra cada turno ate o rank do golpe. Resistido por Fortitude.",
        "components": [
            {
                "effect_key": "damage",
                "rank": 1,
                "is_linked": True,
                "extras": [
                    {"key": "alternate_resistance", "ranks": 1, "description": "Fortitude"},
                    {"key": "increased_duration", "ranks": 3, "description": ""},
                    {"key": "custom_free", "ranks": 1, "description": "Comeca com 2, dobra ate rank"},
                ],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                    {"key": "custom_flaw_minus1", "ranks": 1, "description": "Organismos vivos"},
                ],
                "conditions": None,
                "stat_targets": [],
                "resist_override": "Fortitude",
            },
        ],
    },
    "burn": {
        "label_pt": "Queimadura (Burn)",
        "label_en": "Burn",
        "desc_pt": "Dano fixo + reducao de Stgr. Linked ao dano principal.",
        "components": [
            {
                "effect_key": "damage",
                "rank": 1,
                "is_linked": True,
                "extras": [
                    {"key": "secondary_effect", "ranks": 1, "description": ""},
                ],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                ],
                "conditions": None,
                "stat_targets": [],
                "resist_override": "Fortitude",
            },
            {
                "effect_key": "weaken",
                "rank": 1,
                "is_linked": True,
                "extras": [],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                ],
                "conditions": None,
                "stat_targets": ["Stgr"],
                "resist_override": "Fortitude",
            },
        ],
    },
    "freeze": {
        "label_pt": "Congelamento (Freeze)",
        "label_en": "Freeze",
        "desc_pt": "Affliction (Dazed/Stunned/Transformed) + reducao de Int. Golpes de Fogo descongelam.",
        "components": [
            {
                "effect_key": "affliction",
                "rank": 1,
                "is_linked": True,
                "extras": [
                    {"key": "custom_free", "ranks": 1, "description": "Golpes de Fogo descongelam"},
                ],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                ],
                "conditions": {"degree_1": "Dazed", "degree_2": "Stunned", "degree_3": "Transformed"},
                "stat_targets": [],
                "resist_override": "Fortitude",
            },
            {
                "effect_key": "weaken",
                "rank": 1,
                "is_linked": True,
                "extras": [],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                ],
                "conditions": None,
                "stat_targets": ["Int"],
                "resist_override": "Fortitude",
            },
        ],
    },
    "paralysis": {
        "label_pt": "Paralisia (Paralysis)",
        "label_en": "Paralysis",
        "desc_pt": "Chance de flinch (Fatigued/Immobile/Paralyzed) + reducao de Speed e Initiative.",
        "components": [
            {
                "effect_key": "affliction",
                "rank": 1,
                "is_linked": True,
                "extras": [],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                ],
                "conditions": {"degree_1": "Fatigued", "degree_2": "Immobile", "degree_3": "Paralyzed"},
                "stat_targets": [],
                "resist_override": "Fortitude",
            },
            {
                "effect_key": "weaken",
                "rank": 1,
                "is_linked": True,
                "extras": [],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                ],
                "conditions": None,
                "stat_targets": ["Initiative", "Speed"],
                "resist_override": "Fortitude",
            },
        ],
    },
    "sleep": {
        "label_pt": "Sono (Sleep)",
        "label_en": "Sleep",
        "desc_pt": "Affliction (Fatigued/Exhausted/Asleep). Resistido por Will.",
        "components": [
            {
                "effect_key": "affliction",
                "rank": 1,
                "is_linked": True,
                "extras": [],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                ],
                "conditions": {"degree_1": "Fatigued", "degree_2": "Exhausted", "degree_3": "Asleep"},
                "stat_targets": [],
                "resist_override": "Will",
            },
        ],
    },
    "confusion": {
        "label_pt": "Confusao (Confusion)",
        "label_en": "Confusion",
        "desc_pt": "Affliction (Vulnerable/Defenseless) + Dano fixo ao entrar na affliction. Resistido por Will.",
        "components": [
            {
                "effect_key": "affliction",
                "rank": 1,
                "is_linked": True,
                "extras": [
                    {"key": "custom_free", "ranks": 1, "description": "Golpes de Foco descongelam"},
                ],
                "flaws": [],
                "conditions": {"degree_1": "Vulnerable", "degree_2": "Defenseless", "degree_3": ""},
                "stat_targets": [],
                "resist_override": "Will",
            },
            {
                "effect_key": "damage",
                "rank": 1,
                "is_linked": True,
                "extras": [
                    {"key": "secondary_effect", "ranks": 1, "description": ""},
                ],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Entrar na affliction"},
                ],
                "conditions": None,
                "stat_targets": [],
                "resist_override": None,
            },
        ],
    },
    "flinch": {
        "label_pt": "Flinch (Hesitar)",
        "label_en": "Flinch",
        "desc_pt": "Chance de hesitar: Affliction (Dazed/Stunned). Geralmente Limited a causar dano.",
        "components": [
            {
                "effect_key": "affliction",
                "rank": 1,
                "is_linked": True,
                "extras": [],
                "flaws": [
                    {"key": "limited", "ranks": 1, "description": "Causar dano"},
                    {"key": "limited_degree", "ranks": 1, "description": "Maximo grau 2"},
                ],
                "conditions": {"degree_1": "Dazed", "degree_2": "Stunned", "degree_3": ""},
                "stat_targets": [],
                "resist_override": "Fortitude",
            },
        ],
    },

    # ── Clima / Weather ──
    "rain": {
        "label_pt": "Chuva (Rain)",
        "label_en": "Rain",
        "desc_pt": "Clima: Chuva. Visibilidade -2. Dura 5 turnos (8 com item).",
        "components": [
            {
                "effect_key": "environment",
                "rank": 1,
                "is_linked": False,
                "extras": [
                    {"key": "custom_free", "ranks": 1, "description": "Visibilidade -2"},
                ],
                "flaws": [],
                "conditions": None,
                "stat_targets": [],
                "resist_override": None,
            },
        ],
    },
    "sun": {
        "label_pt": "Sol (Sun)",
        "label_en": "Sun",
        "desc_pt": "Clima: Sol intenso. Calor + Luz intensa.",
        "components": [
            {
                "effect_key": "environment",
                "rank": 1,
                "is_linked": False,
                "extras": [
                    {"key": "custom_free", "ranks": 1, "description": "Calor intenso + Luz"},
                ],
                "flaws": [],
                "conditions": None,
                "stat_targets": [],
                "resist_override": None,
            },
        ],
    },
    "sandstorm": {
        "label_pt": "Tempestade de Areia (Sandstorm)",
        "label_en": "Sandstorm",
        "desc_pt": "Dano por turno + reducao de visibilidade. Area conforme rank.",
        "components": [
            {
                "effect_key": "damage",
                "rank": 1,
                "is_linked": False,
                "extras": [],
                "flaws": [],
                "conditions": None,
                "stat_targets": [],
                "resist_override": None,
            },
            {
                "effect_key": "environment",
                "rank": 1,
                "is_linked": True,
                "extras": [
                    {"key": "custom_free", "ranks": 1, "description": "Visibilidade reduzida (areia)"},
                ],
                "flaws": [],
                "conditions": None,
                "stat_targets": [],
                "resist_override": None,
            },
        ],
    },
    "hail": {
        "label_pt": "Granizo (Hail)",
        "label_en": "Hail",
        "desc_pt": "Dano por turno + visibilidade reduzida. Area conforme rank.",
        "components": [
            {
                "effect_key": "damage",
                "rank": 1,
                "is_linked": False,
                "extras": [],
                "flaws": [],
                "conditions": None,
                "stat_targets": [],
                "resist_override": None,
            },
            {
                "effect_key": "environment",
                "rank": 1,
                "is_linked": True,
                "extras": [
                    {"key": "custom_free", "ranks": 1, "description": "Visibilidade reduzida (neve)"},
                ],
                "flaws": [],
                "conditions": None,
                "stat_targets": [],
                "resist_override": None,
            },
        ],
    },
}
