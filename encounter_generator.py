"""
encounter_generator.py
======================
Gerador de Encontros Orgânicos para o RPG Pokémon sombrio de Ga'Al.

Uso:
    python encounter_generator.py
    python encounter_generator.py --regiao "Coração Verde" --bioma "Floresta" --horario "Noite"
    python encounter_generator.py --quantidade 5
    python encounter_generator.py --output encontros.txt
    python encounter_generator.py --stats 100

O script tenta carregar 'pokedex_Nova.xlsx' automaticamente.
Se o arquivo não existir, usa dados mockados embutidos.
Dependências externas (pandas, openpyxl) são opcionais.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

# Força UTF-8 no terminal do Windows para suportar caracteres especiais
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ===========================================================================
# DATACLASS
# ===========================================================================

@dataclass
class Pokemon:
    nome: str
    numero: str
    tipos: List[str]
    raridade: str
    biomas: List[str]
    regioes: List[str]          # pode ter múltiplas: "Campos Longos/Coração Verde"
    descricao: str
    estrutura_social: str = "Solitário"
    comportamento: List[str] = field(default_factory=list)
    noturno: bool = False


# ===========================================================================
# SISTEMA DE TAGS (extração a partir da descrição da Pokédex)
# ===========================================================================

# Ordem importa: a primeira que bater ganha (para estrutura social)
ESTRUTURA_SOCIAL_KEYWORDS: List[tuple] = [
    ("Matilha",     ["matilha", "alcateia", "lidera a matilha", "hierarquia", " alfa ", "dominância", "dominante"]),
    ("Colônia",     ["colônia", "enxame", "ninho coletivo", "colmeia", "território coletivo", "casta", "operárias"]),
    ("Bando",       ["bando", "em bandos", "migra em", "cardume", "nuvem de", "revoada", "centenas deles", "aos milhares"]),
    ("Par",         ["em par", " dupla", "com parceiro", "dois a dois", "casal", "monogâmico", "companheiro de vida"]),
    ("Oportunista", ["oportunista", "carniça", "carniceiro", "ladrão", "vasculha", "furta", "rouba", " restos "]),
]
ESTRUTURA_SOCIAL_KEYWORDS_FLAT: List[tuple] = [
    (tag, kws) for tag, kws in ESTRUTURA_SOCIAL_KEYWORDS
]
# Solitário é o default — também detectado por palavras-chave próprias
SOLITARIO_KEYWORDS = ["solitário", "sozinho", "isolado", "ermitão", "vive só", "anda só"]

COMPORTAMENTO_KEYWORDS: List[tuple] = [
    ("Furtivo",     ["furtivo", "emboscada", "camuflado", "sorrateiro", "invisível", "imperceptível"]),
    ("Agressivo",   ["agressivo", "violento", "ataca sem provocação", "brutal", "fúria", "berserker", "raivoso"]),
    ("Territorial", ["territorial", "defende seu território", "marca território", "não tolera intrusos", "guarda fronteiras"]),
    ("Parasita",    ["parasita", "drena", "infecta", "hospedeiro", "se alimenta de"]),
    ("Migratório",  ["migra", "sazonal", "nômade", " rota ", "percorre longas distâncias"]),
    ("Noturno",     ["noturno", "ativo à noite", "crepuscular", "só sai à noite", "caça de noite"]),
    ("Pacífico",    ["pacífico", "dócil", "inofensivo", "manso", "gentil", "tímido", "curioso mas inofensivo"]),
]


def extrair_tags(descricao: str) -> tuple[str, List[str], bool]:
    """
    Analisa a descrição da Pokédex e retorna:
    - estrutura_social (str)
    - comportamento (List[str]) — múltiplos possíveis
    - noturno (bool)
    """
    d = descricao.lower()

    # Estrutura Social
    estrutura = "Solitário"
    for tag, keywords in ESTRUTURA_SOCIAL_KEYWORDS_FLAT:
        if any(kw in d for kw in keywords):
            estrutura = tag
            break
    # confirma solitário se nenhum outro bateu e há palavras-chave próprias
    if estrutura == "Solitário" and not any(kw in d for kw in SOLITARIO_KEYWORDS):
        pass  # mantém Solitário como default

    # Comportamentos (múltiplos)
    comportamentos: List[str] = []
    noturno = False
    for tag, keywords in COMPORTAMENTO_KEYWORDS:
        if any(kw in d for kw in keywords):
            if tag == "Noturno":
                noturno = True
                comportamentos.append(tag)
            else:
                comportamentos.append(tag)

    if not comportamentos:
        comportamentos = ["Pacífico"]

    return estrutura, comportamentos, noturno


# ===========================================================================
# SISTEMA DE CLIMA REGIONAL
# ===========================================================================

CLIMAS_POR_REGIAO: dict = {
    "Deserto Irrigado": {
        "Calor Extremo": 40, "Céu Limpo": 30, "Tempestade de Areia": 20,
        "Chuva Torrencial": 8, "Neblina Densa": 2,
    },
    "Baixo Deserto": {
        "Calor Extremo": 45, "Céu Limpo": 30, "Tempestade de Areia": 20,
        "Chuva Torrencial": 3, "Neblina Densa": 2,
    },
    "Baía Morta": {
        "Chuva Ácida": 35, "Neblina Densa": 25, "Céu Limpo": 20,
        "Calor Úmido": 12, "Granizo": 8,
    },
    "Coração Verde": {
        "Chuva Constante": 40, "Neblina Densa": 25, "Céu Limpo": 20,
        "Tempestade": 10, "Calor Úmido": 5,
    },
    "Terras Fluviais": {
        "Chuva Constante": 35, "Céu Limpo": 30, "Neblina Densa": 20,
        "Granizo": 10, "Calor Extremo": 5,
    },
    "Campos Longos": {
        "Céu Limpo": 40, "Calor Extremo": 25, "Chuva Constante": 20,
        "Neblina Densa": 10, "Granizo": 5,
    },
    "Chifre de Ga'Al": {
        "Neblina Densa": 35, "Céu Limpo": 25, "Chuva Gelada": 20,
        "Granizo": 15, "Calor Extremo": 3, "Tempestade de Areia": 2,
    },
}
CLIMA_PADRAO = {"Calor Extremo": 30, "Céu Limpo": 30, "Chuva Constante": 20, "Neblina Densa": 15, "Granizo": 5}

# Climas que reconhecem "Chuva" como variante
_CLIMAS_CHUVA = {"Chuva Constante", "Chuva Torrencial", "Chuva Ácida", "Chuva Gelada"}
_CLIMAS_CALOR = {"Calor Extremo", "Calor Úmido"}
_CLIMAS_NEBLINA = {"Neblina Densa"}
_CLIMAS_AREIA = {"Tempestade de Areia"}
_CLIMAS_GRANIZO = {"Granizo"}

# Bônus de tipo por clima (+15 no peso de raridade do Pokémon)
BONUS_TIPO_POR_CLIMA: dict = {
    "Chuva Constante":      ["Água", "Elétrico"],
    "Chuva Torrencial":     ["Água", "Elétrico"],
    "Chuva Ácida":          ["Água", "Elétrico", "Veneno"],
    "Chuva Gelada":         ["Água", "Gelo"],
    "Tempestade de Areia":  ["Rocha", "Aço", "Terra"],
    "Granizo":              ["Gelo"],
    "Neblina Densa":        ["Fantasma", "Sombrio", "Psíquico"],
    "Calor Extremo":        ["Fogo", "Terra"],
    "Calor Úmido":          ["Fogo", "Veneno"],
    "Tempestade":           ["Elétrico"],
}

# Descrições mecânicas curtas por clima (aparecem no header do output)
CLIMA_EFEITO: dict = {
    "Calor Extremo":        "desidratação rápida, -1 em ações físicas prolongadas",
    "Calor Úmido":          "suor constante, visibilidade reduzida pelo vapor",
    "Céu Limpo":            "sem cobertura, todos veem tudo",
    "Chuva Constante":      "lama, sons amplificados, rastros apagados",
    "Chuva Torrencial":     "visibilidade crítica, movimentos lentos",
    "Chuva Ácida":          "dano por exposição em pele desprotegida",
    "Chuva Gelada":         "hipotermia possível, movimentos lentos",
    "Tempestade de Areia":  "visibilidade zero, dano por abrasão",
    "Neblina Densa":        "visibilidade mínima, sons distorcidos",
    "Granizo":              "dano direto em exposição, barulho ensurdecedor",
    "Tempestade":           "risco de raios, Pokémon Elétricos agitados",
}

# Introduções sensoriais por família de clima (2-3 variações cada)
INTRO_POR_CLIMA: dict = {
    "Calor Extremo": [
        "O ar distorce tudo. A cada três passos, o horizonte vibra e mente sobre o que existe além. "
        "Gargantas estão secas há tempo demais. Vocês avançam com cabeça baixa, poupando cada gota, "
        "quando o chão à frente se mexe.",
        "A areia queima através das solas. Respirar fundo é engolir ar quente como vapor de panela. "
        "O sol não tem misericórdia aqui — e ele não está sozinho nessa intenção.",
        "Nenhuma sombra. Nenhum alívio. O calor suga a vontade de continuar andando. "
        "Vocês param para reagrupar quando percebem que não estão sozinhos há um tempo.",
    ],
    "Calor Úmido": [
        "O ar é uma esponja saturada. Cada respiração parece afogamento parcial. "
        "Insetos e odores se misturam no vapor constante — e algo nessa névoa úmida se move.",
        "Suor nunca evapora aqui. A umidade cola roupa, embaça visão e carrega cheiros longe demais. "
        "Vocês ouvem antes de ver.",
    ],
    "Céu Limpo": [
        "Nenhuma cobertura. O horizonte é uma linha reta onde tudo se enxerga — "
        "e tudo também vê vocês. No meio dessa exposição total, um movimento chama atenção.",
        "A luz bate em tudo com honestidade brutal. Não há sombra para se esconder, "
        "não há névoa para se dissolver. Vocês e o que está à frente se veem ao mesmo tempo.",
        "O silêncio aqui tem textura. Sem vento, sem cobertura vegetal, cada som viaja longe. "
        "Por isso vocês ouvem o que está acontecendo antes de checar se deveriam se aproximar.",
    ],
    "Chuva Constante": [
        "A chuva apaga rastros e esmaga sons. Lama arrasta cada passo e a visibilidade cai para metros. "
        "Vocês avançam agachados sob capas quando o cheiro forte os faz parar.",
        "Água desce pelos troncos como rios verticais. O barulho constante cobre tudo — "
        "o que deveria ser aviso chega tarde demais.",
    ],
    "Chuva Torrencial": [
        "A chuva não deixa ouvir gritos a dois metros. Vocês avançam quase às cegas, "
        "cada passo incerto na lama profunda, quando tropeçam em algo que reage.",
        "Rios formam onde não havia. O terreno muda em tempo real. "
        "Nessa confusão aquática, algo que não é água se move com propósito.",
    ],
    "Chuva Ácida": [
        "A chuva cheira a metal velho e produto químico. Onde toca pele exposta, arde — "
        "não como queimadura, mas como aviso. Vocês buscam abrigo sob estrutura enferrujada "
        "e percebem que não estão sozinhos nesse instinto de sobrevivência.",
        "Pele arde, olhos lacrimejam. A chuva aqui não é água — é o descuido industrial "
        "da Baía Morta precipitando em forma líquida. Vocês correm para se cobrir e param "
        "diante do que já ocupa o único abrigo visível.",
    ],
    "Chuva Gelada": [
        "A chuva congela ao tocar. Dedos perdem sensibilidade antes do esperado. "
        "O chão está coberto de uma crosta fina e traiçoeira — e algo aqui parece indiferente ao frio.",
    ],
    "Tempestade de Areia": [
        "Areia chicoteia pele exposta como vidro moído. Respirar sem pano no rosto é abrir a boca "
        "para lixar os pulmões. A visibilidade é zero. E então, através do uivo do vento, vocês ouvem "
        "algo que não é vento.",
        "O mundo some. Só existe a areia que queima e o som que engole tudo. Vocês avançam ligados uns "
        "aos outros por corda quando um dos puxões não vem de nenhum de vocês.",
    ],
    "Neblina Densa": [
        "A neblina engole forma e som. Vocês avançam às cegas, confiando mais no ouvido que no olho. "
        "Um cheiro forte — orgânico, territorial — avisa que algo está perto antes que possam ver.",
        "Visibilidade de três metros, talvez menos. Cada silhueta na névoa pode ser pedra ou predador. "
        "Vocês param quando percebem que uma das silhuetas se move, e depois para também.",
        "A neblina não filtra sons, só os distorce. Vocês ouvem respiração — próxima demais — "
        "sem conseguir localizar de onde vem.",
    ],
    "Granizo": [
        "Pedras de gelo batem nos ombros como socos pequenos. O barulho é ensurdecedor — "
        "impossível ouvir grito, ordem ou aviso. Vocês cobrem a cabeça e avançam curvados "
        "quando o chão à frente mostra que não está deserto.",
        "O granizo obriga a olhar para baixo. Vocês só veem o que está à frente quando já estão "
        "próximos demais para recuar sem ser notados.",
    ],
    "Tempestade": [
        "Relâmpagos clareiam o terreno em flashes brancos. Entre um e outro, escuridão total. "
        "O trovão cobrindo tudo. Nos instantes de luz, vocês veem o que está acontecendo à frente — "
        "e não dá para saber se a tempestade ou o que está nela é mais perigoso.",
    ],
}


def sortear_clima(regiao: str) -> str:
    tabela = CLIMAS_POR_REGIAO.get(regiao, CLIMA_PADRAO)
    climas = list(tabela.keys())
    pesos = list(tabela.values())
    return random.choices(climas, weights=pesos, k=1)[0]


def intro_clima(clima: str) -> str:
    # Tenta chave exata, depois por família
    for chave, intros in INTRO_POR_CLIMA.items():
        if chave.lower() in clima.lower() or clima.lower() in chave.lower():
            return random.choice(intros)
    return "Vocês avançam pelo terreno com cautela quando percebem que não estão sozinhos."


# ===========================================================================
# RARIDADE E PESOS DE SORTEIO
# ===========================================================================

RARIDADE_PESOS: dict = {
    "Comum":      50,
    "Incomum":    30,
    "Raro":       25,
    "Super Raro": 8,
    "Ultra Raro": 3,
    "Fóssil":     5,
    "Lendário":   0,  # nunca sorteado
}


def calcular_peso(pokemon: Pokemon, tipos_bonus: List[str]) -> int:
    base = RARIDADE_PESOS.get(pokemon.raridade, 10)
    if base == 0:
        return 0
    bonus = 15 if any(t in tipos_bonus for t in pokemon.tipos) else 0
    return base + bonus


# ===========================================================================
# CENAS BIOLÓGICAS
# ===========================================================================

CENAS_POR_ESTRUTURA: dict = {

    "Matilha": [
        "Uma formação em pinça se fecha ao redor de uma presa — um {presa} com pata traseira arrastando "
        "que não tem para onde ir. O alfa rosna uma ordem curta e os outros apertam o cerco.",
        "O alfa come. Os outros ficam a dois metros de distância, queixos tensos, olhos no pedaço que resta. "
        "Qualquer movimento em falso e a hierarquia explode em violência entre eles mesmos.",
        "Dois filhotes seguram uma presa pequena viva entre as patas enquanto os mais velhos observam, "
        "corrigindo cada erro com mordida disciplinar. A presa ainda se debate.",
        "Um membro mais velho da matilha cambaleia. Os outros param. O alfa o olha por três segundos — "
        "depois ataca. O que sobra não é reconhecível. O resto da matilha retoma o caminho.",
        "Marcações frescas de sangue em uma linha de pedras e troncos — a fronteira foi renovada hoje. "
        "Dois membros patrulham o perímetro em sentidos opostos, comunicando-se por sons baixos.",
        "Dois disputam a posição do alfa — corpos entrelaçados, dentes expostos, nenhum recua. "
        "O resto da matilha observa em círculo perfeito, absolutamente imóvel.",
        "Em fila dupla, eles avançam em silêncio total em direção a algo maior. "
        "A coordenação é perturbadora — sem sinal visível, todos viram ao mesmo tempo.",
        "Os filhotes estão no centro. A matilha inteira forma um anel ao redor sem intervalo. "
        "Algo os ameaçou e eles ainda não sabem de onde veio — estão olhando para fora do círculo.",
        "Um estranho da mesma espécie está no centro do grupo sendo avaliado. Cada membro se aproxima, "
        "cheira, empurra — o candidato não reage. Essa passividade é o teste.",
        "O grupo retorna da caça com as patas pesadas e sem nada entre elas. "
        "A tensão entre eles é elétrica — alguém vai pagar pelo fracasso de hoje.",
    ],

    "Bando": [
        "Centenas em migração. Atropelam tudo que está na rota — pedras, arbustos, o que quer que "
        "vocês coloquem na frente. A lógica do bando não tem direção, só momentum.",
        "Uma carcaça no chão e quarenta deles em cima. O barulho é ensurdecedor — "
        "trincos, farfalhar, ossos estourando. Qualquer coisa que se aproximar entra na disputa.",
        "Algo os assustou. O bando foge em pânico absoluto — direto na direção onde vocês estão. "
        "Não há inteligência nessa corrida, só volume e velocidade.",
        "Um ponto d'água e centenas chegando ao mesmo tempo. Empurram, submergem, disputam espaço. "
        "Os mais fracos são afogados sem que ninguém perceba.",
        "Ninhos disputados em uma falésia. Os que chegam jogam os outros para fora. "
        "Abaixo, ossos de gerações de perdedores.",
        "Um predador atacou e partiu. O que sobrou do bando está desorientado — fragmentos menores "
        "vagando em círculos, colidindo entre si, sem direção.",
        "Um grupo de indivíduos doentes foi isolado pelo bando. Ficaram para trás. "
        "Os saudáveis seguiram sem olhar.",
        "Ritual de acasalamento — ruído, agressão, frenesi. Ignoram tudo que não seja o processo. "
        "Caminhar pelo meio é atravessar um evento que não tem pausa.",
        "Alimentam-se com urgência — há algo que não está aqui mas vai chegar. Eles sabem. "
        "A pressa deles é o único aviso disponível.",
        "Formação esférica, todos apontando para fora — algo no ar os ameaça. "
        "O centro do domo pulsa de movimento.",
    ],

    "Solitário": [
        "Não se mexeu em horas. Parece pedra. Mas está respirando — um sinal leve do flanco. "
        "Posicionado exatamente no ponto cego do único caminho disponível.",
        "Arranhões frescos em três pontos diferentes nos últimos cem metros — "
        "a fronteira foi marcada hoje, e os marcos ainda transbordam seiva/sangue/odor.",
        "Lambe uma ferida no flanco com métrica e repetição. Não para. "
        "A ferida é profunda. Quem ou o que fez isso está, por enquanto, com sorte.",
        "Come de costas para vocês. Uma carcaça pela metade embaixo das patas. "
        "Qualquer ruído o faz virar — e ao virar, a postura muda completamente.",
        "Bebe. Para. Olha em todas as direções. Bebe. Para. Olha. O ciclo não quebra. "
        "Está nervoso com algo — provavelmente vocês, ou algo que vocês não viram ainda.",
        "Escava com precisão metódica e enterra o que sobrou de uma caça anterior. "
        "Reserva. Isso significa que vai caçar de novo antes de usar a reserva.",
        "Percorre o perímetro do território em velocidade baixa, parando em cada ponto marcado. "
        "Ao chegar no ponto mais próximo de vocês, para mais tempo que nos outros.",
        "Está seguindo algo maior que ele há tempo suficiente para que o padrão seja claro. "
        "Paciente. Esperando uma abertura.",
        "Dorme em local que permite saída em quatro direções. "
        "Mesmo dormindo, as orelhas se movem independentemente.",
        "O menor som e tudo muda — postura, respiração, direção do olhar. "
        "Zero de tolerância para qualquer coisa que entre no raio que ele definiu.",
    ],

    "Par": [
        "Um dorme. O outro não pisca. Há dias assim — e o que vigia nunca parece cansado, "
        "como se fosse biologicamente impossível para ele vacilar nessa função.",
        "Em pinça perfeita: um flanqueia pela esquerda, o outro pela direita. "
        "Sem sinal visível, coordenação absoluta. A presa no meio não entende o que está acontecendo.",
        "Um está ferido — flanco aberto, movimentos lentos. O outro posiciona o corpo entre o ferido "
        "e qualquer direção de onde algo possa chegar. Não vai sair daqui.",
        "Quatro em standoff. Dois pares, fronteira invisível entre eles. "
        "Nenhum dos quatro pisca. Quem romper o equilíbrio perde a vantagem.",
        "Constroem algo juntos — cada um com uma função diferente, sem sobreposição. "
        "É eficiente de um jeito que lembra trabalho treinado.",
        "Revezam alimentar o que parece ser um filhote único. "
        "A precisão e o cuidado são quase mecânicos de tão repetidos.",
        "Um está morto. O sobrevivente não saiu. Está ao lado, imóvel, há tempo suficiente "
        "para que o corpo já cheiro diferente. O luto aqui tem forma de paralisia.",
        "Displays sincronizados de ameaça — movem ao mesmo tempo, vocalizam no mesmo tom. "
        "O efeito visual é de uma criatura única que é duas.",
        "Dividem a presa segundo regras específicas — nem a mesma parte, nem aleatório. "
        "Há uma ordem nisso que se repete toda vez.",
        "Patrulha em revezamento de lados — nunca os dois no mesmo lado ao mesmo tempo. "
        "O território deles tem cobertura permanente.",
    ],

    "Colônia": [
        "No centro, um que errou — postura submissa, ferido, rodeado. "
        "O grupo ao redor não está agitado. Está metódico. Isso é pior.",
        "Constroem. Cada indivíduo com função clara, sem desperdício de movimento. "
        "Uma estrutura que ontem não existia hoje já tem forma. Passam por cima de qualquer obstáculo no caminho.",
        "Algo os alertou. Em cascata — um chama outro que chama outro — "
        "todos os presentes convergem para o mesmo ponto de atenção ao mesmo tempo.",
        "Fila. Carregam materiais pesados em silêncio absoluto. "
        "A fila não para para nada. Inclui o que estiver no caminho.",
        "Sentinelas em pontos estratégicos distribuídos — cobertura de 360 graus sem intervalo. "
        "Comunicação por toque, não som. Vocês foram notados antes de notá-los.",
        "Algo danificou a estrutura. Todas as castas mobilizadas para reparação — "
        "operárias carregando, soldados protegendo o perímetro, sem hierarquia visível na urgência.",
        "Um intruso entrou no raio da colônia. A resposta é imediata, coordenada e sem proporção aparente. "
        "Não há negociação nessa reação.",
        "Ao centro, atividade de reprodução ou criação. Na periferia, soldados com postura diferente "
        "dos que fazem trabalho. A divisão de papéis é visível de longe.",
        "A fronteira do território avançou hoje — novos marcadores em pontos antes neutros. "
        "Alguém que usava esse espaço vai descobrir isso da pior forma.",
        "Três se separam do grupo principal e avançam em direção ao que o restante evita. "
        "Sacrifício funcional. As outras nem olham.",
    ],

    "Oportunista": [
        "O acampamento foi abandonado rápido — bolsas abertas, lata virada, comida pela metade. "
        "Está no meio de tudo isso, vasculhando com nariz rente ao chão. Cheira a humano recente.",
        "Mantém distância calculada de algo três vezes maior que ele. Segue há horas. "
        "Quando o maior abater algo, o oportunista tem trinta segundos antes que os outros cheguem.",
        "Dois deles na mesma carcaça. Cada um puxando de um lado. "
        "O equilíbrio é frágil — qualquer terceiro inclina a balança e o fraco perde tudo.",
        "A armadilha do caçador está com a isca torta — foi roubada pela metade. "
        "Está tentando pegar o resto sem ativar o mecanismo. Já quase conseguiu.",
        "Um corpo ainda respira, mas devagar. O oportunista está a dois metros, sentado, "
        "aguardando com paciência que não tem nada de inocente.",
        "Nariz no chão, seguindo rastro de sangue que leva em alguma direção. "
        "Totalmente absorto — não ouve, não olha para cima.",
        "O ninho está desguardado — os pais partiram juntos. "
        "Ele sabe exatamente quanto tempo tem. Já começou.",
        "Come de uma lata amassada. Comida humana. Aprendeu que humanos deixam coisas para trás "
        "e adaptou toda a rotina a seguir esse padrão.",
        "Está atrás do grupo há quilômetros. Nunca se aproxima o suficiente para ser confrontado. "
        "Está esperando um tropeço, uma distração, um momento de fraqueza.",
        "Ataca um Pokémon que claramente acabou de fugir de uma batalha — "
        "reservas esgotadas, ferido, sem chance de reação consistente.",
    ],
}


def cena_biologica(pokemon: Pokemon) -> str:
    """Seleciona cena com prioridade para cenas únicas geradas pela Pokédex.
    Cenas específicas têm peso 3x; genéricas peso 1x."""
    presas = ["Bouffalant", "Stantler", "Tauros", "Rapidash", "Zebstrika", "Dodrio"]

    # Pool genérico (por estrutura social)
    genericas = CENAS_POR_ESTRUTURA.get(pokemon.estrutura_social, CENAS_POR_ESTRUTURA["Solitário"])
    genericas_resolvidas = [c.replace("{presa}", random.choice(presas)) for c in genericas]

    # Pool específico (gerado a partir da Pokédex do espécime)
    especificas = gerar_cenas_especificas(pokemon)

    if not especificas:
        return random.choice(genericas_resolvidas)

    # Sorteio com pesos: específicas 5x mais prováveis que genéricas
    # 3 específicas × 5 = 15 pts vs ~10 genéricas × 1 = 10 pts → ~60% específica
    pool  = especificas + genericas_resolvidas
    pesos = [5] * len(especificas) + [1] * len(genericas_resolvidas)
    return random.choices(pool, weights=pesos, k=1)[0]


# ===========================================================================
# CENAS ESPECÍFICAS POR POKÉDEX
# ===========================================================================

import re as _re

# Cada kit: (keywords_detectoras, prefixes, suffixes)
# {fato} = sentença extraída da Pokédex; {nome} = nome do Pokémon
_KITS_CENA: list = [
    # ── CAMUFLAGEM / DISFARCE ──────────────────────────────────────────────
    (
        ["camufla", "invisível", "esconde", "imperceptível", "camuflagem",
         "aparência enganosa", "disfarça", "mimetiza", "parece uma", "idêntico",
         "confunde com", "toma a forma", "se funde", "indistinguível"],
        [
            "Vocês pisam perto demais antes de entender que algo está ali: {fato}.",
            "A sensação de ser observado sem ver o observador tem uma explicação — {fato}.",
            "Não há como saber há quanto tempo está lá. {fato}.",
            "Parecem pedras, raízes, troncos — {fato}.",
            "O ambiente parece vazio até que para de parecer. {fato}.",
        ],
        [
            " O {nome} pode ter visto vocês antes de vocês perceberem qualquer coisa.",
            " Não há como saber quantos há.",
            " O problema de criar algo que não pode ser detectado é que você descobre tarde demais.",
            " Alguma coisa levemente errada no ambiente é tudo que delata a presença.",
            " A distância segura aqui é zero — vocês não sabem onde ele termina e o ambiente começa.",
        ],
    ),
    # ── VENENO / SUBSTÂNCIA ────────────────────────────────────────────────
    (
        ["veneno", "ácido", "tóxico", "toxina", "expele", "envenena",
         "paralisa", "nuvem", "gás venenoso", "líquido corrosivo", "secreção",
         "substância", "irritante", "corrosivo", "libera líquido", "espirra"],
        [
            "O ar tem sabor antes de ter cheiro. {fato}.",
            "Um leve ardor na pele. Um gosto metálico na garganta. {fato}.",
            "Algo está errado com o ar aqui — não é vento, é intenção. {fato}.",
            "Nenhum aviso visual. O primeiro sinal é físico: {fato}.",
            "O chão abaixo está descolorido — {fato}.",
        ],
        [
            " O {nome} não precisa fazer mais nada — só esperar.",
            " Sintomas de envenenamento leve já começaram, para quem prestar atenção no próprio corpo.",
            " Nunca houve aviso. Nunca há.",
            " O chão abaixo do {nome} está sem vegetação. Há quanto tempo ele está aqui?",
            " O {nome} não está em alerta. Está apenas existindo, e isso já é suficiente.",
        ],
    ),
    # ── AROMA / ODOR (Grass, Fairy, etc.) ─────────────────────────────────
    (
        ["aroma", "perfume", "odor", "cheiro", "fragrância", "pólen",
         "libera aroma", "exala", "emana cheiro", "adormece", "relaxa",
         "doce aroma", "libera perfume", "cheiro agradável", "flores liberam"],
        [
            "Um cheiro que não pertence a este lugar — doce demais, persistente demais. {fato}.",
            "A cabeça pesa antes de identificar o motivo. {fato}.",
            "Os animais ao redor parecem calmos de um jeito que não faz sentido. {fato}.",
            "O ar carrega algo além de umidade. {fato}.",
        ],
        [
            " Exposição prolongada tem efeitos que só aparecem depois que os jogadores saírem daqui.",
            " O {nome} não precisa atacar. O ambiente que ele cria já trabalha por ele.",
            " A vontade de ir embora diminui quanto mais tempo se passa perto.",
            " Pokémons ao redor parecem entorpecidos. Vocês podem estar seguindo o mesmo caminho.",
        ],
    ),
    # ── FOGO / CALOR ───────────────────────────────────────────────────────
    (
        ["chama", "fogo", "queima", "incendeia", "chama interna", "aquece",
         "combustão", "brasa", "incandescente", "chamas", "brasas",
         "temperatura extrema", "calor extremo", "calor sufocante"],
        [
            "O calor aqui tem fonte. {fato}.",
            "Uma luz que não é reflexo de nada. {fato}.",
            "Antes de ver, vocês sentem o aumento de temperatura. {fato}.",
            "O cheiro de ar queimado chega antes da visão. {fato}.",
            "A vegetação próxima está chamuscada em padrão radial — {fato}.",
        ],
        [
            " O {nome} usa isso com indiferença — para ele é apenas existir.",
            " Qualquer coisa inflamável nas imediações já está em risco.",
            " A temperatura do local subiu antes de vocês chegarem — e ainda sobe.",
            " O {nome} não está com calor. O {nome} é o calor.",
            " Fumaça não sobe daqui. Aquece mas não queima — por enquanto.",
        ],
    ),
    # ── CONTROLE MENTAL / ILUSÃO ───────────────────────────────────────────
    (
        ["controla", "hipnotiza", "influencia", "domina", "mente",
         "ilude", "alucinação", "manipula", "fascina", "encanta",
         "lê pensamentos", "sente emoções", "percebe intenções", "afeta a mente",
         "sob seu controle", "fica sob", "coloca sob", "force vital",
         "controle", "dominar", "subjuga"],
        [
            "Os animais ao redor se comportam de forma errada — errada demais. {fato}.",
            "Um dos jogadores percebe que sua leitura do espaço mudou levemente. {fato}.",
            "Há algo de errado com o ambiente — não o ambiente físico, mas a percepção dele. {fato}.",
            "Dificuldade de manter foco. Pensamentos que não parecem inteiramente seus. {fato}.",
            "Uma vontade de parar, sentar, esperar — que não veio de dentro. {fato}.",
        ],
        [
            " Discernir o que é real e o que é projeção vai exigir esforço consciente agora.",
            " Pokémons ao redor já estão sob efeito. Quanto tempo vocês estão aqui?",
            " O {nome} está em algum lugar nessa névoa de percepção distorcida.",
            " Primeira coisa que o {nome} distorce: a noção de ameaça.",
            " O {nome} não ataca quem coopera. O problema é o que 'cooperar' significa aqui.",
        ],
    ),
    # ── SOM / PERCUSSÃO / FREQUÊNCIA ──────────────────────────────────────
    (
        ["som", "canto", "grito", "uivo", "ruído", "barulho", "ritmo",
         "vibração", "onda sonora", "frequência", "eco", "tambor", "percussão",
         "toca", "instrumento", "melodia", "ressonância", "estalo", "silvio",
         "rugido", "bate o", "bate seu", "bate em"],
        [
            "Vocês ouvem antes de ver qualquer coisa. {fato}.",
            "O som chega primeiro, e o som não é natural. {fato}.",
            "Um ruído que não pertence a este ambiente. {fato}.",
            "A frequência é baixa o suficiente para ser sentida no peito antes de ser ouvida. {fato}.",
            "O silêncio quebra de um jeito específico. {fato}.",
        ],
        [
            " A fonte ainda não está visível — mas está próxima o suficiente para sentir a vibração.",
            " Outros animais fugiram por causa disso. Vocês chegaram depois deles.",
            " O {nome} não sabe — ou não se importa — com o efeito que produz.",
            " Há ecos de resposta ao longe. Outros {nome} ouviram.",
            " O ritmo não para. Está escalando.",
        ],
    ),
    # ── PREDAÇÃO / CAÇA ────────────────────────────────────────────────────
    (
        ["caça", "predador", "preda", "presa", "devora", "come", "alimenta",
         "captura", "abate", "persegue", "embosca", "afunda dentes",
         "rasga", "dilacera", "engole", "engolindo", "mata", "caçadas",
         "sua presa", "da presa", "pega presa", "pega sua"],
        [
            "Vocês chegam no meio do processo. {fato}.",
            "A cena já está em andamento quando vocês aparecem. {fato}.",
            "Havia algo aqui antes de vocês, e o que havia não tem mais sorte. {fato}.",
            "O cheiro de sangue fresco orienta antes da visão. {fato}.",
            "O barulho de osso estourando chega antes de qualquer imagem. {fato}.",
        ],
        [
            " A presa ainda respira. Por quanto tempo não está claro.",
            " O {nome} está absorto demais para notar chegada discreta — mas não vai durar.",
            " Interromper tem consequências. Não interromper também.",
            " O {nome} está no ciclo. Vocês interromperam o ciclo.",
            " Sangue fresco. Três minutos, talvez menos, até isso atrair mais coisas.",
        ],
    ),
    # ── ELETRICIDADE ───────────────────────────────────────────────────────
    (
        ["elétrico", "eletricidade", "choque", "faísca", "energia elétrica",
         "carga elétrica", "raio", "descarga", "estática", "relâmpago",
         "corrente elétrica", "brilha", "magnetismo"],
        [
            "Os pelos se arrepiam sem razão óbvia. Depois a razão aparece. {fato}.",
            "Um cheiro de ozônio que não devia estar aqui. {fato}.",
            "O metal dos equipamentos está aquecido de forma estranha. {fato}.",
            "Faíscas no chão — rastro de passagem recente. {fato}.",
            "A bússola de alguém do grupo girou 180 graus. {fato}.",
        ],
        [
            " Qualquer metal no corpo de quem estiver mais próximo funciona como antena.",
            " O {nome} acumula carga quanto mais agitado fica. E está ficando agitado.",
            " O ar ao redor do {nome} faísca levemente a cada movimento.",
            " Água no chão está eletrificada. Não há como saber o raio.",
            " Qualquer aparelho eletrônico do grupo deixou de funcionar corretamente.",
        ],
    ),
    # ── GELO / FRIO ────────────────────────────────────────────────────────
    (
        ["gelo", "congela", "frio", "temperatura negativa", "cristal de gelo",
         "solidifica", "congela instantaneamente", "neve", "nevasca", "ventania gelada",
         "ar congelante", "cristaliza"],
        [
            "A temperatura caiu dez graus antes de chegarem perto o suficiente para ver. {fato}.",
            "O bafo congela antes de se dissipar. {fato}.",
            "Rastros de gelo no chão onde não deveria haver. {fato}.",
            "As plantas mais próximas estão cristalizadas. {fato}.",
            "O solo range como se tivesse mudado de natureza. {fato}.",
        ],
        [
            " Exposição prolongada tem consequências físicas reais e progressivas.",
            " O {nome} não está com frio. O {nome} é o frio.",
            " Qualquer líquido descoberto já está parcialmente cristalizado.",
            " A sensação de entorpecimento já começou nas extremidades.",
            " A janela antes da hipotermia é mais curta aqui do que em qualquer outro lugar.",
        ],
    ),
    # ── VEGETAÇÃO / RAIZ / FUNGO ───────────────────────────────────────────
    (
        ["raiz", "planta", "árvore", "vegetal", "hera", "esporo",
         "fungo", "parasita vegetal", "absorve nutrientes", "trepadeira",
         "galho", "folhas", "floresta", "copa das árvores", "brotou"],
        [
            "As plantas aqui estão diferentes — posição errada, curvatura errada. {fato}.",
            "Raízes que não deveriam se mover se movem. {fato}.",
            "O chão está vivo de um jeito que o chão não deveria estar. {fato}.",
            "Um galho que não estava lá antes. Ou estava e vocês não viram. {fato}.",
            "A floresta ao redor ficou quieta de repente — e esse silêncio veio de algo. {fato}.",
        ],
        [
            " Distinguir o {nome} do ambiente ao redor exige percepção que o tempo não permite.",
            " Tudo que parece vegetação aqui pode não ser apenas vegetação.",
            " O {nome} já sabia que vocês estavam vindo. As raízes contaram.",
            " Quem está dentro do raio de influência já foi catalogado.",
            " Sair daqui pode ser mais difícil do que entrar foi.",
        ],
    ),
    # ── FANTASMA / ESPÍRITO / MALDIÇÃO ────────────────────────────────────
    (
        ["osso", "esqueleto", "morto", "fantasma", "espírito", "alma",
         "assombra", "aparição", "energia espectral", "amaldiçoa", "maldição",
         "vítima", "drena vida", "força vital", "espectro", "intangível",
         "sugar a força", "suga a força", "drena a força", "força de vida",
         "força vital de", "energia vital"],
        [
            "O frio não tem fonte física. Os animais ao redor pararam de se mover. {fato}.",
            "Uma sombra que não corresponde a nenhuma forma visível. {fato}.",
            "A sensação de presença antes da presença. {fato}.",
            "O silêncio aqui tem peso. {fato}.",
            "Um dos jogadores sente a exaustão sem ter feito esforço. {fato}.",
        ],
        [
            " Ver o {nome} claramente é mais difícil quanto mais se tenta focar.",
            " Pokémons ao redor já fugiram. Vocês não sabiam que deveriam.",
            " O {nome} não está aqui para vocês especificamente. Mas vocês estão aqui para ele.",
            " Algo foi perturbado. Não fica claro o quê nem quando.",
            " O que o {nome} deixa para trás não é apenas uma sensação.",
        ],
    ),
    # ── FORÇA FÍSICA / MUSCULATURA ─────────────────────────────────────────
    (
        ["músculo", "força", "levanta", "carrega peso", "golpe", "soco",
         "pancada", "destrói", "esmaga", "quebra pedra", "arremessa",
         "potência", "braço", "pata poderosa", "impacto", "colisão",
         "se gaba", "exibe", "impressiona"],
        [
            "O barulho chega antes — impacto surdo, repetitivo, sem pausa. {fato}.",
            "O chão treme em ritmo. {fato}.",
            "Uma pedra do tamanho de um tronco foi movida de lugar recentemente — {fato}.",
            "A vegetação ao redor está achatada num padrão radial. {fato}.",
            "Algo está sendo destruído aqui, metodicamente. {fato}.",
        ],
        [
            " O {nome} não está em alerta. Está treinando. Isso pode ser pior.",
            " Colocar-se no caminho disso não é uma decisão — é um acidente.",
            " O {nome} percebe vocês quando termina. Não antes.",
            " A força aqui não tem inimigo visível. O que ele está preparando?",
            " Não é raiva. É rotina. E a rotina dele é o que vocês estão interrompendo.",
        ],
    ),
    # ── ÁGUA / AMBIENTE AQUÁTICO ───────────────────────────────────────────
    (
        ["mergulha", "nada", "subaquático", "profundeza", "corrente",
         "maré", "oceano", "lago", "rio", "aquático", "flutuante",
         "superfície da água", "debaixo d'água", "respiração aquática"],
        [
            "A superfície da água está errada — movimento que não tem vento como causa. {fato}.",
            "Bolhas sobem em padrão regular de um ponto fixo. {fato}.",
            "A água aqui está mais fria do que o esperado para a profundidade. {fato}.",
            "Algo passou por baixo há pouco — a maré ainda não se acomodou. {fato}.",
        ],
        [
            " Entrar na água agora é uma decisão que precisa de contexto que vocês não têm.",
            " O {nome} tem vantagem absoluta aqui. Vocês têm visibilidade zero nesse meio.",
            " A margem é segura apenas enquanto o {nome} não decidir o contrário.",
            " Quanto tempo está sob a superfície sem respirar é uma questão aberta.",
        ],
    ),
    # ── COMPORTAMENTO SOCIAL / DOMINÂNCIA / EXIBIÇÃO ──────────────────────
    (
        ["rival", "compete", "status", "hierarquia", "prestígio", "impressionar",
         "outros membros", "seus pares", "se gaba de", "orgulhoso",
         "marca presença", "desafia", "demonstra", "exibe seus", "superar"],
        [
            "Há uma audiência que não são vocês. {fato}.",
            "Dois deles estão em display — nenhum ataca, mas nenhum cede. {fato}.",
            "O que parece briga não é: é performance. {fato}.",
            "Algo está sendo provado aqui, para uma plateia que inclui vocês agora. {fato}.",
        ],
        [
            " Interromper isso tem peso social — o {nome} vai precisar reafirmar algo, e vocês são o alvo mais fácil.",
            " A exibição está escalando. Quando parar de ser exibição, acontece rápido.",
            " Os observadores ao redor registraram a chegada de vocês. O exibicionista ainda não.",
            " Qualquer coisa que pareça fraqueza agora vai ser testada.",
        ],
    ),
    # ── SONO / INATIVIDADE / TORPOR ───────────────────────────────────────
    (
        ["dorme", "hibernação", "torpor", "descansa", "recarrega",
         "conserva energia", "quando acorda", "desperta", "sono profundo",
         "imóvel por", "letárgico"],
        [
            "Está imóvel. Respiração lenta e ritmada. {fato}.",
            "Parece inofensivo. Parece. {fato}.",
            "O erro mais comum com esse Pokémon específico: acordar ele. {fato}.",
            "Quieto demais para estar apenas descansando. {fato}.",
        ],
        [
            " O {nome} dormindo é uma janela. O {nome} acordando fecha essa janela rápido.",
            " O que acontece quando ele acorda não está claro — mas os sinais ao redor sugerem que outros já descobriram.",
            " Passar por aqui sem acordar exige precisão que o ambiente não ajuda a ter.",
            " Quando esse Pokémon acorda, leva alguns segundos para orientar. Esses segundos são tudo.",
        ],
    ),
]

_PREFIXES_GENERICOS: list = [
    "O que a Pokédex descreve em texto seco se materializa diante de vocês: {fato}.",
    "Nenhuma descrição prepara para ver isso ao vivo. {fato}.",
    "Vocês chegam no momento exato. {fato}.",
    "Sem aviso. Sem transição. {fato}.",
    "A entrada foi silenciosa. O que vocês encontram não é. {fato}.",
]
_SUFFIXES_GENERICOS: list = [
    " O {nome} ainda não percebeu que tem audiência.",
    " Vocês têm talvez cinco segundos antes que isso mude.",
    " O que acontece a seguir depende do que fizerem agora.",
    " Há uma janela. Pequena. Mas existe.",
    " Havia jeito de não estar aqui. Esse momento já passou.",
]


_INTRO_BLACKLIST = _re.compile(
    r"^(\w[\wç]+ é (um|uma) pok[eé]mon|"
    r"\w[\wç]+ é a (evolu|forma)|"
    r"\w[\wç]+ é (considerado|conhecido|chamado)|"
    r"\w[\wç]+ foi introduzido)",
    _re.IGNORECASE,
)


def _extrair_sentencas(descricao: str) -> list:
    """Divide a descrição em sentenças com ao menos 30 caracteres,
    descartando frases de introdução genérica ('X é um Pokémon do tipo...')."""
    partes = _re.split(r'(?<=[.!?])\s+|\n+', descricao)
    resultado = []
    for s in partes:
        s = s.strip()
        if len(s) < 30:
            continue
        if _INTRO_BLACKLIST.match(s):
            continue
        resultado.append(s)
    return resultado


def _montar_cena_especifica(sentenca: str, nome: str) -> str:
    """Transforma uma sentença da Pokédex em cena observável, usando o kit de categoria mais adequado."""
    s_lower = sentenca.lower()
    for keywords, prefixes, suffixes in _KITS_CENA:
        if any(kw in s_lower for kw in keywords):
            prefixo = random.choice(prefixes).format(fato=sentenca.rstrip(".!?"))
            sufixo  = random.choice(suffixes).format(nome=nome)
            return prefixo + sufixo
    # Nenhuma categoria específica — usa wrapper genérico
    prefixo = random.choice(_PREFIXES_GENERICOS).format(fato=sentenca.rstrip(".!?"))
    sufixo  = random.choice(_SUFFIXES_GENERICOS).format(nome=nome)
    return prefixo + sufixo


def gerar_cenas_especificas(pokemon: Pokemon) -> list:
    """Gera 2-3 cenas únicas baseadas nos fatos biológicos da Pokédex do espécime.
    Prioriza sentenças com verbos de ação/comportamento."""
    sentencas = _extrair_sentencas(pokemon.descricao)
    if not sentencas:
        return []

    # Priorizar sentenças com verbos de comportamento observável
    verbos_prio = [
        "pode", "usa", "ataca", "caça", "vive", "cria", "controla", "emite",
        "produz", "libera", "expele", "camufla", "protege", "alimenta", "devora",
        "habita", "treina", "marca", "defende", "rasteja", "paralisa", "envenena",
        "hipnotiza", "atrai", "fascina", "congela", "queima", "absorve", "drena",
    ]
    com_acao  = [s for s in sentencas if any(v in s.lower() for v in verbos_prio)]
    sem_acao  = [s for s in sentencas if s not in com_acao]
    ordenadas = com_acao + sem_acao

    # Gerar até 3 cenas, evitando duplicatas por prefixo
    cenas: list = []
    usadas: set = set()
    for s in ordenadas:
        if len(cenas) >= 3:
            break
        chave = s[:45]
        if chave not in usadas:
            cenas.append(_montar_cena_especifica(s, pokemon.nome))
            usadas.add(chave)

    return cenas


# ===========================================================================
# SISTEMA DE GRUPO E INTERAÇÕES INTER-ESPÉCIES
# ===========================================================================

def _papel_ecologico(pokemon: Pokemon) -> str:
    """Classifica o Pokémon em papel ecológico primário para lógica de interação."""
    comp = set(pokemon.comportamento)
    soc  = pokemon.estrutura_social
    if "Parasita" in comp:
        return "Parasita"
    if soc == "Oportunista":
        return "Oportunista"
    if soc == "Colônia":
        return "Colonial"
    if soc == "Bando":
        return "Bando"
    if "Agressivo" in comp or soc == "Matilha":
        return "Predador"
    if "Furtivo" in comp and "Pacífico" not in comp:
        return "Predador"
    if "Territorial" in comp:
        return "Territorial"
    if "Pacífico" in comp or "Migratório" in comp:
        return "Presa"
    return "Neutro"


# ── Templates de interação para 2 espécies ────────────────────────────────
# {A} = ativo/dominante, {B} = passivo/subordinado (ou simétrico nos genéricos)
# {soc_A}, {soc_B} = estruturas sociais

_CENAS_INTERACAO_PRED: list = [
    # A = predador, B = presa
    "{A} ({soc_A}) em caça ativa. {B} ({soc_B}) ainda não sabe — mas a postura do {A} "
    "mudou há dois minutos e não há interpretação inocente para isso. "
    "A cena vai terminar de um jeito. Vocês chegaram antes do final.",

    "O {B} ({soc_B}) está ferido — ginga ao se mover, rastro de sangue fresco no chão. "
    "O {A} ({soc_A}) o segue a distância calculada, sem pressa. "
    "Não é perseguição. É espera. O {A} sabe que não precisa correr.",

    "{A} ({soc_A}) e {B} ({soc_B}) dividem o mesmo espaço — mas só um escolheu isso. "
    "O {A} se aproximou durante a última hora. O {B} percebeu, mas tarde demais. "
    "A distância entre eles agora é de pressão, não de segurança.",

    "O {B} ({soc_B}) bebeu. Levantou a cabeça. Viu o {A} ({soc_A}). "
    "Os dois se encararam por três segundos completos. "
    "Depois o {B} começou a recuar, lentamente, sem virar. O {A} não avançou ainda.",
]

_CENAS_INTERACAO_OPOR: list = [
    # A = detentor do recurso, B = oportunista
    "O {A} ({soc_A}) está no meio do processo — alimentando-se, ocupado, absorto. "
    "O {B} ({soc_B}) mantém distância exata de segurança. "
    "Não vai se mover até o {A} terminar ou vacilar — o que vier primeiro.",

    "Carcaça no chão — restos de uma caça do {A} ({soc_A}). "
    "O {B} ({soc_B}) se aproxima pelos flancos, esperando uma janela. "
    "O {A} ainda está perto o suficiente para que essa negociação termine de outro jeito.",

    "O {A} ({soc_A}) abateu algo. O {B} ({soc_B}) chegou antes de vocês — "
    "já circunda o perímetro da refeição, testando a atenção do {A} com movimentos pequenos. "
    "Esse jogo existe há tempo suficiente para ter regras não escritas.",

    "O {B} ({soc_B}) seguiu o {A} ({soc_A}) por quilômetros. Nunca se expõe, nunca recua. "
    "Está esperando um tropeço, uma distração, uma fraqueza. "
    "O {A} sabe que está sendo seguido. Não se importa ainda.",
]

_CENAS_INTERACAO_TERR: list = [
    # Simétrico — A e B em confronto territorial
    "{A} ({soc_A}) e {B} ({soc_B}) em standoff de fronteira — nenhum ataca, nenhum recua. "
    "Displays de ameaça se alternam em ciclo tenso. "
    "Vocês aparecem como terceiro elemento numa equação que já estava instável.",

    "A fronteira entre os territórios do {A} ({soc_A}) e do {B} ({soc_B}) "
    "passa exatamente por onde vocês estão. "
    "Ambos os lados os notam ao mesmo tempo. O conflito entre eles pausa — brevemente.",

    "O {A} ({soc_A}) cruzou a linha — marcações do {B} ({soc_B}) ainda frescas, "
    "ignoradas com intenção. O {B} não ignorou. "
    "Vocês chegam no momento exato em que a resposta começa.",

    "{A} ({soc_A}) e {B} ({soc_B}) se olham por sobre uma linha de marcadores "
    "que nenhum dos dois fez — restos de uma espécie que não está mais aqui. "
    "Território em disputa sem dono legítimo. Isso torna tudo mais imprevisível.",
]

_CENAS_INTERACAO_COLONIAL: list = [
    # A = colônia, B = intruso
    "A colônia de {A} ({soc_A}) identificou o {B} ({soc_B}) como intrusão. "
    "A resposta não tem proporção — é volume, coordenação, sem pausa para avaliação. "
    "O {B} foge. A colônia decide se para na fronteira ou segue.",

    "O {B} ({soc_B}) entrou sem perceber nos limites do {A} ({soc_A}). "
    "A reação começou antes que ele processasse o aviso. Agora foge — "
    "e a trajetória passa por onde vocês estão.",
]

_CENAS_INTERACAO_PARASITA: list = [
    # A = parasita, B = hospedeiro
    "O {A} ({soc_A}) está sobre o {B} ({soc_B}) — ou próximo o suficiente. "
    "O {B} não reagiu. Não consegue reagir. O processo já passou da fase de escolha. "
    "Vocês chegam durante, não antes.",

    "O {B} ({soc_B}) se move de forma estranha — lento, desorientado, sem direção clara. "
    "O {A} ({soc_A}) está próximo demais para ser coincidência. "
    "A relação entre eles não é predação. É algo mais lento e mais completo.",

    "O {B} ({soc_B}) está em movimento mas não em fuga — vai em círculos, sem propósito. "
    "O {A} ({soc_A}) o segue de perto, indiferente ao ambiente ao redor. "
    "A drenagem já começou — há quanto tempo não está claro.",
]

_CENAS_INTERACAO_GENERIC: list = [
    # Simétrico ou neutro
    "{A} ({soc_A}) e {B} ({soc_B}) no mesmo ponto d'água — sem atacar, sem ceder. "
    "Uma trégua funcional que existe porque ambos precisam de água mais do que um do outro agora. "
    "Qualquer elemento externo quebra esse equilíbrio. Vocês são esse elemento.",

    "{A} ({soc_A}) e {B} ({soc_B}) ocupam o mesmo espaço por motivos distintos. "
    "Não há cooperação — há sobreposição tolerada. "
    "Cada um finge que o outro não existe. Até algo mudar.",

    "Dois ocupantes do mesmo corredor. {A} ({soc_A}) e {B} ({soc_B}) "
    "passam a dez metros um do outro sem confronto — avaliação mútua em fração de segundo. "
    "Nenhum achou que valesse. Nenhum vai esquecer.",

    "O {A} ({soc_A}) ocupa a posição elevada. O {B} ({soc_B}), a baixa. "
    "Nenhum viola o espaço do outro — não por respeito, por cálculo. "
    "Vocês chegam quando esse cálculo ainda está sendo feito.",

    "{A} ({soc_A}) está ocupado com algo que não é o {B} ({soc_B}). "
    "{B} está ocupado com algo que não é o {A}. "
    "A proximidade é circunstancial, não escolhida. "
    "E mesmo assim, cada um monitora o outro com metade da atenção.",
]

# ── Templates de interação para 3 espécies ────────────────────────────────
# {A}, {B}, {C} e {soc_A}, {soc_B}, {soc_C}

_CENAS_INTERACAO_3: list = [
    # Cadeia de predação: A caça B, C oportunista
    "{A} ({soc_A}) em caça ativa contra {B} ({soc_B}), que foge com reservas diminuindo. "
    "Atrás de tudo isso, {C} ({soc_C}) segue à distância. "
    "Quando o {A} abater, o {C} vai tentar pegar os restos antes que o {A} volte. "
    "Vocês chegaram no meio dessa cadeia.",

    "{B} ({soc_B}) está ferido e em fuga. {A} ({soc_A}) o persegue. "
    "{C} ({soc_C}) os segue a todos — não para ajudar nenhum dos dois. Para esperar. "
    "Três espécies em movimento, uma cadeia que vai terminar antes de vocês saírem daqui.",

    # Confronto com observador
    "{A} ({soc_A}) e {B} ({soc_B}) em standoff territorial — "
    "displays, vocalizações, nenhum recua. "
    "O {C} ({soc_C}) observa de posição elevada, imóvel. "
    "Não está do lado de nenhum. Está esperando o vencedor se distrair.",

    "A disputa entre {A} ({soc_A}) e {B} ({soc_B}) virou performance — "
    "nenhum vai atacar agora. "
    "O {C} ({soc_C}) está nos arredores, vasculhando o que ficou desguardado "
    "enquanto os dois se encaram. "
    "Vocês aparecem como quarta variável numa equação que já tem três.",

    # Colonial + intruso + oportunista
    "A colônia de {A} ({soc_A}) está expulsando {B} ({soc_B}) do perímetro — "
    "ruidoso, coordenado, sem negociação. "
    "No meio do caos, {C} ({soc_C}) vasculha o que ficou desprotegido na distração. "
    "Há uma janela aqui que não é para vocês — mas vocês estão nela.",

    # Espiral de predação mútua
    "{A} ({soc_A}) caça {B} ({soc_B}) — mas o {B} está ocupado evitando {C} ({soc_C}), "
    "que por sua vez circula o território do {A}. "
    "Três espécies em movimento, cada uma fora do lugar que escolheria. "
    "O equilíbrio aqui se mantém por tensão constante, não por estabilidade.",

    # Genérico 3-way
    "{A} ({soc_A}), {B} ({soc_B}) e {C} ({soc_C}) compartilham o mesmo espaço "
    "por razões distintas. "
    "Nenhum escolheria isso — cada um está aqui por necessidade ou circunstância. "
    "Qualquer movimento brusco colapsa o equilíbrio improvisado. Vocês são o movimento brusco.",

    "Três espécies, três dinâmicas sobrepostas. "
    "{A} ({soc_A}) está focado no {B} ({soc_B}). {B} está mais preocupado com {C} ({soc_C}). "
    "E o {C} notou vocês antes de qualquer um dos outros dois. "
    "A atenção aqui está distribuída de um jeito que não vai durar.",

    # Recurso disputado
    "Uma carcaça. {A} ({soc_A}) tem a posição central — foi ele quem abateu. "
    "{B} ({soc_B}) e {C} ({soc_C}) disputam os arredores, "
    "cada um testando os limites do outro e do {A} ao mesmo tempo. "
    "Três espécies, uma carcaça, nenhuma regra acordada.",

    "Ponto d'água único. {A} ({soc_A}), {B} ({soc_B}) e {C} ({soc_C}) "
    "chegaram ao mesmo tempo por caminhos distintos. "
    "A tensão é geométrica — cada novo olhar acrescenta uma linha de pressão. "
    "Vocês chegam como quarto ponto.",
]


def _selecionar_interacao_2(A: Pokemon, B: Pokemon) -> str:
    """Seleciona cena de interação entre dois Pokémon respeitando papéis ecológicos."""
    pa = _papel_ecologico(A)
    pb = _papel_ecologico(B)

    def fmt(template: str, ativo: Pokemon, passivo: Pokemon) -> str:
        return template.format(
            A=ativo.nome, B=passivo.nome,
            soc_A=ativo.estrutura_social, soc_B=passivo.estrutura_social,
        )

    # Predação
    if pa == "Predador" and pb in ("Presa", "Neutro", "Bando"):
        return fmt(random.choice(_CENAS_INTERACAO_PRED), A, B)
    if pb == "Predador" and pa in ("Presa", "Neutro", "Bando"):
        return fmt(random.choice(_CENAS_INTERACAO_PRED), B, A)

    # Oportunismo: "ativo" é quem tem o recurso, "passivo" é o oportunista
    if pa == "Oportunista" and pb != "Oportunista":
        return fmt(random.choice(_CENAS_INTERACAO_OPOR), B, A)
    if pb == "Oportunista" and pa != "Oportunista":
        return fmt(random.choice(_CENAS_INTERACAO_OPOR), A, B)

    # Parasitismo
    if pa == "Parasita":
        return fmt(random.choice(_CENAS_INTERACAO_PARASITA), A, B)
    if pb == "Parasita":
        return fmt(random.choice(_CENAS_INTERACAO_PARASITA), B, A)

    # Colonial expulsando intruso
    if pa == "Colonial":
        return fmt(random.choice(_CENAS_INTERACAO_COLONIAL), A, B)
    if pb == "Colonial":
        return fmt(random.choice(_CENAS_INTERACAO_COLONIAL), B, A)

    # Standoff territorial
    if pa in ("Territorial", "Predador", "Colonial") and pb in ("Territorial", "Predador", "Colonial"):
        return fmt(random.choice(_CENAS_INTERACAO_TERR), A, B)

    # Fallback universal
    return fmt(random.choice(_CENAS_INTERACAO_GENERIC), A, B)


def _selecionar_interacao_3(grupo: List[Pokemon]) -> str:
    """Gera cena de interação para três Pokémon, selecionando template compatível com os papéis."""
    papeis = [_papel_ecologico(p) for p in grupo]
    A, B, C = grupo[0], grupo[1], grupo[2]

    def fmt(template: str, x: Pokemon, y: Pokemon, z: Pokemon) -> str:
        return template.format(
            A=x.nome, B=y.nome, C=z.nome,
            soc_A=x.estrutura_social, soc_B=y.estrutura_social, soc_C=z.estrutura_social,
        )

    idx_pred     = next((i for i, r in enumerate(papeis) if r == "Predador"), None)
    idx_opor     = next((i for i, r in enumerate(papeis) if r == "Oportunista"), None)
    idx_colonial = next((i for i, r in enumerate(papeis) if r == "Colonial"), None)
    idx_presa    = next((i for i, r in enumerate(papeis)
                         if r in ("Presa", "Bando", "Neutro")
                         and i not in (idx_pred, idx_opor, idx_colonial)), None)

    # 1. Cadeia de predação: predador → presa + terceiro (oportunista ou qualquer outro)
    if idx_pred is not None and idx_presa is not None:
        restante = ({0, 1, 2} - {idx_pred, idx_presa}).pop()
        z_idx = idx_opor if idx_opor is not None else restante
        return fmt(random.choice(_CENAS_INTERACAO_3[:2]),
                   grupo[idx_pred], grupo[idx_presa], grupo[z_idx])

    # 2. Colonial expulsando intruso: A=colônia, B=intruso, C=oportunista/terceiro
    if idx_colonial is not None:
        outros = [i for i in range(3) if i != idx_colonial]
        b_idx = idx_opor if idx_opor is not None and idx_opor != idx_colonial else outros[0]
        c_idx = next(i for i in outros if i != b_idx)
        return fmt(_CENAS_INTERACAO_3[4], grupo[idx_colonial], grupo[b_idx], grupo[c_idx])

    # 3. Standoff entre dois agressivos/territoriais, terceiro observa
    agressivos = [i for i, r in enumerate(papeis) if r in ("Predador", "Territorial")]
    if len(agressivos) >= 2:
        observador_idx = ({0, 1, 2} - {agressivos[0], agressivos[1]}).pop()
        # templates [2] e [3]: A e B são os que se confrontam, C é o observador
        return fmt(random.choice(_CENAS_INTERACAO_3[2:4]),
                   grupo[agressivos[0]], grupo[agressivos[1]], grupo[observador_idx])

    # 4. Fallback: templates neutros/genéricos (sem papel específico obrigatório)
    #    Usa os índices [5:] que não requerem papéis concretos
    return fmt(random.choice(_CENAS_INTERACAO_3[5:]), A, B, C)


def sortear_grupo(candidatos: List[Pokemon], pesos: List[int]) -> List[Pokemon]:
    """Sorteia 1–3 espécies distintas para o encontro.
    Distribuição: 1 espécie → 50%, 2 espécies → 35%, 3 espécies → 15%."""
    n = random.choices([1, 2, 3], weights=[50, 35, 15], k=1)[0]
    n = min(n, max(1, len(candidatos)))

    if n == 1:
        return [random.choices(candidatos, weights=pesos, k=1)[0]]

    # Sorteia sem repetição de espécie (por nome)
    cands = list(candidatos)
    pesos_r = list(pesos)
    selecionados: List[Pokemon] = []
    for _ in range(n):
        if not cands or not any(pesos_r):
            break
        escolhido = random.choices(cands, weights=pesos_r, k=1)[0]
        selecionados.append(escolhido)
        # Remove todas as entradas com o mesmo nome para evitar duplicatas
        indices = [i for i, p in enumerate(cands) if p.nome == escolhido.nome]
        for i in sorted(indices, reverse=True):
            cands.pop(i)
            pesos_r.pop(i)

    return selecionados or [random.choices(candidatos, weights=pesos, k=1)[0]]


def gerar_cena_grupo(grupo: List[Pokemon]) -> str:
    """Gera cena narrativa para o grupo de Pokémon (1, 2 ou 3 espécies)."""
    if len(grupo) == 1:
        return cena_biologica(grupo[0])
    if len(grupo) == 2:
        return _selecionar_interacao_2(grupo[0], grupo[1])
    return _selecionar_interacao_3(grupo)


# ===========================================================================
# SISTEMA DE DIFICULDADE
# ===========================================================================

DIFICULDADES = ["Ambiental", "Predação", "Fator Humano", "Letal"]
DIFICULDADE_PESOS = [45, 30, 20, 5]
DIFICULDADE_LABELS = {
    "Ambiental":    "Tier 1 — Ambiental/Observação",
    "Predação":     "Tier 2 — Predação/Território",
    "Fator Humano": "Tier 3 — Fator Humano",
    "Letal":        "Tier 4 — LETAL",
}

FACCOES_HUMANAS: List[tuple] = [
    ("Clube de Caça de Aspargo", [
        "jipes modificados surgem com redes pneumáticas — atiram sem avisar, "
        "sem distinção entre Pokémon selvagem e o que estiver perto dele",
        "armadilhas são montadas ao redor em silêncio, bloqueando saídas enquanto vocês estavam distraídos — "
        "só percebem quando tentam recuar",
        "negociam por rádio o preço do espécime enquanto o cercam. "
        "Tratam vocês como variável ambiental — um problema, não uma prioridade... por enquanto",
        "dardos tranquilizantes cruzam o espaço sem aviso. "
        "O alvo eram os Pokémon, mas a pontaria deles nunca foi precisa com civis no caminho",
    ]),
    ("Caçadores de Lillac", [
        "aparecem sem ruído — equipamento de ponta, rostos cobertos, já sabem quantos vocês são. "
        "Estavam esperando. Tinham informação antes de chegar",
        "drones silenciosos varrem a área marcando cada Pokémon e cada rosto. "
        "Em algum lugar, alguém compra esse dado",
        "um dos caçadores levanta imobilizador em direção não ao Pokémon selvagem — mas ao Pokémon de um dos jogadores. "
        "Contrato de captura. Espécie rara vale mais com treinador incluído",
    ]),
    ("Milicorpe", [
        "barricadas portáteis bloqueiam a única saída. Dois agentes com tablets exigem documentação, "
        "credenciais e o equivalente em pokédollars de um pedágio não oficial",
        "drones catalogam rostos em tempo real — vocês já estão no sistema deles, "
        "associados ao incidente, antes de dizer uma palavra",
        "dois agentes detêm um viajante desconhecido com violência desnecessária a menos de vinte metros. "
        "Quando terminam, viram e olham para vocês",
    ]),
    ("Rangers de Dainise", [
        "em missão furtiva de contenção — aparecem do nada sinalizando silêncio absoluto urgente. "
        "Algo piorou aqui recentemente e eles estão no limite de conseguir controlar",
        "tentam resgatar um Pokémon ferido com o predador ainda circulando. "
        "Precisam de distração. Olham para vocês com uma expressão que não é pedido — é aviso",
        "cercas elétricas portáteis sobem ao redor da área — eficientes, rápidas, sem aviso. "
        "Quem estava dentro do perímetro quando subiram ficou dentro",
    ]),
    ("Mineradores Ilegais da Malva", [
        "explosão na encosta a duzentos metros — a topografia muda em tempo real, "
        "derrubando parte do terreno que vocês pensavam ser sólido",
        "máquinas pesadas avançam destruindo vegetação e toca sem desviar. "
        "Os Pokémon que estavam lá precisam ir para algum lugar — e o lugar mais próximo são vocês",
        "geradores barulhentos ligados no meio da rota. "
        "O nível de ruído transforma cada criatura no raio de duzentos metros em imprevisível",
    ]),
]

EVENTOS_LETAIS: List[str] = [
    "Delta — O General da Team Keneze — patrulha pessoalmente com esquadrão de elite. "
    "Pokémon Ezenek em formação ao redor dele. Fuga é a única resposta possível. "
    "Qualquer outra é estatística.",
    "Uma onda de corrupção Ezenek varre a área. Pokémon sendo marcados em tempo real — "
    "olhos que mudam de cor, comportamento que inverte. O que era indiferente se torna hostil "
    "em segundos. Inclusive o que estava do lado de vocês.",
    "Patrulha de limpeza da Team Keneze em operação — Pokémon Ezenek em formação tática, "
    "soldados marcados seguindo protocolo que não distingue combatente de civil",
    "Um Pokémon lendário está ferido. Caçadores chegaram antes e erraram — agora fugiram. "
    "O que ficou é uma criatura mítica em fúria cega que não separa quem atacou de quem passou por aqui",
    "Tempestade Ácida se forma no horizonte e avança rápido. "
    "Dano contínuo em pele exposta, vegetação morrendo em tempo real. "
    "Abrigo não é opção — é obrigação em minutos",
    "O terreno colapsa — deslizamento, inundação súbita, explosão de gás subterrâneo. "
    "O bioma inteiro se reorganiza em tempo real e vocês estão no meio",
    "Depósito de munição e combustível da Keneze detona em cadeia a menos de quinhentos metros. "
    "Onda de calor, estilhaços, Pokémon em fuga total em todas as direções",
    "Horda de Pokémon marcados pela corrupção Ezenek avança em formação — "
    "não caçam, não defendem. Executam. Algo os está direcionando",
    "Alpha — O Primeiro Marcado — conduz um ritual de invocação aqui, agora. "
    "A área ao redor já está reagindo: plantas morrendo, água evaporando, animais fugindo há horas",
    "Área de teste de uma nova forma de corrupção Ezenek. "
    "Pokémon selvagens ao redor já foram afetados e o experimento ainda está ativo",
]


def gerar_complicacao(tier: str, pokemon: Pokemon) -> str:
    if tier == "Ambiental":
        opcoes = [
            f"A dificuldade é **Baixa (Ambiental)**: ainda não perceberam vocês. "
            f"Jogada de Furtividade para manter isso assim, ou Observação para entender o padrão antes de decidir o que fazer.",
            f"A dificuldade é **Baixa (Ambiental)**: estão ocupados com o próprio drama. "
            f"Vocês podem observar, passar despercebidos, ou intervir — cada escolha tem consequência.",
            f"A dificuldade é **Baixa (Ambiental)**: por enquanto, vocês não existem para eles. "
            f"Mantenham assim se quiserem.",
        ]
        return random.choice(opcoes)

    elif tier == "Predação":
        opcoes = [
            f"A dificuldade é **Média (Predação/Território)**: vocês cruzaram uma fronteira que "
            f"não estava marcada em nenhum mapa. O(s) {pokemon.nome} identificaram a intrusão "
            f"e estão reavaliando a situação — a favor deles.",
            f"A dificuldade é **Média (Predação/Território)**: vocês se tornaram a presa mais fácil disponível no momento. "
            f"Recuar pode funcionar — ou pode ativar o instinto de perseguição.",
            f"A dificuldade é **Média (Predação/Território)**: o {pokemon.nome} não está atacando ainda — "
            f"está avaliando. Cada movimento importa agora.",
        ]
        return random.choice(opcoes)

    elif tier == "Fator Humano":
        faccao_nome, acoes = random.choice(FACCOES_HUMANAS)
        acao = random.choice(acoes)
        return (
            f"A dificuldade é **Alta (Fator Humano)**: enquanto isso acontece, "
            f"{faccao_nome} — {acao}. "
            f"Vocês estão no meio de dois problemas que não sabem que são três."
        )

    else:  # Letal
        evento = random.choice(EVENTOS_LETAIS)
        return (
            f"A dificuldade é **LETAL (O Fim do Mundo)**: {evento} "
            f"O que está na frente de vocês é o menor dos problemas agora."
        )


# ===========================================================================
# CARREGAMENTO DE DADOS
# ===========================================================================

DADOS_MOCK = [
    {
        "Nome": "Sandile", "Número": "#551",
        "Tipo": "Terra/Sombrio", "Raridade": "Comum",
        "Biomas": "Deserto/Dunas", "Região": "Deserto Irrigado",
        "Descrição da Pokedex": (
            "Vivem em matilhas organizadas sob a areia quente. O alfa lidera emboscadas coordenadas "
            "contra presas maiores, mantendo hierarquia rígida com violência. Membros que erram são "
            "punidos publicamente — o grupo aprende pelo exemplo."
        ),
    },
    {
        "Nome": "Skorupi", "Número": "#451",
        "Tipo": "Veneno/Inseto", "Raridade": "Raro",
        "Biomas": "Deserto/Ruínas", "Região": "Baixo Deserto",
        "Descrição da Pokedex": (
            "Solitário e territorial por natureza. Enterra-se na areia e aguarda imóvel por dias até "
            "que uma presa passe. Não tolera intrusos no raio de dez metros. Seu veneno paralisa "
            "progressivamente — a vítima percebe tarde demais. Ativo à noite, invisível de dia."
        ),
    },
    {
        "Nome": "Bisharp", "Número": "#625",
        "Tipo": "Sombrio/Aço", "Raridade": "Ultra Raro",
        "Biomas": "Floresta/Ruínas/Montanha", "Região": "Coração Verde",
        "Descrição da Pokedex": (
            "Lidera colônias de Pawniard com mão de ferro. Pune violentamente membros fracos ou desleais "
            "como lição coletiva. Território é marcado com lâminas cravadas nas árvores — fronteiras reais "
            "que outros respeitam. A colônia é a unidade, não o indivíduo."
        ),
    },
    {
        "Nome": "Murkrow", "Número": "#198",
        "Tipo": "Voador/Sombrio", "Raridade": "Incomum",
        "Biomas": "Floresta/Cidade/Ruínas", "Região": "Baía Morta",
        "Descrição da Pokedex": (
            "Vivem em bandos ruidosos e territoriais. Roubam objetos brilhantes de humanos e Pokémon "
            "igualmente. Oportunistas natos — seguem eventos de maior porte para lucrar com os restos. "
            "Noturno por preferência, mas adaptável."
        ),
    },
    {
        "Nome": "Gastly", "Número": "#092",
        "Tipo": "Fantasma/Veneno", "Raridade": "Raro",
        "Biomas": "Ruínas/Caverna/Cidade", "Região": "Chifre de Ga'Al",
        "Descrição da Pokedex": (
            "Solitário e furtivo. Dissolve-se em névoa para se camuflagem. Alimenta-se de energia vital "
            "enquanto a vítima dorme — parasita silencioso. Noturno. Invisível à luz direta. "
            "Extremamente territorial em espaços fechados."
        ),
    },
    {
        "Nome": "Larvitar", "Número": "#246",
        "Tipo": "Rocha/Terra", "Raridade": "Incomum",
        "Biomas": "Montanha/Deserto/Caverna", "Região": "Baixo Deserto",
        "Descrição da Pokedex": (
            "Completamente solitário desde o nascimento — emerge já separado dos genitores. "
            "Devora montanha literalmente: ingere rocha para crescer. "
            "Territorial com qualquer coisa que se mova na superfície que habita. Agressivo por default."
        ),
    },
    {
        "Nome": "Zangoose", "Número": "#335",
        "Tipo": "Normal", "Raridade": "Raro",
        "Biomas": "Campos/Floresta/Planície", "Região": "Campos Longos",
        "Descrição da Pokedex": (
            "Vivem em duplas monogâmicas de longa duração. Caçam em par coordenado com eficiência brutal. "
            "Rivalidade ancestral com Seviper — qualquer encontro termina em combate sem negociação. "
            "Se um do par morre, o sobrevivente muda completamente de comportamento."
        ),
    },
    {
        "Nome": "Meowth", "Número": "#052",
        "Tipo": "Normal", "Raridade": "Comum",
        "Biomas": "Cidade/Ruínas/Porto", "Região": "Terras Fluviais",
        "Descrição da Pokedex": (
            "Oportunista clássico. Vasculha lixo, rouba mantimentos, fura armadilhas alheias. "
            "Aprende padrões humanos com rapidez assustadora — sabe quando mercados fecham, "
            "quando guardas trocam de turno, quando alguém está distraído. Ladrão por vocação."
        ),
    },
    {
        "Nome": "Croagunk", "Número": "#453",
        "Tipo": "Veneno/Lutador", "Raridade": "Comum",
        "Biomas": "Pântano/Rio/Floresta Úmida", "Região": "Terras Fluviais",
        "Descrição da Pokedex": (
            "Vivem em duplas parceiras que dividem território e caça. "
            "Usam veneno por toque — não precisam morder. Território marcado com secreções tóxicas "
            "que causam desorientação em qualquer coisa que pise. Pacíficos com quem respeita os limites."
        ),
    },
    {
        "Nome": "Pawniard", "Número": "#624",
        "Tipo": "Sombrio/Aço", "Raridade": "Comum",
        "Biomas": "Floresta/Ruínas", "Região": "Coração Verde",
        "Descrição da Pokedex": (
            "Formam colônias sob comando de um Bisharp. Hierarquia rígida — cada membro sabe seu papel. "
            "Agressivos com intrusos por treinamento, não instinto. O grupo age como unidade; "
            "o indivíduo não tem autonomia de decisão. Punem a fraqueza antes que o Bisharp precise."
        ),
    },
]


def _safe_str(val) -> str:
    """Converte qualquer valor (incluindo NaN do pandas) para string limpa."""
    import math
    if val is None:
        return ""
    try:
        if math.isnan(float(val)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


# Mapa de normalização de raridade (cobre variantes de caixa e sem acento)
_RARIDADE_NORMALIZE: dict = {
    "comum":      "Comum",
    "incomum":    "Incomum",
    "raro":       "Raro",
    "super raro": "Super Raro",
    "super_raro": "Super Raro",
    "ultra raro": "Ultra Raro",
    "ultra_raro": "Ultra Raro",
    "fossil":     "Fóssil",
    "fóssil":     "Fóssil",
    "lendario":   "Lendário",
    "lendário":   "Lendário",
}


def _normalizar_raridade(raw: str) -> str:
    return _RARIDADE_NORMALIZE.get(raw.strip().lower(), raw.strip().title())


def _normalizar_str_comparacao(s: str) -> str:
    """Remove acentos e coloca em minúsculo para comparações flexíveis."""
    import unicodedata
    return unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode("ascii")


def _split_regioes(raw: str) -> List[str]:
    """Divide 'Campos Longos / Coração Verde' em ['Campos Longos', 'Coração Verde']."""
    parts = []
    for sep in ("/", ","):
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            return parts
    return [raw.strip()] if raw.strip() else []


def _processar_linha(row: dict) -> Pokemon:
    # Tenta múltiplas grafias possíveis de cada coluna-chave
    nome     = _safe_str(row.get("Nome", ""))
    numero   = _safe_str(row.get("Número", row.get("Numero", row.get("Nº", row.get("N°", "")))))
    tipos_raw = _safe_str(row.get("Tipo", ""))
    tipos    = [t.strip() for t in tipos_raw.split("/") if t.strip()]
    raridade_raw = _safe_str(row.get("Raridade", "Comum"))
    raridade = _normalizar_raridade(raridade_raw)
    biomas_raw = _safe_str(row.get("Biomas", row.get("Bioma", "")))
    biomas   = [b.strip() for b in biomas_raw.split("/") if b.strip()] or [biomas_raw] if biomas_raw else ["Desconhecido"]
    # Região pode conter múltiplas regiões separadas por "/"
    regiao_raw = _safe_str(row.get("Região", row.get("Regiao", row.get("região", ""))))
    regioes  = _split_regioes(regiao_raw) or ["Desconhecida"]
    descricao = _safe_str(row.get("Descrição da Pokedex",
                           row.get("Descricao da Pokedex",
                           row.get("Descrição", ""))))
    estrutura, comportamento, noturno = extrair_tags(descricao)
    return Pokemon(
        nome=nome, numero=numero, tipos=tipos, raridade=raridade,
        biomas=biomas, regioes=regioes, descricao=descricao,
        estrutura_social=estrutura, comportamento=comportamento, noturno=noturno,
    )


def carregar_pokedex_excel(caminho: str) -> List[Pokemon]:
    """Lê Excel usando openpyxl (nativo, sem pandas).
    Fallback para pandas se openpyxl não estiver disponível."""
    pokemons: List[Pokemon] = []

    # Tenta openpyxl primeiro — já vem com a maioria das instalações Python
    try:
        import openpyxl  # type: ignore
        wb = openpyxl.load_workbook(caminho, read_only=True, data_only=True)
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        # Primeira linha = cabeçalhos
        headers = [str(h).strip() if h is not None else "" for h in next(rows)]
        for row_vals in rows:
            row_dict = dict(zip(headers, row_vals))
            p = _processar_linha(row_dict)
            if p.nome:
                pokemons.append(p)
        wb.close()
        return pokemons
    except ImportError:
        pass  # openpyxl não disponível, tenta pandas

    # Fallback: pandas
    import pandas as pd  # type: ignore
    df = pd.read_excel(caminho)
    for _, row in df.iterrows():
        p = _processar_linha(row.to_dict())
        if p.nome:
            pokemons.append(p)
    return pokemons


def carregar_pokedex_csv(caminho: str) -> List[Pokemon]:
    """Carrega CSV padrão."""
    import csv
    pokemons = []
    with open(caminho, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = _processar_linha(dict(row))
            if p.nome:
                pokemons.append(p)
    return pokemons


def carregar_pokedex(caminho_excel: str = "pokedex Nova.xlsx",
                     caminho_csv: str = "pokedex_gaal.csv") -> tuple[List[Pokemon], bool]:
    """
    Tenta: Excel → CSV → Mock.
    Retorna (lista, é_mock).
    """
    import os
    if os.path.exists(caminho_excel):
        try:
            pokemons = carregar_pokedex_excel(caminho_excel)
            print(f"[✓] Pokédex carregada de '{caminho_excel}' ({len(pokemons)} entradas).")
            return pokemons, False
        except Exception as e:
            print(f"[!] Falha ao ler Excel ({e}). Tentando CSV...")

    if os.path.exists(caminho_csv):
        try:
            pokemons = carregar_pokedex_csv(caminho_csv)
            print(f"[✓] Pokédex carregada de '{caminho_csv}' ({len(pokemons)} entradas).")
            return pokemons, False
        except Exception as e:
            print(f"[!] Falha ao ler CSV ({e}). Usando dados mockados.")

    print("[!] Nenhuma Pokédex encontrada. Usando dados mockados (10 Pokémon de demonstração).\n")
    pokemons = [_processar_linha(d) for d in DADOS_MOCK]
    return pokemons, True


# ===========================================================================
# FILTROS DE CANDIDATOS
# ===========================================================================

HORARIOS_DIURNOS = {"Manhã", "Tarde"}
HORARIOS_NOTURNOS = {"Noite", "Madrugada"}


def _match_regiao(regioes_pokemon: List[str], regiao_buscada: str) -> bool:
    """Checa se a região buscada aparece em qualquer das regiões do Pokémon.
    Usa normalização sem acento para tolerar variações como 'Baia Morta' vs 'Baía Morta'."""
    alvo = _normalizar_str_comparacao(regiao_buscada)
    for r in regioes_pokemon:
        r_norm = _normalizar_str_comparacao(r)
        if alvo in r_norm or r_norm in alvo:
            return True
    return False


def _match_bioma(biomas_pokemon: List[str], bioma_buscado: str) -> bool:
    """Checa se o bioma solicitado é substring de qualquer bioma do Pokémon (sem acento)."""
    alvo = _normalizar_str_comparacao(bioma_buscado)
    for b in biomas_pokemon:
        b_norm = _normalizar_str_comparacao(b)
        if alvo in b_norm or b_norm in alvo:
            return True
    return False


_RARIDADES_LENDARIAS = {"Lendário", "Lendario"}


def filtrar_candidatos(pokedex: List[Pokemon], regiao: str,
                       bioma: str, horario: str) -> List[Pokemon]:
    """Filtra por região, bioma e horário. Relaxa progressivamente se nenhum bater."""
    nao_lendario = [p for p in pokedex if p.raridade not in _RARIDADES_LENDARIAS]

    # Filtro principal: região + bioma
    candidatos = [
        p for p in nao_lendario
        if _match_regiao(p.regioes, regiao) and _match_bioma(p.biomas, bioma)
    ]

    # Fallback 1: só região
    if not candidatos:
        candidatos = [p for p in nao_lendario if _match_regiao(p.regioes, regiao)]

    # Fallback 2: qualquer não-lendário
    if not candidatos:
        candidatos = nao_lendario

    # Filtro de horário (suave — não elimina se restariam 0)
    if horario in HORARIOS_DIURNOS:
        filtrado_horario = [p for p in candidatos if not p.noturno]
        if filtrado_horario:
            candidatos = filtrado_horario
    elif horario in HORARIOS_NOTURNOS:
        # Noturnos têm peso dobrado, não exclusão
        pass  # tratado nos pesos

    return candidatos


# ===========================================================================
# GERADOR PRINCIPAL
# ===========================================================================

def gerar_encontro(pokedex: List[Pokemon], regiao: str, bioma: str,
                   horario: str = "Tarde") -> str:
    """
    Gera e retorna um encontro narrativo completo como string formatada.
    Sorteia 1-3 espécies e gera cenas de interação entre elas quando aplicável.
    """
    # Passo A: Clima
    clima = sortear_clima(regiao)
    tipos_bonus = BONUS_TIPO_POR_CLIMA.get(clima, [])
    efeito = CLIMA_EFEITO.get(clima, "—")

    # Passo B: Candidatos e pesos
    candidatos = filtrar_candidatos(pokedex, regiao, bioma, horario)

    pesos = []
    for p in candidatos:
        w = calcular_peso(p, tipos_bonus)
        # Noturno tem peso reduzido de dia e aumentado à noite
        if horario in HORARIOS_DIURNOS and p.noturno:
            w = max(1, w // 4)
        elif horario in HORARIOS_NOTURNOS and p.noturno:
            w = int(w * 1.5)
        pesos.append(w)

    # Passo C: Sortear grupo de 1-3 espécies distintas
    grupo = sortear_grupo(candidatos, pesos)

    # Passo D: Cena (individual ou interação inter-espécies)
    cena = gerar_cena_grupo(grupo)

    # Passo E: Dificuldade (referencia o Pokémon principal do grupo)
    tier = random.choices(DIFICULDADES, weights=DIFICULDADE_PESOS, k=1)[0]
    complicacao = gerar_complicacao(tier, grupo[0])

    # Passo F: Intro sensorial
    intro = intro_clima(clima)

    # Passo G: Cabeçalho de espécies
    tier_label = DIFICULDADE_LABELS[tier]

    if len(grupo) == 1:
        p = grupo[0]
        tipos_str = " / ".join(p.tipos)
        comportamentos_str = " · ".join(p.comportamento)
        especies_header = (
            f"  {p.nome} ({p.estrutura_social} / {comportamentos_str}) [{tipos_str}]"
        )
    elif len(grupo) == 2:
        A, B = grupo[0], grupo[1]
        la = f"{A.nome} ({A.estrutura_social}) [{' / '.join(A.tipos)}]"
        lb = f"{B.nome} ({B.estrutura_social}) [{' / '.join(B.tipos)}]"
        especies_header = f"  {la}  x  {lb}"
    else:
        linhas = [
            f"  {p.nome} ({p.estrutura_social}) [{' / '.join(p.tipos)}]"
            for p in grupo
        ]
        especies_header = "\n".join(linhas)

    # Passo H: Seção técnica — um bloco por Pokémon
    info_blocos = []
    for idx, p in enumerate(grupo, 1):
        tipos_str = " / ".join(p.tipos)
        comportamentos_str = " · ".join(p.comportamento)
        label = f"Espécie {idx}: " if len(grupo) > 1 else ""
        bloco = (
            f"  {label}{p.nome}"
            + (f" {p.numero}" if p.numero else "")
            + f" | Raridade: {p.raridade} | Tipos: {tipos_str}\n"
            f"  Comportamento: {p.estrutura_social} + {comportamentos_str}\n"
            f"  Pokédex de Ga'Al:\n"
            f"  \"{p.descricao}\""
        )
        info_blocos.append(bloco)

    separador = "\n  " + "-" * 56 + "\n"
    info_tecnica = separador.join(info_blocos)

    saida = (
        f"\n{'=' * 62}\n"
        f"  ENCONTRO EM GA'AL\n"
        f"{'=' * 62}\n"
        f"  Local:   {bioma} -- {regiao}\n"
        f"  Clima:   {clima} ({efeito})\n"
        f"  Horario: {horario}\n"
        f"{'=' * 62}\n\n"
        f"  CENA:\n\n"
        f"  {intro}\n\n"
        f"{especies_header}\n"
        f"  {cena}\n\n"
        f"  {complicacao}\n\n"
        f"{'=' * 62}\n"
        f"  INFORMACOES TECNICAS\n"
        f"  Nivel de Ameaca: {tier_label}\n\n"
        f"{info_tecnica}\n"
        f"{'=' * 62}\n"
    )
    return saida


# ===========================================================================
# MODO STATS
# ===========================================================================

def rodar_stats(pokedex: List[Pokemon], regiao: str, bioma: str,
                horario: str, n: int) -> None:
    """Gera n encontros e imprime distribuição estatística."""
    tiers_cont: Counter = Counter()
    climas_cont: Counter = Counter()
    pokemon_cont: Counter = Counter()
    estrutura_cont: Counter = Counter()
    grupo_cont: Counter = Counter()      # distribuição de tamanhos de grupo

    for _ in range(n):
        clima = sortear_clima(regiao)
        tipos_bonus = BONUS_TIPO_POR_CLIMA.get(clima, [])
        candidatos = filtrar_candidatos(pokedex, regiao, bioma, horario)
        pesos = [calcular_peso(p, tipos_bonus) for p in candidatos]
        if not any(pesos):
            continue
        grupo = sortear_grupo(candidatos, pesos)
        tier = random.choices(DIFICULDADES, weights=DIFICULDADE_PESOS, k=1)[0]
        tiers_cont[tier] += 1
        climas_cont[clima] += 1
        grupo_cont[len(grupo)] += 1
        for pokemon in grupo:
            pokemon_cont[pokemon.nome] += 1
            estrutura_cont[pokemon.estrutura_social] += 1

    print(f"\n{'=' * 62}")
    print(f"  ESTATÍSTICAS — {n} encontros | {bioma} / {regiao}")
    print(f"{'=' * 62}")
    print("\n  TIERS DE DIFICULDADE:")
    for t in DIFICULDADES:
        c = tiers_cont.get(t, 0)
        barra = "█" * (c * 30 // n) if n > 0 else ""
        print(f"    {t:<20} {c:>4}x  ({c * 100 // n:>2}%)  {barra}")
    print("\n  TAMANHO DO GRUPO (espécies por encontro):")
    for sz in sorted(grupo_cont.keys()):
        c = grupo_cont[sz]
        barra = "█" * (c * 30 // n) if n > 0 else ""
        print(f"    {sz} espécie(s)            {c:>4}x  ({c * 100 // n:>2}%)  {barra}")
    print("\n  CLIMAS:")
    for clima, c in climas_cont.most_common():
        barra = "█" * (c * 30 // n) if n > 0 else ""
        print(f"    {clima:<25} {c:>4}x  {barra}")
    print("\n  POKÉMON SORTEADOS (top 10):")
    for nome, c in pokemon_cont.most_common(10):
        print(f"    {nome:<20} {c:>4}x")
    print("\n  ESTRUTURAS SOCIAIS:")
    for est, c in estrutura_cont.most_common():
        print(f"    {est:<20} {c:>4}x  ({c * 100 // n:>2}%)")
    print(f"{'=' * 62}\n")


# ===========================================================================
# ARGPARSE E ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gerador de Encontros Orgânicos para o RPG de Ga'Al",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--regiao", default="Deserto Irrigado",
        help="Região de Ga'Al (ex: 'Coração Verde', 'Baixo Deserto')\nPadrão: Deserto Irrigado",
    )
    parser.add_argument(
        "--bioma", default="Dunas",
        help="Bioma específico dentro da região (ex: 'Floresta', 'Ruínas')\nPadrão: Dunas",
    )
    parser.add_argument(
        "--horario", default="Tarde",
        choices=["Manhã", "Tarde", "Anoitecer", "Noite", "Madrugada"],
        help="Horário do encontro\nPadrão: Tarde",
    )
    parser.add_argument(
        "--quantidade", type=int, default=1,
        help="Número de encontros a gerar\nPadrão: 1",
    )
    parser.add_argument(
        "--output", default=None,
        help="Salvar resultado em arquivo de texto (ex: encontros.txt)",
    )
    parser.add_argument(
        "--stats", type=int, default=0, metavar="N",
        help="Gerar N encontros e exibir distribuição estatística (ex: --stats 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Semente aleatória para reproduzir resultados",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    pokedex, _ = carregar_pokedex()

    if not pokedex:
        print("[ERRO] Pokédex vazia. Verifique o arquivo de dados.", file=sys.stderr)
        sys.exit(1)

    # Modo stats
    if args.stats > 0:
        rodar_stats(pokedex, args.regiao, args.bioma, args.horario, args.stats)
        return

    # Geração normal
    resultados = []
    for i in range(args.quantidade):
        encontro = gerar_encontro(pokedex, args.regiao, args.bioma, args.horario)
        resultados.append(encontro)

    output_texto = "\n".join(resultados)
    print(output_texto)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_texto)
        print(f"\n[✓] Resultado salvo em '{args.output}'.")


if __name__ == "__main__":
    main()
