import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# Configuração da página
st.set_page_config(page_title="Evolução D&D", page_icon="🐉", layout="wide")

# =============================================================================
# 1. ENGENHARIA DE DADOS (Com Cache do Streamlit)
# =============================================================================

# O @st.cache_data evita que o app leia o CSV do disco toda vez que você mexer em um botão
@st.cache_data
def load_and_prep_data():
    data = {}
    
    # Carregar Personagens
    try:
        df_chars = pd.read_csv("data/chars.csv")
        data['LISTA_CLASSES'] = df_chars['class_starting'].dropna().unique().tolist()
        data['LISTA_RACAS'] = df_chars['race'].dropna().unique().tolist()
        
        stats_cols = ['stats_1', 'stats_2', 'stats_3', 'stats_4', 'stats_5', 'stats_6']
        data['stats_mean'] = df_chars[stats_cols].mean().values
        data['stats_std'] = df_chars[stats_cols].std().values
        data['char_count'] = len(df_chars)
        
    except FileNotFoundError:
        st.error("ERRO: 'data/chars.csv' não encontrado. Crie a pasta 'data' e coloque o arquivo lá.")
        st.stop()

    # Carregar Monstros
    try:
        df_monsters = pd.read_csv("data/monsters.csv")
        df_monsters['ac'] = df_monsters['ac'].astype(str).str.extract(r'(\d+)').astype(float)
        df_monsters = df_monsters.dropna(subset=['ac'])
        data['MONSTER_SAMPLE'] = df_monsters.sample(n=50)
        data['monster_status'] = "Dados reais carregados (50 monstros)."
    except FileNotFoundError:
        data['MONSTER_SAMPLE'] = pd.DataFrame({'ac': np.random.randint(12, 22, 50)})
        data['monster_status'] = "AVISO: 'data/monsters.csv' não encontrado. Usando monstros aleatórios."

    # Armas fixas
    data['keywords_armas'] = {
        'Greatsword': {'dmg': 7.0, 'attr': 0},   # 2d6, usa STR (índice 0)
        'Greataxe':   {'dmg': 6.5, 'attr': 0},   # 1d12, usa STR
        'Rapier':     {'dmg': 4.5, 'attr': 1},   # 1d8, usa DEX (índice 1)
        'Longbow':    {'dmg': 4.5, 'attr': 1},   # 1d8, usa DEX
        'Dagger':     {'dmg': 2.5, 'attr': 1},   # 1d4, usa DEX
        'Maul':       {'dmg': 7.0, 'attr': 0},   # 2d6, usa STR
        'Shortsword': {'dmg': 3.5, 'attr': 1},   # 1d6, usa DEX
        'Wand':       {'dmg': 5.5, 'attr': 3}    # 1d10, usa INT (índice 3)
    }
    data['LISTA_ARMAS'] = list(data['keywords_armas'].keys())
    
    return data

# Carrega os dados para a memória da sessão
db = load_and_prep_data()

# =============================================================================
# 2. O ALGORITMO GENÉTICO
# =============================================================================

def create_individual():
    cls = random.choice(db['LISTA_CLASSES'])
    race = random.choice(db['LISTA_RACAS'])
    weapon = random.choice(db['LISTA_ARMAS'])
    
    stats = np.random.normal(db['stats_mean'], db['stats_std']).astype(int)
    stats = np.clip(stats, 8, 20) 
    
    return [cls, race, weapon, *stats]

def fitness(individual):
    cls, race, wpn_name, str_, dex_, con_, int_, wis_, cha_ = individual
    stats_array = [str_, dex_, con_, int_, wis_, cha_]
    
    wpn_stats = db['keywords_armas'][wpn_name]
    attr_idx = wpn_stats['attr'] 
    attr_val = stats_array[attr_idx]
    
    mod = (attr_val - 10) // 2
    prof_bonus = 3 
    attack_bonus = mod + prof_bonus
    avg_dmg = wpn_stats['dmg'] + mod
    
    monster_acs = db['MONSTER_SAMPLE']['ac'].values
    hit_chance = (21 - (monster_acs - attack_bonus)) / 20.0
    hit_chance = np.clip(hit_chance, 0.05, 0.95)
    
    total_dpr = np.sum(hit_chance * avg_dmg)
    
    if cls in ['Wizard', 'Artificer'] and int_ < 14:
        total_dpr *= 0.8
    if race == 'Orc' and wpn_stats['attr'] == 0: 
        total_dpr *= 1.1
        
    return max(0, total_dpr)

def reproduce(parent1, parent2):
    point = random.randint(1, 8)
    return parent1[:point] + parent2[point:]

def mutate(individual, rate=0.1):
    if random.random() < rate:
        gene_idx = random.randint(0, len(individual) - 1)
        if gene_idx == 0: 
            individual[0] = random.choice(db['LISTA_CLASSES'])
        elif gene_idx == 1: 
            individual[1] = random.choice(db['LISTA_RACAS'])
        elif gene_idx == 2: 
            individual[2] = random.choice(db['LISTA_ARMAS'])
        else: 
            swap_idx = random.randint(3, 8)
            individual[gene_idx], individual[swap_idx] = individual[swap_idx], individual[gene_idx]
    return individual

# =============================================================================
# 3. INTERFACE STREAMLIT
# =============================================================================

st.title("🧬 Otimizador Genético de Personagens D&D")
st.markdown("Evolução de atributos de RPG utilizando simulação de dados cruzados (Personagens Kaggle vs Monstros).")

# --- SIDEBAR: Parâmetros ---
st.sidebar.header("Hiperparâmetros do Algoritmo")

pop_size = st.sidebar.number_input("Tamanho da População", min_value=10, max_value=200000, value=200, step=50, 
                                   help="Cuidado: Valores acima de 10.000 podem causar lentidão severa.")
generations = st.sidebar.number_input("Gerações", min_value=1, max_value=5000, value=50, step=10)
mutation_rate = st.sidebar.slider("Taxa de Mutação", min_value=0.01, max_value=1.0, value=0.15, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Status dos Dados")
st.sidebar.text(f"⚔️ Personagens: {db['char_count']}")
st.sidebar.text(f"🐉 {db['monster_status']}")

# --- ÁREA PRINCIPAL ---
col1, col2 = st.columns([1, 2])

with col1:
    iniciar = st.button("🚀 Iniciar Evolução", use_container_width=True)

# Placeholders para atualizar em tempo real
progress_bar = st.progress(0)
status_text = st.empty()
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

if iniciar:
    population = [create_individual() for _ in range(pop_size)]
    best_scores_history = []
    
    # Loop de evolução
    for gen in range(generations):
        scores = [fitness(ind) for ind in population]
        
        best_score = max(scores)
        best_idx = scores.index(best_score)
        best_ind = population[best_idx]
        
        best_scores_history.append(best_score)
        
        # Atualizações Visuais no Streamlit
        progress = (gen + 1) / generations
        progress_bar.progress(progress)
        
        if gen % max(1, (generations // 10)) == 0 or gen == generations - 1:
            status_text.markdown(f"**Geração {gen + 1}/{generations}** | Melhor DPR (Dano): **{best_score:.2f}** | Líder atual: `{best_ind[1]} {best_ind[0]}`")
            # Gráfico de linha em tempo real
            chart_placeholder.line_chart(best_scores_history, height=200)

        # Seleção e Reprodução (mantido do original)
        total_score = sum(scores)
        if total_score == 0: probs = [1/pop_size]*pop_size
        else: probs = [s/total_score for s in scores]
        
        new_pop = [best_ind] # Elitismo
        
        for _ in range(pop_size - 1):
            parents = random.choices(population, weights=probs, k=2)
            child = reproduce(parents[0], parents[1])
            child = mutate(child, mutation_rate)
            new_pop.append(child)
            
        population = new_pop
    
    # --- RESULTADO FINAL ---
    status_text.success("🎉 Evolução Concluída!")
    
    champion = population[scores.index(max(scores))]
    
    st.subheader("🏆 O Personagem Supremo (Alpha)")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Classe", champion[0])
    c2.metric("Raça", champion[1])
    c3.metric("Arma", champion[2])
    
    st.markdown("### Atributos Otimizados")
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    sc1.metric("STR", champion[3])
    sc2.metric("DEX", champion[4])
    sc3.metric("CON", champion[5])
    sc4.metric("INT", champion[6])
    sc5.metric("WIS", champion[7])
    sc6.metric("CHA", champion[8])
    
    st.info("Nota: A função de aptidão atual prioriza o Atributo correto para a Arma escolhida. Você deve notar que se o algoritmo escolheu 'Greataxe', a STR (Força) estará próxima ou no máximo (20).")