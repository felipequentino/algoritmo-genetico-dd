import pandas as pd
import numpy as np
import random
import re

# =============================================================================
# 1. ENGENHARIA DE DADOS (Extraindo os Genes do CSV)
# =============================================================================

print("--- Carregando Bases de Dados ---")

# Carregar Personagens (Gene Pool)
try:
    df_chars = pd.read_csv("data/chars.csv")
    print(f"Dataset de Personagens carregado: {len(df_chars)} registros.")
except FileNotFoundError:
    print("ERRO: 'chars.csv' não encontrado. Certifique-se que o arquivo está na pasta.")
    exit()

# Carregar Monstros (Ambiente de Teste)
# Se não tiver o arquivo, criamos um mock rápido para o código não quebrar
try:
    df_monsters = pd.read_csv("data/monsters.csv")
    # Limpeza básica da coluna Armor Class (AC)
    df_monsters['ac'] = df_monsters['ac'].astype(str).str.extract(r'(\d+)').astype(float)
    df_monsters = df_monsters.dropna(subset=['ac'])
    MONSTER_SAMPLE = df_monsters.sample(n=50) # Amostra para o torneio
    print(f"Dataset de Monstros carregado. Usando amostra de 50 inimigos.")
except:
    print("AVISO: 'dnd_monsters.csv' não encontrado. Gerando monstros aleatórios.")
    MONSTER_SAMPLE = pd.DataFrame({'ac': np.random.randint(12, 22, 50)})

# --- Extração de Genes Válidos ---

# 1. Lista de Classes e Raças únicas
LISTA_CLASSES = df_chars['class_starting'].dropna().unique().tolist()
LISTA_RACAS = df_chars['race'].dropna().unique().tolist()

# 2. Mineração de Armas do Inventário
# O inventário é uma string suja. Vamos procurar palavras-chave de armas comuns.
keywords_armas = {
    'Greatsword': {'dmg': 7.0, 'attr': 0},   # 2d6, usa STR (índice 0)
    'Greataxe':   {'dmg': 6.5, 'attr': 0},   # 1d12, usa STR
    'Rapier':     {'dmg': 4.5, 'attr': 1},   # 1d8, usa DEX (índice 1) - Finesse
    'Longbow':    {'dmg': 4.5, 'attr': 1},   # 1d8, usa DEX
    'Dagger':     {'dmg': 2.5, 'attr': 1},   # 1d4, usa DEX
    'Maul':       {'dmg': 7.0, 'attr': 0},   # 2d6, usa STR
    'Shortsword': {'dmg': 3.5, 'attr': 1},   # 1d6, usa DEX
    'Wand':       {'dmg': 5.5, 'attr': 3}    # Exemplo: Firebolt (1d10), usa INT (índice 3)
}

LISTA_ARMAS = list(keywords_armas.keys())

# 3. Distribuição de Stats (Para gerar valores realistas)
# Vamos pegar a média e desvio padrão de cada coluna de stat para gerar novos valores
stats_cols = ['stats_1', 'stats_2', 'stats_3', 'stats_4', 'stats_5', 'stats_6']
stats_mean = df_chars[stats_cols].mean().values
stats_std = df_chars[stats_cols].std().values

print(f"Genes Extraídos: {len(LISTA_CLASSES)} Classes, {len(LISTA_RACAS)} Raças.")

# =============================================================================
# 2. O ALGORITMO GENÉTICO (Raw Implementation)
# =============================================================================

# Estrutura do Indivíduo (Genoma):
# [Classe, Raça, Arma_Nome, Stat1, Stat2, Stat3, Stat4, Stat5, Stat6]
# Indices dos Stats: 0=STR, 1=DEX, 2=CON, 3=INT, 4=WIS, 5=CHA

def create_individual():
    """Cria um indivíduo aleatório baseado nos dados minerados."""
    # Sorteia classe e raça das listas do CSV
    cls = random.choice(LISTA_CLASSES)
    race = random.choice(LISTA_RACAS)
    weapon = random.choice(LISTA_ARMAS)
    
    # Gera stats usando distribuição normal baseada no CSV (para ser realista)
    # Ex: Se a média de STR no CSV é 12, gera algo perto de 12.
    stats = np.random.normal(stats_mean, stats_std).astype(int)
    stats = np.clip(stats, 8, 20) # Limita entre 8 e 20 (regras D&D 5e)
    
    return [cls, race, weapon, *stats]

def fitness(individual):
    """
    Calcula o Dano Médio Por Rodada (DPR) contra a amostra de monstros.
    """
    # Decodificar genoma
    cls, race, wpn_name, str_, dex_, con_, int_, wis_, cha_ = individual
    stats_array = [str_, dex_, con_, int_, wis_, cha_]
    
    # Pegar dados da arma
    wpn_stats = keywords_armas[wpn_name]
    
    # Identificar atributo de ataque
    # A arma diz qual atributo usa (0=STR, 1=DEX, 3=INT...)
    attr_idx = wpn_stats['attr'] 
    
    # Se a classe for 'Hexblade' (Warlock), pode usar CHA (índice 5) - Regra específica
    # Isso é onde o AG brilha: ele deve descobrir combos. Vamos simplificar.
    
    attr_val = stats_array[attr_idx]
    
    # Cálculo D&D
    mod = (attr_val - 10) // 2
    prof_bonus = 3 # Assumindo nível médio
    attack_bonus = mod + prof_bonus
    avg_dmg = wpn_stats['dmg'] + mod
    
    # Simulação Vetorizada contra Monstros
    monster_acs = MONSTER_SAMPLE['ac'].values
    
    # Chance de Acerto
    hit_chance = (21 - (monster_acs - attack_bonus)) / 20.0
    hit_chance = np.clip(hit_chance, 0.05, 0.95)
    
    # Dano Esperado
    total_dpr = np.sum(hit_chance * avg_dmg)
    
    # --- PENALIDADES E BÔNUS (Fitness Shaping) ---
    
    # Penalidade: Stats mentais baixos para classes mágicas
    if cls in ['Wizard', 'Artificer'] and int_ < 14:
        total_dpr *= 0.8
    
    # Bônus: Sinergia de Raça (Exemplo simples)
    if race == 'Orc' and wpn_stats['attr'] == 0: # Orc gosta de Força
        total_dpr *= 1.1
        
    return max(0, total_dpr)

def reproduce(parent1, parent2):
    """Crossover de Ponto Único"""
    # O genoma tem tamanho 9 (3 categóricos + 6 stats)
    point = random.randint(1, 8)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, rate=0.1):
    """Mutação adaptativa"""
    if random.random() < rate:
        gene_idx = random.randint(0, len(individual) - 1)
        
        # Mutação depende do tipo de gene
        if gene_idx == 0: # Classe
            individual[0] = random.choice(LISTA_CLASSES)
        elif gene_idx == 1: # Raça
            individual[1] = random.choice(LISTA_RACAS)
        elif gene_idx == 2: # Arma
            individual[2] = random.choice(LISTA_ARMAS)
        else: # Stats (índices 3 a 8)
            # Troca de valores (Swap) é melhor que random para stats
            # Ex: Trocar Força com Inteligência
            swap_idx = random.randint(3, 8)
            individual[gene_idx], individual[swap_idx] = individual[swap_idx], individual[gene_idx]
            
    return individual

# =============================================================================
# 3. LOOP PRINCIPAL (Execução)
# =============================================================================

def run_genetic_algorithm():
    POP_SIZE = 200000
    GENERATIONS = 5000
    MUTATION_RATE = 0.15
    
    # 1. População Inicial
    population = [create_individual() for _ in range(POP_SIZE)]
    
    print(f"\nIniciando evolução por {GENERATIONS} gerações...")
    
    for gen in range(GENERATIONS):
        # 2. Avaliação
        scores = [fitness(ind) for ind in population]
        
        # Estatísticas da Geração
        best_score = max(scores)
        best_idx = scores.index(best_score)
        best_ind = population[best_idx]
        
        if gen % 10 == 0:
            print(f"Gen {gen}: Melhor Score = {best_score:.1f} | Build: {best_ind[1]} {best_ind[0]} com {best_ind[2]}")
        
        # 3. Seleção (Roleta Ponderada - WEIGHTED RANDOM CHOICES)
        # Normaliza scores para probabilidades
        total_score = sum(scores)
        if total_score == 0: probs = [1/POP_SIZE]*POP_SIZE
        else: probs = [s/total_score for s in scores]
        
        new_pop = []
        
        # Elitismo: Mantém o melhor absoluto da geração passada
        new_pop.append(best_ind)
        
        for _ in range(POP_SIZE - 1):
            # Escolhe 2 pais baseado na probabilidade (Fitness)
            parents = random.choices(population, weights=probs, k=2)
            
            # Reprodução
            child = reproduce(parents[0], parents[1])
            
            # Mutação
            child = mutate(child, MUTATION_RATE)
            
            new_pop.append(child)
            
        population = new_pop

    return best_ind

# Rodar
if __name__ == "__main__":
    champion = run_genetic_algorithm()
    
    print("\n" + "="*40)
    print("RESULTADO DA OTIMIZAÇÃO")
    print("="*40)
    print(f"Classe: {champion[0]}")
    print(f"Raça:   {champion[1]}")
    print(f"Arma:   {champion[2]}")
    print("-" * 20)
    print("Atributos Otimizados:")
    print(f"STR: {champion[3]} | DEX: {champion[4]} | CON: {champion[5]}")
    print(f"INT: {champion[6]} | WIS: {champion[7]} | CHA: {champion[8]}")
    print("="*40)