import numpy as np

# Configurações
tamanho_populacao = 100
tamanho_individuo = 10
taxa_mutacao = 0.05

# 1. Inicialização
populacao = np.random.randint(0, 2, (tamanho_populacao, tamanho_individuo))
geracao = 0

def genetic_algorithm(population) -> int: # retorna um individuo
    while True:
        pesos = np.sum(population, axis=1) # olha a linha de população e soma ela inteira
        new_population = []
        for _ in range(population):
            # tamanho populacao = [0,1,1]
            # probabilidades = [0.4, 0.9, 0.3] as odds de cada um serem escolhidas
            
            soma_pesos = np.sum(pesos)
            if soma_pesos == 0:
                probabilidades = np.ones(tamanho_populacao) / tamanho_populacao
            else:
                probabilidades = pesos / soma_pesos
            
            i_x = np.random.choice(tamanho_populacao, p=probabilidades)
            i_y = np.random.choice(tamanho_populacao, p=probabilidades)
            pai_x, pai_y = population[i_x], population[i_y]
            ponto = np.random.randint(1, tamanho_individuo)
            filho = np.concatenate([pai_x[:ponto], pai_y[ponto:]])

            if np.random.rand() < taxa_mutacao:
                new_population.append(filho)
            
        population = new_population
        pesos = np.sum(population, axis=1)
        melhor_fitness = np.max(pesos)
        if melhor_fitness == tamanho_individuo:
            return melhor_fitness
    


ga = genetic_algorithm(population=populacao)
print(ga)