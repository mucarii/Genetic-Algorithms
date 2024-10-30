# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
# Função de aptidão
def fitness_function(x):
    return x * np.sin(10 * np.pi * x) + 1


# %%
# Inicialização da população
def initialize_population(size):
    return np.random.rand(size)


# %%
# Seleção por torneio
def selection(population, fitness):
    # Seleciona os melhores indivíduos da população com base na aptidão
    selected_indices = np.argsort(fitness)[
        -len(population) // 2 :
    ]  # Pegue os índices dos melhores indivíduos
    return population[selected_indices]  # Retorne os indivíduos selecionados


# %%
# Cruzamento
def crossover(parent1, parent2):
    crossover_point = np.random.rand()
    return parent1 * crossover_point + parent2 * (1 - crossover_point)


# %%
# Mutação
def mutation(cromossome, mutation_rate):
    if np.random.rand() < mutation_rate:
        return np.clip(cromossome + np.random.normal(0, 0.1), 0, 1)
    return cromossome


# %%
# Algoritmo Genético
def genetic_algorithm(generations, population_size, mutation_rate):
    # Inicializa a população
    population = initialize_population(population_size)

    for generation in range(generations):
        # Avaliação
        fitness = fitness_function(population)

        # Seleção
        selected = selection(population, fitness)

        # Cruzamento e mutação
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = np.random.choice(selected, 2, replace=False)
            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate)
            new_population.append(child)

        population = np.array(new_population)

        # Exibe a melhor solução a cada 10 gerações
        if generation % 10 == 0:
            print(f"Geração {generation}: Melhor aptidão = {np.max(fitness)}")

    return population


# %%
# Parâmetros do Algoritmo
generations = 100
population_size = 20
mutation_rate = 0.1

# %%
# Executa o Algoritmo Genético
final_population = genetic_algorithm(generations, population_size, mutation_rate)

# %%
# Exibe o resultado
best_solution = final_population[np.argmax(fitness_function(final_population))]
best_fitness = fitness_function(best_solution)

print(f"Solução ótima: x = {best_solution}, Aptidão = {best_fitness}")

# %%
# Plota a função e a solução encontrada
x = np.linspace(0, 1, 100)
y = fitness_function(x)

plt.plot(x, y, label="Função $f(x)$")
plt.scatter(best_solution, best_fitness, color="red", label="Solução ótima")
plt.title("Algoritmo Genético")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
