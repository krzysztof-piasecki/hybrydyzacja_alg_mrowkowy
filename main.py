import random
import numpy as np


# Inicjalizacja feromonów
def initialize_pheromone_matrix(length):
    return np.ones((length, length))


# Inicjalizacja mrówek
def initialize_ants(num_ants, length):
    ants = []
    for _ in range(num_ants):
        ant = list(range(1, length + 1))
        random.shuffle(ant)
        ants.append(ant)
    return ants


# Wybór następnego elementu
def choose_next_element(pheromone_matrix, attractiveness_matrix, visited, current_index, alpha, beta):
    unvisited = [i for i in range(len(pheromone_matrix)) if i not in visited]
    probabilities = []

    for index in unvisited:
        pheromone = pheromone_matrix[current_index][index]
        attractiveness = attractiveness_matrix[current_index][index]
        probability = pheromone ** alpha * attractiveness ** beta
        probabilities.append(probability)

    probabilities_sum = sum(probabilities)
    probabilities = [p / probabilities_sum for p in probabilities]

    next_index = random.choices(unvisited, probabilities)[0]
    return next_index


# Obliczenie jakości rozwiązania
def calculate_fitness(solution, dna_sequence):
    reconstructed_sequence = ''.join([dna_sequence[i - 1] for i in solution])
    fitness = levenshtein_distance(dna_sequence, reconstructed_sequence)
    return fitness


# Aktualizacja feromonów
def update_pheromones(pheromone_matrix, ants, decay_rate, best_fitness):
    pheromone_matrix *= (1 - decay_rate)
    for ant in ants:
        delta_pheromone = 1 / best_fitness
        for i in range(len(ant) - 1):
            current_index = ant[i] - 1
            next_index = ant[i + 1] - 1
            pheromone_matrix[current_index][next_index] += delta_pheromone


# Algorytm mrówkowy
def ant_colony_optimization(num_ants, num_iterations, decay_rate, alpha, beta, dna_sequence):
    length = len(dna_sequence)
    pheromone_matrix = initialize_pheromone_matrix(length)
    attractiveness_matrix = np.random.rand(length, length)
    ants = initialize_ants(num_ants, length)
    best_solution = ants[0].copy()
    best_fitness = calculate_fitness(best_solution, dna_sequence)

    for iteration in range(num_iterations):
        for ant in ants:
            visited = set()
            visited.add(ant[0])

            for i in range(len(ant) - 1):
                current_index = ant[i] - 1
                next_index = choose_next_element(pheromone_matrix, attractiveness_matrix, visited, current_index, alpha,
                                                 beta)
                ant[i + 1] = next_index + 1
                visited.add(next_index)

            fitness = calculate_fitness(ant, dna_sequence)

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = ant.copy()

        update_pheromones(pheromone_matrix, ants, decay_rate, best_fitness)

    return best_solution


# Funkcja pomocnicza - odległość Levenshteina
def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def calculate_error_rate(dna_sequence, best_solution):
    reconstructed_sequence = ''.join([dna_sequence[i - 1] for i in best_solution])
    error_count = levenshtein_distance(dna_sequence, reconstructed_sequence)
    error_rate = error_count / len(dna_sequence)
    return error_rate


# Przykład użycia algorytmu
num_ants = 50
num_iterations = 1
decay_rate = 0.1
alpha = 1
beta = 2

with open('dna_sequences.txt', 'r') as f:
    dna_sequences = [line.strip() for line in f]


test_set = []
# Iteracja po liście sekwencji DNA
for num_ants in range(50, 300, 50):
    error_rate = 0
    for seq in dna_sequences:
        best_solution = ant_colony_optimization(num_ants, num_iterations, decay_rate, alpha, beta, seq)
        error_rate = error_rate + calculate_error_rate(seq, best_solution)
    error_desc = "Ilosc mrowek: " + str(num_ants) + " procentowa ilosc błedu: " + str(error_rate/len(dna_sequences))
    test_set.append(error_desc)

with open('ant_chng.txt', 'w') as f:
    for errors in test_set:
        f.write(errors + '\n')

