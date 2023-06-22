import random
from functools import partial
from multiprocessing import Pool

import numpy as np
from numpy import floor


def initialize_pheromone_matrix(length):
    return np.ones((length, length))


def initialize_ants(num_ants, length):
    ants = []
    for _ in range(num_ants):
        ant = list(range(1, length + 1))
        random.shuffle(ant)
        ants.append(ant)
    return ants


def calculate_fitness(solution, dna_sequence):
    reconstructed_sequence = ''.join([dna_sequence[i - 1] for i in solution])
    fitness = levenshtein_distance(dna_sequence, reconstructed_sequence)
    return fitness


def update_pheromones(pheromone_matrix, ants, decay_rate, best_fitness):
    pheromone_matrix *= (1 - decay_rate)
    delta_pheromone = 1 / best_fitness
    for ant in ants:
        current_indices = np.array(ant[:-1]) - 1
        next_indices = np.array(ant[1:]) - 1
        pheromone_matrix[current_indices, next_indices] += delta_pheromone


def choose_next_element(pheromone_matrix, attractiveness_matrix, visited, current_index, alpha, beta):
    unvisited = np.array([i for i in range(len(pheromone_matrix)) if i not in visited])
    pheromone = pheromone_matrix[current_index][unvisited]
    attractiveness = attractiveness_matrix[current_index][unvisited]
    probabilities = pheromone ** alpha * attractiveness ** beta
    probabilities /= np.sum(probabilities)
    next_index = np.random.choice(unvisited, p=probabilities)
    return next_index


def update_ant(ant, pheromone_matrix, attractiveness_matrix, alpha, beta, dna_sequence):
    visited = set()
    visited.add(ant[0])

    for i in range(len(ant) - 1):
        current_index = ant[i] - 1
        next_index = choose_next_element(pheromone_matrix, attractiveness_matrix, visited, current_index, alpha, beta)
        ant[i + 1] = next_index + 1
        visited.add(next_index)

    fitness = calculate_fitness(ant, dna_sequence)

    return ant, fitness


def ant_colony_optimization(num_ants, num_iterations, decay_rate, alpha, beta, dna_sequence):
    length = len(dna_sequence)
    pheromone_matrix = initialize_pheromone_matrix(length)
    attractiveness_matrix = np.random.rand(length, length)
    ants = initialize_ants(num_ants, length)
    best_solution = ants[0].copy()
    best_fitness = calculate_fitness(best_solution, dna_sequence)

    pool = Pool()

    for iteration in range(num_iterations):
        update_ant_partial = partial(update_ant, pheromone_matrix=pheromone_matrix,
                                     attractiveness_matrix=attractiveness_matrix, alpha=alpha, beta=beta,
                                     dna_sequence=dna_sequence)
        results = pool.map(update_ant_partial, ants)
        ants, fitnesses = zip(*results)

        for i in range(len(ants)):
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_solution = ants[i].copy()

        update_pheromones(pheromone_matrix, ants, decay_rate, best_fitness)

    pool.close()
    pool.join()

    return best_solution


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


if __name__ == '__main__':
    num_ants = 50
    num_iterations = 10
    decay_rate = 0.1
    alpha = 1
    beta = 2

    with open('dna_sequences.txt', 'r') as f:
        dna_sequences = [line.strip() for line in f]

    test_set = []
    max_numbers = len(dna_sequences) * 5
    count_percantage = 0
    # Iteracja po liście sekwencji DNA
    for alpha in range(1, 6, 1):
        error_rate = 0
        for seq in dna_sequences:
            count_percantage += 1
            best_solution = ant_colony_optimization(num_ants, num_iterations, decay_rate, alpha, beta, seq)
            error_rate = error_rate + calculate_error_rate(seq, best_solution)
            print("Zrealizowano już: " + str(floor((count_percantage / max_numbers) * 100)) + " %")

        error_desc = "Wspolczynnik alpha: " + str(alpha) + " procentowa ilosc błedu: " + str(
            error_rate / len(dna_sequences))
        test_set.append(error_desc)

    with open('decay_alpha_beta_change.txt', 'w') as f:
        for errors in test_set:
            f.write(errors + '\n')
