import random
import time
import math
import numpy as np

def make_gene(grid_size, initial=None):
    gene = list(range(1, grid_size + 1))
    if initial is None:
        random.shuffle(gene)
        return gene
    
    mapp = {v: i for i, v in enumerate(gene)}
    for i in range(grid_size):
        if initial[i] != 0 and gene[i] != initial[i]:
            j = mapp[initial[i]]
            gene[i], gene[j] = initial[i], gene[i]
            mapp[gene[j]] = j
    return gene

def make_chromosome(grid_size, initial=None):
    if initial is None:
        initial = [[0] * grid_size for _ in range(grid_size)]
    return [make_gene(grid_size, row) for row in initial]

def make_population(count, grid_size, initial=None):
    return [make_chromosome(grid_size, initial) for _ in range(count)]

def get_fitness(chromosome):
    grid_size = len(chromosome)
    subgrid_size = int(math.sqrt(grid_size))
    fitness = 0
    
    # Check columns
    for col in range(grid_size):
        seen = set()
        for row in range(grid_size):
            if chromosome[row][col] in seen:
                fitness -= 1
            seen.add(chromosome[row][col])
    
    # Check subgrids
    for i in range(0, grid_size, subgrid_size):
        for j in range(0, grid_size, subgrid_size):
            seen = set()
            for x in range(subgrid_size):
                for y in range(subgrid_size):
                    value = chromosome[i+x][j+y]
                    if value in seen:
                        fitness -= 1
                    seen.add(value)
    
    return fitness

def crossover(ch1, ch2):
    crossover_point = random.randint(1, len(ch1) - 1)
    return (ch1[:crossover_point] + ch2[crossover_point:],
            ch2[:crossover_point] + ch1[crossover_point:])

def mutation(ch, pm, initial):
    grid_size = len(ch)
    return [make_gene(grid_size, initial[i]) if random.random() < pm else row for i, row in enumerate(ch)]

def get_mating_pool(population):
    fitness_list = [(get_fitness(ch), ch) for ch in population]
    fitness_list.sort(reverse=True)
    total_fitness = sum(fit for fit, _ in fitness_list)
    weights = [fit / total_fitness for fit, _ in fitness_list]
    return random.choices(population, weights=weights, k=len(population))

def get_offsprings(population, initial, pm, pc):
    new_pool = []
    for i in range(0, len(population), 2):
        ch1, ch2 = population[i], population[min(i+1, len(population)-1)]
        if random.random() < pc:
            ch1, ch2 = crossover(ch1, ch2)
        new_pool.extend([mutation(ch1, pm, initial), mutation(ch2, pm, initial)])
    return new_pool

def genetic_algorithm(initial, population_size=1000, repetitions=1000, pm=0.1, pc=0.95):
    grid_size = len(initial)
    population = make_population(population_size, grid_size, initial)
    for _ in range(repetitions):
        mating_pool = get_mating_pool(population)
        population = get_offsprings(mating_pool, initial, pm, pc)
        best_fitness = max(get_fitness(ch) for ch in population)
        if best_fitness == 0:
            return population
    return population

def solveGA(puzzle, population_size=1000, repetitions=1000, pm=0.1, pc=0.95):
    start_time = time.time()
    result = genetic_algorithm(puzzle, population_size, repetitions, pm, pc)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    best_fitness = max(get_fitness(ch) for ch in result)
    print(f"Best fitness: {best_fitness}")
    
    for ch in result:
        if get_fitness(ch) == best_fitness:
            solution = np.array(ch)
            mask = np.array(puzzle) != 0
            solution[mask] = np.array(puzzle)[mask]
            return solution.tolist()
    
    return None

# # Test the optimized solver
# puzzle_9x9 = [
#     [5, 3, 0, 0, 7, 0, 0, 0, 0],
#     [6, 0, 0, 1, 9, 5, 0, 0, 0],
#     [0, 9, 8, 0, 0, 0, 0, 6, 0],
#     [8, 0, 0, 0, 6, 0, 0, 0, 3],
#     [4, 0, 0, 8, 0, 3, 0, 0, 1],
#     [7, 0, 0, 0, 2, 0, 0, 0, 6],
#     [0, 6, 0, 0, 0, 0, 2, 8, 0],
#     [0, 0, 0, 4, 1, 9, 0, 0, 5],
#     [0, 0, 0, 0, 8, 0, 0, 7, 9]
# ]

# print("Solving 9x9 Sudoku:")
# solution_9x9 = solveGA(puzzle_9x9)
# if solution_9x9:
#     for row in solution_9x9:
#         print(" ".join(map(str, row)))
# else:
#     print("No solution found for 9x9 puzzle.")

# puzzle_16x16 = [
#     [ 1,  0,  0,  0,  5,  0,  0,  0,  9,  0,  0,  0, 13,  0,  0,  0],
#     [ 0,  2,  0,  0,  0,  6,  0,  0,  0, 10,  0,  0,  0, 14,  0,  0],
#     [ 0,  0,  3,  0,  0,  0,  7,  0,  0,  0, 11,  0,  0,  0, 15,  0],
#     [ 0,  0,  0,  4,  0,  0,  0,  8,  0,  0,  0, 12,  0,  0,  0, 16],
#     [ 5,  0,  0,  0,  1,  0,  0,  0, 13,  0,  0,  0,  9,  0,  0,  0],
#     [ 0,  6,  0,  0,  0,  2,  0,  0,  0, 14,  0,  0,  0, 10,  0,  0],
#     [ 0,  0,  7,  0,  0,  0,  3,  0,  0,  0, 15,  0,  0,  0, 11,  0],
#     [ 0,  0,  0,  8,  0,  0,  0,  4,  0,  0,  0, 16,  0,  0,  0, 12],
#     [ 9,  0,  0,  0, 13,  0,  0,  0,  1,  0,  0,  0,  5,  0,  0,  0],
#     [ 0, 10,  0,  0,  0, 14,  0,  0,  0,  2,  0,  0,  0,  6,  0,  0],
#     [ 0,  0, 11,  0,  0,  0, 15,  0,  0,  0,  3,  0,  0,  0,  7,  0],
#     [ 0,  0,  0, 12,  0,  0,  0, 16,  0,  0,  0,  4,  0,  0,  0,  8],
#     [13,  0,  0,  0,  9,  0,  0,  0,  5,  0,  0,  0,  1,  0,  0,  0],
#     [ 0, 14,  0,  0,  0, 10,  0,  0,  0,  6,  0,  0,  0,  2,  0,  0],
#     [ 0,  0, 15,  0,  0,  0, 11,  0,  0,  0,  7,  0,  0,  0,  3,  0],
#     [ 0,  0,  0, 16,  0,  0,  0, 12,  0,  0,  0,  8,  0,  0,  0,  4]
# ]

# print("\nSolving 16x16 Sudoku:")
# solution_16x16 = solveGA(puzzle_16x16, population_size=2000, repetitions=2000, pm=0.2, pc=0.9)
# if solution_16x16:
#     for row in solution_16x16:
#         print(" ".join(f"{num:2d}" for num in row))
# else:
#     print("No solution found for 16x16 puzzle.")