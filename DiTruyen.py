import random as rndm
import time
import math
# Nhập các thư viện cần thiết: random để tạo số ngẫu nhiên, time để đo thời gian thực thi, và math để tính toán

def make_gene(grid_size, initial=None):
    if initial is None:
        initial = [0] * grid_size
    mapp = {}
    gene = list(range(1, grid_size + 1))
    rndm.shuffle(gene)
    for i in range(grid_size):
        mapp[gene[i]] = i
    for i in range(grid_size):
        if initial[i] != 0 and gene[i] != initial[i]:
            temp = gene[i], gene[mapp[initial[i]]]
            gene[mapp[initial[i]]], gene[i] = temp
            mapp[initial[i]], mapp[temp[0]] = i, mapp[initial[i]]
    return gene
# Hàm tạo một gen (một hàng trong bảng Sudoku)
# - Nếu không có giá trị ban đầu, tạo một list các số từ 1 đến grid_size và xáo trộn
# - Nếu có giá trị ban đầu, đảm bảo các số đó được đặt đúng vị trí

def make_chromosome(grid_size, initial=None):
    if initial is None:
        initial = [[0] * grid_size for _ in range(grid_size)]
    chromosome = []
    for i in range(grid_size):
        chromosome.append(make_gene(grid_size, initial[i]))
    return chromosome
# Hàm tạo một nhiễm sắc thể (một bảng Sudoku hoàn chỉnh)
# - Tạo một list các gen, mỗi gen tương ứng với một hàng trong bảng Sudoku

def make_population(count, grid_size, initial=None):
    if initial is None:
        initial = [[0] * grid_size for _ in range(grid_size)]
    population = []
    for _ in range(count):
        population.append(make_chromosome(grid_size, initial))
    return population
# Hàm tạo quần thể ban đầu
# - Tạo một số lượng nhiễm sắc thể (bảng Sudoku) theo count đã cho

def get_fitness(chromosome):
    """Calculate the fitness of a chromosome (puzzle)."""
    grid_size = len(chromosome)
    subgrid_size = int(math.sqrt(grid_size))
    fitness = 0
    for i in range(grid_size): # For each column
        seen = {}
        for j in range(grid_size): # Check each cell in the column
            if chromosome[j][i] in seen:
                seen[chromosome[j][i]] += 1
            else:
                seen[chromosome[j][i]] = 1
        for key in seen: # Subtract fitness for repeated numbers
            fitness -= (seen[key] - 1)
    for m in range(subgrid_size): # For each subgrid
        for n in range(subgrid_size):
            seen = {}
            for i in range(subgrid_size * n, subgrid_size * (n + 1)):  # Check cells in subgrid
                for j in range(subgrid_size * m, subgrid_size * (m + 1)):
                    if chromosome[j][i] in seen:
                        seen[chromosome[j][i]] += 1
                    else:
                        seen[chromosome[j][i]] = 1
            for key in seen: # Subtract fitness for repeated numbers
                fitness -= (seen[key] - 1)
    return fitness
# Hàm tính độ phù hợp của một nhiễm sắc thể (bảng Sudoku)
# - Kiểm tra số lượng số trùng lặp trong mỗi cột và mỗi ô vuông con
# - Trừ điểm fitness cho mỗi số trùng lặp

def pch(ch):
    grid_size = len(ch)
    for i in range(grid_size):
        for j in range(grid_size):
            print(f"{ch[i][j]:2d}", end=" ")
        print("")
# Hàm in ra bảng Sudoku

def crossover(ch1, ch2):
    grid_size = len(ch1)
    new_child_1 = []
    new_child_2 = []
    for i in range(grid_size):
        x = rndm.randint(0, 1)
        if x == 1:
            new_child_1.append(ch1[i])
            new_child_2.append(ch2[i])
        elif x == 0:
            new_child_2.append(ch1[i])
            new_child_1.append(ch2[i])
    return new_child_1, new_child_2
# Hàm lai ghép hai nhiễm sắc thể (bảng Sudoku) để tạo ra hai nhiễm sắc thể con mới

def mutation(ch, pm, initial):
    grid_size = len(ch)
    for i in range(grid_size):
        x = rndm.randint(0, 100)
        if x < pm * 100:
            ch[i] = make_gene(grid_size, initial[i])
    return ch
# Hàm đột biến một nhiễm sắc thể
# - Với xác suất pm, tạo lại một gen (hàng) mới trong nhiễm sắc thể


def r_get_mating_pool(population):
    fitness_list = []
    pool = []
    for chromosome in population:
        fitness = get_fitness(chromosome)
        fitness_list.append((fitness, chromosome))
    fitness_list.sort()
    weight = list(range(1, len(fitness_list) + 1))
    for _ in range(len(population)):
        ch = rndm.choices(fitness_list, weight)[0]
        pool.append(ch[1])
    return pool
# Hàm chọn lọc các nhiễm sắc thể để tạo thành nhóm giao phối
# - Sử dụng phương pháp chọn lọc theo thứ hạng

def w_get_mating_pool(population):
    fitness_list = []
    pool = []
    for chromosome in population:
        fitness = get_fitness(chromosome)
        fitness_list.append((fitness, chromosome))
    weight = [fit[0] - fitness_list[0][0] for fit in fitness_list]
    for _ in range(len(population)):
        ch = rndm.choices(fitness_list, weights=weight)[0]
        pool.append(ch[1])
    return pool
# Hàm chọn lọc các nhiễm sắc thể để tạo thành nhóm giao phối
# - Sử dụng phương pháp chọn lọc theo trọng số

def get_offsprings(population, initial, pm, pc):
    new_pool = []
    i = 0
    while i < len(population):
        ch1 = population[i]
        ch2 = population[(i + 1) % len(population)]
        x = rndm.randint(0, 100)
        if x < pc * 100:
            ch1, ch2 = crossover(ch1, ch2)
        new_pool.append(mutation(ch1, pm, initial))
        new_pool.append(mutation(ch2, pm, initial))
        i += 2
    return new_pool
# Hàm tạo ra thế hệ mới từ quần thể hiện tại
# - Thực hiện lai ghép và đột biến để tạo ra các cá thể mới

# Main genetic algorithm function
def genetic_algorithm(initial, population_size=1000, repetitions=1000, pm=0.1, pc=0.95):
    grid_size = len(initial)
    population = make_population(population_size, grid_size, initial)
    for _ in range(repetitions):
        mating_pool = r_get_mating_pool(population)
        rndm.shuffle(mating_pool)
        population = get_offsprings(mating_pool, initial, pm, pc)
        fit = [get_fitness(c) for c in population]
        m = max(fit)
        if m == 0:
            return population
    return population
# Hàm thuật toán di truyền chính
# - Tạo quần thể ban đầu
# - Lặp lại quá trình chọn lọc, lai ghép và đột biến cho đến khi đạt được số lần lặp tối đa hoặc tìm thấy giải pháp

def solveGA(puzzle, population_size=1000, repetitions=1000, pm=0.1, pc=0.95):
    tic = time.time()
    result = genetic_algorithm(puzzle, population_size, repetitions, pm, pc)
    toc = time.time()
    print("Time taken: ", toc - tic)
    
    fit = [get_fitness(c) for c in result]
    m = max(fit)
    print("Best fitness:", m)

    # Tìm ra chromosome với fitness cao nhất
    for c in result:
        if get_fitness(c) == m:
            # Tạo một bản sao của bảng kết quả
            board_copy = [row[:] for row in c]
            return board_copy
    
    return None
# Hàm giải Sudoku sử dụng thuật toán di truyền
# - Gọi hàm genetic_algorithm để tìm giải pháp
# - In ra thời gian thực thi và giải pháp tốt nhất tìm được

# puzzle = [
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
# # Định nghĩa một bảng Sudoku 9x9 cần giải

# solveGA(puzzle)
# # Gọi hàm solveGA để giải bảng Sudoku 9x9

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
# # Định nghĩa một bảng Sudoku 16x16 cần giải

# solveGA(puzzle_16x16, population_size=2000, repetitions=2000, pm=0.2, pc=0.9)
# # Gọi hàm solveGA để giải bảng Sudoku 16x16 với các tham số điều chỉnh