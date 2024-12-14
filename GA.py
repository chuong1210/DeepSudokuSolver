import numpy as np
import random
import operator
from past.builtins import range

random.seed()

class Sudoku:
    def __init__(self, grid_size=9):
        self.Nd = grid_size  # Number of digits (9 for 9x9, 16 for 16x16)
        self.given = None

    def load(self, p):
        self.given = Fixed(p, self.Nd)
        return

    def solve(self):
        Nc = 1000  # Number of candidates (i.e. population size).
        Ne = int(0.05 * Nc)  # Number of elites.
        Ng = 10000  # Number of generations.
        Nm = 0  # Number of mutations.

        # Mutation parameters.
        phi = 0
        sigma = 1
        mutation_rate = 0.06

        # Check given one first
        if not self.given.no_duplicates():
            return (-1, 1)

        self.population = Population(self.Nd)
        print("Creating an initial population.")
        if self.population.seed(Nc, self.given) == 1:
            pass
        else:
            return (-1, 1)

        # For up to 10000 generations...
        stale = 0
        for generation in range(0, Ng):

                # Check for a solution.
                best_fitness = 0.0
                #best_fitness_population_values = self.population.candidates[0].values
                for c in range(0, Nc):
                    fitness = self.population.candidates[c].fitness
                    if (fitness == 1):
                        print("Solution found at generation %d!" % generation)
                        return (generation, self.population.candidates[c])

                    # Find the best fitness and corresponding chromosome
                    if (fitness > best_fitness):
                        best_fitness = fitness
                        #best_fitness_population_values = self.population.candidates[c].values

                print("Generation:", generation, " Best fitness:", best_fitness)
                #print(best_fitness_population_values)

                # Create the next population.
                next_population = []

                # Select elites (the fittest candidates) and preserve them for the next generation.
                self.population.sort()
                elites = []
                for e in range(0, Ne):
                    elite = Candidate(self.Nd)
                    elite.values = np.copy(self.population.candidates[e].values)
                    elites.append(elite)

                # Create the rest of the candidates.
                for count in range(Ne, Nc, 2):
                    # Select parents from population via a tournament.
                    t = Tournament()
                    parent1 = t.compete(self.population.candidates)
                    parent2 = t.compete(self.population.candidates)

                    ## Cross-over.
                    cc = CycleCrossover(self.Nd)
                    child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

                    # Mutate child1.
                    child1.update_fitness()
                    old_fitness = child1.fitness
                    success = child1.mutate(mutation_rate, self.given)
                    child1.update_fitness()
                    if (success):
                        Nm += 1
                        if (child1.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                            phi = phi + 1

                    # Mutate child2.
                    child2.update_fitness()
                    old_fitness = child2.fitness
                    success = child2.mutate(mutation_rate, self.given)
                    child2.update_fitness()
                    if (success):
                        Nm += 1
                        if (child2.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                            phi = phi + 1

                    # Add children to new population.
                    next_population.append(child1)
                    next_population.append(child2)

                # Append elites onto the end of the population. These will not have been affected by crossover or mutation.
                for e in range(0, Ne):
                    next_population.append(elites[e])

                # Select next generation.
                self.population.candidates = next_population
                self.population.update_fitness()

                # Calculate new adaptive mutation rate (based on Rechenberg's 1/5 success rule).
                # This is to stop too much mutation as the fitness progresses towards unity.
                if (Nm == 0):
                    phi = 0  # Avoid divide by zero.
                else:
                    phi = phi / Nm

                if (phi > 0.2):
                    sigma = sigma / 0.998
                elif (phi < 0.2):
                    sigma = sigma * 0.998

                mutation_rate = abs(np.random.normal(loc=0.0, scale=sigma, size=None))

                # Check for stale population.
                self.population.sort()
                if (self.population.candidates[0].fitness != self.population.candidates[1].fitness):
                    stale = 0
                else:
                    stale += 1

                # Re-seed the population if 100 generations have passed
                # with the fittest two candidates always having the same fitness.
                if (stale >= 100):
                    print("The population has gone stale. Re-seeding...")
                    self.population.seed(Nc, self.given)
                    stale = 0
                    sigma = 1
                    phi = 0
                    mutation_rate = 0.06

        print("No solution found.")
        return (-2, 1)

class Population:
    def __init__(self, Nd):
        self.candidates = []
        self.Nd = Nd

    def seed(self, Nc, given):
        self.candidates = []

        helper = Candidate(self.Nd)
        helper.values = [[[] for j in range(self.Nd)] for i in range(self.Nd)]
        for row in range(self.Nd):
            for column in range(self.Nd):
                for value in range(1, self.Nd + 1):
                    if ((given.values[row][column] == 0) and 
                        not (given.is_column_duplicate(column, value) or 
                             given.is_block_duplicate(row, column, value) or 
                             given.is_row_duplicate(row, value))):
                        helper.values[row][column].append(value)
                    elif given.values[row][column] != 0:
                        helper.values[row][column].append(given.values[row][column])
                        break

        # Seed a new population
        for p in range(Nc):
            g = Candidate(self.Nd)
            for i in range(self.Nd):
                row = np.zeros(self.Nd)
                for j in range(self.Nd):
                    if given.values[i][j] != 0:
                        row[j] = given.values[i][j]
                    elif given.values[i][j] == 0:
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                ii = 0
                while len(set(row)) != self.Nd:
                    ii += 1
                    if ii > 500000:
                        return 0
                    for j in range(self.Nd):
                        if given.values[i][j] == 0:
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                g.values[i] = row

            self.candidates.append(g)

        self.update_fitness()
        return 1


    def update_fitness(self):
        """ Update fitness of every candidate/chromosome. """
        for candidate in self.candidates:

            candidate.update_fitness()
        return

    def sort(self):
        """ Sort the population based on fitness. """
        self.candidates = sorted(self.candidates, key=operator.attrgetter('fitness'))
        return


class Candidate:
    def __init__(self, Nd):
        self.values = np.zeros((Nd, Nd))
        self.fitness = None
        self.Nd = Nd

    def update_fitness(self):
        column_count = np.zeros(self.Nd)
        block_count = np.zeros(self.Nd)
        column_sum = 0
        block_sum = 0

        self.values = self.values.astype(int)
        for j in range(self.Nd):
            for i in range(self.Nd):
                column_count[self.values[i][j] - 1] += 1

            for k in range(len(column_count)):
                if column_count[k] == 1:
                    column_sum += (1/self.Nd)/self.Nd
            column_count = np.zeros(self.Nd)

        block_size = int(np.sqrt(self.Nd))
        for i in range(0, self.Nd, block_size):
            for j in range(0, self.Nd, block_size):
                block = self.values[i:i+block_size, j:j+block_size].flatten()
                for value in block:
                    block_count[value - 1] += 1

                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1/self.Nd)/self.Nd
                block_count = np.zeros(self.Nd)

        if int(column_sum) == 1 and int(block_sum) == 1:
            fitness = 1.0
        else:
            fitness = column_sum * block_sum

        self.fitness = fitness

    def mutate(self, mutation_rate, given):
            """ Mutate a candidate by picking a row, and then picking two values within that row to swap. """

            r = random.uniform(0, 1.1)
            while r > 1:  # Outside [0, 1] boundary - choose another
                r = random.uniform(0, 1.1)

            success = False
            if r < mutation_rate:  # Mutate.
                while not success:
                    row1 = random.randint(0, 8)
                    row2 = random.randint(0, 8)
                    row2 = row1

                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)
                    while from_column == to_column:
                        from_column = random.randint(0, 8)
                        to_column = random.randint(0, 8)

                        # Check if the two places are free to swap
                    if given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0:
                        # ...and that we are not causing a duplicate in the rows' columns.
                        if not given.is_column_duplicate(to_column, self.values[row1][from_column]) and not given.is_column_duplicate(from_column, self.values[row2][to_column]) and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column]) and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column]):
                            # Swap values.
                            temp = self.values[row2][to_column]
                            self.values[row2][to_column] = self.values[row1][from_column]
                            self.values[row1][from_column] = temp
                            success = True

            return success


class Fixed:
    def __init__(self, values, Nd):
        self.values = values
        self.Nd = Nd

    def is_row_duplicate(self, row, value):
        for column in range(self.Nd):
            if self.values[row][column] == value:
                return True
        return False

    def is_column_duplicate(self, column, value):
        for row in range(self.Nd):
            if self.values[row][column] == value:
                return True
        return False

    def is_block_duplicate(self, row, column, value):
        block_size = int(np.sqrt(self.Nd))
        i = block_size * (row // block_size)
        j = block_size * (column // block_size)

        for x in range(block_size):
            for y in range(block_size):
                if self.values[i+x][j+y] == value:
                    return True
        return False

    def make_index(self, v):
        block_size = int(np.sqrt(self.Nd))
        return (v // block_size) * block_size

    def no_duplicates(self):
        for row in range(self.Nd):
            for col in range(self.Nd):
                if self.values[row][col] != 0:
                    cnt1 = list(self.values[row]).count(self.values[row][col])
                    cnt2 = list(self.values[:,col]).count(self.values[row][col])

                    block_size = int(np.sqrt(self.Nd))
                    block_values = self.values[self.make_index(row):self.make_index(row)+block_size,
                                               self.make_index(col):self.make_index(col)+block_size]
                    cnt3 = list(block_values.flatten()).count(self.values[row][col])

                    if cnt1 > 1 or cnt2 > 1 or cnt3 > 1:
                        return False
        return True


class Tournament(object):
    """ The crossover function requires two parents to be selected from the population pool. The Tournament class is used to do this.

    Two individuals are selected from the population pool and a random number in [0, 1] is chosen. If this number is less than the 'selection rate' (e.g. 0.85), then the fitter individual is selected; otherwise, the weaker one is selected.
    """

    def __init__(self):

        return

    def compete(self, candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1 = candidates[random.randint(0, len(candidates) - 1)]
        c2 = candidates[random.randint(0, len(candidates) - 1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if (f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        # selection_rate = 0.85
        selection_rate = 0.80
        r = random.uniform(0, 1.1)
        while (r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
        if (r < selection_rate):
            return fittest
        else:
            return weakest


class CycleCrossover(object):
    """ Crossover relates to the analogy of genes within each parent candidate
    mixing together in the hopes of creating a fitter child candidate.
    Cycle crossover is used here (see e.g. A. E. Eiben, J. E. Smith.
    Introduction to Evolutionary Computing. Springer, 2007). """

    def __init__(self,Nd):
        self.Nd = Nd

        return

    def crossover(self, parent1, parent2, crossover_rate):
        """ Create two new child candidates by crossing over parent genes. """
        child1 = Candidate(self.Nd)
        child2 = Candidate(self.Nd)

        # Make a copy of the parent genes.
        child1.values = np.copy(parent1.values)
        child2.values = np.copy(parent2.values)

        r = random.uniform(0, 1.1)
        while (r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)

        # Perform crossover.
        if (r < crossover_rate):
            # Pick a crossover point. Crossover must have at least 1 row (and at most Nd-1) rows.
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while (crossover_point1 == crossover_point2):
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)

            if (crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp

            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2):
        child_row1 = np.zeros(self.Nd)
        child_row2 = np.zeros(self.Nd)

        remaining = range(1, self.Nd + 1)
        cycle = 0

        while ((0 in child_row1) and (0 in child_row2)):  # While child rows not complete...
            if (cycle % 2 == 0):  # Even cycles.
                # Assign next unused value.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]

                while (next != start):  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:  # Odd cycle - flip values.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]

                while (next != start):  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]

                cycle += 1

        return child_row1, child_row2

    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if (parent_row[i] in remaining):
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if (parent_row[i] == value):
                return i

# The Tournament and CycleCrossover classes can remain largely unchanged
# Just make sure to replace any hardcoded 9 values with Nd

# Example usage
sudoku = Sudoku(9)  # For 9x9 grid
# sudoku = Sudoku(16)  # For 16x16 grid
# Load your puzzle here
# sudoku.load(your_puzzle)
# generation, solution = sudoku.solve()

