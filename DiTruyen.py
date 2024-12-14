import numpy as np
import random
import operator
from past.builtins import range

random.seed()

Nd = 9  # Number of digits (in the case of standard Sudoku puzzles, this is 9x9).

class QuanThe(object):
    """ A set of candidate solutions to the Sudoku puzzle.
    These ungviens are also known as the chromosomes in the population. """

    def __init__(self):
        self.ungviens = []
        return

    def seed(self, Nc, given):
        self.ungviens = []

        ultil = UngVien()
        ultil.values = [[[] for j in range(0, Nd)] for i in range(0, Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    if ((given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))):
                        # Value is available.
                        ultil.values[row][column].append(value)
                    elif given.values[row][column] != 0:
                        # Given/known value from file.
                        ultil.values[row][column].append(given.values[row][column])
                        break

        for p in range(0, Nc):
            g = UngVien()
            for i in range(0, Nd):  
                row = np.zeros(Nd)

                for j in range(0, Nd):  # New column j value in row i.

                    if given.values[i][j] != 0:
                        row[j] = given.values[i][j]
                    elif given.values[i][j] == 0:
                        row[j] = ultil.values[i][j][random.randint(0, len(ultil.values[i][j]) - 1)]

                ii = 0
                while len(list(set(row))) != Nd:
                    ii += 1
                    if ii > 500000:
                        return 0
                    for j in range(0, Nd):
                        if given.values[i][j] == 0:
                            row[j] = ultil.values[i][j][random.randint(0, len(ultil.values[i][j]) - 1)]

                g.values[i] = row
            self.ungviens.append(g)
        self.capnhatdothichnghi()

        # print("Seeding complete.")

        return 1

    def capnhatdothichnghi(self):
        """ Update fitness of every candidate/chromosome. """
        for candidate in self.ungviens:

            candidate.capnhatdothichnghi()
        return

    def SapXep(self):
        """ Sort the population based on fitness. """
        self.ungviens = sorted(self.ungviens, key=operator.attrgetter('fitness'))
        return

class UngVien(object):
    """ A candidate solutions to the Sudoku puzzle. """

    def __init__(self):
        self.values = np.zeros((Nd, Nd))
        self.fitness = None
        return

    def capnhatdothichnghi(self):
    
        number_column = np.zeros(Nd)
        block_value = np.zeros(Nd)
        tong_column = 0
        tong_block = 0

        self.values = self.values.astype(int)
        # For each column....
        for j in range(0, Nd):
            for i in range(0, Nd):
                number_column[self.values[i][j] - 1] += 1

            # unique
            # tong_column += (1.0 / len(set(number_column))) / Nd
            # set
            # for k in range(len(number_column)):
            #     if number_column[k] != 0:
            #         tong_column += (1/Nd)/Nd
            # duplicate
            for k in range(len(number_column)):
                if number_column[k] == 1:
                    tong_column += (1/Nd)/Nd
            number_column = np.zeros(Nd)

        # For each block...
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_value[self.values[i][j] - 1] += 1
                block_value[self.values[i][j + 1] - 1] += 1
                block_value[self.values[i][j + 2] - 1] += 1

                block_value[self.values[i + 1][j] - 1] += 1
                block_value[self.values[i + 1][j + 1] - 1] += 1
                block_value[self.values[i + 1][j + 2] - 1] += 1

                block_value[self.values[i + 2][j] - 1] += 1
                block_value[self.values[i + 2][j + 1] - 1] += 1
                block_value[self.values[i + 2][j + 2] - 1] += 1

                # unique
                # tong_block += (1.0 / len(set(block_value))) / Nd
                # set
                # for k in range(len(block_value)):
                #     if block_value[k] != 0:
                #         tong_block += (1/Nd)/Nd
                # duplicate
                for k in range(len(block_value)):
                    if block_value[k] == 1:
                        tong_block += (1/Nd)/Nd
                block_value = np.zeros(Nd)

        # Calculate overall fitness.
        if int(tong_column) == 1 and int(tong_block) == 1:
            fitness = 1.0
        else:
            fitness = tong_column * tong_block

        self.fitness = fitness
        return

    def dotbien(self, mutation_rate, given):
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


class CheckGrid(UngVien):

    def __init__(self, values):
        self.values = values
        return

    def is_row_duplicate(self, row, value):
        for column in range(0, Nd):
            if self.values[row][column] == value:
                return True
        return False

    def is_column_duplicate(self, column, value):
        """ Check duplicate in a column. """
        for row in range(0, Nd):
            if self.values[row][column] == value:
                return True
        return False

    def is_block_duplicate(self, row, column, value):
        """ Check duplicate in a 3 x 3 block. """
        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))

        if ((self.values[i][j] == value)
            or (self.values[i][j + 1] == value)
            or (self.values[i][j + 2] == value)
            or (self.values[i + 1][j] == value)
            or (self.values[i + 1][j + 1] == value)
            or (self.values[i + 1][j + 2] == value)
            or (self.values[i + 2][j] == value)
            or (self.values[i + 2][j + 1] == value)
            or (self.values[i + 2][j + 2] == value)):
            return True
        else:
            return False

    def make_index(self, v):
        if v <= 2:
            return 0
        elif v <= 5:
            return 3
        else:
            return 6

    def no_duplicates(self):
        for row in range(0, Nd):
            for col in range(0, Nd):
                if self.values[row][col] != 0:

                    cnt1 = list(self.values[row]).count(self.values[row][col])
                    cnt2 = list(self.values[:,col]).count(self.values[row][col])

                    block_values = [y[self.make_index(col):self.make_index(col)+3] for y in
                                    self.values[self.make_index(row):self.make_index(row)+3]]
                    block_values_ = [int(x) for y in block_values for x in y]
                    cnt3 = block_values_.count(self.values[row][col])

                    if cnt1 > 1 or cnt2 > 1 or cnt3 > 1:
                        return False
        return True

class Tournament(object):
    """ The crossover function requires two parents to be selected from the population pool. The Tournament class is used to do this.

    Two individuals are selected from the population pool and a random number in [0, 1] is chosen. If this number is less than the 'selection rate' (e.g. 0.85), then the fitter individual is selected; otherwise, the weaker one is selected.
    """

    def __init__(self):
        return

    def compete(self, ungviens):
        """ Pick 2 random ungviens from the population and get them to compete against each other. """
        c1 = ungviens[random.randint(0, len(ungviens) - 1)]
        c2 = ungviens[random.randint(0, len(ungviens) - 1)]
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


class VongDoiCrossover(object):

    def __init__(self):
        return

    def crossover(self, cha1, cha2, crossover_rate):
        """ Create two new child ungviens by crossing over parent genes. """
        con1 = UngVien()
        con2 = UngVien()

        # Make a copy of the parent genes.
        con1.values = np.copy(cha1.values)
        con2.values = np.copy(cha2.values)

        r = random.uniform(0, 1.1)
        while (r > 1):  
            r = random.uniform(0, 1.1)

        if (r < crossover_rate):
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
                con1.values[i], con2.values[i] = self.crossover_rows(con1.values[i], con2.values[i])

        return con1, con2

    def crossover_rows(self, row1, row2):
        child_row1 = np.zeros(Nd)
        child_row2 = np.zeros(Nd)

        remaining = range(1, Nd + 1)
        cycle = 0

        while ((0 in child_row1) and (0 in child_row2)): 
            if (cycle % 2 == 0):  
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]

                while (next != start): 
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else: 
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]

                while (next != start):  
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


class Sudoku(object):

    def __init__(self):
        self.given = None
        return

    def load(self, p):
        self.given = CheckGrid(p)
        return

    def solve(self):

        Nc = 1000  
        Ne = int(0.05 * Nc) 
        Ng = 10000  
        Nm = 0 

        phi = 0
        sigma = 1
        mutation_rate = 0.06

        # Check given one first
        if self.given.no_duplicates() == False:
            return (-1, 1)

        self.population = QuanThe()
        print("create an initial population.")
        if self.population.seed(Nc, self.given) ==  1:
            pass
        else:
            return (-1, 1)

        # For up to 10000 generations...
        stale = 0
        for generation in range(0, Ng):

            # Check for a solution.
            best_fitness = 0.0
            #best_fitness_population_values = self.population.ungviens[0].values
            for c in range(0, Nc):
                fitness = self.population.ungviens[c].fitness
                if (fitness == 1):
                    print("Solution found at generation %d!" % generation)
                    return (generation, self.population.ungviens[c])

                # Find the best fitness and corresponding chromosome
                if (fitness > best_fitness):
                    best_fitness = fitness
                    #best_fitness_population_values = self.population.ungviens[c].values

            print("Generation:", generation, " Best fitness:", best_fitness)
            #print(best_fitness_population_values)

            # Create the next population.
            next_population = []

            # Select elites (the fittest ungviens) and preserve them for the next generation.
            self.population.SapXep()
            elites = []
            for e in range(0, Ne):
                elite = UngVien()
                elite.values = np.copy(self.population.ungviens[e].values)
                elites.append(elite)

            # Create the rest of the ungviens.
            for count in range(Ne, Nc, 2):
                # Select parents from population via a tournament.
                t = Tournament()
                cha1 = t.compete(self.population.ungviens)
                cha2 = t.compete(self.population.ungviens)

                ## Cross-over.
                vd = VongDoiCrossover()
                con1, con2 = vd.crossover(cha1, cha2, crossover_rate=1.0)

                # Mutate con1.
                con1.capnhatdothichnghi()
                old_fitness = con1.fitness
                success = con1.dotbien(mutation_rate, self.given)
                con1.capnhatdothichnghi()
                if (success):
                    Nm += 1
                    if (con1.fitness > old_fitness): 
                        phi = phi + 1

                con2.capnhatdothichnghi()
                old_fitness = con2.fitness
                success = con2.dotbien(mutation_rate, self.given)
                con2.capnhatdothichnghi()
                if (success):
                    Nm += 1
                    if (con2.fitness > old_fitness): 
                        phi = phi + 1

                next_population.append(con1)
                next_population.append(con2)

            for e in range(0, Ne):
                next_population.append(elites[e])

            # Select next generation.
            self.population.ungviens = next_population
            self.population.capnhatdothichnghi()

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
            self.population.SapXep()
            if (self.population.ungviens[0].fitness != self.population.ungviens[1].fitness):
                stale = 0
            else:
                stale += 1

            # Re-seed the population if 100 generations have passed
            # with the fittest two ungviens always having the same fitness.
            if (stale >= 100):
                print("Load lại vì data đã cũ load loại")
                self.population.seed(Nc, self.given)
                stale = 0
                sigma = 1
                phi = 0
                mutation_rate = 0.06

        print("Không thấy đáp án")
        return (-2, 1)