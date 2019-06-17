from random import *
import copy
import sys
import datetime

def read(filename):
    """
    Function that reads in a file containing a soduku puzzle in the format:
        Empty cells are full stops '.',
        The character '!'is a semantic column separator and should be ignored,
        The string '---!---!---' is a semantic row separator and should be ignored.
    @param filename: filename of text file containing the puzzle in the directory
    @return: the Soduku puzzle represented as a matrix, where uninitialised values are represented as 0s
    """
    puzzle = []
    for line in open(filename, 'r'):
        row = []
        if (line != '---!---!---\n'):
            for char in line:
                if (char == '.'):
                    row.append(0)
                elif (char == '!' or char == '\n'):
                    pass
                else:
                    row.append(int(char))
        if (row != []):
            puzzle.append(row)
    return puzzle

def write(file_name, arr):
    """
    Function that writes the elemnts of an array into a textfile.
    @param file_name: the name of the file that is to be written out to (or created.
    @param arr: the array, the elements of which are to be written to a text file
    """
    f = open(file_name, "w+")
    for element in range(len(arr)):
        f.write(str(arr[element][0]) + "\n")
    f.close
    return None

def beautify(sudoku):
    """
    Function that prints a soduku puzzle, separating cells and subgrids.
    @param soduku: a matrix representing a soduku puzzle.
    @return: <Nothing>
    """
    for index, row in enumerate(sudoku):
        print (str(row[0:3]).strip('[]') + ' | ' + str(row[3:6]).strip('[]') + ' | ' + str(row[6:9]).strip('[]'))
        if (index % 3 == 2 and index != 8):
            print ('-' * 27)
    return None

def transpose(matrix):
    """
    Function that transposes a matrix.
    @param matrix: the matrix to be transposed
    @return: the transposed matrix
    """
    result = copy.deepcopy(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[j][i] = matrix[i][j]
    return result


class Sudoku:

    def __init__(self,\
                 puzzle,\
                 population_size,\
                 truncation_rate,\
                 k_tournament,\
                 mutation_rate,\
                 swaps_per_mutation,\
                 ):
        """
        Function that initialises the sudoku class.
        @param puzzle: the uninialised puzzle.
        @param population_size: the size of the population to be created.
        @param truncation_rate: the proportion of the generation to truncaat, before crossover, at each evolving stage
        @param k_trounament: the number of candidate solutions evaluated per tournament selection round.
        @param mutation_rate: the probability of a child being mutated.
        @param swaps_per_mutation: the number of cell swaps per mutation.
        """
        self.puzzle = puzzle
        self.population_size = population_size
        self.truncation_rate = truncation_rate
        self.k_tournament = k_tournament
        self.mutation_rate = mutation_rate
        self.swaps_per_mutation = swaps_per_mutation



    def populate(self, sudoku):
        """
        Function to initialise a grid using a random seed.
        @param sudoku: the matrix representing the puzzle.
        @return: a matrix, replacing each 0 with a random integer ranging from 1 to 0.

        """
        puzzle = copy.deepcopy(sudoku)
        for row in puzzle:
            self.hard_constraint(row)
        return puzzle


    def hard_constraint(self, row):
        """
        Function that makes sure that the only cells that are not already in the row are appended.
        Ensuring that numbers in the row are unique.
        @param row: the row to be populated
        @return: an array with some permutation from 1-9
        """
        legal_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        fixed = list(filter(lambda a: a !=0, row))
        legal_numbers = [x for x in legal_numbers if x not in fixed]
        for i in range(len(row)):
            if (row[i] == 0):
                index = randint(0,len(legal_numbers)-1)
                num = legal_numbers[index]
                row[i] = num
                legal_numbers.remove(num)
        return row


    def check(self, s1, s2):
        """
        Function that checks that a soduku puzzle has the same fixed positions as the initial puzzle.
        @param s1: the uninitalised sudoku puzzle.
        @param s2: the sudoku puzzle to be checked.
        @return: a boolean indicateing whether or not a populated puzzle has retained its fixed positions.
        """
        for index, row in enumerate(s1):
            for i in range(len(row)):
                if row[i] != 0:
                    if row[i] == s2[index][i]:
                        return True
                    else:
                        return False


    def is_fixed(self, initial_sudoku, row, column):
        """
        Function that checks if a cell in a puzzle is a fixed position.
        @param initial_sudoku: the uninitialised puzzle.
        @param row: the row coordinate of the cell to be checked.
        @param column: the column coordinate of the cell to be checked.
        @return: a boolean indicating whether or not a cell is a fixed position.
        """
        if (initial_sudoku[row][column] == 0):
            return False
        else:
            return True


    def create_population(self, init_sodoku, pop_size):
        """
        Function to create a population of candidate solutions.
        @param init_sudoku: uninitialised puzzle
        @param pop_size: size of the population to be created
        @return: an array of candidate solutions using the above method.
        """
        population = []
        for i in range(pop_size):
            population.append(self.populate(init_sodoku))
        return population


    def fitness_function(self, population):
        """
        Function to calculate the fitness of every candidate solution (cs) in a population.
            f(x) is the sum of the: repeats per column + repeats per row + repeats per 3x3 subgrid.
        @param population: the population of candidate solutions whose fitnesses are to be calculated.
        @return: an array populated with candidate solutions and thier corresponding fitness value.
        """
        fitness = []
        for cs in population:
            fitness.append(self.calculate_fitness(cs))
        return fitness


    def calculate_fitness(self, cs):
        """
        Function to calculate the fitness of a candidate solution.
        @param cs: the candidate solution whose fitness is to be calculated.
        @return: a list where the first element is the fitness and the seconds is the puzzle.
        """
        pair = [self.fitness_row(cs) + self.fitness_column(cs) + self.fitness_subgrid(cs), cs]
        return pair


    def fitness_row(self, puzzle):
        """
        Function to calculate the fitness of a puzzle regarding its rows,
        where the fitness of the row is the number of duplicate elements it contains.
        @param puzzle: the puzzle to be checked.
        @return: the total number of duplicates that can be found in each row.
        """
        duplicates = 0
        for row in puzzle:
            duplicates += len(row) - len(set(row))
        return duplicates


    def fitness_column(self, puzzle):
        """
        Function to calculate the fitness of a puzzle regarding its columns,
        where the fitness of the column is the number of duplicate elements it contains.
        @param puzzle: the puzzle to be checked.
        @return: the total number of duplicates that can be found in each column.
        """
        return self.fitness_row(transpose(puzzle))


    def fitness_subgrid(self, puzzle):
        """
        Function to calculate the fitness of a puzzle regarding each 3x3 subgrid,
        where the fitness of the subgrid is the number of duplicate elements it contains.
        @param puzzle: the puzzle to be checked.
        @return: the total number of duplicates that can be found in each subgrid.
        """
        all_subgrids = []
        duplicates = 0

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                temp = []
                for row in puzzle[i:i+3]:
                    holder = list(row[j:j+3])
                    temp.append(holder)
                subgrid = [y for x in temp for y in x] #flatten
                all_subgrids.append(subgrid)

        for grid in all_subgrids:
            duplicates += len(grid) - len(set(grid))
        return duplicates


    def greedy_selection(self, truncation_rate, population):
        """
        Function that returns the top x individuals from the population,
        by eliminating the bottom n individuals, specified the truncation rate.
        @param truncation_rate: the proportion of individuals to truncate from the population.
        @param population: the population to be truncated.
        @return: a truncated copy of the population.

        """
        new_pop = copy.deepcopy(population)
        n_individuals = int((1-truncation_rate) * len(population))
        shuffle(new_pop)
        new_pop.sort(key = lambda x: int(x[0]))
        return new_pop[0:n_individuals]


    def tournament(self, select_size, k_size, population):
        """
        Function that returns k individuals from the population, using a tournament based system.
        @param select_size: the number of individuals to select.
        @param k_size: the number of individuals evaluated per round.
        @param population: the population to select the individuals from.
        @return: an array of the selected candidate solutions.
        """
        assert select_size < len(population), "Tournament Error: Selection Size > Population Size"
        winners = []
        for x in range(select_size):
            winners.append(self.perform_round(k_size, population))
        return winners


    def perform_round(self, k_size, participants):
        """
        Function that simulates a round in a tournament, selecting one individual as a result.
        @param k_size: the number of individuals evaluated per round.
        @param participants: the population to select individuals from.
        @return: the selected candidate solution.
        """
        best= [216,0]
        candidates = []
        for i in range(k_size):
            individual = randint(0, len(participants)) - 1
            candidates.append(participants[individual])
        candidates.sort(key = lambda x: int(x[0]))
        return candidates[0]


    def uniform_crossover(self, parentA, parentB):
        """
        Function that takes in two parents and return two children using uniform corssover.
        @param parentA: a sudoku puzzle representing parent A.
        @param parentB: a sudoku puzzle representing parent B.
        @return: an array containing two children (two candidate solutions).
        """
        childA = []
        childB = []
        for n in range(9):
            prob = randint(0,1)
            if (prob == 1):
                childA.append(parentA[n])
                childB.append(parentB[n])
            else:
                childB.append(parentA[n])
                childA.append(parentB[n])
        parentA = transpose(childA)
        parentB = transpose(childB)

        childA = []
        childB = []

        #transpose the matrix and perform a unifrom crossover to allow a change in columns
        for n in range(9):
            prob = randint(0,1)
            if (prob == 1):
                childA.append(parentA[n])
                childB.append(parentB[n])
            else:
                childB.append(parentA[n])
                childA.append(parentB[n])

        return [transpose(childA), transpose(childB)]


    def uniform_crossover_f(self, parentA, parentB):
        """
        An extension of the uniform_crossover function that returns the children with thier respective fitnesses.
        @param parentA: puzzle representing parent A.
        @param parentB: puzzle representing parent B.
        @return: an array containing two candidate solutions and thier respective fitnesses.
        """
        return self.fitness_function(self.uniform_crossover(parentA, parentB))



    def mutate(self, cs, initial_sudoku, mutation_rate=0.4, n=1):
        """
        Function that mutates a candidate solution by swapping two legal cells in a randomly chosen row.
        @param cs: the candidate solition.
        @param initial_sudoku: the uninitialised puzzle.
        @param mutation_rate: the probability of a mutation taking place (Default=0.4).
        @param n: the number of cell swaps to take place per mutation (Default=1).
        @return: a candidate solution that may or not be a mutated version of cs.
        """
        count = 0
        if (mutation_rate > randint(0,100) / 100):
            while (count < n):
                temp = self.single_swap(cs, initial_sudoku)
                if (type(temp) != bool):
                    cs = temp
                    count += 1
        return cs


    def single_swap(self, cs, initial_sudoku):
        """
        Function that swaps two random cells if it is a legal move.
        @param cs: the candidate soltuion to be mutated.
        @param initial_sudoku: the uninitialised sudoku puzzle.
        @return: modified cs if a swap has occured, otherwise return false
        """
        row = randint(0,8)
        col1 = randint(0,8)
        col2 = randint(0,8)

        if (not(self.is_fixed(initial_sudoku, row, col1)) and not (self.is_fixed(initial_sudoku, row, col2))\
            and col1 != col2 and (cs[row][col1]!= cs[row][col2])):
            cs[row][col1], cs[row][col2] = cs[row][col2], cs[row][col1]
            return cs
        else:
            return False


    def uniform_breeding(self, mating_pool, no_of_offspring, k_tournament, initial_sudoku, mutation_rate=0.4, mutation_n=1):
        """
        Function that iterates over a mating pool and breeds them until n number of children been produced.
        @param mating_pool: the population to select parents from.
        @param no_of_offspring: the number offspring to produce.
        @param k_tournament: the number of individuals evaluated per round of the tournament selection.
        @param initial_sudoko: the uninialised sudoku puzzle.
        @param mutation_rate: the probability of a mutation occurring per child (Default=0.4)
        @param mutation_n: the number of swaps per mutation (Default=0.1)
        @return: an array of candidate solutions, with thier respective fitnesses.
        """
        offspring = []
        while (len(offspring) < no_of_offspring):
            parents = self.tournament(2, k_tournament, mating_pool)
            children = self.uniform_crossover(parents[0][1], parents[1][1])

            childA = self.calculate_fitness(self.mutate(children[0], initial_sudoku, mutation_rate, mutation_n))
            childB = self.calculate_fitness(self.mutate(children[1], initial_sudoku, mutation_rate, mutation_n))

            offspring.append(childA)
            offspring.append(childB)

        return offspring


    def evolve(self, population, truncation_rate, k_tournament, initial_sudoku, mutation_rate=0.4, mutation_n=1):
        """
        Function that evolves Generation N into Generation N+1.
        @param population: the generation to be evolved.
        @param truncation_rate: the proportion of the population to truncate before crossover.
        @param k_tournament: the number of candidate solutions evaluated per round of the tournament selection.
        @param initial_sudoku: the unitialised puzzle.
        @param mutation_rate: the probability of each child produced being mutated (Default=0.4).
        @param mutation_n: the number of swaps made per mutation (Default=1).
        @return: best_individual: the best candidate solution of the individual.
        @return: next_gen: an array containing the candidate solutions of the evolved Generation N+1.
        @return: average_fitness: the average fitness of the Generation N+1
        """
        #Extract a mating pool from the population
        mating_pool = self.greedy_selection(truncation_rate, population)

        #Create offspring
        offspring = self.uniform_breeding(mating_pool, len(population), k_tournament, initial_sudoku, mutation_rate=0.4, mutation_n=1)

        #Survivour Strategy (Generation N+1 =  best 25% of GenerationN and best 75% of the offspring)
        mating_pool = mating_pool[0:int(0.3*len(population))]
        offspring.sort(key = lambda x: int(x[0]) )
        offspring = offspring[0: int(0.7*len(population))]

        #Next Generation
        next_gen = offspring + mating_pool
        next_gen.sort(key = lambda x: int(x[0]))

        #Find Best Individual
        best_individual = min(next_gen)

        #Find Average Fitness
        average_fitness =  sum(i[0] for i in next_gen)/len(next_gen)

        return best_individual, next_gen, average_fitness


    def run(self):
        """
        Function that runs an Evolutionary Algorithm on the sudoku puzzle.
        @return: an array of the best individuals from each generation.
        """

        gen_count = 1
        best_individuals = []

        init_sudoku = self.puzzle

        #termination criteria
        cs_produced = self.population_size
        LIMIT = 500000

        #Create an initial population
        init_pop = self.create_population(init_sudoku, self.population_size)
        population = self.fitness_function(init_pop)
        best_individuals.append(min(population))

        #Initial Population Information
        print("Gen " + str(gen_count) + " Best: " + str(best_individuals[-1][0]))
        print("Gen " + str(gen_count) + " Avg: " + str(sum(i[0] for i in population)/len(population)))
        print("")

        #Keep producing new generations until the termination criteron is achieved

        while ( cs_produced < LIMIT and best_individuals[-1][0] != 0 ):

            best, new_gen, average = self.evolve(population, self.truncation_rate, self.k_tournament, init_sudoku, self.mutation_rate, self.swaps_per_mutation)

            population = new_gen

            best_individuals.append(best)

            cs_produced += len(new_gen)

            gen_count += 1
            print("Gen " + str(gen_count) + " Best: " + str(best_individuals[-1][0]))
            print("Gen " + str(gen_count) + " Avg: " + str(average))
            print("")

        #Termination information
        if( best_individuals[-1][0] == 0 ):
            print("SOLUTION HAS BEEN FOUND")
            print(" ")
            beautify(best_individuals[-1][1])
        else:
            print("TERMINATION CRITERIA SATISFIED")
            print("Final Generation: " + str(gen_count))
            print(" ")
            print("BEST SOlUTION HAS FITNESS: " + str(best_individuals[-1][0]))
            beautify(best_individuals[-1][1])

        return best_individuals





if __name__ == "__main__":

    #program command variables in the format:
        #argv[1] = file name containing the puzzle
        #argv[2] = population size

    uninitialised_puzzle = read(sys.argv[1])
    pop_size = int(sys.argv[2])

    sudoku = Sudoku(puzzle=uninitialised_puzzle,\
               population_size=pop_size,\
               truncation_rate=0.4,\
               k_tournament=3,\
               mutation_rate=0.5,\
               swaps_per_mutation=2)


    fitness_list = sudoku.run()
