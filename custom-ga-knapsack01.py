from pathlib import Path
from random import randint, choices, random
import csv
import numpy
import sys
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from statistics import stdev

NUM_GEN = 300
POP_SIZE = 300
ELITISM = 0.9
MUTATION_CHANCE = 0.1
ALPHA = 2.1


def read_file(file_name):
    data = []
    with open(Path(file_name), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            val = list(numpy.fromstring(row[0], dtype=int, sep=' '))
            if val is not None:
                data.append(val)
        fdata = data[1:data[0][0]+1]
        return data[0], fdata


def calc_fitness_knapsack(individual, values, weights, capacity):
    value_total = 0
    weight_total = 0

    for i in range(len(individual)):
        value_total += values[i] * individual[i]
        weight_total += weights[i] * individual[i]

    curr_capacity = max(0, weight_total - capacity)
    fit = value_total - ALPHA * curr_capacity

    return fit


# Deep copy arrays and then do elitism.
def elitism_selection(pop, fits):
    clone = deepcopy(pop)
    clone_fit = deepcopy(fits)
    elitists = []

    for __ in range(round(ELITISM * POP_SIZE)):
        best_ind = max(range(len(clone_fit)), key=clone_fit.__getitem__)
        elitists.append(clone[best_ind])
        del clone[best_ind]
        del clone_fit[best_ind]

    return elitists


def roulette_selection(prev_pop, prev_fits, k):
    total_fitness = sum(prev_fits)
    fitness_proportions = [x / total_fitness for x in prev_fits]
    # Uses the proportional weighting to select TWO parents
    pop = []
    try:
        pop = choices(population=prev_pop, weights=fitness_proportions, k=k)
        return pop
    except ValueError:
        print(fitness_proportions)
        print('************* ValueError *************')
        print(
            'This is due to choices() requiring a set of weights that sum to greater than 0')
        print('ABORTING RUN, PLEASE TRY AGAIN.')
        exit()

# Take two indiv's (pop) and do crossover


def crossover_selection(pop, indiv_len):
    # Crossover is pointless if you just swap the whole list
    rand = randint(1, indiv_len - 2)
    crossover = []
    crossover.append(pop[0][:rand] + pop[1][rand:])
    crossover.append(pop[1][:rand] + pop[0][rand:])

    return crossover


def mutation_selection(pop, indiv_len):
    for indiv in pop:
        if random() <= MUTATION_CHANCE:
            flip_ind = randint(0, indiv_len-1)
            indiv[flip_ind] = 0 if indiv[flip_ind] == 1 else 1
            # print('MUTATION! AT INDEX ' + str(flip_ind), clone, '->', indiv)

    return pop


def create_new_pop(prev_pop, prev_fits, indiv_len):
    # TODO Should I be removing the ELITE individuals once they are selected??
    new_pop = elitism_selection(prev_pop, prev_fits)

    while (len(new_pop) < POP_SIZE):
        # Select TWO parents with roulette wheel
        k = 2 if POP_SIZE - len(new_pop) > 2 else 1
        roulette = roulette_selection(prev_pop, prev_fits, k)

        # apply one-point crossover to them
        crossover = crossover_selection(
            roulette, indiv_len) if len(roulette) == 2 else roulette

        # 5% chance of mutation (flip)
        mutation = mutation_selection(crossover, indiv_len)

        # put the children into the pop
        new_pop += mutation

    return new_pop


def run_knapsack(data, individual_len, capacity):
    # Randomly initialise individual to have either 0 or 1 for knapsack
    pop = [[1 if random() <= 0.1 else 0 for __ in range(individual_len)]
           for ___ in range(POP_SIZE)]
    pop_fit = [None] * POP_SIZE

    values = [row[0] for row in data]
    weights = [row[1] for row in data]

    best_feasible = None
    best_fit = 0
    fitness_history = []

    for __ in range(NUM_GEN):
        # Calc fitness of current pop
        for i, ind in enumerate(pop):
            pop_fit[i] = calc_fitness_knapsack(ind, values, weights, capacity)

        # Find new best feasible
        best_ind = max(range(len(pop_fit)), key=pop_fit.__getitem__)
        if pop_fit[best_ind] > best_fit:
            best_feasible = pop[best_ind]
            best_fit = pop_fit[best_ind]

        pop = create_new_pop(pop, pop_fit, individual_len)
        pop_fit.sort()
        fitness_history.append(sum(pop_fit[-5:]) / 5)
        pop_fit = [None] * POP_SIZE

    return best_feasible, fitness_history


def run():
    info, data = read_file(sys.argv[1])

    # Setup up initial population and variables
    individual_len = info[0]
    capacity = info[1]
    solution, fitness_history = run_knapsack(data, individual_len, capacity)

    vs = [x[0] for x in data]
    total = 0
    if solution is not None:
        for i, value in enumerate(vs):
            if solution[i] == 1:
                total += value

    return fitness_history, total


def main():
    histories = []
    # for count in range(5):
    start = datetime.now()
    # print('STARTING RUN NUMBER: ' + str(count))
    fitness_history, total = run()
    # print('Round ' + str(count) + ' total: ', total)
    histories.append(total)
    end = datetime.now()
    print('Time elapsed: ' + str(end - start))

    print(total)
    plt.plot(list(range(NUM_GEN)), fitness_history, label='Mean Fitness')
    plt.legend()
    plt.title('Fitness through the generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()

    # print('Standard Deviation:', stdev(histories))
    # print('Mean Fitness:', sum(histories) / len(histories))

if __name__ == '__main__':
    main()
