from ortools.algorithms import pywrapknapsack_solver
from pathlib import Path
import csv
import numpy

def readFile(fileName):
    data = []
    with open(Path(fileName), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([])
            data[-1] = list(numpy.fromstring(row[0], dtype=float, sep=' '))
        return data[0], data[1:]

def main():
    info, data = readFile("knapsack-data/10_269")

    # Create the solver.
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

    values = [row[0] for row in data]
    weights = [[row[1] for row in data]]
    capacities = [info[1]]

    solver.Init(values, weights, capacities)
    computed_value = solver.Solve()

    packed_items = []
    packed_weights = []
    total_weight = 0

    print('Total value =', computed_value)

    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
            
    print('Total weight:', total_weight)
    print('Packed items:', packed_items)
    print('Packed_weights:', packed_weights)


if __name__ == '__main__':
    main()