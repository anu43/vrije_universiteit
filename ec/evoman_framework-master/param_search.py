# Import libs
from deap import base, creator, tools, algorithms
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import random
import time
import sys
import os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

experiment_name = 'param_search'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def simulation(env, x):
    '''gives the individuals of one population and returns stats from the env'''
    f, p, e, t = env.play(pcont=x)
    return f,


def evaluate(x):
    '''simulate with the given individual'''
    return simulation(env, x)


# Declare variables of the environment
n_hidden = 10  # the number of hidden layers

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  logs='off')

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

# genetic algorithm params
run_mode = 'train'  # train or test

# Declare variables of the simulation
# The number of actions (individuals)
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
# Initialize best param value
best_param = - np.inf
# The parameters to test
# If -eaSimple was chosen
if sys.argv[1] == '-eaSimple':
    params = {
        'npop': [30, 50, 100],
        'ngen': [20, 35, 50],
        'cxpb': [0.2, 0.5, 0.75],
        'mutpb': [0.2, 0.5, 0.8],
        'tournsize': [3, 5]
    }
# Otherwise
else:
    params = {
        'npop': [30, 50, 100],
        'ngen': [20, 35, 50],
        'cxpb': [0.2, 0.5, 0.75],
        'mutpb': [0.2, 0.5, 0.8],
        'tournsize': [3, 5],
        'mu': [0.5, 0.75]
    }

# Turn parameters into a grid
params = ParameterGrid(params)

# Use DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Run two different EA
for idx, param in enumerate(params):
    # Trace
    print(f'\nRunning {idx}/{len(params)}: {param}\n')
    # Use DEAP
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float,
                     n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
    toolbox.register("select", tools.selTournament, tournsize=param['tournsize'])

    # Statistics
    stat_fit = tools.Statistics(lambda ind: ind.fitness.values)  # Fitness statistics
    stat_size = tools.Statistics(key=len)  # Size statistics
    stats = tools.MultiStatistics(fitness=stat_fit, size=stat_size)
    stats.register('avg', np.mean)  # the average by np.mean
    stats.register('std', np.std)  # the standard deviation by np.std
    stats.register('min', np.min)  # the average by np.min
    stats.register('max', np.max)  # the average by np.max

    # Populate
    pop = toolbox.population(n=param['npop'])  # size: (npop, n_vars)

    # Track time
    ini = time.perf_counter()

    # Run simulation due to given algorithm name
    if sys.argv[1] == '-eaSimple':
        # Declare algorithm name
        algorithm_name = 'eaSimple'
        # Run simulations
        final_pop, verb = algorithms.eaSimple(pop, toolbox, param['cxpb'],
                                              param['mutpb'], param['ngen'], stats,
                                              verbose=True)
    elif sys.argv[1] == '-eaMuPlusLambda':
        # Declare algorithm name
        algorithm_name = 'eaMuPlusLambda'
        # Run simulations
        final_pop, verb = algorithms.eaMuPlusLambda(pop, toolbox,
                                                    int(param['mu'] * param['npop']),
                                                    int(param['npop']/5), param['cxpb'],
                                                    param['mutpb'], param['ngen'],
                                                    stats, verbose=True)
    elif sys.argv[1] == '-eaMuCommaLambda':
        # Declare algorithm name
        algorithm_name = 'eaMuCommaLambda'
        # Run simulations
        final_pop, verb = algorithms.eaMuCommaLambda(pop, toolbox,
                                                     int(param['mu'] * param['npop']),
                                                     int(1.25 * param['npop']),
                                                     param['cxpb'],
                                                     param['mutpb'],
                                                     param['ngen'],
                                                     stats, verbose=True)

    # Track time
    print(f"SIMULATION RUN FOR {round((time.perf_counter() - ini) / 60, 2)} mins")

    # Check if path exists
    if not os.path.exists(f"{experiment_name}/run_{idx}/{algorithm_name}"):
        # Create the directory
        os.makedirs(f"{experiment_name}/run_{idx}/{algorithm_name}")

    # Save fitness statistics
    # Params concatenation for file names
    path = f"{experiment_name}/run_{idx}/{algorithm_name}"
    fit = pd.DataFrame(verb.chapters['fitness'])[
        ['gen', 'nevals', 'avg', 'std', 'max', 'min']
    ]
    fit.to_csv(f"{path}/stats_fit.csv")
    # Calculate the metric for keeping the best param
    fit['metric'] = fit['avg'] + 0.5 * fit['max'] - 0.5 * fit['std']
    # Declare metric
    metric = fit['metric'].mean()
    # If metric is better than best_param
    if best_param < metric:
        # Trace
        print(f'Found better param setup. Last {best_param}. New {metric}')
        # Save it as the new one
        best_param = metric
        # Save the param setup
        best_setup = param

    # Save size statistics
    pd.DataFrame(verb.chapters['size'])[
        ['gen', 'nevals', 'avg', 'std', 'max', 'min']
    ].to_csv(f"{path}/stats_size.csv")

    # Save the best solution
    best_solution = tools.selBest(final_pop, k=1)  # size: (1, n_vars)
    # Save the worst solution
    worst_solution = tools.selWorst(final_pop, k=1)  # size: (1, n_vars)
    # Save the best solution to a txt file
    np.savetxt(f"{path}/best.txt",
               np.array(best_solution).T)
    # Save the worst solution to a txt file
    np.savetxt(f"{path}/worst.txt",
               np.array(worst_solution).T)

# Print out the best param
print(f'\nBest param setup: {best_setup}')
