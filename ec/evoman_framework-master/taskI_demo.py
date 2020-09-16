# Import libs
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, tools, algorithms
import pandas as pd
import numpy as np
import random
import time
import sys
import os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

experiment_name = 'taskI'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def simulation(env, x):
    '''gives the individuals of one population and returns stats from the env'''
    f, p, e, t = env.play(pcont=x)
    return f,


def evaluate(x):
    '''simulate with the given individual'''
    return simulation(env, x)


# def helper(enemy):
#     # Update the number of neurons for this specific example
#     n_hidden_neurons = 0
#     # Create the environment
#     env = Environment(experiment_name=experiment_name,
#                       playermode="ai",
#                       enemies=[enemy],
#                       player_controller=player_controller(n_hidden_neurons),
#                       speed="fastest",
#                       enemymode="static",
#                       level=2)
#     # Logging
#     # Load specialist controller
#     sol = np.loadtxt('solutions_demo/demo_'+str(enemy)+'.txt')
#     print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(enemy)+' \n')
#     # Play the given env
#     env.play(sol)


# with ProcessPoolExecutor() as executor:
#     enemies = [1, 2, 3]
#     results = executor.map(helper, enemies)

# for result in results:
#     print(result)

# Declare variables of the environment
n_hidden = 10  # the number of hidden layers

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

# genetic algorithm params
run_mode = 'train'  # train or test

# Declare variables of the simulation
# The number of actions (individuals)
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
dom_u = 1  # upper bound of uniform dist
dom_l = -1  # lower bound of uniform dist
npop = 10  # number of population
ngen = 5  # number of generation
mutation = 0.2  # the mutation probability
last_best = 0
cxpb = 0.5  # the cross-over probability
mutpb = 0.2  # the mutation probability

# Use DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, dom_l, dom_u)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float,
                 n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
toolbox.register("select", tools.selTournament, tournsize=3)

# Statistics
stat_fit = tools.Statistics(lambda ind: ind.fitness.values)  # Fitness statistics
stat_size = tools.Statistics(key=len)  # Size statistics
stats = tools.MultiStatistics(fitness=stat_fit, size=stat_size)
stats.register('avg', np.mean)  # the average by np.mean
stats.register('std', np.std)  # the standard deviation by np.std
stats.register('min', np.min)  # the average by np.min
stats.register('max', np.max)  # the average by np.max

# Populate
pop = toolbox.population(n=npop)  # size: (npop, n_vars)

# Run simulation due to given algorithm name
if sys.argv[1] == '-eaSimple':
    # Declare algorithm name
    algorithm_name = 'eaSimple'
    # Run simulations
    final_pop, verb = algorithms.eaSimple(pop, toolbox, cxpb,
                                          mutpb, ngen, stats,
                                          verbose=True)
elif sys.argv[1] == '-eaMuPlusLambda':
    # Declare algorithm name
    algorithm_name = 'eaMuPlusLambda'
    # Run simulations
    final_pop, verb = algorithms.eaMuPlusLambda(pop, toolbox, int(3*npop/4), int(npop/5),
                                                cxpb, mutpb, ngen, stats, verbose=True)

# Check if path exists
if not os.path.exists(f'{experiment_name}/{algorithm_name}'):
    os.makedirs(f'{experiment_name}/{algorithm_name}')
# Save fitness statistics
pd.DataFrame(verb.chapters['fitness'])[
    ['gen', 'nevals', 'avg', 'std', 'max', 'min']
].to_csv(f'{experiment_name}/{algorithm_name}/stats_fit.csv')
# Save size statistics
pd.DataFrame(verb.chapters['size'])[
    ['gen', 'nevals', 'avg', 'std', 'max', 'min']
].to_csv(f'{experiment_name}/{algorithm_name}/stats_size.csv')

# Save the best solution
best_solution = tools.selBest(pop, k=1)  # size: (1, n_vars)
# Save the best solution to a txt file
np.savetxt(f'{experiment_name}/{algorithm_name}/best.txt', np.array(best_solution).T)
