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
                  speed="fastest",
                  logs='off')

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

# genetic algorithm params
run_mode = 'train'  # train or test

# Declare variables of the simulation
# The number of actions (individuals)
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
dom_u = 1  # upper bound of uniform dist
dom_l = -1  # lower bound of uniform dist
npop = 100  # number of population
ngen = 30  # number of generation
cxpb = 0.5  # the cross-over probability
mutpb = 0.2  # the mutation probability
EAs = [
    {
        'name': 'ea1',
        'dom_u': 1,
        'dom_l': -1,
        'npop': 2,
        'ngen': 3,
        'cxpb': 0.5,
        'mutpb': 0.2,
        'tournsize': 3
    },
    {
        'name': 'ea2',
        'dom_u': 0.5,
        'dom_l': -0.5,
        'npop': 5,
        'ngen': 2,
        'cxpb': 0.75,
        'mutpb': 0.3,
        'tournsize': 5
    }
]

# Use DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Run two different EA
for EA in EAs:
    # Use DEAP
    toolbox.register("attr_float", random.uniform, EA['dom_l'], EA['dom_u'])
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float,
                     n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
    toolbox.register("select", tools.selTournament, tournsize=EA['tournsize'])

    # Statistics
    stat_fit = tools.Statistics(lambda ind: ind.fitness.values)  # Fitness statistics
    stat_size = tools.Statistics(key=len)  # Size statistics
    stats = tools.MultiStatistics(fitness=stat_fit, size=stat_size)
    stats.register('avg', np.mean)  # the average by np.mean
    stats.register('std', np.std)  # the standard deviation by np.std
    stats.register('min', np.min)  # the average by np.min
    stats.register('max', np.max)  # the average by np.max

    # Populate
    pop = toolbox.population(n=EA['npop'])  # size: (npop, n_vars)

    # Track time
    ini = time.perf_counter()

    # Run simulation due to given algorithm name
    if sys.argv[1] == '-eaSimple':
        # Declare algorithm name
        algorithm_name = 'eaSimple'
        # Run simulations
        final_pop, verb = algorithms.eaSimple(pop, toolbox, EA['cxpb'],
                                              EA['mutpb'], EA['ngen'], stats,
                                              verbose=True)
    elif sys.argv[1] == '-eaMuPlusLambda':
        # Declare algorithm name
        algorithm_name = 'eaMuPlusLambda'
        # Run simulations
        final_pop, verb = algorithms.eaMuPlusLambda(pop, toolbox,
                                                    int(3*npop/4), int(npop/5),
                                                    EA['cxpb'], EA['mutpb'],
                                                    EA['ngen'], stats, verbose=True)
    elif sys.argv[1] == '-eaMuCommaLambda':
        # Declare algorithm name
        algorithm_name = 'eaMuCommaLambda'
        # Run simulations
        final_pop, verb = algorithms.eaMuCommaLambda(pop, toolbox,
                                                     int(3*npop/4), int(6*npop/5),
                                                     EA['cxpb'], EA['mutpb'],
                                                     EA['ngen'], stats, verbose=True)

    # Track time
    print(f"SIMULATION {EA['name']} RUN FOR {round((time.perf_counter() - ini) / 60, 2)} mins")

    # Check if path exists
    if not os.path.exists(f"{experiment_name}/{EA['name']}/{algorithm_name}"):
        os.makedirs(f"{experiment_name}/{EA['name']}/{algorithm_name}")
    # Save fitness statistics
    # Params concatenation for file names
    path = f"{experiment_name}/{EA['name']}/{algorithm_name}"
    params = f"{EA['npop']}_{EA['ngen']}_{EA['cxpb']}_{EA['mutpb']}_{EA['tournsize']}"
    pd.DataFrame(verb.chapters['fitness'])[
        ['gen', 'nevals', 'avg', 'std', 'max', 'min']
    ].to_csv(f"{path}/stats_fit_{params}.csv")
    # Save size statistics
    pd.DataFrame(verb.chapters['size'])[
        ['gen', 'nevals', 'avg', 'std', 'max', 'min']
    ].to_csv(f"{path}/stats_size_{params}.csv")

    # Save the best solution
    best_solution = tools.selBest(final_pop, k=1)  # size: (1, n_vars)
    # Save the worst solution
    worst_solution = tools.selWorst(final_pop, k=1)  # size: (1, n_vars)
    # Save the best solution to a txt file
    np.savetxt(f"{path}/best_{params}.txt",
               np.array(best_solution).T)
    # Save the worst solution to a txt file
    np.savetxt(f"{path}/worst_{params}.txt",
               np.array(worst_solution).T)
