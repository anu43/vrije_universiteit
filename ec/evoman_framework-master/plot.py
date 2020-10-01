# Import libraries
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
import sys
import os
sys.path.insert(0, 'evoman')
from demo_controller import player_controller
from environment import Environment

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
# The EA params
EAs = [
    {
        'name': 'ea1',
        'algorithm': '-eaSimple',
        'dom_u': 1,
        'dom_l': -1,
        'npop': 50,
        'ngen': 50,
        'cxpb': 0.2,
        'mutpb': 0.2,
        'tournsize': 3
    },
    {
        'name': 'ea2',
        'algorithm': '-eaMuPlusLambda',
        'dom_u': 0.5,
        'dom_l': -0.5,
        'npop': 30,
        'ngen': 50,
        'cxpb': 0.5,
        'mutpb': 0.5,
        'mu': 0.5,
        'tournsize': 5
    }
]

# Use DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Run two different EA
for i in range(10):
    # Set seed to a different value each time
    np.random.seed(i)

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
    if EA[] == '-eaSimple':
        # Declare algorithm name
        algorithm_name = 'eaSimple'
        # Run simulations
        final_pop, verb = algorithms.eaSimple(pop, toolbox, param['cxpb'],
                                              param['mutpb'], param['ngen'], stats,
                                              verbose=True)
    elif EA[] == '-eaMuPlusLambda':
        # Declare algorithm name
        algorithm_name = 'eaMuPlusLambda'
        # Run simulations
        final_pop, verb = algorithms.eaMuPlusLambda(pop, toolbox,
                                                    int(param['mu'] * param['npop']),
                                                    int(param['npop']/5), param['cxpb'],
                                                    param['mutpb'], param['ngen'],
                                                    stats, verbose=True)
    elif EA[] == '-eaMuCommaLambda':
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


# Create figure and axes for the subplots
axs = plt.figure(figsize=(15, 8)).subplots(1, 4)

# Declare path for plotting
path1 = f"{experiment_name}/ea1/{algorithm_name}"  # EA1
path2 = f"{experiment_name}/ea1/{algorithm_name}"  # EA2
# Read fitness csv files within the experiment folders
for filename in os.listdir(f"{path1}"):
    # If it starts with stats_fit
    if filename.startswith('stats_fit'):
        # Read the csv file
        df = pd.read_csv(f'{path1}/{filename}')[
            ['gen', 'nevals', 'avg', 'std', 'max', 'min']
        ]
        # Plot by generation
        axs[0].set_title('AVG')
        axs[0].plot(df['gen'], df['avg'], label='EA1')
        axs[0].legend(loc='upper right')
        axs[1].set_title('STD')
        axs[1].plot(df['gen'], df['std'], label='EA1')
        axs[1].legend(loc='upper right')
        axs[2].set_title('MAX')
        axs[2].plot(df['gen'], df['max'], label='EA1')
        axs[2].legend(loc='upper right')
        axs[3].set_title('MIN')
        axs[3].plot(df['gen'], df['min'], label='EA1')
        axs[3].legend(loc='upper right')

# Read fitness csv files within the experiment folders
for filename in os.listdir(f"{path2}"):
    # If it starts with stats_fit
    if filename.startswith('stats_fit'):
        # Read the csv file
        df = pd.read_csv(f'{path2}/{filename}')[
            ['gen', 'nevals', 'avg', 'std', 'max', 'min']
        ]
        # Plot by generation
        axs[0].set_title('AVG')
        axs[0].plot(df['gen'], df['avg'], label='EA2')
        axs[0].legend(loc='upper right')
        axs[1].set_title('STD')
        axs[1].plot(df['gen'], df['std'], label='EA2')
        axs[1].legend(loc='upper right')
        axs[2].set_title('MAX')
        axs[2].plot(df['gen'], df['max'], label='EA2')
        axs[2].legend(loc='upper right')
        axs[3].set_title('MIN')
        axs[3].plot(df['gen'], df['min'], label='EA2')
        axs[3].legend(loc='upper right')

# Show the plots after running the experiment
plt.show()
