# Import libraries
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import sys
import os
sys.path.insert(0, 'evoman')
from demo_controller import player_controller
from environment import Environment

experiment_name = 'report'
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
                  enemies=[int(sys.argv[1])],
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
# EAs = [
#     {
#         'name': 'ea1',
#         'algorithm': '-eaSimple',
#         'dom_u': 1,
#         'dom_l': -1,
#         'npop': 50,
#         'ngen': 50,
#         'cxpb': 0.2,
#         'mutpb': 0.2,
#         'tournsize': 3
#     },
#     {
#         'name': 'ea2',
#         'algorithm': '-eaMuPlusLambda',
#         'dom_u': 0.5,
#         'dom_l': -0.5,
#         'npop': 30,
#         'ngen': 50,
#         'cxpb': 0.5,
#         'mutpb': 0.5,
#         'mu': 0.5,
#         'tournsize': 5
#     }
# ]

EAs = [
    {
        'name': 'ea1',
        'algorithm': '-eaSimple',
        'dom_u': 1,
        'dom_l': -1,
        'npop': 5,
        'ngen': 3,
        'cxpb': 0.2,
        'mutpb': 0.2,
        'tournsize': 3
    },
    {
        'name': 'ea2',
        'algorithm': '-eaMuPlusLambda',
        'dom_u': 0.5,
        'dom_l': -0.5,
        'npop': 3,
        'ngen': 2,
        'cxpb': 0.5,
        'mutpb': 0.5,
        'mu': 0.5,
        'tournsize': 5
    }
]

# Declare empty arrays
ea1_means = list()  # Tracking for mean vals of EA1
ea2_means = list()  # Tracking for mean vals of EA2
ea1_stdOfmeans = list()  # Tracking for the std of mean vals of EA1
ea2_stdOfmeans = list()  # Tracking for the std of mean vals of EA2
ea1_maxs = list()  # Tracking for the max vals of EA1
ea2_maxs = list()  # Tracking for the max vals of EA2
ea1_stdOfmaxs = list()  # Tracking for the std of the max vals of EA1
ea2_stdOfmaxs = list()  # Tracking for the std of the max vals of EA2

fitnesses = list()

# Declare the number of runs
run = 10

# Run 10 times independently
for i in range(run):
    # Set seed to a different value each time
    np.random.seed(i)  # np seed
    random.seed(i)  # python built-in

    # Trace
    print(f'\nRUN {i+1}/10')

    # Run two different EA
    for EA in EAs:
        # Declare simulation labels
        fit_name = f"fit_{i}_{EA['name']}"
        ind_name = f"ind_{i}_{EA['name']}"
        # Run simulation due to given algorithm name
        if EA['algorithm'] == '-eaSimple':
            # Use DEAP
            creator.create(fit_name, base.Fitness, weights=(1.0,))
            creator.create(ind_name, np.ndarray, fitness=getattr(creator, fit_name))
            toolbox = base.Toolbox()
            toolbox.register("attr_float", random.uniform, EA['dom_l'], EA['dom_u'])
            toolbox.register("individual", tools.initRepeat,
                             getattr(creator, ind_name), toolbox.attr_float,
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

            # Declare algorithm name
            algorithm_name = 'eaSimple'
            # Run simulations
            final_pop, verb = algorithms.eaSimple(pop, toolbox, EA['cxpb'],
                                                  EA['mutpb'], EA['ngen'], stats,
                                                  verbose=True)
        elif EA['algorithm'] == '-eaMuPlusLambda':
            # Use DEAP
            creator.create(fit_name, base.Fitness, weights=(1.0,))
            creator.create(ind_name, np.ndarray, fitness=getattr(creator, fit_name))
            toolbox = base.Toolbox()
            toolbox.register("attr_float", random.uniform, EA['dom_l'], EA['dom_u'])
            toolbox.register("individual", tools.initRepeat,
                             getattr(creator, ind_name), toolbox.attr_float,
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

            # Declare algorithm name
            algorithm_name = 'eaMuPlusLambda'
            # Run simulations
            final_pop, verb = algorithms.eaMuPlusLambda(pop, toolbox,
                                                        int(EA['mu'] * EA['npop']),
                                                        int(EA['npop']/5), EA['cxpb'],
                                                        EA['mutpb'], EA['ngen'],
                                                        stats, verbose=True)

        # Track time
        print(f"SIMULATION {EA['name']} RUN FOR {round((time.perf_counter() - ini) / 60, 2)} mins")

        # Check if path exists
        if not os.path.exists(f"{experiment_name}/run_{i+1}/{EA['name']}/{algorithm_name}"):
            os.makedirs(f"{experiment_name}/run_{i+1}/{EA['name']}/{algorithm_name}")
        # Save fitness statistics
        # Params concatenation for file names
        path = f"{experiment_name}/run_{i+1}/{EA['name']}/{algorithm_name}"
        params = f"{EA['npop']}_{EA['ngen']}_{EA['cxpb']}_{EA['mutpb']}_{EA['tournsize']}"
        # Create a frame for fitness logs
        dfFit = pd.DataFrame(verb.chapters['fitness'])[
            ['gen', 'nevals', 'avg', 'std', 'max', 'min']
        ]
        # Append mean and max values to the belonging array
        globals()[f"{EA['name']}_means"].append(dfFit['avg'].mean())  # mean vals
        globals()[f"{EA['name']}_stdOfmeans"].append(np.std(globals()[f"{EA['name']}_means"]))
        globals()[f"{EA['name']}_maxs"].append(dfFit['max'].mean())  # max vals
        globals()[f"{EA['name']}_stdOfmaxs"].append(np.std(globals()[f"{EA['name']}_maxs"]))
        # Save it
        dfFit.to_csv(f"{path}/stats_fit_{params}.csv")
        # Create a frame for size logs
        dfSize = pd.DataFrame(verb.chapters['size'])[
            ['gen', 'nevals', 'avg', 'std', 'max', 'min']
        ]
        # Save it
        dfSize.to_csv(f"{path}/stats_size_{params}.csv")

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

# Declare array of runs
runs = np.arange(1, run + 1)

# Create figure and axes for the subplots
axs = plt.figure(figsize=(10, 8)).subplots(4, 1)

# Plot average mean across all runs
axs[0].set_title('Average mean')
axs[0].plot(runs, ea1_means, label='EA1')
axs[0].plot(runs, ea2_means, label='EA2')
axs[0].legend(loc='upper right')
# Plot the std of average mean across all runs
axs[1].set_title('STD of average mean')
axs[1].plot(runs, ea1_stdOfmeans, label='EA1')
axs[1].plot(runs, ea2_stdOfmeans, label='EA2')
axs[1].legend(loc='upper right')
# Plot average max across all runs
axs[2].set_title('Average max')
axs[2].plot(runs, ea1_maxs, label='EA1')
axs[2].plot(runs, ea2_maxs, label='EA2')
axs[2].legend(loc='upper right')
# Plot the std of average max across all runs
axs[3].set_title('STD of average max')
axs[3].plot(runs, ea1_stdOfmaxs, label='EA1')
axs[3].plot(runs, ea2_stdOfmaxs, label='EA2')
axs[3].legend(loc='upper right')

# Tight layout
plt.tight_layout()
# Save figure
plt.savefig(f'./{experiment_name}/{run}runs')
