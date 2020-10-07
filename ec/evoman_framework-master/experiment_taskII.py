# imports other libs
import time
import random
import logging
from deap import base, creator, tools, algorithms
import pandas as pd
import numpy as np
import os

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# Disable the game screen
os.environ["SDL_VIDEODRIVER"] = "dummy"


def simulation(env, x):
    '''
    gives the individuals of one population and returns stats from the env
    '''
    f, p, e, t = env.play(pcont=x)
    return f,


def evaluate(x):
    '''simulate with the given individual'''
    return simulation(env, x)


# Set seed for both numpy and random libs
np.random.seed(43)
random.seed(43)

# Declare global variables
experiment_name = 'ea1-group1'  # Declare experiment name
n_hidden_neurons = 10  # Declare the number of hidden
best_param = - np.inf  # Initialize the best parameter

# Declare parameters to search in a combination
param = {
    'npop': 30,
    'ngen': 30,
    'cxpb': 0.5,
    'mutpb': 0.5,
    'tournsize': 5,
    'typeOfCrossover': 'cxBlend',
    'typeOfMutation': 'mutShuffleIndexes',
    'typeOfTournament': 'selTournament'
}

# Set the random value for lambda_
param['lambda_'] = np.random.randint(param['npop'] - 5,
                                     param['npop'], 1)[0]

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=[1, 2, 4, 5, 8],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  logs='off',
                  speed="fastest")


# Declare variables of the simulation
# The number of actions (individuals)
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# Create log file if it does not exists
# Check if path exists
if not os.path.exists(f"{experiment_name}/"):
    # Create the directory
    os.makedirs(f"{experiment_name}/")

# Declare empty arrays
ea1_means = list()  # Tracking for mean vals of EA1
ea1_stdOfmeans = list()  # Tracking for the std of mean vals of EA1
ea1_maxs = list()  # Tracking for the max vals of EA1
ea1_stdOfmaxs = list()  # Tracking for the std of the max vals of EA1

fitnesses = list()

# Keep logs of runs
logging.basicConfig(filename=f'{experiment_name}/app{sys.argv[1]}.log',
                    filemode='w', format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

# Run two different EA
for idx in range(10):
    # Trace
    print(f'\nRUN {idx+1}/10')
    # Try
    try:
        # Declare creator and individual names
        FIT_MAX = f"FitnessMax_{idx}"
        IND = f"Individual_{idx}"
        # Use DEAP
        creator.create(FIT_MAX, base.Fitness, weights=(1.0,))
        creator.create(IND, np.ndarray, fitness=getattr(creator, FIT_MAX))

        toolbox = base.Toolbox()
        toolbox.register("attr_uniform", random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat,
                         getattr(creator, IND), toolbox.attr_uniform,
                         n=n_vars)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        # If the type of crossover
        if param['typeOfCrossover'] == 'cxTwoPoint':
            toolbox.register("mate", tools.cxTwoPoint)  # tweak
        elif param['typeOfCrossover'] == 'cxUniform':
            toolbox.register("mate", tools.cxUniform, indpb=0.5)  # tweak
        elif param['typeOfCrossover'] == 'cxBlend':
            toolbox.register("mate", tools.cxBlend, alpha=0.5)  # tweak
        # If the type of mutation
        if param['typeOfMutation'] == 'mutGaussian':
            toolbox.register("mutate", tools.mutGaussian,
                             mu=0, sigma=0.05, indpb=0.10)  # tweak
        elif param['typeOfMutation'] == 'mutShuffleIndexes':
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.10)  # tweak
        elif param['typeOfMutation'] == 'mutUniformInt':
            toolbox.register("mutate", tools.mutUniformInt,
                             low=-1, up=1, indpb=0.10)  # tweak
        toolbox.register("select", tools.selTournament,
                         tournsize=param['tournsize'])

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

        if sys.argv[1] == '-eaMuPlusLambda':
            # Declare algorithm name
            algorithm_name = 'eaMuPlusLambda'
            # Run simulations
            final_pop, verb = algorithms.eaMuPlusLambda(pop, toolbox,
                                                        param['npop'],
                                                        param['lambda_'],
                                                        param['cxpb'],
                                                        param['mutpb'],
                                                        param['ngen'],
                                                        stats, verbose=True)
        elif sys.argv[1] == '-eaMuCommaLambda':
            # Declare algorithm name
            algorithm_name = 'eaMuCommaLambda'
            # Run simulations
            final_pop, verb = algorithms.eaMuCommaLambda(pop, toolbox,
                                                         param['npop'],
                                                         param['lambda_'],
                                                         param['cxpb'],
                                                         param['mutpb'],
                                                         param['ngen'],
                                                         stats, verbose=True)

        # Track time
        print(f"SIMULATION RUN FOR {round((time.perf_counter() - ini) / 60, 2)} mins")
        # Log time
        logging.warning(f"SIMULATION RUN FOR {round((time.perf_counter() - ini) / 60, 2)} mins")

        # Check if path exists
        if not os.path.exists(f"{experiment_name}/run_{idx + 1}/"):
            # Create the directory
            os.makedirs(f"{experiment_name}/run_{idx + 1}/")

        # Save fitness statistics
        # Params concatenation for file names
        path = f"{experiment_name}/run_{idx + 1}/"
        dfFit = pd.DataFrame(verb.chapters['fitness'])[
            ['gen', 'nevals', 'avg', 'std', 'max', 'min']
        ]
        # Append mean and max values to the belonging array
        ea1_means.append(dfFit['avg'].mean())
        ea1_stdOfmeans.append(np.std(ea1_means))
        ea1_maxs.append(dfFit['max'].mean())
        ea1_stdOfmaxs.append(np.std(ea1_maxs))
        # Write fitness logs to a csv file
        dfFit.to_csv(f"{path}/stats_fit.csv")

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

    # If there is a regular error
    except Exception as e:
        # Trace
        print('------------')
        print(f'Error occured in {idx + 1}th run. {e}')
        print('Going to the next round')
        print('------------')
        # Log
        logging.error('------------')
        logging.error(f'Error occured in {idx + 1}th run. {e}')
        logging.error('Going to the next round')
        logging.error('------------')

# Save report results to a txt file
np.savetxt(f'{experiment_name}/ea1_means', ea1_means)
np.savetxt(f'{experiment_name}/ea1_stdOfmeans', ea1_stdOfmeans)
np.savetxt(f'{experiment_name}/ea1_maxs', ea1_maxs)
np.savetxt(f'{experiment_name}/ea1_stdOfmaxs', ea1_stdOfmaxs)
