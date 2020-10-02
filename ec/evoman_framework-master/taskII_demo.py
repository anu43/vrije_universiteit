# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import random
import logging
from deap import base, creator, tools, algorithms
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
from math import fabs,sqrt
import glob, os

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
experiment_name = 'taskII'  # Declare experiment name
n_hidden_neurons = 10  # Declare the number of hidden
RUN = 0  # The number of run
best_param = - np.inf  # Initialize the best parameter

# Declare parameters to search in a combination
params = {
    'npop': [40],
    'ngen': [35],
    'cxpb': [0.9],
    'mutpb': [0.1],
    'tournsize': [3],
    'typeOfCrossover': ['cxTwoPoint', 'cxUniform', 'cxBlend'],
    'typeOfMutation': ['mutGaussian', 'mutShuffleIndexes', 'mutUniformInt'],
    'typeOfTournament': ['selTournament']
}
# Turn parameters into a grid
params = ParameterGrid(params)

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=[1, 3, 5, 7, 8],
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

# Keep logs of runs
logging.basicConfig(filename=f'{experiment_name}/app{sys.argv[1]}.log',
                    filemode='w', format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

# Run two different EA
for idx, param in enumerate(params):
    # Trace
    print(f'Running {idx + 1}/{len(params)}: {param}')
    # Log
    logging.warning(f'Running {idx + 1}/{len(params)}: {param}')
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

        # Set the random value for lambda_
        lambda_ = np.random.randint(param['npop'] - 10,
                                    param['npop'], 1)[0]

        # Track time
        ini = time.perf_counter()

        if sys.argv[1] == '-eaMuPlusLambda':
            # Declare algorithm name
            algorithm_name = 'eaMuPlusLambda'
            # Run simulations
            final_pop, verb = algorithms.eaMuPlusLambda(pop, toolbox,
                                                        param['npop'],
                                                        lambda_,
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
                                                         lambda_,
                                                         param['cxpb'],
                                                         param['mutpb'],
                                                         param['ngen'],
                                                         stats, verbose=True)

        # Track time
        print(f"SIMULATION RUN FOR {round((time.perf_counter() - ini) / 60, 2)} mins")
        # Log time
        logging.warning(f"SIMULATION RUN FOR {round((time.perf_counter() - ini) / 60, 2)} mins")

        # Check if path exists
        if not os.path.exists(f"{experiment_name}/run_{idx + 1}/{algorithm_name}"):
            # Create the directory
            os.makedirs(f"{experiment_name}/run_{idx + 1}/{algorithm_name}")

        # Save fitness statistics
        # Params concatenation for file names
        path = f"{experiment_name}/run_{idx + 1}/{algorithm_name}"
        fit = pd.DataFrame(verb.chapters['fitness'])[
            ['gen', 'nevals', 'avg', 'std', 'max', 'min']
        ]
        fit.to_csv(f"{path}/stats_fit.csv")
        # Calculate the metric for keeping the best param
        # fit['metric'] = fit['avg'] + 0.5 * fit['max'] - 0.5 * fit['std']
        # Declare metric
        metric = fit['max'].mean()
        # If metric is better than best_param
        if best_param < metric:
            # Trace
            print(f'Found better param setup. Last {best_param}. New {metric}')
            # Trace
            logging.warning(f'Found better param setup. Last {best_param}. New {metric}')
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

# Print out the best param
print(f'Best param setup: {best_setup} and its score: {best_param}')
# Log the best param
logging.warning(f'Best param setup: {best_setup}')
