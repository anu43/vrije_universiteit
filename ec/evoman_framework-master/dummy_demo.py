################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, tools
import numpy as np
import random
import sys
import os
sys.path.insert(0, 'evoman')
from demo_controller import player_controller
from environment import Environment

experiment_name = 'taskI'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


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

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# Declare variables
npop = 5  # the number of population
gens = 15  # the number of generation
mutation = 0.2  # the mutation probability
last_best = 0


def simulation(env, x):
    """runs simulation"""
    fitness, p_life, e_life, time = env.play(pcont=x)
    return fitness


def evaluate(x):
    return simulation(env, x)


# DEAP configuration
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float,
                 n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
toolbox.register("select", tools.selTournament, tournsize=3)
