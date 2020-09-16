# Import libs
from environment import Environment
from demo_controller import player_controller
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, tools, algorithms
import numpy as np
import random
import time
import sys
import os
sys.path.insert(0, 'evoman')

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

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

# genetic algorithm params
run_mode = 'train'  # train or test

# Declare variables of the simulation
n_hidden = 10  # the number of hidden layers
# The number of actions (individuals)
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
dom_u = 1  # upper bound of uniform dist
dom_l = -1  # lower bound of uniform dist
npop = 10  # number of population
gens = 5  # number of generation
mutation = 0.2  # the mutation probability
last_best = 0
