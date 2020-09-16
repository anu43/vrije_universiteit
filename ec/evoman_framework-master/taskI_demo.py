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
