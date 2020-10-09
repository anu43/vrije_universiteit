# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# Disable the game screen
os.environ["SDL_VIDEODRIVER"] = "dummy"


def simulation(env, x):
    '''
    gives the individuals of one population and returns stats from the env
    '''
    f, p, e, t = env.play(pcont=x)
    return f, p, e, t


def evaluate(x):
    '''simulate with the given individual'''
    return np.array(list(map(lambda y: simulation(env, y), x)))


# Set the number of hidden neurons
n_hidden_neurons = 10
# Read the best solution from previous runs
bsol = np.loadtxt(f'ea1-group1/run_5/best.txt')

# Declare empty lists for player and enemy lives
p_lives = list()
e_lives = list()

for enemy in np.arange(1, 9):
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name='best_run',
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      logs='off',
                      speed="fastest"
                    )
    # Trace
    print(f"\nTESTING ENEMY {enemy}")
    # Run 5 times
    for k in range(5):
        # Trace
        print(f'\tTrial {k+1}')
        # Receive variables from the evaluation
        f, p, e, t = evaluate([bsol])[0]
        # Append the energy points of the player and the enemy
        p_lives.append(p)  # player lives
        e_lives.append(e)  # enemy lives

    print(f'Player life: {np.mean(p_lives)}. Enemy life: {np.mean(e_lives)}')

    # Flush the lists
    p_lives = list()
    e_lives = list()
