# imports other libs
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
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
    return f, p, e, t


def evaluate(x):
    '''simulate with the given individual'''
    return np.array(list(map(lambda y: simulation(env, y), x)))


# Set seed for both numpy and random libs
np.random.seed(43)
random.seed(43)

# Declare global variables
experiment_name = 'boxplot4TaskII'  # Declare experiment name
n_hidden_neurons = 10  # Declare the number of hidden

# Declare groups
groups = ('ea1-group1', 'ea1-group2')

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=np.arange(1, 9),
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

# Declare an empty list for individual gains
ind_gains4group1 = list()
ind_gains4group2 = list()
# Declare the number of runs
run = 10

# Run over experiments
for i in range(run):
    # Test over groups
    for group in groups:
        # Declare an empty array for keeping the track of 5 runs
        ind_gains = list()
        # Declare names for reading files
        path = f"{group}/run_{i+1}/"
        # Trace
        print(f"\nTESTING {i+1}th run -- {group}")
        # Run 5 times
        for k in range(5):
            # Trace
            print(f'\tTrial {k+1}')
            # Read the best solution from previous runs
            bsol = np.loadtxt(f'{path}/best.txt')
            # Receive variables from the evaluation
            f, p, e, t = evaluate([bsol])[0]
            # Calculate individual gain = player energy - enemy energy
            ind_gain = p - e
            # Append it to calculate the mean later
            ind_gains.append(ind_gain)
        # Take the mean of the values and append to the general list
        globals()[f"ind_gains4{group.split('-')[1]}"].append(np.mean(ind_gains))

# Concatenate the data to insert it to boxplot
data = [ind_gains4group1, ind_gains4group2]
# Save the data to a txt file
np.savetxt(f'{experiment_name}/data.txt', np.array(data))

# Plot the boxplot
# Set title
plt.title('Individual Gains over 10 runs')
# Plot
plt.boxplot(data, labels=[group.upper() for group in groups])
# Save the boxplot
plt.savefig(f'{experiment_name}/ind_gain')

# Do a statistical test
print('t-test:', stats.ttest_ind(ind_gains4group1, ind_gains4group2))
print('one way ANOVA:', stats.f_oneway(ind_gains4group1, ind_gains4group2))
