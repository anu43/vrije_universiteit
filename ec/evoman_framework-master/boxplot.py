# Import libraries
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

experiment_name = 'boxplot'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def simulation(env, x):
    '''gives the individuals of one population and returns stats from the env'''
    f, p, e, t = env.play(pcont=x)
    return f, p, e, t


def evaluate(x):
    '''simulate with the given individual'''
    return np.array(list(map(lambda y: simulation(env, y), x)))


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

# The EA params
EAs = [
    {
        'name': 'ea1',
        'algorithm': 'eaSimple',
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
        'algorithm': 'eaMuPlusLambda',
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

# Declare the number of runs
run = 10
# Declare an empty list for individual gains
ind_gains4ea1 = list()
ind_gains4ea2 = list()

# Run over experiments
for i in range(run):
    # Test over EAs
    for EA in EAs:
        # Trace
        print(f"\nTESTING {i+1}th run -- {EA['name']}")
        # Declare an empty array for keeping the track of 5 runs
        ind_gains = list()
        # Declare names for reading files
        path = f"report/run_{i+1}/{EA['name']}/{EA['algorithm']}"
        params = f"{EA['npop']}_{EA['ngen']}_{EA['cxpb']}_{EA['mutpb']}_{EA['tournsize']}"
        # Run 5 times
        for k in range(5):
            # Trace
            print(f'\tTrial {k+1}')
            # Read the best solution from previous runs
            bsol = np.loadtxt(f'{path}/best_{params}.txt')
            # Receive variables from the evaluation
            f, p, e, t = evaluate([bsol])[0]
            # Calculate individual gain = player energy - enemy energy
            ind_gain = p - e
            # Append it to calculate the mean later
            ind_gains.append(ind_gain)
        # Take the mean of the values and append to the general list
        globals()[f"ind_gains4{EA['name']}"].append(np.mean(ind_gains))

# Concatenate the data to insert it to boxplot
data = [ind_gains4ea1, ind_gains4ea2]
# Save the data to a txt file
np.savetxt('boxplot/data.txt', np.array(data))

# Plot the boxplot
# Set title
plt.title('Individual Gains over 10 runs')
# Plot
plt.boxplot(data, labels=['EA1', 'EA2'])
# Save the boxplot
plt.savefig('boxplot/ind_gain')

# Do a statistical test
print('t-test:', stats.ttest_ind(ind_gains4ea1, ind_gains4ea2))

# TODO: one-way ANOVA among all enemies
