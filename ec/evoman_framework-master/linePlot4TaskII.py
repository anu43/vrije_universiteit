# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Declare array of runs
runs = np.arange(1, 11)

# Import records from EA1-GROUP1
# Group 1
group1_means = np.loadtxt('ea1-group1/ea1_means')
group1_stdOfmeans = np.loadtxt('ea1-group1/ea1_stdOfmeans')
group1_maxs = np.loadtxt('ea1-group1/ea1_maxs')
group1_stdOfmaxs = np.loadtxt('ea1-group1/ea1_stdOfmaxs')
# Group 2
group2_means = np.loadtxt('ea1-group2/ea1_means')
group2_stdOfmeans = np.loadtxt('ea1-group2/ea1_stdOfmeans')
group2_maxs = np.loadtxt('ea1-group2/ea1_maxs')
group2_stdOfmaxs = np.loadtxt('ea1-group2/ea1_stdOfmaxs')

# Create figure and axes for the subplots
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(10, 8))

# Plot average mean across all runs
axs[0].set_title('Average mean', fontsize=15)
axs[0].plot(runs, group1_means, label='EA1-GROUP1')
axs[0].plot(runs, group2_means, label='EA1-GROUP2')
axs[0].legend(loc='upper right')
# Plot the std of average mean across all runs
axs[1].set_title('STD of average mean', fontsize=15)
axs[1].plot(runs, group1_stdOfmeans, label='EA1-GROUP1')
axs[1].plot(runs, group2_stdOfmeans, label='EA1-GROUP2')
axs[1].legend(loc='upper right')
# Plot average max across all runs
axs[2].set_title('Average max', fontsize=15)
axs[2].plot(runs, group1_maxs, label='EA1-GROUP1')
axs[2].plot(runs, group2_maxs, label='EA1-GROUP2')
axs[2].legend(loc='upper right')
# Plot the std of average max across all runs
axs[3].set_title('STD of average max', fontsize=15)
axs[3].plot(runs, group1_stdOfmaxs, label='EA1-GROUP1')
axs[3].plot(runs, group2_stdOfmaxs, label='EA1-GROUP2')
axs[3].legend(loc='upper right')

# Set a common x and y labels
fig.text(0.5, 0, 'The number of runs', ha='center', fontsize=20)
fig.text(0, 0.5, 'Fitness', va='center', rotation='vertical', fontsize=20)

# Tight layout
plt.tight_layout()
# Save figure
plt.savefig('linePlot4TaskII_EA1.pdf', format='pdf')
