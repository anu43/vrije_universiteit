# Import libraries
import seaborn as sns
import pandas as pd

# Set the theme
# sns.set_theme(style="whitegrid")

# Set up a grid to plot survival probability against several variables
g = sns.catplot(x="numberOfRuns", y="subClassRunTimes",
                kind="point", hue="heuristic", col="method",
                data=pd.read_csv('datasets/results/runtimeTable.csv'))

# Save plot
g.savefig('datasets/results/runtimeTableSubClassRunTimes.pdf', format='pdf')

# Set up a grid to plot survival probability against several variables
g = sns.catplot(x="numberOfRuns", y="letheRunTimes",
                kind="point", hue="heuristic", col="method",
                data=pd.read_csv('datasets/results/runtimeTable.csv'))

# Save plot
g.savefig('datasets/results/runtimeTableLetheRunTimes.pdf', format='pdf')
