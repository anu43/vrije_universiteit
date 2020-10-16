# Import libraries
import pylab
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


# Import frame
df = pd.read_csv('datasets/results/runtimeTable.csv')

# Declare figure and the axis of plots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
# Flatten the axis to iterate later
ax = axes.flatten()

# Declare a counter
counter = 0


# Declare an empty list to record the shapiro results
shapiros: list = list()
# Declare an empty list for the p values
ps: list = list()
# Declare an empty list for tags
tags: list = list()
# Declare empty lists for methods and heuristics
methods = list()
heuristics = list()

# Iterate through methods
for method in ['ALCHTBoxForgetter',
               'SHQTBoxForgetter',
               'ALCOntologyForgetter']:
    # Iterate through heuristics
    for heuristic in ['insertSigOneByOneHeuristic',
                      'insertSigTwoByTwoHeuristic',
                      'insertSigRandomlyHeuristic']:
        # Prepare the QQ plots
        data = df[
            (df['method'] == method) & (df['heuristic'] == heuristic)
        ].subClassRunTimes
        sm.qqplot(data, fit=True, line='45', ax=ax[counter])

        # Increase the counter by one
        counter += 1

        # Shapiro-Wilk test
        stat, p = stats.shapiro(data)

        # Append the statistics
        shapiros.append(stat)  # Shapiro results
        ps.append(p)  # P values
        tags.append(True if p > 0.05 else False)

        # Append methods and heuristics
        methods.append(method)
        heuristics.append(heuristic)

plt.savefig('datasets/results/letheRunTimesQQ.pdf', format='pdf')

# Create a frame for the data
frame: dict = {
    'Method': methods,
    'Heuristic': heuristics,
    'ShapiroStatistics': shapiros,
    'p-value': ps,
    'isNormal': tags
}

# Import data
df = pd.read_csv('datasets/results/runtimeTable.csv')

# Perform two-way ANOVA
model = ols('letheRunTimes ~ C(method) + C(heuristic) + C(method):C(heuristic)',
            data=df).fit()
pd.DataFrame(data=sm.stats.anova_lm(model, typ=2)).to_csv('datasets/results/letheRunTimesANOVA.csv')
