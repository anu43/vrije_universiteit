import lgbm as lgb
import numpy as np
import pandas as pd
from time import time

# Import training data
train = pd.read_csv('../../../../data/training.csv')

# Prepare X and y
target = 'target'  # Set column name for y
# Receive all columns except target column for X
X_train = train.loc[:, train.columns != target]
y_train = train.loc[:, target]

# Set their relevant shape for LGBM ranker
X = np.array(X_train.values, X_train.columns)
y = y_train.values

# Create LGBMRanker model
ranker = lgb.LGBMRanker()
# Set start time
start = time()
# Train data
ranker.fit(X, y, group=[X_train.shape[0]])
# Set stop time
stop = time()
# Print training time
print('duration', stop - start)
