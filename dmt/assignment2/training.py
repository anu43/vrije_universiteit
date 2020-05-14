import lightgbm as lgb
import numpy as np
import pandas as pd
from time import time

# Import training/test sets
train = pd.read_csv('../../../../data/training.csv')
test = pd.read_csv('../../../../data/test.csv')
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
ranker.fit(X, y, group=[X_train.shape[0]], early_stopping_rounds=50)
# Set stop time
stop = time()
# Print training time
print('training time', (stop - start) / 60, 'mins')

# Create X_test
X_test = np.array(test.values, test.columns)
# Predict
test_pred = ranker.predict(X_test)
# Put the predictions to the test frame
test["ranking_rates"] = test_pred

# Write the result to csv
test.to_csv('../../../../data/predicted_test.csv', index=False)
