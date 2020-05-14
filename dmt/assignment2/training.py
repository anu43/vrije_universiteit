import lightgbm as lgb
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split

# Import training/test sets
train = pd.read_csv('../../../../data/training.csv', nrows=20000)
test = pd.read_csv('../../../../data/test.csv', nrows= 20000)
# Prepare X and y
target = 'target'  # Set column name for y
# Receive all columns except target column for X
X_train = train.loc[:, train.columns != target]
y_train = train.loc[:, target]

# Split training data as training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2,
                                                  random_state=43)

# Set the queries for LGBM
query_train = [X_train.shape[0]]
query_val = [X_val.shape[0]]

# Create LGBMRanker model
ranker = lgb.LGBMRanker()
# Set start time
start = time()
# Train data
ranker.fit(X_train, y_train, group=query_train,
        eval_set=[(X_val, y_val)], eval_group=[query_val],
        eval_at=[5, 10, 20], early_stopping_rounds=50,
        verbose=False)

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
