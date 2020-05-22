import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from sklearn.model_selection import train_test_split


def closestNumber(n, m):

    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if((n * m) > 0):
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    # if true, then n1 is the required closest number
    if (abs(n - n1) < abs(n - n2)):
        return n1

    # else n2 is the required closest number
    return n2


def extract_groups(df) -> list:

    # Define an empty list for groups
    groups = list()

    # Set ids from 'srch_id'
    ids = df.srch_id.unique()
    # Define the length of ids
    lenID = len(ids)

    # Return the same or closest length that is divisible by 4
    lenID = closestNumber(lenID, 4)

    # Iterate through ids in train set
    for i, id_ in enumerate(ids):
        # Print the progress in percentage
        if i == lenID / 4:
            print('25% at:', datetime.now().time())
        elif i == lenID / 2:
            print('50% at:', datetime.now().time())
        elif i == 3 * lenID / 4:
            print('75% at:', datetime.now().time())

        # Append the length of the specific id to the groups list
        groups.append(len(df[df.srch_id == id_]))

    return groups


def set_types(train, test):

    # Define bool variables
    boolVars = [
        'prop_brand_bool',
        'promotion_flag',
        'srch_saturday_night_bool',
        'random_bool',
        'with_family',
        'is_Fall',
        'is_Spring',
        'is_Summer',
        'is_Winter'
    ]
    # Change categorical columns' type
    train[boolVars] = train[boolVars].astype('bool')
    test[boolVars] = test[boolVars].astype('bool')

    return train, test


def prepareSets4XGBRanker(train):

    # Prepare X and y
    target = 'target'  # Set column name for y
    # Receive all columns except target column for X
    X_train = train.loc[:, train.columns != target]
    # Get the target column only
    y_train = train.loc[:, target]

    # Split training data as training and validation
    print('splitting training/validation sets')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      random_state=43)

    # Sort training/validation sets
    print('sorting training/validation sets')
    X_train.sort_values('srch_id', inplace=True)
    y_train = y_train.reindex(X_train.index)
    X_val.sort_values('srch_id', inplace=True)
    y_val = y_val.reindex(X_val.index)

    # Set the queries for LGBM
    print('extracting training groups')
    # Set start time
    start = time()
    groups_train = extract_groups(X_train)
    # Set stop time
    stop = time()
    print('extracting training groups took', (stop - start) / 60, 'mins')
    print('extracting validation groups')
    # Set start time
    start = time()
    groups_val = extract_groups(X_val)
    # Set stop time
    stop = time()
    print('extracting validation groups took', (stop - start) / 60, 'mins')

    # Drop 'srch_id' from training/validation sets
    X_train.drop('srch_id', axis=1, inplace=True)
    X_val.drop('srch_id', axis=1, inplace=True)

    return X_train, y_train, X_val, y_val, groups_train, groups_val


def train_lgbm_model(train, test):
    '''
    Returns the tested dataset after the training
    Ranking results are in the column called ranking_rates

    Parameters
    ----------
    train : Pandas' DataFrame object
        Train dataset.
    test : Pandas' DataFrame object
        Test dataset.

    Returns
    -------
    test : Pandas' DataFrame object
        Test dataset with ranking_rates column.
    ranker.eval_results_ : array
        Evaluation metrics in the training session.

    '''

    categoricalCols = [
        'site_id', 'visitor_location_country_id',
        'prop_country_id', 'prop_id', 'promotion_flag',
        'srch_destination_id'
    ]

    # Split and prepare groups of train/val sets
    X_train, y_train, X_val, y_val, groups_train, groups_val = prepareSets4XGBRanker(train)

    # Sort test data by id
    test.sort_values('srch_id', inplace=True)
    # Drop srch_id from test set
    X_test = test.drop('srch_id', axis=1)

    ranker = lgb.LGBMRanker()

    # Set start time
    start = time()
    # Fit the model
    print('started training')
    ranker.fit(X_train, y_train, group=groups_train, eval_metric='ndcg@n',
               eval_set=[(X_val, y_val)], eval_group=[groups_val],
               early_stopping_rounds=50, feature_name=X_train.columns.to_list(),
               categorical_feature=categoricalCols, verbose=False)
    # Set stop time
    stop = time()
    # Print training time
    print('LGBM training time', (stop - start) / 60, 'mins')

    # Reindex columns for prediction
    # cols = list(ranker.feature_name_)
    # X_test = test.reindex(cols, axis=1)
    # Predict
    print('predicting')
    preds = ranker.predict(X_test)
    # Put the predictions to the test frame
    test["ranking_rates"] = preds
    # Sort values by first 'srch_id', then 'ranking_rates'
    test.sort_values(['srch_id', 'ranking_rates'],
                     ascending=[True, False],
                     inplace=True)
    print('done predicting')

    # Return test set with results
    return test, ranker.evals_result_['valid_0']['ndcg@1']


def train_xgbRanker_model(train, test, with_val=False):
    '''
    Returns the tested dataset after the training
    Ranking results are in the column called ranking_rates

    Parameters
    ----------
    train : Pandas' DataFrame object
        Train dataset.
    test : Pandas' DataFrame object
        Test dataset.
    with_val: Bool
        Indicator to whether train with/out validation set

    Returns
    -------
    test : Pandas' DataFrame object
        Test dataset with ranking_rates column.
    ranker.eval_results_ : array
        Evaluation metrics in the training session.

    '''

    # Define parameters for XGBRanker model
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg',
        'learning_rate': 0.02,
        'gamma': 0.5,
        'min_child_weight': 1,
        'max_depth': 10,
        'n_estimators': 500
    }

    # Create the XGBRanker model
    ranker = xgb.sklearn.XGBRanker(**params)

    # If the training would be with validation set
    if with_val:
        # Split and prepare groups of train/val sets
        X_train, y_train, X_val, y_val, groups_train, groups_val = prepareSets4XGBRanker(train)

        # Set start time
        start = time()
        # Fit the model
        print('started training with eval set')
        ranker.fit(X_train, y_train, group=groups_train,
                   eval_set=[(X_val, y_val)], eval_group=[groups_val],
                   early_stopping_rounds=50, verbose=False)
        # Set stop time
        stop = time()
        # Print training time
        print('XGB Ranker training time', (stop - start) / 60, 'mins')
        
        # Plot the tree
        # print('Plotting the model tree')
        # xgb.plot_tree(ranker, rankdir='LR')

    else:
        # Prepare X_train and y_train
        target = 'target'  # Set column name for y
        # Receive all columns except target column for X
        X_train = train.loc[:, train.columns != target]
        # Get the target column only
        y_train = train.loc[:, target]

        # Set the grouping for training set
        print('extracting training groups at:', datetime.now().time())
        # Set start time
        start = time()
        groups_train = extract_groups(X_train)
        # Set stop time
        stop = time()
        print('extracting training groups took', (stop - start) / 60, 'mins')

        # Drop 'srch_id' from training set
        X = X_train.drop('srch_id', axis=1)

        # Set start time
        start = time()
        # Fit the model
        print('started training')
        ranker.fit(X, y_train, group=groups_train, verbose=False)
        # Set stop time
        stop = time()
        # Print training time
        print('XGB Ranker training time', (stop - start) / 60, 'mins')
        
        # Plot the tree
        # print('Plotting the model tree')
        # xgb.plot_tree(ranker, rankdir='LR')

    # Sort test data by id
    # test.sort_values('srch_id', inplace=True)

    # Drop srch_id from test set
    X_test = test.drop('srch_id', axis=1)

    # Reindex columns for prediction
    cols = list(ranker.get_booster().feature_names)
    X_test = test.reindex(cols, axis=1)

    # Predict
    print('predicting')
    preds = ranker.predict(X_test)
    # Put the predictions to the test frame
    test["ranking_rates"] = preds
    # Sort values by first 'srch_id', then 'ranking_rates'
    test.sort_values(['srch_id', 'ranking_rates'],
                     ascending=[True, False],
                     inplace=True)
    print('done predicting')

    # Return test set with results
    return test, ranker.evals_result['eval_0']['ndcg']


def plot_eval_results(xgbRankerResults, LGBMRankerResults):

    # Set min and max value for y range
    # minval = min(min(xgbRankerResults), min(LGBMRankerResults))
    # maxval = max(max(xgbRankerResults), max(LGBMRankerResults))

    # Plot xgbRankerResults and LGBMRankerResults
    plt.subplot(221)
    plt.plot(xgbRankerResults)
    
    # Set title
    plt.title('NDGC Metric Evaluation\n for XGBRanker')
    # Set xlabel
    plt.xlabel('Number of Evaluations')
    # Set ylabel
    plt.ylabel('NDCG Metric Range')

    plt.subplot(222)
    plt.plot(LGBMRankerResults)
    
    # Set title
    plt.title('NDGC Metric Evaluation\n for LGBMRanker')
    # Set xlabel
    plt.xlabel('Number of Evaluations')

    plt.figure(figsize=(20,20))    

    # Show plot
    plt.show()


# Import training/test sets
train = pd.read_csv('../../../../data/4trainingTesting/training.csv')
test = pd.read_csv('../../../../data/4trainingTesting/test.csv')

# Set types of train/test
train, test = set_types(train, test)

# Train LGBM model
testL, LGBMRankerResults = train_lgbm_model(train, test)

# Train XGBoostRanker model
testX, xgbRankerResults = train_xgbRanker_model(train, test, True)

# Plot results
plot_eval_results(xgbRankerResults, LGBMRankerResults)

# Write the result to csv
test.to_csv('../../../../data/predicted_test.csv', index=False)
