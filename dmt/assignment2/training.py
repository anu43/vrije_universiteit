import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split


categoricalCols = [
    'site_id', 'visitor_location_country_id',
    'prop_country_id', 'prop_id','promotion_flag',
    'srch_destination_id'
]


def extract_groups(df) -> list:

    # Define an empty list for groups
    groups = list()

    # Iterate through ids in train set
    for id_ in df.srch_id.unique():
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

    '''

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
    ranker.fit(X_train, y_train, group=groups_train,
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
    return test


def train_xgboost_model(train, test):
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

    '''

    # Prepare X and y
    target = 'target'  # Set column name for y
    # Receive all columns except target column for X
    X_train = train.loc[:, train.columns != target]
    # Get the target column only
    y_train = train.loc[:, target]

    # Create a DMatrix for XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # Create XGBoost model
    parameters = {
        'max_depth': 10,
        'eta': 1,
        'silent': 1,
        'objective': 'reg:squarederror',
        'eval_metric': 'ndcg',
        'learning_rate': .05
    }

    # Set start time
    start = time()
    # Fit the model
    xg = xgb.train(parameters,
                   dtrain, 50)
    # Set stop time
    stop = time()
    # Print training time
    print('XGBoost training time', (stop - start) / 60, 'mins')

    # Predict
    cols = xg.feature_names
    test = test.reindex(cols, axis=1)
    dtest = xgb.DMatrix(test)
    preds = xg.predict(dtest)
    test['ranking_rates'] = preds
    # test.sort_values(['srch_id', 'ranking_rates'],
    #                  ascending=[True, False],
    #                  inplace=True)

    # Return test set with results
    return test


def train_xgbRanker_model(train, test):
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

    '''

    # Split and prepare groups of train/val sets
    X_train, y_train, X_val, y_val, groups_train, groups_val = prepareSets4XGBRanker(train)

    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg',
        'learning_rate': 0.1,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'max_depth': 10,
        'n_estimators': 10
    }

    # Sort test data by id
    test.sort_values('srch_id', inplace=True)
    # Drop srch_id from test set
    X_test = test.drop('srch_id', axis=1)

    ranker = xgb.sklearn.XGBRanker(**params)

    # Set start time
    start = time()
    # Fit the model
    print('started training')
    ranker.fit(X_train, y_train, group=groups_train,
               eval_set=[(X_val, y_val)], eval_group=[groups_val],
               early_stopping_rounds=50, verbose=False)
    # Set stop time
    stop = time()
    # Print training time
    print('XGB Ranker training time', (stop - start) / 60, 'mins')

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
    return test


# Import training/test sets
train = pd.read_csv('../../../../data/training.csv')
test = pd.read_csv('../../../../data/test.csv')

# Set types of train/test
train, test = set_types(train, test)

# Train LGBM model
test = train_lgbm_model(train, test)

# Train XGBoost model
test = train_xgboost_model(train, test)

# Train XGBoostRanker model
test = train_xgbRanker_model(train, test)

# Write the result to csv
test.to_csv('../../../../data/predicted_test.csv', index=False)
