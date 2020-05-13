#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:46:03 2020

@author: anu
"""

import pandas as pd


def append_list(l: list, *argv) -> list:
    '''
    Returns the updated list with given args.

    Parameters
    ----------
    l : list
        List to be updated.
    *argv : list
        Update list.

    Returns
    -------
    l : list
        Updated list.

    '''
    # Iterate through list to be added
    for arg in argv:
        # Append each item
        l.append(arg)

    # Return list
    return l


def pos_meaning(row):
    '''
    Return the importance according to the position value

    Parameters
    ----------
    row : pandas' dataframe's row
        A single row from the dataframe.

    Returns
    -------
    str
        Importance value.

    '''
    if row.position <= 10:
        return 'Important'
    elif row.position <= 20:
        return 'SlighlyImportant'
    elif row.position <= 30:
        return 'Average'
    else:
        return 'Low'


def target_col(row):
    '''
    Returns new metric calculation for the target column

    Parameters
    ----------
    row : pandas' dataframe's row
        A single row from the dataframe.

    Returns
    -------
    float
        New metric value.

    '''
    metric = int(row.booking_bool * 5 + row.click_bool)
    if row.pos_meaning == 'Important':
        return 5 * metric
    elif row.pos_meaning == 'SlightlyImportant':
        return 3 * metric
    elif row.pos_meaning == 'Average':
        return 2 * metric
    else:
        return metric


# Import evaluated training data
train = pd.read_csv('../../../../data/eval_training_set.csv')

# Import test data
test = pd.read_csv('../../../../data/eval_test_set.csv')

# Copy datasets
# train, test = df_train.copy(), df_test.copy()

# Create another column called pos_meaning to represent position in later
train['pos_meaning'] = train.apply(pos_meaning, axis=1)

# Create target column
train['target'] = train.apply(target_col, axis=1)

# Drop unnecessary columns for training set
# Set columns list to be dropped
dropCols = [
    'booking_bool',
    'click_bool',
    'position',
    'pos_meaning'
]

# Drop columns for training set
train.drop(dropCols, inplace=True, axis=1)

# Set a list for categorical values
categoricalCols = [
    'srch_id', 'site_id', 'visitor_location_country_id',
    'prop_country_id', 'prop_id', 'prop_brand_bool',
    'promotion_flag', 'srch_destination_id',
    'random_bool', 'with_family', 'is_Fall',
    'is_Spring', 'is_Summer', 'is_Winter'
]
# Change categorical columns' type
train[categoricalCols] = train[categoricalCols].astype('category')
