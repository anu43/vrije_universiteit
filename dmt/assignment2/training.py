#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:46:03 2020

@author: anu
"""

import lightgbm as lgb
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


# Import evaluated training data
df_train = pd.read_csv('../../../../data/eval_training_set.csv')

# Import test data
df_test = pd.read_csv('../../../../data/test_set_VU_DM.csv')

# Copy datasets
train, test = df_train.copy(), df_test.copy()

# Create target column
train['target'] = df_train['booking_bool'] * 5 + df_train['click_bool']

# Drop unnecessary columns for training and test set
# Set columns list to be dropped
dropCols = [
    'srch_id',
    'site_id'
]

# Drop columns from test set
test.drop(dropCols, inplace=True, axis=1)
# Update columns list to be dropped
dropCols = append_list(dropCols, 'booking_bool', 'click_bool')
# Drop columns for training set
train.drop(dropCols, inplace=True, axis=1)
