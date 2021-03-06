#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:33:14 2020

@author: anu
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np

# Import original training data
df = pd.read_csv('../../../../data/training_set_VU_DM.csv')

# Import evaluated training data
df = pd.read_csv('../../../../data/eval_training_set.csv')

# Import test data
df = pd.read_csv('../../../../data/test_set_VU_DM.csv')

# Convert date column to date_time type
df['date_time'] = pd.to_datetime(df['date_time'])

# Define an empty list for columns to be deleted
dropCols = list()


def flatten(l):
    '''
    Flattens the multidimensional list to one dimensional

    Parameters
    ----------
    l : list

    Returns
    -------
    list
        The flattened list.

    '''
    return [item for sublist in l for item in sublist]


def list_missing_features_fraction(df):
    '''
    Prints the list of missing values in a given dataframe by descended sorting

    Parameters
    ----------
    df : Pandas DataFrame object
        The given dataset.

    Returns
    -------
    None.

    '''
    # Get the percentage of missing values for each column
    missing_frac = 1 - df.count() / len(df)
    return missing_frac[missing_frac > 0.0].sort_values(ascending=False)


def get_columns_names_has_missing(df):
    '''
    Returns the list of columns where the column has missing values

    Parameters
    ----------
    df : Pandas DataFrame object
        The given dataset.

    Returns
    -------
    list
        The list of columns where the column has missing values.

    '''
    return df.columns[df.isnull().any()].tolist()


def set_X_y_by_missing_vals(df, target):
    '''
    Returns X and y according to the given target name by
    splitting the dataset as looking to the missing values

    Parameters
    ----------
    df : Pandas DataFrame's object
        The given dataset.
    target: str
        The target column's name.

    Returns
    -------
    X : Pandas DataFrame's object
        The rest of the variables except the target value.
    y : Pandas DataFrame's object
        The target value.

    '''
    subdf = df[df[target].notnull()]
    X = subdf.loc[:, subdf.columns != target]
    y = subdf[target]

    return X, y


def season(value):
    '''
    A lookup function for seasonal transformation
    according to the time variable

    Parameters
    ----------
    value : M8[ns]
        Pandas date_time variable.

    Returns
    -------
    str
        related season.

    '''
    # Set seasonal ranges
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)

    if value in spring:
        return 'Spring'
    if value in summer:
        return 'Summer'
    if value in fall:
        return 'Fall'
    else:
        return 'Winter'


# Set the new column season
df['season'] = df.set_index('date_time').index.dayofyear.map(season)
# Convert it as boolean representation
subdf = df.season.str.get_dummies()
subdf.columns = ['is_' + col for col in subdf.columns]
df = pd.concat([df, subdf], axis=1)
df = df.drop('season', axis=1)

# Add date_time column to dropCols list to be deleted later
dropCols.append(['date_time'])

# Create a dictionary for the mean customer review score by the id of the countries
avg_rev_scores = df.groupby(['prop_country_id'])['prop_review_score'].mean().to_dict()
# Fill missing prop_review_score according to the id of the country
df['prop_review_score'] = df.apply(
    lambda row: avg_rev_scores[row['prop_country_id']] if np.isnan(
        row['prop_review_score']) else row['prop_review_score'],
    axis=1)


def is_child(row):
    '''
    An indicator to show whether it is a family vacation or not

    Parameters
    ----------
    row : Pandas DataFrame object
        Frame's row.

    Returns
    -------
    int
        A boolean value to indicate whether it is a family vacation.

    '''
    # If there is a child
    if row['srch_children_count'] > 0:
        # Set it as True
        return 1
    # False otherwise
    return 0


# Add additional column to indicate whether it is a family vacation
df['with_family'] = df.apply(is_child, axis=1)

# Delete rows where price_usd feature is below 10
df = df[df['price_usd'] > 10]


def per_night(row):
    '''
    Calculates per night per room price if price_usd is higher than 150K.
    Otherwise remains the same amount.

    Parameters
    ----------
    row : Pandas DataFrame object
        Frame's row.

    Returns
    -------
    float
        Returns the same or calculated new price.

    '''
    if row.price_usd > 150000:
        if row.srch_children_count > 0:
            person_count = row.srch_adults_count + row.srch_children_count * 0.5
            return row.price_usd / (row.srch_length_of_stay * person_count)
        else:
            return row.price_usd / (row.srch_length_of_stay * row.srch_adults_count)
    return row.price_usd


# Create per night per room feature
df['per_night_per_room'] = df.apply(per_night, axis=1)

# Add the following columns to dropCols list to be deleted later
dropCols.append([
    'price_usd',
    'srch_length_of_stay',
    'srch_adults_count',
    'srch_children_count'
])

# Deal with 'visitor_hist_starrating' and 'visitor_hist_adr_usd'
# by inserting 0 for NaN values
# Set lookup dictionary for filling
visitor_hist_lookup = {
    'visitor_hist_starrating': 0,
    'visitor_hist_adr_usd': 0
}
# Fill according to the lookup dictionary
df = df.fillna(visitor_hist_lookup)

# Add the columns which the very big fraction of their values are missing
dropCols.append([
    'comp1_rate',
    'comp1_inv',
    'comp1_rate_percent_diff',
    'comp2_rate_percent_diff',
    'comp3_rate_percent_diff',
    'comp4_rate',
    'comp4_inv',
    'comp4_rate_percent_diff',
    'comp5_rate_percent_diff',
    'comp6_rate',
    'comp6_inv',
    'comp6_rate_percent_diff',
    'comp7_rate',
    'comp7_inv',
    'comp7_rate_percent_diff',
    'comp8_rate_percent_diff',
    'gross_bookings_usd'
])

# Flatten the list
dropCols = flatten(dropCols)
# Drop the columns in dropCols list
df = df.drop(dropCols, axis=1)

# Generate correlation matrix
corr = df.corr()
# Visualize the correlation matrix
plt.figure(figsize=(40, 40))
sns.heatmap(corr, annot=True, cmap="RdYlGn")

# ----------------- FILLING MISSING VALUES BY PREDICTING -----------------
X, y = set_X_y_by_missing_vals(df, 'prop_location_score2')
data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)

xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

df4Predict = df[df.prop_location_score2.isnull()]
X = df4Predict.loc[:, df4Predict.columns != 'prop_location_score2']
y = df4Predict.prop_location_score2

df.loc[y.index, 'prop_location_score2'] = xg_reg.predict(X)
# ----------------- FILLING MISSING VALUES BY PREDICTING -----------------
