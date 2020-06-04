#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:11:29 2020

@author: anu
"""


# Import libraries
import pandas as pd

# Define main path for importing
path = '../data/ml4qs/created_by_phyphox/'
# Define file names in a lookup dictionary
file_names = {
    'Gyroscope_rotation_rate.csv': None,
    'Gyroscope_rotation_rate2.csv': None,
    'LocationGPS.csv': None,
    'Magnetometer.csv': None,
    'Pressure.csv': None
}

# Import data from given path
for file_name in file_names:
    # Import csv and put it into dictionary
    file_names[file_name] = pd.read_csv(path + file_name)
    # Convert Time (s) feature to timedelta
    file_names[file_name].loc[:, 'time'] = pd.to_timedelta(
        file_names[file_name].loc[:, 'Time (s)'], 'ms')
    # Drop Time (s) column
    file_names[file_name].drop('Time (s)', axis=1, inplace=True)
    # Set new converted time as index
    file_names[file_name].set_index('time', inplace=True)
