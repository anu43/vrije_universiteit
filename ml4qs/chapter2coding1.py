#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:11:29 2020

@author: anu
"""


# Import libraries
import matplotlib.pyplot as plt
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


def visualize_gyroscope_data(file_name: str):
    'Visualizes the gyroscope data with several different granularities'
    # Set data
    data = file_names[file_name]
    # Create axes
    fig, axs = plt.subplots(4, figsize=(10, 15))

    # Plot features
    axs[0].plot(data.index, data.iloc[:, 0])  # X axis
    axs[1].plot(data.index, data.iloc[:, 1], 'tab:orange')  # Y axis
    axs[2].plot(data.index, data.iloc[:, 2], 'tab:green')  # Z axis
    axs[3].plot(data.index, data.iloc[:, 3], 'tab:red')  # Absolute value

    # Set titles
    axs[0].set_title('X Axis')
    axs[1].set_title('Y Axis')
    axs[2].set_title('Z Axis')
    axs[3].set_title('Absolute Rate')

    # Show plot
    plt.show()


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

# Visualize frames
visualize_gyroscope_data('Gyroscope_rotation_rate.csv')  # Gyroscope data
visualize_gyroscope_data('Gyroscope_rotation_rate2.csv')  # Gyroscope data
