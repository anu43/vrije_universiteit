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
    'Visualizes the gyroscope data'
    # Set data
    data = file_names[file_name]
    # Create axes
    fig, axs = plt.subplots(3, figsize=(10, 15))

    # Plot features
    axs[0].plot(data.index, data.iloc[:, 0])  # X axis
    axs[1].plot(data.index, data.iloc[:, 1], 'tab:orange')  # Y axis
    axs[2].plot(data.index, data.iloc[:, 2], 'tab:green')  # Z axis
    # axs[3].plot(data.index, data.iloc[:, 3], 'tab:red')  # Absolute value

    # Set titles
    axs[0].set_title('X Axis')
    axs[1].set_title('Y Axis')
    axs[2].set_title('Z Axis')
    # axs[3].set_title('Absolute Rate')

    # Save plot
    plt.savefig('./gyroscope', quality=95)
    # Show plot
    plt.show()


def visualize_gps_data(file_name: str):
    'Visualizes Location (GPS) data'
    # Set data
    data = file_names[file_name]
    # Create axes
    fig, axs = plt.subplots(2, 3, figsize=(10, 15))

    # Plot features
    # First row
    axs[0, 0].plot(data.index, data.iloc[:, 0])  # Latitude
    axs[0, 1].plot(data.index, data.iloc[:, 1])  # Longitude
    axs[0, 2].plot(data.index, data.iloc[:, 2])  # Altitude
    # Second row
    axs[1, 0].plot(data.index, data.iloc[:, 4])  # Speed
    axs[1, 1].plot(data.index, data.iloc[:, 5])  # Direction
    axs[1, 2].plot(data.index, data.iloc[:, 6])  # Distance
    # # Third row
    # axs[2, 0].plot(data.index, data.iloc[:, 7])  # Horizontal Accuracy
    # axs[2, 1].plot(data.index, data.iloc[:, 8])  # Vertical Accuracy
    # axs[2, 2].plot(data.index, data.iloc[:, 9])  # Satellites

    # Set titles
    # First row
    axs[0, 0].set_title('Latitude')
    axs[0, 1].set_title('Longitude')
    axs[0, 2].set_title('Altitude')
    # Second row
    axs[1, 0].set_title('Speed')
    axs[1, 1].set_title('Direction')
    axs[1, 2].set_title('Distance')
    # # Third row
    # axs[2, 0].set_title('Horizontal Accuracy')
    # axs[2, 1].set_title('Vertical Accuracy')
    # axs[2, 2].set_title('Satellites')

    # Save plot
    plt.savefig('./gps', quality=95)
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
visualize_gps_data('LocationGPS.csv')  # GPS data
