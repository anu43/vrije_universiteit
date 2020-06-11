# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import data
df = pd.read_csv('../data/ml4qs/created_by_phyphox/Gyroscope_rotation_rate2.csv')
# Convert time to timedelta
df.time = pd.to_timedelta(df.time, 'S')
# Set time feature as index
df.set_index('time', inplace=True)

# Visualize X-Axis
plt.plot(df.gyro_x)

# Create activity column
df['activity'] = np.nan
# Define the set of rules for assigning the type of activity according to time
# If time is between '00:02:18' and '00:02:50', set as standing
ind = df[(df.time >= '00:02:18') & (df.time <= '00:02:50')].index
df.at[ind, 'activity'] = 'standing'
# If time is between '00:05:20' and '00:06:05', set as standing
ind = df[(df.time >= '00:05:20') & (df.time <= '00:06:05')].index
df.at[ind, 'activity'] = 'standing'
# If time is greater than '00:10:25', set as running
ind = df[df.time >= '00:10:25'].index
df.at[ind, 'activity'] = 'running'
# Else set as walking
df[df['activity'] == np.nan] = 'walking'

# Write it to csv without indexing
df.to_csv('./gyro.csv', index=False)
