# Import libraries
import pandas as pd

# Import data
df = pd.read_csv('../data/ml4qs/created_by_phyphox/Gyroscope_rotation_rate2.csv')
# Convert time to timedelta
df.time = pd.to_timedelta(df.time, 'S')
# Set time feature as index
df.set_index('time', inplace=True)
