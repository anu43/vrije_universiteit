# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
df = pd.read_csv('../data/ml4qs/created_by_phyphox/Gyroscope_rotation_rate2.csv')
df = df.set_index(pd.to_timedelta(df.time, 'ms'))

# Set figure and axes
fig = plt.figure()
ax = plt.axes(projection="3d")

# Plot 3D
ax.scatter3D(df.gyro_x, df.gyro_y, df.gyro_z, c=df.gyro_z, cmap='hsv')

# Show plot
plt.show()
