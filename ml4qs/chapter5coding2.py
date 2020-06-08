# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
df = pd.read_csv('../data/ml4qs/created_by_phyphox/Gyroscope_rotation_rate2.csv')
df = df.set_index(pd.to_timedelta(df.time, 'ms'))
