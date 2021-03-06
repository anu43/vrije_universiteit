# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import xgboost as xgb
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

# Read processed csv
df = pd.read_csv('gyro.csv')

# Apply rolling to get previous movements from each axis by exponential weights
weights = np.arange(1, 11)  # Exponential weights for window 10
# Apply rolling to X-axis
df['gyro_x_window'] = df.gyro_x.rolling(10).apply(
    lambda axs: np.dot(axs, weights) / weights.sum()
)
# Apply rolling to X-axis
df['gyro_y_window'] = df.gyro_y.rolling(10).apply(
    lambda axs: np.dot(axs, weights) / weights.sum()
)
# Apply rolling to X-axis
df['gyro_z_window'] = df.gyro_z.rolling(10).apply(
    lambda axs: np.dot(axs, weights) / weights.sum()
)

train = df.drop('time', axis=1)

# Split data into train/val/test sets
X = train.loc[:, train.columns != 'activity']
y = train.activity
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=43)  # train/test

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2, random_state=43)  # train/val

# Apply XGBoost Classifier
model = xgb.XGBClassifier()
# Fit the model
# eval_set = tuple(zip(X_val.values, y_val.values))  # created eval set
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=False)

# Predict the test samples
predictions = model.predict(X_test)

# Create confusion matrix
results = confusion_matrix(y_test, predictions)
print(results)
# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
# Calculate precision/recall/f1
report = classification_report(y_test, predictions)
print(report)

# Receive evaluation results
eval_results = model.evals_result()
# Plot evaluation results
plt.plot(eval_results['validation_0']['merror'])
plt.xlabel('Epochs')  # X-Axis label
plt.ylabel('Error rate')  # Y-Axis label
plt.title('The Evaluation Results of XGBoost')  # Title
# Save figure
plt.savefig('xgboost_eval.png', quality=95)

# Apply Random Forest Classifier
model_forest = RandomForestClassifier()
# Split data into train/test sets
train.dropna(inplace=True)  # Drop NaN values for the model
X = train.loc[:, train.columns != 'activity']
y = train.activity
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=43)

# Fit the model
model_forest.fit(X_train, y_train)

# Predict the test samples
predictions = model_forest.predict(X_test)

# Create confusion matrix
results = confusion_matrix(y_test, predictions)
print(results)
# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
# Calculate precision/recall/f1
report = classification_report(y_test, predictions)
print(report)

# Apply SVM
model_SVM = svm.SVC()
# Fit the model
model_SVM.fit(X_train, y_train)

# Predict the test samples
predictions = model_SVM.predict(X_test)

# Create confusion matrix
results = confusion_matrix(y_test, predictions)
print(results)
# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
# Calculate precision/recall/f1
report = classification_report(y_test, predictions, zero_division=1)
print(report)
