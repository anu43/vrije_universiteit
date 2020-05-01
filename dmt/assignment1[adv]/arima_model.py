#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:18:50 2020

@author: saurabhjain
"""


import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import seaborn as sns


data_agg = pd.read_csv('./data/frameWithNewFeatures.csv')
data_agg = data_agg.set_index(['time'])
series = pd.DataFrame(data_agg['mood'])

benchmark_pred = data_agg['prevMood']

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
bench_test = benchmark_pred[size:len(benchmark_pred)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history	, order=(8,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
benchmark_error = mean_squared_error(test, bench_test)
print('Test MSE: %.3f' % error)
print('Test MSE: %.3f' % benchmark_error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()