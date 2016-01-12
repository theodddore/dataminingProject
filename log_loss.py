import numpy as np
from math import log

def log_loss(trueLabels, predictedLabels, trips, eps=1e-15):

   	sums = predictedLabels.sum(axis=1)
   	for i in range(predictedLabels.shape[0]):
		predictedLabels[i,:]/sums[i]
	
	predictedClippedLabels = np.clip(predictedLabels, eps, 1 - eps)

	logloss = 0	
	for i in range(trueLabels.size):
		logloss -= log(predictedClippedLabels[i,np.where(trips==trueLabels[i])[0][0]])
	
	logloss /= trueLabels.size
	
	return logloss
