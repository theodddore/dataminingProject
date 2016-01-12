import numpy as np
import csv as csv
from log_loss import log_loss
from sklearn.cross_validation import KFold
from classify import classify

# Load data
csv_file_object = csv.reader(open('train.csv', 'rb')) # Load in the csv file
header = csv_file_object.next() 					  # Skip the fist line as it is a header
rows=[] 											  # Create a variable to hold the data

for row in csv_file_object: # Skip through each row in the csv file,
    rows.append(row[0:]) 	# adding each row to the data variable
data = np.array(rows) 		# Then convert from a list to an array



daysIndex = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6} # A dictionary keyed by day to its index



numVisits = len(np.unique(data[:,1])) # Number of distinct visits

dept = np.unique(data[:,5])

numDep = len(dept)
#X = np.zeros((numVisits, len(daysIndex))) # Matrix containing the day of each visit

counter = 7

length = numDep + len(daysIndex)

departmentIndex = {}

for j in dept:
	departmentIndex[j] = counter
	counter += 1

#departmentIndex = {department: i + 6 for i, department in enumerate(dept)}

X = np.zeros((numVisits, length)) 
Y = np.zeros(numVisits) # A matrix containing the trip types of the visits

previousVisit = 0
index = -1
for i in range(data.shape[0]):
	if data[i,1] != previousVisit: 		# If visit number has changed, initialize a new visit
		index += 1						# The index of the new visit
		num_products = 1				# Set the number of products to 1
		previousVisit = data[i,1]			# Set previous visit number to the current visit
		X[index,daysIndex[data[i,2]]] = 1	# Set the index of the day of the visit to 1
		X[index,departmentIndex[data[i,5]]] = 1
		Y[index] = int(data[i,0])	    # Store the type of the trip of the current visit
	else: 								# If visit number has not changed, it's still the same visit
		num_products += 1				# Increase the number of products of the current visit
		X[index,departmentIndex[data[i,5]]] += 1

kf = KFold(X.shape[0], n_folds=10) # Initialize cross validation



iterations = 0 # Variable that will store the total iterations  
totalLogloss = 0 # Variable that will store the correctly predicted intances  

for trainIndex, testIndex in kf:
	trainSet = X[trainIndex]
	testSet = X[testIndex]
	trainLabels = Y[trainIndex]
	testLabels = Y[testIndex]

	predictions, trips = classify(trainSet, trainLabels, testSet)
	logloss = log_loss(testLabels, predictions, trips)	
	print 'Log Loss: ', logloss
	totalLogloss += logloss
	iterations += 1
print 'Average Log Loss: ', totalLogloss/iterations
