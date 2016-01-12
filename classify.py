from sklearn.linear_model import LogisticRegression

def classify(trainSet, trainLabels, testSet):
	
	clf = LogisticRegression()
	clf.fit(trainSet, trainLabels)
	predictedLabels = clf.predict_proba(testSet)

	return predictedLabels, clf.classes_
