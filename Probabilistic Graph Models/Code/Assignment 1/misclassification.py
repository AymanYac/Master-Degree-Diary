# Run with anaconda2-4.1.1
import qdamodel as qdmod
import generativemodel as genmod
import linearregression as linreg
import logisticregression as logreg
import logging
import numpy as np

# Setup the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Initalize the variable to store the test data in the structure
# A-B-C/lda-logreg-linreg-qda/train-test/misclassified-total
n = np.zeros((3, 4, 2, 2))

# Parameter to decide if the figure are to be saved or not
savefig = True

# Go through all the datasets
datasets = ["A", "B", "C"]
for i in range(0,3):
	dataset = datasets[i]
	logging.info("---- Dataset %s ----"%dataset)

	# Get the training and testing file name
	filetrain = "classification_data_HWK1/classification%s.train"%dataset
	filetest = "classification_data_HWK1/classification%s.test"%dataset

	# Test the generative model
	x_mean, x_var, pi, u, sig = genmod.fit(filetrain, savefig)
	n_mis_train, n_total_train = genmod.misclassification(filetrain, 
		x_mean, x_var, pi, u, sig, savefig)
	n_mis_test, n_total_test = genmod.misclassification(filetest, 
		x_mean, x_var, pi, u, sig, savefig)
	n[i][0][0][:] = np.array([n_mis_train, n_total_train])[:]
	n[i][0][1][:] = np.array([n_mis_test, n_total_test])[:]

	# Test the logistic regression
	x_mean, x_var, w = logreg.fit(filetrain, savefig)
	n_mis_train, n_total_train = logreg.misclassification(filetrain,
		x_mean, x_var, w, savefig)
	n_mis_test, n_total_test = logreg.misclassification(filetest, 
		x_mean, x_var, w, savefig)
	n[i][1][0][:] = np.array([n_mis_train, n_total_train])[:]
	n[i][1][1][:] = np.array([n_mis_test, n_total_test])[:]

	# Test the linear regression
	x_mean, x_var, w, sig = linreg.fit(filetrain, savefig)
	n_mis_train, n_total_train = linreg.misclassification(filetrain, 
		x_mean, x_var, w, sig, savefig)
	n_mis_test, n_total_test = linreg.misclassification(filetest, 
		x_mean, x_var, w, sig, savefig)
	n[i][2][0][:] = np.array([n_mis_train, n_total_train])[:]
	n[i][2][1][:] = np.array([n_mis_test, n_total_test])[:]

	# Test the QDA model
	x_mean, x_var, pi, u, sig0, sig1 = qdmod.fit(filetrain, savefig)
	n_mis_train, n_total_train = qdmod.misclassification(filetrain,
		x_mean, x_var, pi, u, sig0, sig1)
	n_mis_test, n_total_test = qdmod.misclassification(filetest,
		x_mean, x_var, pi, u, sig0, sig1)
	n[i][3][0][:] = np.array([n_mis_train, n_total_train])[:]
	n[i][3][1][:] = np.array([n_mis_test, n_total_test])[:]

for i in range(0,3):
	logging.info("---- Dataset %s Testing Results ----"%datasets[i])
	logging.info("generative model train: %d out of %d, %0.1f%% error"%
		(n[i][0][0][0], n[i][0][0][1], 100*n[i][0][0][0]/n[i][0][0][1]))
	logging.info("generative model test: %d out of %d, %0.1f%% error"%
		(n[i][0][1][0], n[i][0][1][1], 100*n[i][0][1][0]/n[i][0][1][1]))
	logging.info("logistic regression train: %d out of %d, %0.1f%% error"%
		(n[i][1][0][0], n[i][1][0][1], 100*n[i][1][0][0]/n[i][1][0][1]))
	logging.info("logistic regression test: %d out of %d, %0.1f%% error"%
		(n[i][1][1][0], n[i][1][1][1], 100*n[i][1][1][0]/n[i][1][1][1]))
	logging.info("linear regression train: %d out of %d, %0.1f%% error"%
		(n[i][2][0][0], n[i][2][0][1], 100*n[i][2][0][0]/n[i][2][0][1]))
	logging.info("linear regression test: %d out of %d, %0.1f%% error"%
		(n[i][2][1][0], n[i][2][1][1], 100*n[i][2][1][0]/n[i][2][1][1]))
	logging.info("qda model train: %d out of %d, %0.1f%% error"%
		(n[i][3][0][0], n[i][3][0][1], 100*n[i][3][0][0]/n[i][3][0][1]))
	logging.info("qda model test: %d out of %d, %0.1f%% error"%
		(n[i][3][1][0], n[i][3][1][1], 100*n[i][3][1][0]/n[i][3][1][1]))


