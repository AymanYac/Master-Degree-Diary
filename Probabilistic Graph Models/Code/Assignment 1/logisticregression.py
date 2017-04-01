# Run with anaconda2-4.1.1
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import logging

def logistic(z):
   "Compute the logistic function equal to 1/(1+exp(-z))."
   # Mitigate overflow
   z[z<-500] = -500
   z[z>500] = 500
   return np.reciprocal(1+np.exp(-z))

def fit(file, savefig=False):
	"Fit the logistic regression with the data in the file."

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

	# Retrieve the dataset
	dataset = file[file.find(".")-1]
	logging.info("---- Dataset %s Logistic Fitting ----"%dataset)

	# Declare the variables used for storing the data
	x = np.array([])
	y = np.array([])

	# Get the data from the files
	with open(file, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter='\t')
	     for row in spamreader:
	     	x = np.append(x,[float(row[0]), float(row[1])])
	     	y = np.append(y, float(row[2]))
	x = x.reshape((y.shape[0], 2))
	logging.info("%d elements loaded from file"%y.shape[0])

	# Center the data points
	x_mean = np.mean(x, axis=0)
	x_centered = x - x_mean
	logging.info("data mean: (%f, %f)"%(x_mean[0], x_mean[1]))

	# Normalize the data points
	x_var = np.sqrt(np.sum(np.square(x_centered), axis=0)/y.shape[0])
	x_norm = x_centered/x_var
	logging.info("data var: (%f, %f)"%(x_var[0], x_var[1]))
	x = x_norm

	# Build the design matrix
	X = np.ones((y.shape[0], 3))
	X[:,0] = x[:,0]
	X[:,1] = x[:,1]

	# Try a first version of the algorithm that may not converge
	# for instance if the data is linearly separable
	w = np.zeros(3)
	i = 0
	diff = 1
	while i < 100 and diff>10e-6 and np.sum(np.fabs(w))<10e10:
		w_old = w
		eta = logistic(np.dot(X, w))
		s = np.dot(X.T, y-eta)
		m = np.dot(np.dot(X.T, np.diag(eta*(1-eta))), X)
		w = w_old + np.dot(inv(m), s)
		diff = np.sum(np.fabs(w-w_old))
		i += 1

	# Check if the algorithm converged and if not do the
	# variant of the IRLS with normaliation of w.
	if diff>10e-6 or np.sum(np.fabs(w))>10e10:
		logging.info("Regular IRLS didn't converge")
		w = np.zeros(3)
		i = 0
		diff = 1
		while i < 100 and diff>10e-6 and np.sum(np.fabs(w))<10e10:
			w_old = w
			eta = logistic(np.dot(X, w))
			s = np.dot(X.T, y-eta)
			m = np.dot(np.dot(X.T, np.diag(eta*(1-eta))), X)
			w = w_old + np.dot(inv(m), s)
			w = w/np.sum(np.fabs(w)) # new step: normalization
			diff = np.sum(np.fabs(w-w_old))
			i += 1
		if diff<10e-6 and np.sum(np.fabs(w))<10e10:
			logging.info("Normalized IRLS converged after %d iterations"%i)
	else:
		logging.info("Regular IRLS converged after %d iterations"%i)
	logging.info("w: (%f, %f, %f)"%(w[0], w[1], w[2]))

	# Compute the line defined by p(y=0|x)=p(y=1|x)
	# which is equal to w.T*x=0
	x_ep = np.array([[-10., 0], [10., 0]])
	x_ep[:,0] = (x_ep[:,0]-x_mean[0])/x_var[0]
	x_ep[:,1] = -(w[0]*x_ep[:,0] + w[2])/w[1]
	x_ep = x_ep*x_var+x_mean

	if savefig:
		# Initialize the plot
		plt.figure(figsize=(9,9))

		# Plot the data points, y=1 in red and y=0 in blue
		plt.plot(x[y==0][:,0]*x_var[0]+x_mean[0],
			x[y==0][:,1]*x_var[1]+x_mean[1], 
			'bx', label="data points y=0")
		plt.plot(x[y==1][:,0]*x_var[0]+x_mean[0],
			x[y==1][:,1]*x_var[1]+x_mean[1], 
			'rx', label="data points y=1")

		# Plot the equiprobability line
		plt.plot(x_ep[:,0], x_ep[:,1], 'g-', label="equiprobability line")

		# Configure  and display the plot
		plt.axis([-10,10,-10,10])
		plt.title("Logistic regression - dataset " + dataset)
		plt.legend(numpoints=1)
		plt.gcf().savefig(
			"Report/Figures/2-logisticregression-%s.png"%dataset,
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("saved as 2-logisticregression-%s.png"%dataset)

	# Return the parameters
	return x_mean, x_var, w

def misclassification(file, x_mean, x_var, w, savefig=False):
	"Classify the data in the file using the logistic regression and "
	"identify misclassified items."

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

	# Retrieve the dataset and the kind (test or train)
	dataset = file[file.find(".")-1]
	kind = file[file.find(".")+1:]
	logging.info("---- Dataset %s %s Logistic Testing ----"%(dataset, kind))

	# Declare the variables used for storing the data
	x = np.array([])
	y = np.array([])

	# Get the data from the files
	with open(file, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter='\t')
	     for row in spamreader:
	     	x = np.append(x,[float(row[0]), float(row[1])])
	     	y = np.append(y, float(row[2]))
	x = x.reshape((y.shape[0], 2))
	logging.info("%d elements loaded from file"%y.shape[0])

	# Center the data points
	x_centered = x - x_mean

	# Normalize the data points
	x_norm = x_centered/x_var
	x = x_norm

	# Build the design matrix
	X = np.ones((y.shape[0], 3))
	X[:,0] = x[:,0]
	X[:,1] = x[:,1]

	# Retrieve the misclassified elements
	y_model = np.fabs(np.dot(X, w).reshape(y.shape[0])>0)
	y_wrong = np.fabs(y!=y_model)
	n_misclassified = np.sum(y_wrong)
	logging.info("%d elements misclassified"%n_misclassified)

	# Compute the line defined by p(y=0|x)=p(y=1|x)
	# which is equal to w.T*x=0
	x_ep = np.array([[-10., 0], [10., 0]])
	x_ep[:,0] = (x_ep[:,0]-x_mean[0])/x_var[0]
	x_ep[:,1] = -(w[0]*x_ep[:,0] + w[2])/w[1]
	x_ep = x_ep*x_var+x_mean

	if savefig:
		# Initialize the plot
		plt.figure(figsize=(9,9))

		# Plot the data points, y=1 in red and y=0 in blue
		plt.plot(x[(y==0)*(y_wrong==0)][:,0]*x_var[0]+x_mean[0], 
			x[(y==0)*(y_wrong==0)][:,1]*x_var[1]+x_mean[1], 
			'bx', label="data points y=0")
		plt.plot(x[(y==1)*(y_wrong==0)][:,0]*x_var[0]+x_mean[0], 
			x[(y==1)*(y_wrong==0)][:,1]*x_var[1]+x_mean[1], 
			'rx', label="data points y=1")

		# Plot the equiprobability line
		plt.plot(x_ep[:,0], x_ep[:,1], 'g-', label="equiprobability line")

		# Plot the misclassified points in black
		plt.plot(x[y_wrong==1][:,0]*x_var[0]+x_mean[0], 
			x[y_wrong==1][:,1]*x_var[1]+x_mean[1], 
			'kv', label="misclassified points")

		# Configure  and display the plot
		plt.axis([-10,10,-10,10])
		plt.title("Logistic Regression - dataset %s %s"%(dataset, kind))
		plt.legend(numpoints=1)
		plt.gcf().savefig(
			"Report/Figures/4-logisticregression-%s-%s.png"%(dataset, kind), 
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("saved as 4-logisticregression-%s-%s.png"%(dataset, kind))

	# Return the misclassified points and the points total number
	return n_misclassified, y.shape[0]

if __name__ == "__main__":
	for dataset in ["A", "B", "C"]:
		filetrain = "classification_data_HWK1/classification%s.train"%dataset
		filetest = "classification_data_HWK1/classification%s.test"%dataset
		x_mean, x_var, w = fit(filetrain, True)
		misclassification(filetrain, x_mean, x_var, w, True)
		misclassification(filetest, x_mean, x_var, w, True)


