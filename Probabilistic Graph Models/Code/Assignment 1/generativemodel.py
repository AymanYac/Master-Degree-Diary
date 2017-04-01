# Run with anaconda2-4.1.1
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import logging

def fit(file, savefig=False):
	"Fit the generative model with the data in the file."
	
	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

	# Retrieve the dataset
	dataset = file[file.find(".")-1]
	logging.info("---- Dataset %s Generative Fitting ----"%dataset)

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

	# Compute the MLE for pi
	pi = np.mean(y)
	logging.info("pi: %f"%pi)

	# Compute the MLE for the normal distributions means
	u = np.zeros((2,2))
	u[0,:] = np.array([np.mean(x[y==0][:,0]), np.mean(x[y==0][:,1])])
	u[1,:] = np.array([np.mean(x[y==1][:,0]), np.mean(x[y==1][:,1])])
	logging.info("u0: (%f, %f)"%(u[0,0], u[0,1]))
	logging.info("u1: (%f, %f)"%(u[1,0], u[1,1]))

	# Compute the MLE for the covariance matrix sigma
	sig = np.zeros((2,2))
	sig = np.dot((x-u[y.astype(int)]).T, x-u[y.astype(int)])/y.shape[0]
	logging.info("sig line 1: (%f, %f)"%(sig[0,0], sig[0,1]))
	logging.info("sig line 2: (%f, %f)"%(sig[1,0], sig[1,1]))

	# Compute the line defined by p(y=0|x)=p(y=1|x)
	# which is equal to ax=b
	u0 = u[0,:].reshape((2,1))
	u1 = u[1,:].reshape((2,1))
	sig_inv = inv(sig)
	a = np.dot((u0-u1).T, sig_inv).reshape(2)
	b = (np.dot(u0.T, np.dot(sig_inv, u0))-np.dot(u1.T, np.dot(sig_inv, u1)))/2
	b = b.reshape(1)
	x_ep = np.array([[-10., 0], [10., 0]])
	x_ep[:,0] = (x_ep[:,0]-x_mean[0])/x_var[0]
	x_ep[:,1] = (b - a[0]*x_ep[:,0])/a[1]
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

		# Plot the normal means u0 and u1 in solid circle
		plt.plot(u[0][0]*x_var[0]+x_mean[0], u[0][1]*x_var[1]+x_mean[1], 
			'bo', label="normal mean for y=0")
		plt.plot(u[1][0]*x_var[0]+x_mean[0], u[1][1]*x_var[1]+x_mean[1], 
			'ro', label="normal mean for y=1")

		# Plot the equiprobability line
		plt.plot(x_ep[:,0], x_ep[:,1], 'g-', label="equiprobability line")

		# Configure  and display the plot
		plt.axis([-10,10,-10,10])
		plt.title("Generative model (LDA) - dataset " + dataset)
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/1-generativemodel-%s.png"%dataset, \
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("Saved as 1-generativemodel-%s.png"%dataset)

	# Return the parameters
	return x_mean, x_var, pi, u, sig

def misclassification(file, x_mean, x_var, pi, u, sig, savefig=False):
	"Classify the data in the file using the generative model and "
	"identify misclassified items."

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

	# Retrieve the dataset and the kind (test or train)
	dataset = file[file.find(".")-1]
	kind = file[file.find(".")+1:]
	logging.info("---- Dataset %s %s QDA Testing ----"%(dataset, kind))

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

	# Retrieve the normal means
	u0 = u[0,:].reshape((1,2))
	u1 = u[1,:].reshape((1,2))

	# Find the greater probability between p(x|y=1) and p(x|y=0)
	p0 = -np.sum(np.multiply(np.dot((x-u0), inv(sig)), (x-u0)), axis=1)
	p1 = -np.sum(np.multiply(np.dot((x-u1), inv(sig)), (x-u1)), axis=1)

	# Retrieve the misclassified elements
	y_wrong = np.fabs(y!=np.fabs(p0<p1))
	n_misclassified = np.sum(y_wrong)
	logging.info("%d elements misclassified"%n_misclassified)

	# Compute the line defined by p(y=0|x)=p(y=1|x)
	# which is equal to ax=b
	u0 = u[0,:].reshape((2,1))
	u1 = u[1,:].reshape((2,1))
	sig_inv = inv(sig)
	a = np.dot((u0-u1).T, sig_inv).reshape(2)
	b = (np.dot(u0.T, np.dot(sig_inv, u0))-np.dot(u1.T, np.dot(sig_inv, u1)))/2
	b = b.reshape(1)
	x_ep = np.array([[-10., 0], [10., 0]])
	x_ep[:,0] = (x_ep[:,0]-x_mean[0])/x_var[0]
	x_ep[:,1] = (b - a[0]*x_ep[:,0])/a[1]
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
		plt.title("Generative model (LDA) - dataset %s %s"%(dataset, kind))
		plt.legend(numpoints=1)
		plt.gcf().savefig(
			"Report/Figures/4-generativemodel-%s-%s.png"%(dataset, kind),
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("saved as 4-generativemodel-%s-%s.png"%(dataset, kind))

	# Return the misclassified points and the points total number
	return n_misclassified, y.shape[0]

if __name__ == "__main__":
	for dataset in ["A", "B", "C"]:
		filetrain = "classification_data_HWK1/classification%s.train"%dataset
		filetest = "classification_data_HWK1/classification%s.test"%dataset
		x_mean, x_var, pi, u, sig = fit(filetrain, True)
		misclassification(filetrain, x_mean, x_var, pi, u, sig, True)
		misclassification(filetest, x_mean, x_var, pi, u, sig, True)
		

