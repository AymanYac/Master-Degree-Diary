# Run with anaconda2-4.1.1
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import logging

def fit(file, savefig=False):
	"Fit the qda model with the data in the file."
	
	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

	# Retrieve the dataset
	dataset = file[file.find(".")-1]
	logging.info("---- Dataset %s QDA Fitting ----"%dataset)

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

	# Compute the MLE for the covariance matrix sigma0
	sig0 = np.zeros((2,2))
	sig0 = np.dot((x[y==0]-u[y[y==0].astype(int)]).T, 
		x[y==0]-u[y[y==0].astype(int)])/y[y==0].shape[0]
	logging.info("sig0 line 1: (%f, %f)"%(sig0[0,0], sig0[0,1]))
	logging.info("sig0 line 2: (%f, %f)"%(sig0[1,0], sig0[1,1]))

	# Compute the MLE for the covariance matrix sigma1
	sig1 = np.zeros((2,2))
	sig1 = np.dot((x[y==1]-u[y[y==1].astype(int)]).T, 
		x[y==1]-u[y[y==1].astype(int)])/y[y==1].shape[0]
	logging.info("sig1 line 1: (%f, %f)"%(sig1[0,0], sig1[0,1]))
	logging.info("sig1 line 2: (%f, %f)"%(sig1[1,0], sig1[1,1]))

	# Compute the line defined by p(y=0|x)=p(y=1|x)
	# which is equal to xtAx+Bx+C=0 or ay2+by+c=0
	u0 = u[0,:].reshape((2,1))
	u1 = u[1,:].reshape((2,1))
	A = inv(sig0) - inv(sig1)
	B = np.zeros(2)
	B[:] = (2*np.dot(u1.T, inv(sig1)) - 2*np.dot(u0.T, inv(sig0)))[:]
	C = np.dot(np.dot(u0.T, inv(sig0)), u0) \
		- np.dot(np.dot(u1.T, inv(sig1)), u1) \
		+ np.log(det(sig0)) \
		- np.log(det(sig1))
	x_ep = np.zeros((2001, 3))
	x_ep[:,0] =  np.arange(-10, 10+20./(x_ep.shape[0]-1),
		20./(x_ep.shape[0]-1))[:]
	x_ep[:,0] = (x_ep[:,0]-x_mean[0])/x_var[0]
	a = A[1,1]*np.ones((x_ep.shape[0], 1))
	b = (2*A[0,1]*x_ep[:,0] + B[1]).reshape(x_ep.shape[0], 1)
	c = C[0][0] + B[0]*x_ep[:,0] + A[0,0]*np.square(x_ep[:,0])
	c = c.reshape(x_ep.shape[0], 1)
	# Disable warning for sqrt of neg and retrieve as nan
	old_settings = np.seterr(all='ignore')
	x_ep[:,1] = ((-b+np.sqrt(np.square(b)-4*a*c))/(2.*a)).reshape(x_ep.shape[0])
	x_ep[:,2] = ((-b-np.sqrt(np.square(b)-4*a*c))/(2.*a)).reshape(x_ep.shape[0])
	np.seterr(**old_settings)
	# Unnormalize
	x_ep[:,0] = x_ep[:,0]*x_var[0]+x_mean[0]
	x_ep[:,1] = x_ep[:,1]*x_var[1]+x_mean[1]
	x_ep[:,2] = x_ep[:,2]*x_var[1]+x_mean[1]
	# Reunite the two lines
	x_notnan = np.isnan(x_ep)[:,1]==0
	l = x_ep[x_notnan].shape[0]
	x_ep2 = np.zeros((2*l, 2))
	if np.where(x_notnan==False)[0][0]!=0:
		x_ep2[0:l,0][:] = x_ep[x_notnan, 0][:]
		x_ep2[0:l,1][:] = x_ep[x_notnan, 1][:]
		x_ep2[l:2*l,0][:] = x_ep[x_notnan, 0][::-1]
		x_ep2[l:2*l,1][:] = x_ep[x_notnan, 2][::-1]
	else:
		x_ep2[0:l,0][:] = x_ep[x_notnan, 0][::-1]
		x_ep2[0:l,1][:] = x_ep[x_notnan, 1][::-1]
		x_ep2[l:2*l,0][:] = x_ep[x_notnan, 0][:]
		x_ep2[l:2*l,1][:] = x_ep[x_notnan, 2][:]

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
		plt.plot(x_ep2[:,0], x_ep2[:,1], 'g-', label="equiprobability line")

		# Configure  and display the plot
		plt.axis([-10,10,-10,10])
		plt.title("Generative model (QDA) - dataset " + dataset)
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/5-qdamodel-%s.png"%dataset, \
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("Saved as 5-qdamodel-%s.png"%dataset)

	# Return the parameters
	return x_mean, x_var, pi, u, sig0, sig1

def misclassification(file, x_mean, x_var, pi, u, sig0, sig1, savefig=False):
	"Classify the data in the file using the QDA model and "
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
	p0 = np.sum(np.multiply(np.dot((x-u0), inv(sig0)), (x-u0)), axis=1) \
		+ np.log(det(sig0))
	p1 = np.sum(np.multiply(np.dot((x-u1), inv(sig1)), (x-u1)), axis=1) \
		+ np.log(det(sig1))

	# Retrieve the misclassified elements
	y_wrong = np.fabs(y!=np.fabs(p0>p1))
	n_misclassified = np.sum(y_wrong)
	logging.info("%d elements misclassified"%n_misclassified)

	# Compute the line defined by p(y=0|x)=p(y=1|x)
	# which is equal to xtAx+Bx+C=0 or ay2+by+c=0
	u0 = u[0,:].reshape((2,1))
	u1 = u[1,:].reshape((2,1))
	A = inv(sig0) - inv(sig1)
	B = np.zeros(2)
	B[:] = (2*np.dot(u1.T, inv(sig1)) - 2*np.dot(u0.T, inv(sig0)))[:]
	C = np.dot(np.dot(u0.T, inv(sig0)), u0) \
		- np.dot(np.dot(u1.T, inv(sig1)), u1) \
		+ np.log(det(sig0)) \
		- np.log(det(sig1))
	x_ep = np.zeros((2001, 3))
	x_ep[:,0] =  np.arange(-10, 10+20./(x_ep.shape[0]-1),
		20./(x_ep.shape[0]-1))[:]
	x_ep[:,0] = (x_ep[:,0]-x_mean[0])/x_var[0]
	a = A[1,1]*np.ones((x_ep.shape[0], 1))
	b = (2*A[0,1]*x_ep[:,0] + B[1]).reshape(x_ep.shape[0], 1)
	c = C[0][0] + B[0]*x_ep[:,0] + A[0,0]*np.square(x_ep[:,0])
	c = c.reshape(x_ep.shape[0], 1)
	# Disable warning for sqrt of neg and retrieve as nan
	old_settings = np.seterr(all='ignore')
	x_ep[:,1] = ((-b+np.sqrt(np.square(b)-4*a*c))/(2.*a)).reshape(x_ep.shape[0])
	x_ep[:,2] = ((-b-np.sqrt(np.square(b)-4*a*c))/(2.*a)).reshape(x_ep.shape[0])
	np.seterr(**old_settings)
	# Unnormalize
	x_ep[:,0] = x_ep[:,0]*x_var[0]+x_mean[0]
	x_ep[:,1] = x_ep[:,1]*x_var[1]+x_mean[1]
	x_ep[:,2] = x_ep[:,2]*x_var[1]+x_mean[1]
	# Reunite the two lines
	x_notnan = np.isnan(x_ep)[:,1]==0
	l = x_ep[x_notnan].shape[0]
	x_ep2 = np.zeros((2*l, 2))
	if np.where(x_notnan==False)[0][0]!=0:
		x_ep2[0:l,0][:] = x_ep[x_notnan, 0][:]
		x_ep2[0:l,1][:] = x_ep[x_notnan, 1][:]
		x_ep2[l:2*l,0][:] = x_ep[x_notnan, 0][::-1]
		x_ep2[l:2*l,1][:] = x_ep[x_notnan, 2][::-1]
	else:
		x_ep2[0:l,0][:] = x_ep[x_notnan, 0][::-1]
		x_ep2[0:l,1][:] = x_ep[x_notnan, 1][::-1]
		x_ep2[l:2*l,0][:] = x_ep[x_notnan, 0][:]
		x_ep2[l:2*l,1][:] = x_ep[x_notnan, 2][:]

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
		plt.plot(x_ep2[:,0], x_ep2[:,1], 'g-', label="equiprobability line")

		# Plot the misclassified points in black circle
		plt.plot(x[y_wrong==1][:,0]*x_var[0]+x_mean[0], 
			x[y_wrong==1][:,1]*x_var[1]+x_mean[1], 
			'kv', label="misclassified points")

		# Configure  and display the plot
		plt.axis([-10,10,-10,10])
		plt.title("Generative model (LDA) - dataset %s %s"%(dataset, kind))
		plt.legend(numpoints=1)
		plt.gcf().savefig(
			"Report/Figures/4-qdamodel-%s-%s.png"%(dataset, kind),
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("saved as 4-qdamodel-%s-%s.png"%(dataset, kind))

	# Return the misclassified points and the points total number
	return n_misclassified, y.shape[0]

	return

if __name__ == "__main__":
	for dataset in ["A", "B", "C"]:
		filetrain = "classification_data_HWK1/classification%s.train"%dataset
		filetest = "classification_data_HWK1/classification%s.test"%dataset
		x_mean, x_var, pi, u, sig0, sig1 = fit(filetrain, True)
		misclassification(filetrain, x_mean, x_var, pi, u, sig0, sig1, True)
		misclassification(filetest, x_mean, x_var, pi, u, sig0, sig1, True)


