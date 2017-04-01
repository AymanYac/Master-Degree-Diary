# Run with anaconda2-4.1.1
import qdamodel as qdmod
import generativemodel as genmod
import linearregression as linreg
import logisticregression as logreg
import csv
import matplotlib.pyplot as plt
import logging
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

# Setup the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Parameter to decide if the figure are to be saved or not
savefig = True

# Go through all the datasets
datasets = ["A", "B", "C"]
for i in range(0,3):
	# Get the training and testing file name
	dataset = datasets[i]
	logging.info("---- Dataset %s ----"%dataset)
	filetrain = "classification_data_HWK1/classification%s.train"%dataset

	# Declare the variables used for storing the data
	x = np.array([])
	y = np.array([])

	# Get the data from the files
	with open(filetrain, 'rb') as csvfile:
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

	# Initialize the plot
	plt.figure(figsize=(9,9))

	# Plot the data points, y=1 in red and y=0 in blue
	plt.plot(x[y==0][:,0]*x_var[0]+x_mean[0], 
		x[y==0][:,1]*x_var[1]+x_mean[1], 
		'bx', label="data points y=0")
	plt.plot(x[y==1][:,0]*x_var[0]+x_mean[0], 
		x[y==1][:,1]*x_var[1]+x_mean[1], 
		'rx', label="data points y=1")

	# Retrieve the parameters for the LDA model
	x_mean, x_var, pi, u, sig = genmod.fit(filetrain)
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
	# Plot the equiprobability line
	plt.plot(x_ep[:,0], x_ep[:,1], 'g-', label="LDA model")

	# Retrieve the parameters for the logreg model
	x_mean, x_var, w = logreg.fit(filetrain)
	# Compute the line defined by p(y=0|x)=p(y=1|x)
	# which is equal to w.T*x=0
	x_ep = np.array([[-10., 0], [10., 0]])
	x_ep[:,0] = (x_ep[:,0]-x_mean[0])/x_var[0]
	x_ep[:,1] = -(w[0]*x_ep[:,0] + w[2])/w[1]
	x_ep = x_ep*x_var+x_mean
	# Plot the equiprobability line
	plt.plot(x_ep[:,0], x_ep[:,1], 'y-', label="Logistic regression")

	# Retrieve the parameters for the lingref model
	x_mean, x_var, w, sig = linreg.fit(filetrain)	
	# Compute the line defined by p(y=1|x) = 0.5
	x_ep = np.array([[-10., 0], [10., 0]])
	x_ep[:,0] = (x_ep[:,0]-x_mean[0])/x_var[0]
	x_ep[:,1] = (0.5-w[2]-w[0]*x_ep[:,0])/w[1]
	x_ep = x_ep*x_var+x_mean
	# Plot the equiprobability line
	plt.plot(x_ep[:,0], x_ep[:,1], 'm-', label="Linear regression")

	# Retrieve the parameters for the QDA model
	x_mean, x_var, pi, u, sig0, sig1 = qdmod.fit(filetrain)
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
	# Plot the equiprobability line
	plt.plot(x_ep2[:,0], x_ep2[:,1], 'c-', label="QDA model")

	# Configure  and display the plot
	plt.axis([-10,10,-10,10])
	plt.title("Equiprobability lines - dataset %s"%dataset)
	plt.legend(numpoints=1)
	plt.gcf().savefig(
		"Report/Figures/6-equiproblines-%s.png"%dataset,
		dpi=150, bbox_inches='tight')
	plt.close(plt.gcf())
	logging.info("saved as 6-equiproblines-%s.png"%dataset)
