# Run with anaconda2-4.1.1
import csv
import numpy as np
import matplotlib.pyplot as plt
import logging
from kmean import kmean

def em_general_fit(filename, k, n_init, savefig=False):
	"""Implement the EM algorithm for a Gaussian mixture with general Sigma."""

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
	logging.info("---- EM algorithm for general covariance ----")

	# Declare the variables used for storing the data
	x = np.array([])

	# Get the data from the files
	with open(filename, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter=' ')
	     for row in spamreader:
	     	x = np.append(x,[float(row[0]), float(row[1])])
	x = x.reshape((x.shape[0]/2, 2))
	logging.info("%d elements loaded from file"%x.shape[0])

	# Initialize the variable for the EM algirithm using k-means
	logging.info("---- K-mean initialisation ----")
	centroids,ck,_ = kmean(filename, 4, 10, False, False)
	q = np.zeros([x.shape[0],k])
	mu = np.zeros([k,2])
	mu = np.copy(centroids)
	sigma = np.zeros([k,2,2])
	sigmainv = np.zeros([k,2,2])
	sigmadet = np.zeros(k)
	pi = np.zeros(k)
	for i in np.arange(0, k):
		pi[i] = np.sum(ck==i)/float(x.shape[0])
		sigma[i,:,:] = \
			np.dot((x[ck==i]-mu[i,:]).T, x[ck==i]-mu[i,:]/np.sum(ck==i))
		sigmainv[i,:,:] = np.linalg.inv(sigma[i,:,:])
		sigmadet[i] = np.linalg.det(sigma[i,:,:])
	mu_old = np.zeros([k,2])
	sigma_old = np.zeros([k,2,2])
	logging.info("---- K-mean initialisation ended ----")

	# Loop
	logging.info("---- EM steps looping ----")
	l = 0;
	while np.sum(np.square(mu_old-mu))+np.sum(np.square(sigma_old-sigma))>1e-6:
		# Record old values
		mu_old = np.copy(mu)
		sigma_old = np.copy(sigma)

		# Do the expectation step
		for i in np.arange(0,k):
			q[:,i] = pi[i]/(2*np.pi*np.sqrt(sigmadet[i])) * \
				np.exp(-0.5*np.diag(np.dot(x-mu[i,:], 
				np.dot(sigmainv[i,:,:], np.transpose((x-mu[i,:]))))))
		q = q/np.sum(q, axis=1).reshape((x.shape[0],1))

		# Do the maximization step
		pi = np.sum(q, axis=0)
		for i in np.arange(0,k):
			mu[i,:] = np.sum(x*q[:,i].reshape((x.shape[0],1)), axis=0)/pi[i]
			sigma[i,:,:] = np.zeros([2,2])
			for j in np.arange(0,x.shape[0]):
				u = (x[j,:]-mu[i,:]).reshape(2,1)
				sigma[i,:,:] += q[j,i]*np.dot(u, u.T)
			sigma[i,:,:] = sigma[i,:,:]/pi[i]
			sigmainv[i,:,:] = np.linalg.inv(sigma[i,:,:])
			sigmadet[i] = np.linalg.det(sigma[i,:,:])
		pi = pi/np.sum(pi)

		# Update the number of iterations
		l += 1

	logging.info("EM algorithm converged after %d iterations"%l)
	for i in np.arange(0,k):
		logging.info("gaussian mixture %i"%i)
		logging.info(" -pi: %f"%pi[i])
		logging.info(" -mu: (%f, %f)"%(mu[i,0],mu[i,1]))
		logging.info(" -sigma: (%f, %f)"%(sigma[i,0,0],sigma[i,0,1]))
		logging.info("         (%f, %f)"%(sigma[i,1,0],sigma[i,1,1]))


	if savefig:
		# Initialize the plot
		plt.figure(figsize=(9,9))

		# Plot the data points with their corresponding label and mu
		ck = np.argmax(q, axis=1)
		plt.plot(x[ck==0][:,0], x[ck==0][:,1],'bx', label="z = 1")
		plt.plot(x[ck==1][:,0], x[ck==1][:,1],'rx', label="z = 2")
		plt.plot(x[ck==2][:,0], x[ck==2][:,1],'mx', label="z = 3")
		plt.plot(x[ck==3][:,0], x[ck==3][:,1],'gx', label="z = 4")
		plt.plot(mu[0,0], mu[0,1],'bo', label="mu 1")
		plt.plot(mu[1,0], mu[1,1],'ro', label="mu 2")
		plt.plot(mu[2,0], mu[2,1],'mo', label="mu 3")
		plt.plot(mu[3,0], mu[3,1],'go', label="mu 4")

		# Display the ellipsoids
		theta = np.arange(0,2*np.pi*(1+1./100),2*np.pi/100)
		z_ellipse = np.zeros([theta.shape[0],k], dtype=complex)
		for i in np.arange(0,k):
			eigval, eigvect = np.linalg.eig(sigma[i,:,:])
			beta = -np.arctan(eigvect[0,1]/eigvect[0,0])
			z_ellipse[:,i] = np.sqrt(4.605)*np.cos(theta)*eigval[0] + \
				np.sqrt(4.605)*1j*np.sin(theta)*eigval[1]
			z_ellipse[:,i] = z_ellipse[:,i]*np.exp(1j*beta) + \
				mu[i,0] + 1j*mu[i,1]
		plt.plot(z_ellipse[:,0].real, z_ellipse[:,0].imag,'b-')
		plt.plot(z_ellipse[:,1].real, z_ellipse[:,1].imag,'r-')
		plt.plot(z_ellipse[:,2].real, z_ellipse[:,2].imag,'m-')
		plt.plot(z_ellipse[:,3].real, z_ellipse[:,3].imag,'g-')

		# Configure  and display the plot
		plt.axis([-15,15,-15,15])
		plt.title("EM algorithm with general covariance")
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/4-c-em-general.png", dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as 4-c-em-general.png")

	return pi, mu, sigma

def em_general_test(filename, pi, mu, sigma, savefig=False):
	"""Classify all the datapoints from filename using the gaussian mixture."""

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
	logging.info("---- EM algorithm general TEST ----")

	# Declare the variables used for storing the data
	x = np.array([])

	# Get the data from the files
	with open(filename, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter=' ')
	     for row in spamreader:
	     	x = np.append(x,[float(row[0]), float(row[1])])
	x = x.reshape((x.shape[0]/2, 2))
	logging.info("%d elements loaded from file"%x.shape[0])

	# Retrieve the dimension k
	k = pi.shape[0]

	# Compute the probabilities
	q = np.zeros([x.shape[0],k])
	sigmainv = np.zeros([k,2,2])
	sigmadet = np.zeros(k)
	for i in np.arange(0,k):
		sigmainv[i,:,:] = np.linalg.inv(sigma[i,:,:])
		sigmadet[i] = np.linalg.det(sigma[i,:,:])
		q[:,i] = pi[i]/(2*np.pi*np.sqrt(sigmadet[i])) * \
			np.exp(-0.5*np.diag(np.dot(x-mu[i,:], 
			np.dot(sigmainv[i,:,:], np.transpose((x-mu[i,:]))))))
	lhood = np.sum(np.log(np.sum(q, axis=1).reshape((x.shape[0],1))))
	logging.info("marginal log likelihood: %f"%lhood)
	q = q/np.sum(q, axis=1).reshape((x.shape[0],1))

	if savefig:
		# Initialize the plot
		plt.figure(figsize=(9,9))

		# Plot the data points with their corresponding label and mu
		ck = np.argmax(q, axis=1)
		plt.plot(x[ck==0][:,0], x[ck==0][:,1],'bx', label="z = 1")
		plt.plot(x[ck==1][:,0], x[ck==1][:,1],'rx', label="z = 2")
		plt.plot(x[ck==2][:,0], x[ck==2][:,1],'mx', label="z = 3")
		plt.plot(x[ck==3][:,0], x[ck==3][:,1],'gx', label="z = 4")

		# Configure  and display the plot
		plt.axis([-15,15,-15,15])
		plt.title("EM algorithm test with general covariance")
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/4-b-em-general-test.png", dpi=150, 
			bbox_inches='tight')
		plt.show()
		plt.close(plt.gcf())
		logging.info("figure saved as 4-b-em-general-test.png")

	return lhood

if __name__ == "__main__":
	filename_train = "classification_data_HWK2/EMGaussian.data"
	filename_test = "classification_data_HWK2/EMGaussian.test"
	savefig = False
	pi, mu, sigma = em_general_fit(filename_train, 4, 10, savefig)
	lhood_train = em_general_test(filename_train, pi, mu, sigma, savefig)
	lhood_test = em_general_test(filename_test, pi, mu, sigma, savefig)
	logging.info("---- EM general algorithm likelihood computation ----")
	logging.info("likelihood on training dataset: %f"%lhood_train)
	logging.info("likelihood on test dataset: %f"%lhood_test)
		

