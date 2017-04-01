# Run with anaconda2-4.1.1
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
import logging
from scipy.stats import multivariate_normal
from em_general import em_general_fit

def small_sum_log(x_log):
	"""Take small numbers as log and return their sum as log."""
	x_log_max = np.max(x_log)
	return x_log_max+np.log(np.sum(np.exp(x_log-x_log_max)))

def get_data(filename):
	"""Retrieve the data from the file."""
	x = np.array([])
	with open(filename, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter=' ')
	     for row in spamreader:
	     	x = np.append(x,[float(row[0]), float(row[1])])
	x = x.reshape((x.shape[0]/2, 2))
	T = x.shape[0]
	return x,T

def likelihood(u, mu, sigma, A, pi, gamma_log, xi_log):
	"""Compute the log likelihood of a HMM sequence."""
	T = u.shape[0]
	lhood = 0
	temp = np.log(pi)*np.exp(gamma_log[0,:])
	lhood += np.sum(temp[~np.isnan(temp)])
	for t in np.arange(0,T-1):
		for i in np.arange(0,4):
			for j in np.arange(0,4):
				lhood += np.log(A[i,j])*np.exp(xi_log[t,i,j])
	for t in np.arange(0,T):
		for q in np.arange(0,4):
			lhood += np.exp(gamma_log[t,q])*np.log(
				multivariate_normal.pdf(u[t,:], mu[q,:], sigma[q,:]))

	return lhood

def hmm_recursions(filename, A, pi, mu, sigma, savefig):
	"""Compute the alpha and beta recursions and store their log value"""

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

	# Get the data from the files
	u, T = get_data(filename)

	# Initialize the variables for the alpha and beta recursions
	beta_log = np.zeros([T,4])	
	multivar_norm = np.zeros(4)
	for i in np.arange(0,4):
		multivar_norm[i] = multivariate_normal.pdf(u[0,:], mu[i,:], sigma[i,:])
	alpha_log = np.zeros([T,4])
	alpha_log[0,:] = np.log(pi[:]) + np.log(multivar_norm[:])

	# Loop through all points and compute alpha and beta recursions
	for t in np.arange(0,T-1):
		for q in np.arange(0,4):
			# Compute alpha
			alpha_log[t+1,q] = small_sum_log(np.log(A[:,q])+alpha_log[t,:]) + \
				np.log(multivariate_normal.pdf(u[t+1,:], mu[q,:], sigma[q,:]))

			# Compute beta
			for k in np.arange(0,4):
				multivar_norm[k] = multivariate_normal.pdf(
					u[t+1,:], mu[k,:], sigma[k,:])
			beta_log[T-(t+2),q] = small_sum_log(np.log(multivar_norm) + 
				np.log(A[q,:]) + beta_log[T-(t+1),:])

	# Loop through all points and compute gamma = p(q_t|u_1,...,u_T)
	gamma_log = alpha_log + beta_log
	for t in np.arange(0,T):
		gamma_log[t,:] = gamma_log[t,:] - small_sum_log(gamma_log[t,:])

	# Loop through all points and compute xi = p(q_t,q_{t+1}|u_1,...,u_T)
	xi_log = np.zeros([T-1,4,4])
	for t in np.arange(0,T-1):
		for i in np.arange(0,4):
			for j in np.arange(0,4):
				xi_log[t,i,j] = alpha_log[t,i] + gamma_log[t+1,j] + \
					np.log(A[i,j]) - alpha_log[t+1,j] + \
					np.log(multivariate_normal.pdf(u[t+1,:],mu[j,:],sigma[j,:]))

	if savefig:
		# Initialize the plot
		f, axarr = plt.subplots(4, figsize=(14,7))
		axarr[0].set_title("Probability gamma - %s"%filename[-4::])

		# Plot the data points with their corresponding label and mu
		w = 1/1.5
		x = np.arange(0,100)
		axarr[0].bar(x, np.exp(gamma_log[0:100,0]), w, color='b', label="q = 1")
		axarr[1].bar(x, np.exp(gamma_log[0:100,1]), w, color='r', label="q = 2")
		axarr[2].bar(x, np.exp(gamma_log[0:100,2]), w, color='m', label="q = 3")
		axarr[3].bar(x, np.exp(gamma_log[0:100,3]), w, color='c', label="q = 4")

		# Configure  and display the plot
		axarr[0].legend(numpoints=1)
		axarr[1].legend(numpoints=1)
		axarr[2].legend(numpoints=1)
		axarr[3].legend(numpoints=1)
		majorLocator = MultipleLocator(5)
		minorLocator = MultipleLocator(1)
		axarr[0].xaxis.set_major_locator(majorLocator)
		axarr[1].xaxis.set_major_locator(majorLocator)
		axarr[2].xaxis.set_major_locator(majorLocator)
		axarr[3].xaxis.set_major_locator(majorLocator)
		axarr[0].xaxis.set_minor_locator(minorLocator)
		axarr[1].xaxis.set_minor_locator(minorLocator)
		axarr[2].xaxis.set_minor_locator(minorLocator)
		axarr[3].xaxis.set_minor_locator(minorLocator)
		plt.axis([0,100,0,1])
		plt.gcf().savefig("Report/Figures/q-prob-%s.png"%filename[-4::], 
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as q-prob-%s.png"%filename[-4::])

	# Return the alpha and beta recursions as log plus the probability
	return alpha_log, beta_log, gamma_log, xi_log

def hmm_fit(filename_train, filename_test, savefig):
	"""Compute the best parameters fitting the HMM model."""

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
	logging.info("---- HMM Model Fitting ----")

	# Get the data from the files
	u, T = get_data(filename_train)
	u_test, _ = get_data(filename_test)

	# Define the initial HMM parameters
	A = np.ones([4,4])/6 + np.eye(4)*2/6
	pi = np.ones(4)/4
	_, mu, sigma = em_general_fit(filename_train, 4, 10, False)

	# Loop the EM algorithm
	logging.info("---- EM steps looping ----")
	A_old = np.zeros(A.shape)
	pi_old = np.zeros(pi.shape)
	mu_old = np.zeros(mu.shape)
	sigma_old = np.zeros(sigma.shape)
	l = 0;
	lhood_train = np.array([])
	lhood_test = np.array([])
	s = 1
	savefig2 = savefig
	while s>1e-6:
		s = np.sum(np.square(mu_old-mu)) + \
			np.sum(np.square(sigma_old-sigma)) + \
			np.sum(np.square(pi_old-pi)) + \
			np.sum(np.square(A_old-A))
		logging.info("iteration %d: parameter difference %0.6f"%(l,s))

		# Record old values
		A_old = np.copy(A)
		pi_old = np.copy(pi)
		mu_old = np.copy(mu)
		sigma_old = np.copy(sigma)

		# Compute the recursions
		if l > 0:
			savefig2 = False;
		alpha_log, beta_log, gamma_log, xi_log = \
			hmm_recursions(filename_train, A, pi, mu, sigma, savefig2)
		_, _, gamma_log_test, xi_log_test = \
			hmm_recursions(filename_test, A, pi, mu, sigma, False)

		# Compute the log likelihood
		lhood_train = np.append(lhood_train, \
			likelihood(u, mu, sigma, A, pi, gamma_log, xi_log))
		lhood_test = np.append(lhood_test, \
			likelihood(u_test, mu, sigma, A, pi, gamma_log_test, xi_log_test))

		# Update the initial hidden state distribution pi
		pi = np.exp(gamma_log[0,:])

		# Update the transition matrix A
		A = np.zeros([4,4])
		for i in np.arange(0,4):
			for j in np.arange(0,4):
				A[i,j] = np.exp(small_sum_log(xi_log[:,i,j]) - \
					small_sum_log(gamma_log[0:T-1,i]))
		# Take care of small numerical errors
		A = np.multiply(A,np.reciprocal(np.sum(A,1).reshape(4,1)))

		# Update the gaussian means
		for q in np.arange(0,4):
			for t in np.arange(1,T):
				a = np.multiply(u,np.exp(gamma_log[:,q]).reshape([T,1]))
				mu[q,:] = np.sum(a,0)/np.sum(np.exp(gamma_log[:,q]))

		# Update the gaussian covariance matrices
		for k in np.arange(0,4):
			# sigma[i,:,:] = np.zeros([2,2])
			sigma[k,:] = np.zeros([2,2])
			for t in np.arange(0,T):
				v = (u[t,:]-mu[k,:]).reshape(2,1)
				sigma[k,:] += np.dot(v, v.T)*np.exp(gamma_log[t,k])
			sigma[k,:] = sigma[k,:]/np.sum(np.exp(gamma_log[:,k]))

		# Update the number of iterations
		l += 1

	logging.info("EM algorithm converged after %d iterations"%l)
	logging.info("markov chain parameters")
	logging.info(" -pi: (%f, %f, %f, %f)"%(pi[0],pi[1],pi[2],pi[3]))
	logging.info(" -A:  (%f, %f, %f, %f)"%(A[0,0],A[0,1],A[0,2],A[0,3]))
	logging.info("      (%f, %f, %f, %f)"%(A[1,0],A[1,1],A[1,2],A[1,3]))
	logging.info("      (%f, %f, %f, %f)"%(A[2,0],A[2,1],A[2,2],A[2,3]))
	logging.info("      (%f, %f, %f, %f)"%(A[3,0],A[3,1],A[3,2],A[3,3]))
	for i in np.arange(0,4):
		logging.info("gaussian mixture %i"%i)
		logging.info(" -mu:    (%f, %f)"%(mu[i,0],mu[i,1]))
		logging.info(" -sigma: (%f, %f)"%(sigma[i,0,0],sigma[i,0,1]))
		logging.info("         (%f, %f)"%(sigma[i,1,0],sigma[i,1,1]))

	if savefig:
		plt.figure(figsize=(9,9))
		plt.plot(lhood_train, label="train likelihood")
		plt.plot(lhood_test, label="test likelihood")
		plt.xlabel("iteration number")
		plt.ylabel("log-likelihood")
		plt.title("log-likelihood versus iteration number")
		plt.legend()
		plt.gcf().savefig("Report/Figures/lhood-iter.png", 
			dpi=150, bbox_inches='tight')
		plt.show
		plt.close(plt.gcf())
		logging.info("figure saved as lhood-iter.png")

	logging.info("---- Likelihood computation ----")
	logging.info("likelihood on training dataset: %f"%lhood_train[-1])
	logging.info("likelihood on test dataset: %f"%lhood_test[-1])

	return pi, A, mu, sigma

def hmm_viterbi(filename, pi, A, mu, sigma, savefig):
	"""Implement the viterbi decoding for the HMM model."""

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
	logging.info("---- Viterbi Decoding ----")

	# Get the data from the files
	u, T = get_data(filename)

	# Initialize the variables
	delta_log = -np.ones([4,T])*np.inf
	state = -np.ones([4,T])

	# Initialize the viterbi algorithm
	for q in np.arange(0,4):
		for k in np.arange(0,4):
			p_log = np.log(pi[k]*A[k,q]*multivariate_normal.pdf(u[0,:],
				mu[k,:], sigma[k,:]))
			if p_log > delta_log[q,0]:
				delta_log[q,0] = p_log
				state[q,0] = k

	# Do the viterbi decoding for all intermediary steps
	for t in np.arange(1,T-1):
		for q in np.arange(0,4):
			for k in np.arange(0,4):
				p_log = np.log(A[k,q]) + delta_log[k,t-1] + np.log(
					multivariate_normal.pdf(u[t,:], mu[k,:], sigma[k,:]))
				if p_log > delta_log[q,t]:
					delta_log[q,t] = p_log
					state[q,t] = k

	# Make the final pass of the viterbi decoding
	for q in np.arange(0,4):
		delta_log[q,T-1] = delta_log[k,T-2] + \
			np.log(multivariate_normal.pdf(u[T-1,:], mu[q,:], sigma[q,:]))
		state[q,T-1] = q

	# Compute the viterbi decoding
	q_best = np.argmax(delta_log[:,T-1])
	decoding = state[q_best,:]

	if savefig:
		# Plot the viterbi decoding
		plt.figure(figsize=(9,9))
		plt.plot(u[decoding==0][:,0], u[decoding==0][:,1],'bx', label="q = 1")
		plt.plot(u[decoding==1][:,0], u[decoding==1][:,1],'rx', label="q = 2")
		plt.plot(u[decoding==2][:,0], u[decoding==2][:,1],'mx', label="q = 3")
		plt.plot(u[decoding==3][:,0], u[decoding==3][:,1],'gx', label="q = 4")
		plt.plot(mu[:,0], mu[:,1],'ko', label="cluster centers")

		# Configure  and display the plot
		plt.axis([-15,15,-15,15])
		plt.title("Viterbi decoding - %s"%filename[-4::])
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/viterbi-%s.png"%filename[-4::], 
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as viterbi-%s.png"%filename[-4::])

	return decoding

def hmm_marginal_vs_viterbi(filename, pi, A, mu, sigma, savefig):
	"""Compare the marginal likelihood estimation versus Viterbi decoding."""

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

	# Retrieve the marginal probability
	_,_,gamma_log,_ = hmm_recursions(filename, A, pi, mu, sigma, False)

	# Plot the marginal state probability 
	if savefig:
		# Initialize the plot
		f, axarr = plt.subplots(4, figsize=(14,7))
		axarr[0].set_title("Probability gamma - %s"%filename[-4::])

		# Plot the data points with their corresponding label and mu
		w = 1/1.5
		x = np.arange(0,100)
		axarr[0].bar(x, np.exp(gamma_log[0:100,0]), w, color='b', label="q = 1")
		axarr[1].bar(x, np.exp(gamma_log[0:100,1]), w, color='r', label="q = 2")
		axarr[2].bar(x, np.exp(gamma_log[0:100,2]), w, color='m', label="q = 3")
		axarr[3].bar(x, np.exp(gamma_log[0:100,3]), w, color='c', label="q = 4")

		# Configure  and display the plot
		axarr[0].legend(numpoints=1)
		axarr[1].legend(numpoints=1)
		axarr[2].legend(numpoints=1)
		axarr[3].legend(numpoints=1)
		majorLocator = MultipleLocator(5)
		minorLocator = MultipleLocator(1)
		axarr[0].xaxis.set_major_locator(majorLocator)
		axarr[1].xaxis.set_major_locator(majorLocator)
		axarr[2].xaxis.set_major_locator(majorLocator)
		axarr[3].xaxis.set_major_locator(majorLocator)
		axarr[0].xaxis.set_minor_locator(minorLocator)
		axarr[1].xaxis.set_minor_locator(minorLocator)
		axarr[2].xaxis.set_minor_locator(minorLocator)
		axarr[3].xaxis.set_minor_locator(minorLocator)
		plt.axis([0,100,0,1])
		plt.gcf().savefig("Report/Figures/q-prob-%s.png"%filename[-4::], 
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as q-prob-%s.png"%filename[-4::])

	# Compute the marginal and viterbi
	decoding = hmm_viterbi(filename, pi, A, mu, sigma, False)
	decoding = decoding[0:100]
	state = np.argmax(gamma_log, axis=1)
	state = state[0:100]
	u,_ = get_data(filename)

	if savefig:
		# Plot the viterbi decoding
		plt.figure(figsize=(9,9))
		plt.plot(u[decoding==0][:,0], u[decoding==0][:,1],'bx', label="q = 1")
		plt.plot(u[decoding==1][:,0], u[decoding==1][:,1],'rx', label="q = 2")
		plt.plot(u[decoding==2][:,0], u[decoding==2][:,1],'mx', label="q = 3")
		plt.plot(u[decoding==3][:,0], u[decoding==3][:,1],'gx', label="q = 4")
		plt.plot(mu[:,0], mu[:,1],'ko', label="cluster centers")
		plt.axis([-15,15,-15,15])
		plt.title("Final Viterbi decoding - %s"%filename[-4::])
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/final-viterbi-%s.png"%filename[-4::], 
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as final-viterbi-%s.png"%filename[-4::])

		# Plot the marginal likelihood maximization
		plt.figure(figsize=(9,9))
		plt.plot(u[state==0][:,0], u[state==0][:,1],'bx', label="q = 1")
		plt.plot(u[state==1][:,0], u[state==1][:,1],'rx', label="q = 2")
		plt.plot(u[state==2][:,0], u[state==2][:,1],'mx', label="q = 3")
		plt.plot(u[state==3][:,0], u[state==3][:,1],'gx', label="q = 4")
		plt.plot(mu[:,0], mu[:,1],'ko', label="cluster centers")
		plt.axis([-15,15,-15,15])
		plt.title("Final marginal likelihood maximization - %s"%filename[-4::])
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/final-marginal-likelihood-%s.png"%
			filename[-4::], dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as final-marginal-likelihood-%s.png"%
			filename[-4::])

		# Plot the viterbi decoding
		plt.figure(figsize=(14,5))
		plt.title("Viterbi decoding - %s"%filename[-4::])
		plt.xlabel("time")
		plt.ylabel("hidden variable state")
		w = 1/1.5
		x = np.arange(0,100)
		plt.bar(x[decoding==0], 1+decoding[decoding==0], w, 
			color='b', label="q = 1")
		plt.bar(x[decoding==1], 1+decoding[decoding==1], w, 
			color='r', label="q = 2")
		plt.bar(x[decoding==2], 1+decoding[decoding==2], w, 
			color='m', label="q = 3")
		plt.bar(x[decoding==3], 1+decoding[decoding==3], w, 
			color='c', label="q = 4")
		plt.axis([0,100,0,5])
		plt.axes().xaxis.set_major_locator(majorLocator)
		plt.axes().xaxis.set_minor_locator(minorLocator)
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/viterbi-state-%s.png"%filename[-4::], 
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as viterbi-state-%s.png"%filename[-4::])

		# Plot the marginal likelihood maximization
		plt.figure(figsize=(14,5))
		plt.title("Marginal likelihood estimation - %s"%filename[-4::])
		plt.xlabel("time")
		plt.ylabel("hidden variable state")
		w = 1/1.5
		x = np.arange(0,100)
		plt.bar(x[state==0], 1+state[state==0], w, color='b', label="q = 1")
		plt.bar(x[state==1], 1+state[state==1], w, color='r', label="q = 2")
		plt.bar(x[state==2], 1+state[state==2], w, color='m', label="q = 3")
		plt.bar(x[state==3], 1+state[state==3], w, color='c', label="q = 4")
		plt.axis([0,100,0,5])
		plt.axes().xaxis.set_major_locator(majorLocator)
		plt.axes().xaxis.set_minor_locator(minorLocator)
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/marginal-likelihood-state-%s.png"%
			filename[-4::], dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as marginal-likelihood-%s.png"%filename[-4::])

	return

if __name__ == "__main__":
	filename_train = "classification_data_HWK3/EMGaussian.data"
	filename_test = "classification_data_HWK3/EMGaussian.test"
	savefig = True

	# Estimate the HMM paramaters
	pi, A, mu, sigma = hmm_fit(filename_train, filename_test, savefig)

	# Decode the sequence on the training dataset
	hmm_viterbi(filename_train, pi, A, mu, sigma, savefig)

	# Compare the marginal and viterbi estimation
	hmm_marginal_vs_viterbi(filename_test, pi, A, mu, sigma, savefig)



