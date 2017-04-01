# Run with anaconda2-4.1.1
import csv
import numpy as np
import matplotlib.pyplot as plt
import logging

def kmean(filename, k, n_init, savefig_cluster=False, savefig_distorsion=False):
	"""Return the k-mean clustering"""

	# Setup the logger
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
	logging.info("---- K-mean clustering ----")

	# Declare the variables used for storing the data
	x = np.array([])

	# Get the data from the files
	with open(filename, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter=' ')
	     for row in spamreader:
	     	x = np.append(x,[float(row[0]), float(row[1])])
	x = x.reshape((x.shape[0]/2, 2))
	logging.info("%d elements loaded from file"%x.shape[0])

	# Try n_init different initializations
	j = 0
	distorsion = np.zeros(n_init)
	while j<n_init:
		# Initialize the k-mean centers
		centroids = np.random.rand(4,2)
		centroids[:,0] = centroids[:,0]*np.min(x[:,0]) + \
			(1-centroids[:,0])*np.max(x[:,0])
		centroids[:,1] = centroids[:,1]*np.min(x[:,1]) + \
			(1-centroids[:,1])*np.max(x[:,1])
		valid_centroids = True

		# Loop until we reach a local minimum
		ck = np.zeros(x.shape[0])
		old_centroids = np.zeros([k,2])
		while not np.array_equal(old_centroids, centroids):
			# Save the old centroids before update
			old_centroids = np.copy(centroids)

			# Assign the point to their nearest centroids
			dist_tot = 0
			for i in np.arange(x.shape[0]):
				dist = np.diag(np.dot(centroids-x[i,:], 
					np.transpose(centroids-x[i,:])))
				ck[i] = np.argmin(dist)
				dist_tot += dist[int(ck[i])]

			# Check if at least one point is assigned to the centroid
			if not np.array_equal(np.unique(ck), np.arange(0,4)):
				valid_centroids = False
				break

			# Recompute the centroids
			centroids[0,:] = np.mean(x[ck==0], axis=0)
			centroids[1,:] = np.mean(x[ck==1], axis=0)
			centroids[2,:] = np.mean(x[ck==2], axis=0)
			centroids[3,:] = np.mean(x[ck==3], axis=0)

		# If we have a valid centroid, then compare it to the best
		if valid_centroids:
			if j == 0:
				# If it's the first valid centroid then it's the best
				best_centroids = np.copy(centroids)
				best_dist = dist_tot
				best_ck = np.copy(ck)
				logging.info("best centroids updated at "
					"iteration %d, dist=%d"%(j+1,best_dist))
			elif dist_tot < best_dist:
				# If we have a better centroid, record it
				best_centroids = np.copy(centroids)
				best_dist = dist_tot
				best_ck = np.copy(ck)
				logging.info("best centroids updated at "
					"iteration %d, dist=%d"%(j+1,best_dist))

			# Record the distorsion and update iteration
			distorsion[j] = dist_tot
			j += 1

			# Save the figure representing the data and the clusters
			if savefig_cluster:
				# Initialize the plot
				plt.figure(figsize=(9,9))

				# Plot the data points with their correspond label
				plt.plot(x[ck==0][:,0], x[ck==0][:,1],'bx', label="ck = 1")
				plt.plot(x[ck==1][:,0], x[ck==1][:,1],'rx', label="ck = 2")
				plt.plot(x[ck==2][:,0], x[ck==2][:,1],'mx', label="ck = 3")
				plt.plot(x[ck==3][:,0], x[ck==3][:,1],'gx', label="ck = 4")
				plt.plot(centroids[0,0], centroids[0,1],'bo', 
					label="centroid 1")
				plt.plot(centroids[1,0], centroids[1,1],'ro', 
					label="centroid 2")
				plt.plot(centroids[2,0], centroids[2,1],'mo', 
					label="centroid 3")
				plt.plot(centroids[3,0], centroids[3,1],'go', 
					label="centroid 4")

				# Configure  and display the plot
				plt.axis([-15,15,-15,15])
				plt.title("K-mean with %i clusters, distorsion %0.2f"
					%(k,dist_tot))
				plt.legend(numpoints=1)
				plt.gcf().savefig("Report/Figures/4-a-kmean-%i-new.png"%j, 
					dpi=150, bbox_inches='tight')
				plt.close(plt.gcf())
				logging.info("figure saved as 4-a-kmean-%i-new.png"%j)


	# Save the figure representing the data and the clusters
	if savefig_distorsion:
		# Initialize the plot
		plt.figure(figsize=(9,9))

		# Plot the data points with their correspond label
		plt.plot(np.arange(1,n_init+1), distorsion,'k-o', label="distorsion")

		# Configure  and display the plot
		np
		plt.axis([1,n_init,np.floor(np.min(distorsion)/500)*500,
			(np.floor(np.max(distorsion)/500)+1)*500])
		plt.title("K-mean distorsion for different initialization")
		plt.legend(numpoints=1)
		plt.gcf().savefig("Report/Figures/4-a-kmean-distorsion-new.png", 
			dpi=150, bbox_inches='tight')
		plt.close(plt.gcf())
		logging.info("figure saved as 4-a-kmean-distorsion-new.png")

	# Display the final centroids
	logging.info("best centroid 1: (%f, %f)"%(best_centroids[0,0], 
		best_centroids[0,1]))
	logging.info("best centroid 2: (%f, %f)"%(best_centroids[1,0], 
		best_centroids[1,1]))
	logging.info("best centroid 3: (%f, %f)"%(best_centroids[2,0], 
		best_centroids[2,1]))
	logging.info("best centroid 4: (%f, %f)"%(best_centroids[3,0], 
		best_centroids[3,1]))
	
	# Return the parameters
	return best_centroids, best_ck, best_dist

if __name__ == "__main__":
	filename = "classification_data_HWK2/EMGaussian.data"
	kmean(filename, 4, 100, False, False)
		

