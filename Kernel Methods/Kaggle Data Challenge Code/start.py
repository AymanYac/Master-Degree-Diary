import os
import numpy
from scipy import linalg

class svm11:
    def __init__(self, kernel):
        self.kernel = kernel
        self.X = None
        self.alpha = None
        self.bias = None
        self.class1 = None
        self.class2 = None

    def _dual_objective(self, i, j, ai, K, y, alpha):
        aj = alpha[i] + alpha[j] - ai
        ret = 2 * ai * y[i] + 2 * aj * y[j]
        ret -= ai * K[i, i] * ai + aj * K[j, j] * aj + 2 * ai * K[i, j] * aj

        n = K.shape[0]
        for k in range(n):
            if k != i and k != j:
                ret -= 2 * (ai * K[i, k] + aj * K[j, k]) * alpha[k]

        return ret

    def _take_step(self, i, j, K, y, alpha, E, epsilon=1e-4):
        s = alpha[i] + alpha[j]
        L = max(-(1 - y[i]) * self.C / 2, s - (1 + y[j]) * self.C / 2)
        H = min((1 + y[i]) * self.C / 2, s + (1 - y[j]) * self.C / 2)

        if L == H:
            return False

        eta = 2 * K[i, j] - K[i, i] - K[j, j]

        if abs(eta) > 0:
            n = K.shape[0]
            alpha_new = y[j] - y[i] + s * (K[i, j] - K[j, j])
            for k in range(n):
               if k != i and k != j:
                   alpha_new += (K[i, k] - K[j, k]) * alpha[k]
            alpha_new /= eta

            if alpha_new < L:
                alpha_new = L
            elif alpha_new > H:
                alpha_new = H
        else:
            Lobj = self._dual_objective(i, j, L, K, y, alpha)
            Hobj = self._dual_objective(i, j, H, K, y, alpha)

            if Lobj < Hobj - epsilon:
                alpha_new = H
            elif Hobj < Lobj - epsilon:
                alpha_new = L
            else:
                alpha_new = alpha[i]

        if abs(alpha_new - alpha[i]) < 1e-5 * (alpha_new + alpha[i] + 1e-5):
            return False

        alpha[i] = alpha_new
        alpha[j] = s - alpha[i]

        E[i] = numpy.dot(K[i, :], alpha) + self.bias - y[i]
        E[j] = numpy.dot(K[j, :], alpha) + self.bias - y[j]
        b1 = -E[i] + self.bias
        b2 = -E[j] + self.bias

        if alpha[i] * y[i] > epsilon and alpha[i] * y[i] < self.C - epsilon:
           bias = b1
        elif alpha[j] * y[j] > epsilon and alpha[j] * y[j] < self.C - epsilon:
           bias = b2
        else:
           bias = (b1 + b2) / 2

        E[i] += bias - self.bias
        E[j] += bias - self.bias
        self.bias = bias
        return True

    def _solve_dual(self, K, y, C, iterations=50, epsilon=1e-4):
        n = y.shape[0]
        alpha = numpy.zeros(n)
        self.bias = 0
        self.C = C

        E = numpy.zeros(n)
        for i in range(n):
            E[i] = -y[i]

        loop_all = True

        for it in range(iterations):
            num_changed = 0

            for i in range(n):
                E[i] = numpy.dot(K[i, :], alpha) + self.bias - y[i]
                if (loop_all \
                    or (alpha[i] * y[i] > 0 and alpha[i] * y[i] < self.C)) \
                    and ((E[i] * y[i] < -epsilon and alpha[i] * y[i] < self.C) \
                    or (E[i] * y[i] > epsilon and alpha[i] * y[i] > 0)):

                    non_bound = 0
                    for j in range(n):
                        if j != i and alpha[j] * y[j] > epsilon and alpha[j] * y[j] < self.C - epsilon:
                            non_bound = 1

                    done = False
                    if non_bound > 0:
                        k = -1
                        for j in range(n):
                            if j != i and alpha[j] * y[j] > epsilon and alpha[j] * y[j] < self.C - epsilon \
                                and (j == -1 or (E[i] > 0 and E[j] < E[k]) \
                                or (E[i] < 0 and E[j] > E[k])):
                                k = j
                        if self._take_step(i, k, K, y, alpha, E):
                            done = True

                    if not done:
                        start = random.randint(0, n - 1)
                        for j in range(start, n):
                            if j != i and alpha[j] * y[j] > epsilon and alpha[j] * y[j] < self.C - epsilon:
                                if self._take_step(i, j, K, y, alpha, E):
                                    done = True
                                    break

                    if not done:
                        start = random.randint(0, n - 1)
                        for j in range(start, n):
                            if j != i:
                                if self._take_step(i, j, K, y, alpha, E):
                                    done = True
                                    break

                    if done:
                        num_changed += 1

            if loop_all:
                loop_all = False
            elif num_changed > 0:
                loop_all = True

        return alpha

    def fit(self, X, y, K):
        n = X.shape[0]
        self.class1 = numpy.min(y)
        self.class2 = numpy.max(y)
        ind1 = (y == self.class1)
        ind2 = (y == self.class2)
        y2 = numpy.zeros(n)
        y2[ind1] = -1
        y2[ind2] = 1
        alpha = self._solve_dual(K, y2, C)
        ind = (numpy.abs(alpha) > 1e-9)
        n_support_vectors = numpy.sum(ind)
        self.X = X[ind, :]
        self.alpha = alpha[ind]

    def predict(self, X,K):
        n = X.shape[0]
        y = numpy.zeros(n, dtype=numpy.int32)
        pred = numpy.dot(K, self.alpha)

        for i, f in enumerate(pred):
            if f + self.bias >= 0:
                y[i] = self.class2
            else:
                y[i] = self.class1

        return y

    def _calc_accuracy(self, X, y):
        ypred = self.predict(X)
        return numpy.sum(ypred == y) * 100.0 / y.shape[0]


def Create11M(Xtr, Ytr, N):
	print 'Splitting train data to validation set'
	Binaries=[] #Matrix of binary 1vs1 classifiers
	for i in xrange(10):
            tmp = []
            for j in range(i + 1, 10):
                tmp.append(svm11(N))
            Binaries.append(tmp)

	sd = numpy.random.rand(Xtr.shape[0]) < 0.8
	Xtrain = Xtr[sd,:]
	Xtest = Xtr[~sd,:]
	Ytrain = Ytr[sd]
	Ytest = Ytr[~sd]
	K=N[sd,:]
	K=K[:,sd]
	class_indct = []
	for i in range(10):
		s = (Ytrain == i)
		class_indct.append(s)
	for i in xrange(10):
		for j in xrange(i+1,10):
			if i>0:
				CURSOR_UP_ONE = '\x1b[1A'
				ERASE_LINE = '\x1b[2K'
				#print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)
			print 'Progress '+str(100.0*(i+1)/11)+' %'
			s = numpy.logical_or(class_indct[i],class_indct[j])
			local = K[s,:]
			local = local[:,s]
			Binaries[i][j - i - 1].fit(Xtrain[s, :], Ytrain[s], K=local)
	return Binaries,Xtest,Ytest


def write_output(Y, filename):
    assert(Y.shape[0] == n_test)
    f = open(filename, 'w')
    f.write('Id,Prediction\n')

    for i, y in enumerate(Y):
        f.write("{0:d},{1:d}\n".format(i + 1, y))

    print("Ytest output to : %s" % filename)

def noyau(kernel,kparam,features,test):
	if test == None:
		test = features
	if kernel == 'lin':
		return numpy.dot(features, test.T)
	elif kernel == 'rbf':
		N = numpy.zeros((features.shape[0], test.shape[0]))
		for i in xrange(test.shape[0]):
			if i>0:
				CURSOR_UP_ONE = '\x1b[1A'
				ERASE_LINE = '\x1b[2K'
				#print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)
			print 'Progress '+str(100.0*(i+1)/features.shape[0])+' %'
			N[:, i] = numpy.linalg.norm(features - test[i, :], axis=1) ** 2
		N = N / (2 * (kparam ** 2))
		return numpy.exp(-N)
	elif kernel == 'poly':
		return (numpy.dot(features, test.T) +1) ** kparam
	elif kernel =='lplc' :
		N = numpy.zeros((features.shape[0], test.shape[0]))
		for i in xrange(test.shape[0]):
			if i>0:
				CURSOR_UP_ONE = '\x1b[1A'
				ERASE_LINE = '\x1b[2K'
				print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)
			print 'Progress '+str(100.0*(i+1)/features.shape[0])+' %'
			N[:, i] = numpy.sum(numpy.abs(features - test[i, :]), axis=1)
		N = N / (kparam ** 2)
		return numpy.exp(-N)


def kpca(features,test,kernel,kparam):
	dim0 = features.shape[0]
	N = noyau(kernel,kparam,features,None)
	print 'Phase I > Training kPCA'
	U = numpy.ones((dim0, dim0)) / dim0
	TRC = numpy.mean(N, axis=1) - numpy.mean(N)
	numpy.mean(N, axis=1) - numpy.mean(N)
	N = numpy.dot(numpy.dot(numpy.eye(dim0) - U, N), numpy.eye(dim0) - U)
	print 'Decomposition propre'
	eigenValues, eigenVectors = linalg.eigh(N)
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	idx = eigenValues > 0
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:, idx]
	alpha = eigenVectors
	for k, m in enumerate(eigenValues):
		alpha[:, k] /= numpy.sqrt(m)
	alpha = alpha[:, :500] #Take only 500 PCs
	print 'Phase II > Fitting kPCA'
	Ntr = noyau(kernel,kparam,features,features)
	Nte = noyau(kernel,kparam,test,features)
	Ntr = Ntr - numpy.mean(Ntr, axis=1)[:, numpy.newaxis] - TRC
	Nte = Nte - numpy.mean(Nte, axis=1)[:, numpy.newaxis] - TRC
	return numpy.dot(Ntr, alpha),numpy.dot(Nte, alpha)




def Color_Grads(image):
	hist = numpy.zeros((4, 4, 9)) #Patch size. 9 non oriented intervals
	for i in xrange(32):
		for j in xrange(32):
			gx = gy = 0
			gx = gx+image[i+1,j] if i+1 < 32 else gx
			gx = gx-image[i-1,j] if i > 0 else gx
			gy = gy+image[i,j+1] if j+1 < 32 else gy
			gy = gy-image[i,j-1] if j-1 > 0 else gy

			if(gx == 0 and gy ==0): #Marginals, hist is zeroed at init
				continue
			gamma = numpy.sqrt(gx**2 + gy**2)
			theta = numpy.arctan(1.0*gy/gx) if gx>0 else 0.5*numpy.pi
			hist_interval = int(numpy.floor((theta+ 0.5*numpy.pi) / (numpy.pi / 9.0)))
			Flag=True
			if(hist_interval >= 9 ):
				first_int = 0
				second_int = 8 # last_interval
				Flag=False
			elif(hist_interval == 0):
				second_int = 1 if theta > 0.5*(9/numpy.pi - numpy.pi) else 8
				first_int = hist_interval
			elif(hist_interval ==8):
				second_int = 7 if theta < 8.5*(9/numpy.pi) - numpy.pi/2 else 0
				first_int = hist_interval
			else:
				first_int=hist_interval
				second_int = first_int-1 if theta<(hist_interval+0.5)*(9/numpy.pi) - numpy.pi/2 else first_int +1
			
			absolute_hist_x = (theta + numpy.pi / 2) / (numpy.pi / 9.0) if Flag else 0
			if absolute_hist_x < hist_interval + 0.5:
				ratio_to_second = (absolute_hist_x - hist_interval) + 0.5
			else:
				ratio_to_second = (hist_interval - absolute_hist_x +1 ) + 0.5
            
			#print first_int
			#print second_int
			hist[i / 8, j / 8, first_int] = ratio_to_second * gamma + hist[i / 8, j / 8, first_int]
			hist[i / 8, j / 8, second_int] = (1 - ratio_to_second) * gamma + hist[i / 8, j / 8, second_int]

		normalized_HOG = numpy.zeros((3, 3, 9 * 4))
		for i in xrange(3):
			for j in xrange(3):
				tmp = hist[i:i + 2, j:j + 2, :].flatten()
				if(numpy.linalg.norm(tmp)>0):
					tmp = tmp / numpy.linalg.norm(tmp)
				normalized_HOG[i, j, :] = tmp


		return normalized_HOG.flatten()


def Histogram_Oriented_Grads(X,matrix):
	if matrix:
		RGB_HOG = []
		for i in xrange(X.shape[0]):
			CURSOR_UP_ONE = '\x1b[1A'
			ERASE_LINE = '\x1b[2K'
			print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)
			print 'Extracting HOG for image '+str(i+1)+' of '+str(X.shape[0])
			RGB_HOG.append(Histogram_Oriented_Grads(X[i,:,:,:],matrix=False))
		return numpy.array(RGB_HOG)
	else:
		COL_HOG = []
		for i in xrange(X.shape[2]):#colors on the last dim : shape of X[i,:::] is 32,32,3
			COL_HOG.append(Color_Grads(X[:,:,i]))
		return numpy.array(COL_HOG).flatten()


def format(images_matrix):
	nrows = images_matrix.shape[0]
	images_matrix = numpy.reshape(images_matrix, (nrows, 3, 32, 32))
	images_matrix = numpy.swapaxes(images_matrix, 1, 2)
	images_matrix = numpy.swapaxes(images_matrix, 2, 3)
	return images_matrix
def load_images():
	return numpy.load('Xtrain.npy'),numpy.load('Xtest.npy'),numpy.load('Ytrain.npy')
	Xtr = numpy.genfromtxt('Xtr.csv', delimiter=',',usecols=range(3072))
	Xte = numpy.genfromtxt('Xte.csv', delimiter=',',usecols=range(3072))
	Ytr = numpy.genfromtxt ('Ytr.csv',delimiter=",",skip_header=1)
	Ytr=Ytr[:,1]
	Ytr=numpy.array([Ytr])
	return format(Xtr),format(Xte),Ytr



if __name__ == '__main__':
	print '32x32 RGB loading'
	Xtr,Xte,Ytr = load_images()
	print 'Appliying HOG to training Data'
	#Xtr = Histogram_Oriented_Grads(Xtr,matrix=True)
	Xtr = numpy.load('Xtrain_hog.npy')
	print 'Shape of the HOG train matrix : ' + str(Xtr.shape)
	print 'Appliying HOG to testing Data'
	#Xte = Histogram_Oriented_Grads(Xte,matrix=True)
	Xte = numpy.load('Xtest_hog.npy')
	print 'Shape of the HOG test matrix : ' + str(Xte.shape)
	print 'Fitting KernelPCA to HOG matrices'
	Xtr,Xte = kpca(Xtr,Xte,'poly',2) #3rd arg can be in [rbf,lin,poly,lplc], 4th arg is kernel parametre(sigma,poly deg)
	print 'Shape of the reduced train matrix : ' + str(Xtr.shape)
	print 'Shape of the reduced test matrix : ' + str(Xte.shape)
	print 'Fitting SVM on reduced train matrix'
	print 'Phase I'
	N = noyau('rbf',10,Xtr,Xtr)#1st arg can be in [rbf,lin,poly,lplc], 2nd arg is kernel parametre(sigma,poly deg)
	Binaries,Xtest,Ytest = Create11M(Xtr, Ytr,N)
	scores = numpy.zeros((Xtr.shape[0], 10))
	for i in xrange(10):
		for j in xrange(i + 1, 10):
			if i>0:
				CURSOR_UP_ONE = '\x1b[1A'
				ERASE_LINE = '\x1b[2K'
				print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)
				print 'Progress '+str(100.0*(i+1)/11)+' %'
				y = Binaries[i][j - i - 1].predict(Xtest,N)
				for k, pred in enumerate(y):
					scores[k][pred] += 1
	ypred = numpy.argmax(scores, axis=1)
	print 'Accuracy on validation data is'+str(numpy.sum(ypred == y) * 100.0 / y.shape[0])
	print("Predicting on test data")
	#Ytest = model.predict(Xtest)
	#write_output(Ytest, 'results/Yte_' + output_suffix +str(sigma)+ '.csv')