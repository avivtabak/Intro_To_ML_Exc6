import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np
def plot_vector_as_image(image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title('title', size=12)
	plt.show()

def get_pictures_by_name(name='Ariel Sharon'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)
	return selected_images, h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,download_if_missing=True)
	return lfw_people

def transform_data(x:np.ndarray,V:np.ndarray)->np.ndarray:
	"""

	:param x: vector in R^d (where d = h*w of the picture)
	:param V: V=[v1,...,vk] where vi (column i^th of V) in R^k and v1,...,vk
			is an orthonormal basis to the sapce of dim k that V span
	:return: x^\hat = VV^T*x
	"""
	return V @V.T @x
######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	n = X.shape[0]
	d = X.shape[1]
	#Σ :=1/n * (X^TX)
	Sigma = 1/n * (X.T@X)
	#Σ = UDU^T
	#np.linalg.svd returns eignvectors sorted in desc order
	u,s,vh = np.linalg.svd(Sigma)
	U = vh[:k,:]
	S = s[:k]

	return U, S


#QUESTION 1.B
selected_images, h, w = get_pictures_by_name('Gerhard Schroeder')
X = np.asarray([x.flatten() for x in selected_images])
(U,S)= PCA(X,10)
for i in range(10):
	plot_vector_as_image(U[i],h,w)


#QUESTION 1.C
ks = [1,5,10,30,50,100]
random_indices = np.random.choice(X.shape[0], size=5, replace=False)
random_pictures = X[random_indices, :]
for k in ks:
	(U, S) = PCA(X, k)
	random_pictures_transformed = random_pictures.copy()
	fig, axes  = plt.subplots(nrows=5, ncols=2,constrained_layout=True)
	for row in range(len(axes)):
		random_pictures_transformed[row] = transform_data(random_pictures_transformed[row],U.T)
		original = random_pictures[row]
		transformed = random_pictures_transformed[row]
		l2_norm = np.linalg.norm(original-transformed)
		axes[:,0][row].set_ylabel("l2 distance:%.2f"%l2_norm, fontsize=9)
		axes[row, 0].imshow(original.reshape((h, w)), cmap=plt.cm.gray)
		axes[row, 1].imshow(transformed.reshape((h, w)), cmap=plt.cm.gray)

	fig.suptitle('k={}'.format(k), fontsize=16)
	plt.show()



