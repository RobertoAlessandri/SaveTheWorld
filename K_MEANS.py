## performs support vector machine classification

import cv2
import os
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.model_selection import GridSearchCV


def load_data(class_a_path, class_b_path):

	labels = []
	generate_arrays = True  # Create arrays where we store the dataset
	for img_name in os.listdir(class_a_path):
		if not img_name.startswith('.'):
			# Read the image
			img = cv2.imread(class_a_path + img_name, 0)
			img = np.reshape(img, (1, img.shape[0], img.shape[1]))
			img_vector = np.reshape(img.ravel(), (1, -1))

			# Create arrays where we store the dataset executed only at beginning
			if generate_arrays:
				images = img
				images_vector = img_vector
				generate_arrays = False
			else:
				images = np.concatenate((images, img), axis=0)
				images_vector = np.concatenate((images_vector, img_vector), axis=0)

			labels.append(0)

	for img_name in os.listdir(class_b_path):
		if not img_name.startswith('.'):
			# Read the image
			img = cv2.imread(class_b_path + img_name, 0)
			img = np.reshape(img, (1, img.shape[0], img.shape[1]))
			img_vector = np.reshape(img.ravel(), (1, -1))
			images = np.concatenate((images, img), axis=0)
			images_vector = np.concatenate((images_vector, img_vector), axis=0)

			labels.append(1)

	return images, images_vector, labels


def train_svm():
	# Load data
	class_names = ['a', 'b']

	images, images_vector, labels = load_data(class_a_path='images/class_a/', class_b_path='images/class_b/')

	pca = PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)
	svc = SVC(kernel='rbf', class_weight='balanced')
	model = make_pipeline(pca, svc)

	with warnings.catch_warnings():
		# ignore all caught warnings
		warnings.filterwarnings("ignore")
		# execute code that will generate warnings
		xtrain, xtest, ytrain, ytest = train_test_split(images_vector, labels, random_state=42)

	param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
	grid = GridSearchCV(model, param_grid)

	print('Fit the SVM model')
	grid.fit(xtrain, ytrain)

	print(grid.best_params_)

	model = grid.best_estimator_

	# Save the model
	dump(model, 'modelSVM.joblib')

	return model


###############################################################

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import shutil
	
# We extract features from the images using InceptionV3 model
# Function to Extract features from the images
def image_feature(direc):
    model = InceptionV3(weights = 'imagenet', include_top = False)
    features = [];
    img_name = [];
    for i in tqdm(direc):
        fnam = 'cluster' + '/' + i
        img = image.load_img(fname, target_size = (224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features, img_name

# Once we extracted the features and name we store it in img_features and img_name

img_path = os.listdir('cluster')
img_features, img_name = image_feature(img_path)

# Using of KMean clustering. K = 2 because there are two classes only
#Creating Clusters
k = 2
clusters = KMeans(k, random_state = 40)
clusters.fit(img_features)

# img_name extraction and conversion to dataframe
# addiction of another column to shows which image belongs to which cluster
# saving of images in their respective clustare 

image_cluster = pd.DataFrame(img_name, columns = ['image'])
image_cluster["clusterid"] = clusters.labels_
image_cluster # 0 denotes cat and 1 denotes dog

# separate class a and b in separate folders
# Made folder to seperate images
os.mkdir('cats')
os.mkdir('dogs')

# Images will be seperated according to cluster they belong
for i in range(len(image_cluster)):
	if image_cluster['clusterid'][i] == 0:
		shutil.move(os.path.join('cluster', image_cluster['image'][i]), 'classA')
	else:
		shutil.move(os.path.join('cluster', image_cluster['image'][i]), 'classB')
# we just separated the images in different files based on the clusterid
