## performs support vector machine classification

from turtle import shape
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

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import shutil

def image_feature_a(direc):
    model = InceptionV3(weights = 'imagenet', include_top = False)
    features = [];
    img_name = [];
    for i in tqdm(direc):
        fnam = 'C:/Users/rien27/Documents/GitHub/SaveTheWorld/images/class_a' + '/' + i
        img = image.load_img(fnam, target_size = (700, 393)) #224 x 224 prima
        x = img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features, img_name

def image_feature_b(direc):
    model = InceptionV3(weights = 'imagenet', include_top = False)
    features = [];
    img_name = [];
    for i in tqdm(direc):
        fnam = 'C:/Users/rien27/Documents/GitHub/SaveTheWorld/images/class_b' + '/' + i
        img = image.load_img(fnam, target_size = (700, 393))
        x = img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features, img_name

def image_feature(direc):
    model = InceptionV3(weights = 'imagenet', include_top = False)
    features = [];
    img_name = [];
    class_a_path = os.listdir(direc + 'class_a/')
    class_b_path = os.listdir(direc + 'class_b/')
    print('Extracting features from class A')
    for i in tqdm(class_a_path):
        if (i != 27):
            if(i != '.DS_Store'):
                fnam_a = direc + 'class_a/' + i # + /, before i
            else: print
        else: print(i)    
        if (fnam_a != '.DS_Store'):
            img_a = image.load_img(fnam_a, target_size = (224, 224))
            x = img_to_array(img_a)
            x = np.expand_dims(x, axis = 0)
            x = preprocess_input(x)
            feat_a = model.predict(x)
            feat_a = feat_a.flatten()
            features.append(feat_a)
            img_name.append(i)
        else: 
            print('fnam_a = ', fnam_a)   

    print ('Extracting features from class B')
    for j in tqdm(class_b_path):
        if(j != 27):
            if(j != '.DS_Store'):
                fnam_b = direc + 'class_b/' + j
            else: print(j)    
        else: print(j)    
        if (fnam_b != '.DS_Store'):
            img_b = image.load_img(fnam_b, target_size = (224, 224)) # 700 x 393
            y = img_to_array(img_b)
            y = np.expand_dims(y, axis = 0)
            y = preprocess_input(y)
            feat_b = model.predict(y)
            feat_b = feat_b.flatten()
            features.append(feat_b)
            img_name.append(j)
        else:
            print('fnam_b = ', fnam_b)

    return features, img_name



def load_data(class_a_path, class_b_path):

	labels = []
	generate_arrays = True  # Create arrays where we store the dataset
	for img_name in os.listdir(class_a_path):
		if not img_name.startswith('.'):
			# Read the image
			img = cv2.imread(class_a_path + img_name, 0)
			img = np.reshape(img, (1, img.shape[0], img.shape[1]))
			img_vector = np.reshape(img.ravel(), (1, -1))
			#img_vector_to_classify = img_vector.copy()
			#img_vector_to_classify = img_vector_to_classify[1:54:-1]
			#print('img_vector.shape = ', img_vector.shape)
			#print('img_vector_to_classify.shape = ', img_vector_to_classify.shape)


			# Create arrays where we store the dataset executed only at beginning
			if generate_arrays:
				images = img
				images_vector = img_vector
				#images_vector = img_vector_to_classify
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
			#img_vector_to_classify = np.reshape(img.ravel(), (1, -1))
			images = np.concatenate((images, img), axis=0)
			images_vector = np.concatenate((images_vector, img_vector), axis=0)

			labels.append(1)

    #

	return images, images_vector, labels


def train_svm():
	# Load data
	class_names = ['a', 'b']

	images, images_vector, labels = load_data(class_a_path='images/class_a/', class_b_path='images/class_b/')
	
	print('labels.shape = ', np.shape(labels))

	#img_path = os.listdir('/Users/macbook/Desktop/POLIMI/CORSI/II/I/CPC/Git/Save_The_World/SaveTheWorld/images/')
	img_features, img_name = image_feature('C:/Users/rien27/Documents/GitHub/SaveTheWorld/images/')
	#print(img_features)

	#img_path_b = os.listdir('/images/class_b/')
	#img_features_b, img_name_b = image_feature_b(img_path_b)

    #img_features, img_name = image_feature(images)

	#feature_vector = np.concatenate(img_features_a, img_features_b, axis = 0)
	#name_vector = np.concatenate(img_name_a, img_name_b, axis = 0)

	#print('images_vector.shape = ', images_vector.shape)
	pca = PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)
	svc = SVC(kernel='rbf', class_weight='balanced')
	model = make_pipeline(pca, svc, verbose = True)

	with warnings.catch_warnings():
		# ignore all caught warnings
		warnings.filterwarnings("ignore")
		# execute code that will generate warnings

		print('img_features.shape = ', np.shape(img_features))
		print('img_features[1].shape = ', np.shape(img_features[1]))
		img_features = np.delete(img_features, [500, 501], axis = 0)
		print('img_features.shape after deleting = ', np.shape(img_features))

		xtrain, xtest, ytrain, ytest = train_test_split(img_features, labels, random_state=42)

	param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
	grid = GridSearchCV(model, param_grid)

	print('Fit the SVM model')
	grid.fit(xtrain, ytrain)

	print(grid.best_params_)

	model = grid.best_estimator_

	# Save the model
	dump(model, 'modelSVM.joblib')

	return model



