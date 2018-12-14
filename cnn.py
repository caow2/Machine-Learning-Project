from keras.preprocessing import image
import cv2
import numpy as np
import processing as p
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras import optimizers
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge

path = 'char74k/image/goodImage/Bmp'

def buildLabelArray():
	arr = []
	for num in range(10): # 0 - 9
		arr.append(str(num))
	for char in range(ord('A'), ord('Z') + 1):
		arr.append(chr(char))
	for char in range(ord('a'), ord('z') + 1):
		arr.append(chr(char))
	return arr

def load_images(grayScale=True):
	# Read images and convert to row format
	x = []
	y = []
	for directory, subdirs, files in os.walk(path):
		for file in files:
			if(file.endswith(".png")):
				index = int(file[3:6]) - 1 
				im = p.preprocess(cv2.imread("%s/%s" % (directory, file)), grayScale=grayScale) # Grayscale
				im = im / 255 # pixels [0, 1] range for faster convergence
				x.append(im)
				y.append(index) # keras only uses numerical labels
	x = np.asarray(x, dtype=np.float32)
	y = np.asarray(y, dtype=np.float32)
	return x, y

def getTrainExamples(x, y, train_size):
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
	return x_train, x_test, y_train, y_test

def cnn(x, y, n, model, pretrained=False, num_epoch=5):
	portion = .75 # 75% of data used for training
	x_train, x_test, y_train, y_test = getTrainExamples(x, y, train_size=portion)
	# Reshape data to fit model
	training_examples = int(len(x) * portion)
	test_examples = len(x) - training_examples
	channel = 1
	if pretrained:
		channel = 3
	x_train = x_train.reshape(training_examples, n, n, -1) # convert each example to (n, n , 1)
	x_test = x_test.reshape(test_examples, n, n, -1)

	# one-hot encode y labels. i.e. '0' -> [1, 0, 0, ...], '1' -> [0, 1, 0, ...]
	# Each will be a vector of len 62.
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	# use a Sequential model. Kernel size 3 implies 3x3 filter matrix
	model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epoch)


# Return model with specified convolutional layers and constant filter_size
def getModel_Layers(n, layers, filter_size):
	model = Sequential()
	model.add(Conv2D(filter_size, kernel_size=3, init='he_normal',activation='relu', input_shape=(n,n, 1)))
	model.add(MaxPooling2D(pool_size=(2,2)))

	for layer in range(1, layers):
		model.add(Conv2D(filter_size, kernel_size=3,init='he_normal', activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	'''model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))'''
	model.add(Dense(62, activation='softmax')) # 62 possible outputs
	return model

# Return model with specified convolutional layers and increasing filter size
def getModel_Filters(n, layers):
	size = 128
	model = Sequential()
	model.add(Conv2D(size, kernel_size=3, init='he_normal',activation='relu', input_shape=(n,n, 1)))
	model.add(MaxPooling2D(pool_size=(2,2)))

	for layer in range(1, layers):
		size *= 2
		model.add(Conv2D(size, kernel_size=3,init='he_normal', activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	'''model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))'''
	model.add(Dense(62, activation='softmax')) # 62 possible outputs
	return model

# n represents image size for the cnn
def cnn_experiment(n):
	x, y = load_images(grayScale=False)

	# Try 3 layers with increasing size - 128, 256, 512
	'''print("\n\nCNN with increasing layers and filter size:")
	for layer in range(0, 3):
		filter_size = 2** (layer + 7)
		print("%s Layers, Filter size of %s" % (str(layer + 1), str(filter_size)))
		model = getModel_Filters(n, layer + 1)
		print(model.summary())
		cnn(x, y, n, model)
	
	print("\n\nCNN with increasing layers and contant filter size:")
	#128 to 512
	for pwr in range(7, 10):
		size = 2 ** pwr
		print("Filter size of %s" % str(size))
		for layer in range(0, 3):
			print("%s Layers" % str(layer + 1))
			model = getModel_Layers(n, layer + 1, size)
			cnn(x, y, n, model)
	'''
	'''
	print("\n\nResNet:")
	base_model = ResNet50(include_top=False,weights='imagenet',input_shape=(64,64,3))
	print(base_model.summary())
	out = base_model.output
	out = Flatten()(out)
	out = Dense(4096, activation='relu')(out)
	out = Dense(4096, activation='relu')(out)
	out = Dense(62, activation='softmax')(out)
	model = Model(inputs=base_model.input, outputs=out)
	cnn(x, y, n, model, pretrained=True)
	'''
	print("\n\nNASNet:")
	base_model = NASNetLarge(include_top=False,weights='imagenet',input_shape=(n,n,3))
	print(base_model.summary())
	out = base_model.output
	out = Flatten()(out)
	out = Dense(4096, activation='relu')(out)
	out = Dense(4096, activation='relu')(out)
	out = Dense(62, activation='softmax')(out)
	model = Model(inputs=base_model.input, outputs=out)
	cnn(x, y, n, model, pretrained=True)


label_arr = buildLabelArray()
cnn_experiment(64)