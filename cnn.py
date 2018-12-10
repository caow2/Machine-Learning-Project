from keras.preprocessing import image
import cv2
import numpy as np
import processing as p
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras import optimizers
from keras.utils import to_categorical

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

def load_images(n):
	# Read images and convert to row format
	x = []
	y = []
	for directory, subdirs, files in os.walk(path):
		for file in files:
			if(file.endswith(".png")):
				index = int(file[3:6]) - 1 
				im = p.preprocess(cv2.imread("%s/%s" % (directory, file))) # Grayscale
				im = im / 255 # pixels [0, 1] range for faster convergence
				im = cv2.resize(im, (n, n))
				x.append(im)
				y.append(index) # keras only uses numerical labels
	x = np.asarray(x, dtype=np.float32)
	y = np.asarray(y, dtype=np.float32)
	return x, y

def getTrainExamples(x, y, train_size):
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
	return x_train, x_test, y_train, y_test

def cnn(x, y, n, num_epoch=5):
	portion = .75 # 75% of data used for training
	x_train, x_test, y_train, y_test = getTrainExamples(x, y, train_size=portion)
	# Reshape data to fit model
	training_examples = int(len(x) * portion)
	test_examples = len(x) - training_examples
	x_train = x_train.reshape(training_examples, n, n, 1) # convert each example to (n, n , 1)
	x_test = x_test.reshape(test_examples, n, n, 1)

	# one-hot encode y labels. i.e. '0' -> [1, 0, 0, ...], '1' -> [0, 1, 0, ...]
	# Each will be a vector of len 62.
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	# use a Sequential model. Kernel size 3 implies 3x3 filter matrix
	model = Sequential()
	model.add(Conv2D(128, kernel_size=3, init='he_normal',activation='relu', input_shape=(n,n, 1)))
	#model.add(Conv2D(128, kernel_size=3, init='he_normal',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(256, kernel_size=3,init='he_normal', activation='relu'))
	#model.add(Conv2D(256, kernel_size=3, init='he_normal',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(512, kernel_size=3, init='he_normal', activation='relu'))
	#model.add(Conv2D(512, kernel_size=3, init='he_normal',activation='relu'))
	#model.add(Conv2D(512, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(62, activation='softmax')) # 62 possible outputs

	model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epoch)

def cnn_experiment(n):
	x, y = load_images(n)
	cnn(x, y, n)

label_arr = buildLabelArray()

cnn_experiment(64)