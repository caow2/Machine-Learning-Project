import numpy as np
import processing as p
import pandas as pd
import os
import cv2
import random
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

path = 'char74k/image/goodImage/Bmp'
rand_min = 0
rand_max = 61

# Build the array to access the corresponding label by index.
# Images are partitioned into different folders according to their label.
# Sample001 = 0, Sample011 = A, Sample037 = a, Sample062 = z -> [1, 2, ..., z]
def buildLabelArray():
	arr = []
	for num in range(10): # 0 - 9
		arr.append(str(num))
	for char in range(ord('A'), ord('Z') + 1):
		arr.append(chr(char))
	for char in range(ord('a'), ord('z') + 1):
		arr.append(chr(char))
	return arr

# Build the dataframe for n bins for all of the char74k 
# Run Edge or Corner detection depending on corner parameter
def buildDataFrame(n, corner=True):
	# Build the dictionary to map the columns to.
	# ['Bin 0 (0 - 127)', 'Bin 1 (128 - 255)']
	cols = []
	for col in range(n):
		lower, upper = p.calculateLowerUpper(col, n)
		cols.append("Bin %s (%s - %s)" % (str(col), str(lower), str(upper)))
	cols.append('Label') # 'A', 'B', etc.
	row_list = buildRows(path, n, corner)
	df = pd.DataFrame(row_list, columns=cols)
	return df

# Build a list of all rows generated from images 
# Recursively step down through all Sample0XX directories from the path and process images
def buildRows(path, n, corner):
	rows = []
	for directory, subdirs, files in os.walk(path):
		#print(directory)
		for file in files: 
			if(file.endswith(".png")):
				index = int(file[3:6]) - 1 # Images are in the format img0XX-00011.png. The XX tells us the label
				#print("\t%s" % file)
				image = p.preprocess(cv2.imread("%s/%s" % (directory, file)))
				if(corner):
					image = p.orb(image)
				else:
					image = p.canny(image)
				image_row = p.convert_to_array(image, n).astype(object)
				image_row = np.append(image_row, label_arr[index]) #use vstack. Resulting df is 3 x 7k
				rows.append(image_row)
	return rows

# Finds the average height and width amongst sample images
# Used for finding the height and width to resize images to
def meanImageSize(path):
	height = 0
	width = 0
	num = 0
	for directory, subdirs, files in os.walk(path):
		for file in files:
			if(file.endswith(".png")):
				print("%s/%s" % (directory, file))
				image = cv2.imread("%s/%s" % (directory, file))
				h, w, channels = image.shape
				height += h
				width += w
				num += 1
	height /= num
	width /= num
	return height, width

# Partition dataframe into training and testing sets
# Training size is [0,1] and represents a % of the entire dataset.
def getTrainExamples(dataFrame, train_size):
	y = dataFrame['Label'].to_frame()
	x = dataFrame.drop(columns=['Label'])
	return x, y

# Run K Nearest Neighbors on the using the given datasets
# 10 Fold Cross Validation
# confusion is for calculating Confusion Matrix - only for x and y with low number of classes
def kNeighbor(k, x, y, classes, confusion=False):
	#convert to 1D array for KNN purposes
	y = np.ravel(y, order='C')
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, x, y, cv=10)
	if(confusion): #print confusion matrix and the labels											Predicted
		y_prediction = cross_val_predict(knn, x, y, cv=10)		#							Actual	 a 	,	b
		c = confusion_matrix(y, y_prediction, labels=classes)	# order by labels. i.e. [a,b]->	a  True a  , True a but predict b
		print("\tk = %s Confusion Matrix for %s" % (str(k),str(classes)))	#					b  True b but predict a , True b
		print(c)
	return np.mean(scores)

### Experiments ###

# Test edges for n bins and c classes. i.e. c = 2 would do a binary KNN for 2 selected classes
def edgeKNN(n, c=62, confusion=False):
	print("Running edgeKNN for n = " + str(n))
	df = buildDataFrame(n, corner=False)
	accuracy = [] # [k = 1, k = 3, k = 5, k = 7, ..., k = 15] up to k = 15
	std = []
	k = 1
	while k <= 15:
		k_acc = [] # run experiment 5 times for each k
		# Extract the appropriate classes, split into x and y, then pass into knn
		for n in range(0,5):
			dataframe, classes = extractClasses(df, c)
			x, y = getTrainExamples(dataframe, .75)
			mean_accuracy = kNeighbor(k, x, y, classes,confusion=confusion)
			k_acc.append(mean_accuracy)
		accuracy.append(np.mean(k_acc))
		std.append(np.std(k_acc))
		k += 2
	return accuracy, std

# Test corner for n bins
def cornerKNN(n, c=62, confusion=False):
	print("Running cornerKNN for n = " + str(n))
	df = buildDataFrame(n, corner=True)
	print(df)
	
	accuracy = [] # [k = 1, k = 3, k = 5, k = 7, ...]
	std = []
	k = 1
	while k <= 15:
		k_acc = [] # run experiment 5 times for each n
		for n in range(0,5):
			dataframe, classes = extractClasses(df, c)
			x, y = getTrainExamples(dataframe, .75)
			mean_accuracy = kNeighbor(k, x, y, classes,confusion=confusion)
			k_acc.append(mean_accuracy)
		accuracy.append(np.mean(k_acc))
		std.append(np.std(k_acc))
		k += 2
	return accuracy, std

def extractClasses(df, c):
	if c > 62:
		return None
	classes = set()
	while(len(classes) < c):
		label = label_arr[random.randint(rand_min,rand_max)]
		classes.add(label)
	return df.loc[df['Label'].isin(classes)], list(classes)


# Test edge for n bins from powers of 2 up to 2^6 due to computation time
# c for number of classes -> c = 2 indicates experiment on 2 label classes
def experimentEdge(c=62, confusion=False):
	pwr = 1
	bin_accuracy = [] # [n = 2 bins, n = 4 bins, ..., n = 256 bins]
	bin_std = []
	while pwr <= 6:
		n = 2 ** pwr
		acc, std = edgeKNN(n, c, confusion)
		bin_accuracy.append(acc)
		bin_std.append(std)
		pwr += 1
	return bin_accuracy, bin_std

def experimentCorner(c=62, confusion=False):
	pwr = 1
	bin_accuracy = []
	bin_std = []
	while pwr <= 6:
		n = 2 ** pwr
		acc, std = cornerKNN(n, c, confusion)
		bin_accuracy.append(acc)
		bin_std.append(std)
		pwr += 1
	return bin_accuracy, bin_std

# Run experiments for 2, 3, 4, 5 and 62 label classes.
# Only print confusion matrix for classes from 2 and 3 classes
def edgeTest():
	print("\nEdge experiments: ")

	print("\n\n2 Classes:")
	acc, std = experimentEdge(c=2, confusion=True)
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

	print("\n\n3 Classes:")
	acc, std = experimentEdge(c=3, confusion=True)
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

	print("\n\n4 Classes:")
	acc, std = experimentEdge(c=4)
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

	print("\n\n5 Classes:")
	acc, std = experimentEdge(c=5)
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

	print("\n\n62 Classes:")
	acc, std = experimentEdge()
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

def cornerTest():
	print("\nCorner experiments: ")
	print("\n\n2 Classes:")
	acc, std = experimentCorner(c=2, confusion=True)
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

	print("\n\n3 Classes:")
	acc, std = experimentCorner(c=3, confusion=True)
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

	print("\n\n4 Classes:")
	acc, std = experimentCorner(c=4)
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

	print("\n\n5 Classes:")
	acc, std = experimentCorner(c=5)
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

	print("\n\n62 Classes:")
	acc, std = experimentCorner()
	print("Accuracy")
	print(acc)
	print("Standard Deviation")
	print(std)

label_arr = buildLabelArray()
#edgeTest()
cornerTest()



