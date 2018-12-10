import numpy as np
import processing as p
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
import keras
from keras.utils import to_categorical

path = 'char74k/image/goodImage/Bmp'

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
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
	return x_train, x_test, y_train, y_test

# Run K Nearest Neighbors on the using the given datasets
def kNeighbor(k, x_train, x_test, y_train, y_test):
	#convert to 1D array for KNN purposes
	y_train = np.ravel(y_train, order='C')
	y_test = np.ravel(y_test, order='C')

	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(x_train, y_train)
	return knn.score(x_test, y_test) # mean accuracy 

### Experiments ###

# Test edges for n bins
def edgeKNN(n):
	print("Running edgeKNN for n = " + str(n))
	df = buildDataFrame(n, corner=False)
	# Try on k for (powers of 2) - 1, up to (2^9) - 1 = 511
	accuracy = [] # [k = 1, k = 3, k = 7, k = 15, ...]
	std = []
	power = 1
	while power < 9:
		k = (2 ** power) - 1
		x_train, x_test, y_train, y_test = getTrainExamples(df, .75)
		k_acc = [] # run experiment 5 times for each n
		for n in range(0,5):
			k_acc.append(kNeighbor(k, x_train, x_test, y_train, y_test))
		accuracy.append(np.mean(k_acc))
		std.append(np.std(k_acc))
		power+= 1
	return accuracy, std

# Test corner for n bins
def cornerKNN(n):
	print("Running cornerKNN for n = " + str(n))
	df = buildDataFrame(n, corner=True)
	# Try on k for (powers of 2) - 1, up to (2^9) - 1 = 511
	accuracy = [] # [k = 1, k = 3, k = 7, k = 15, ...]
	std = []
	power = 1
	while power < 9:
		k = (2 ** power) - 1
		x_train, x_test, y_train, y_test = getTrainExamples(df, .75)
		k_acc = [] # run experiment 10 times for each n
		for n in range(0,5):
			k_acc.append(kNeighbor(k, x_train, x_test, y_train, y_test))
		accuracy.append(np.mean(k_acc))
		std.append(np.std(k_acc))
		power+= 1
	return accuracy, std



# Test edge for n bins from powers of 2 up to 256
def experimentEdge():
	pwr = 1
	bin_accuracy = [] # [n = 2 bins, n = 4 bins, n = 8 bins, ...]
	bin_std = []
	while pwr <= 8:
		n = 2 ** pwr
		acc, std = edgeKNN(n)
		bin_accuracy.append(acc)
		bin_std.append(std)
		pwr += 1
	return bin_accuracy, bin_std

def experimentCorner():
	pwr = 1
	bin_accuracy = []
	bin_std = []
	while pwr <= 8:
		n = 2 ** pwr
		acc, std = cornerKNN(n)
		bin_accuracy.append(acc)
		bin_std.append(std)
		pwr += 1
	return bin_accuracy, bin_std

label_arr = buildLabelArray()

print("\nEdge experiments: ")
edge_acc, edge_std = experimentEdge()
print("Edge accuracy:")
print(edge_acc)
print("Edge standard deviation:")
print(edge_std)


print("\nCorner experiments: ")
corner_acc, corner_std = experimentCorner()
print("Corner accuracy:")
print(corner_acc)
print("Corner standard deviation:")
print(corner_std)



