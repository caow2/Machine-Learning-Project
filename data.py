import numpy as np
import processing as p
import pandas as pd
import os
import cv2

# Images are partitioned into different folders according to their label.
# Sample001 = 0, Sample011 = A, Sample037 = a, Sample062 = z
def buildLabelArray():
	arr = []
	for num in range(10): # 0 - 9
		arr.append(str(num))
	for char in range(ord('A'), ord('Z') + 1):
		arr.append(chr(char))
	for char in range(ord('a'), ord('z') + 1):
		arr.append(chr(char))
	return arr

# Build the dataframe for n bins for all of the char74k data
# Run Edge or Corner detection depending on corner parameter
def buildDataFrame(n, corner=True):
	# Build the dictionary to map the columns to.
	# ['Bin 0 (0 - 127)', 'Bin 1 (128 - 255)']
	cols = []
	for col in range(n):
		lower, upper = p.calculateLowerUpper(col, n)
		cols.append("Bin %s (%s - %s)" % (str(col), str(lower), str(upper)))
	cols.append('Label') # 'A', 'B', etc.
	path = 'char74k/image/goodImage/Bmp'
	row_list = buildRows(path, n, corner)
	df = pd.DataFrame(row_list, columns=cols)
	return df

# Build a list of all rows generaeted from images 
# Recursively step down through all Sample0XX directories from the path and process images
def buildRows(path, n, corner):
	rows = []
	os.chdir(path)
	for directory, subdirs, files in os.walk("."):
		print(directory)
		for file in files: 
			if(file.endswith(".png")):
				index = int(file[3:6]) - 1 # Images are in the format img0XX-00011.png. The XX tells us the label
				print("\t%s" % file)
				image = p.preprocess(cv2.imread("%s/%s" % (directory, file)))
				if(corner):
					image = p.orb(image)
				else:
					image = p.canny(image)
				image_row = p.convert_to_array(image, n).astype(object)
				image_row = np.append(image_row, label_arr[index]) #use vstack. Resulting df is 3 x 7k
				rows.append(image_row)
	return rows

# Select n training examples from each Label
def getTrainExamples(dataFrame, n):
	train = df.groupby('Label').apply(lambda x: x.sample(n)).reset_index(drop=True)
	return train



label_arr = buildLabelArray()
df = buildDataFrame(2)
print(df)