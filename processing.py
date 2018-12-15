import cv2
import numpy as np

### Utility Functions ###

# Preprocesses the given image and converts it to a n x n image. 
# Depending on the parameters, a Gaussian Blur may be applied to smooth out the image and reduce noise.
def preprocess(image, grayScale=True):
	gray = image
	if grayScale:
		gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
	# Gaussian filtering and blur #
	n = 64
	result = cv2.resize(gray, (n , n))
	result = cv2.GaussianBlur(result, (5,5), 0)
	return result

# Convert grayscale image to an np array with n elements (for n bins).
# Each element is a count of the Grayscale pixels for that bin.
# i.e. 2 bins results in [a,b] where a is the count of pixels that range from 0 - 127, and b from 128 - 255.
def convert_to_array(image, n):
	arr = np.zeros(n)
	i, j = np.shape(image)
	for row in range(i):
		for col in range(j):
			pixel = image[row, col]
			arr[findBin(pixel, n)] += 1
	#normalize the bin array. Divide by number of pixels
	arr = arr / (i * j)
	return arr

# Binary search to find the bin that this pixel belongs in
def findBin(pixel, n):
    left = 0
    right = n - 1
    while left <= right:
        mid = int((left + right) / 2)
        lower, upper = calculateLowerUpper(mid, n)
        if lower <= pixel <= upper:
            return mid
        elif pixel < lower:
            right = mid - 1
        else:
            left = mid + 1
    return -1

# Calculate the lower and upper bounds of the bin at index i.
# i.e. For n = 2 bins, index 0 is 0 - 127.
# Note that the pixel intensities might not be evenly distributed - the last bin's range may
# be greater than all the other bins.
def calculateLowerUpper(index, n):
	size = int(256 / n)
	lower = index * size
	upper = ((index + 1) * size) - 1 if index != (n - 1) else 255 #last bin has to have upper of 255
	return lower, upper



### Edge detection ###

# Takes in a preprocessed image and returns the edge map for the image.
# Canny thresholds are automatically selected based on the median pixel intensity of the image.
# The returned image is the original image with the edges overlayed over it (in white pixels).
# This is achieved by a bitwise or
def canny(image, sigma=.33):
	v = np.median(image)
	lower = int(max(0, (1.0) - sigma) * v)
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	image = np.bitwise_or(image, edged)
	return image


### Corner detection ###

# Takes in a preprocessed image and returns an image with black edges
# Corners are initially detected using Oriented FAST and Rotated BRIEF (ORB)
def orb(image):
	o = cv2.ORB_create()
	keypoints = o.detect(image, None)
	corner_img = cv2.drawKeypoints(image, keypoints, None, color=(255,255,255))
	result = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY) #drawKeypoints converts to RGB
	return result




