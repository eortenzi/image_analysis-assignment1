import sklearn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
"""
BACKGROUND: This method transforms our RGB image, into a grayscaled image. It is repeated for every pixel in the image.
STEP 1: The 'image' we pass as an argument is a NumPy array, and stores the dimensions of the array. 
It looks like this: [x, y, 3]
 - The x refers to the height of the image, so the number of rows.
  - The y refers to the width of the image, so the number of columns.
   - The 3, refers to the number of colour channels. In an RBG image it is 3, in a thresholded image it is 2.

STEP 2: We use nested 'for' loops to loop through the whole image.
 - 'x' represents the row, 'y' represents the column.
  - 'x' loops from 0 to the number of rows in the image represented by 'image.shape[0]'
   - 'y' loops from 0 to the number of columns in the image represented by 'image.shape[1]'

STEP 3: Next, the integer 'redChannel' looks into the NumPy array, and stores the red intensity of the current pixel. (E.g. redChannel = 120)
 It repeats this process for all of the pixels on each iteration. Same applies for the 'greenChannel' and 'blueChannel', with '1 = GREEN' and '2 = BLUE' respectively.

STEP 4: Next, we create an int 'pixelIntensity' that stores the grayscale pixel intensity that will be assigned to the current pixel.
 - This is determined by fixed weights assigned to each colour channel. This is because the human eye is more sensitive to certain
  - colours, therefore, making the weights NOT an even 0.333 each.
   - We store the intensity in an 'int' because the result of our calculations could be a decimal number, but a standard 8-bit image
    only represents WHOLE intensity values!

STEP 5: Lastly, we need to set every pixel in the image to grayscale. We take the 'x' & 'y' coordinates of each pixel,
 - and then we set the R, G, and B channels all to the grayscale intensity which we calculated just now!
"""
def grayscaleImage(image):
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            redChannel = image[x, y, 0]
            greenChannel = image[x, y, 1]
            blueChannel = image[x, y, 2]

            pixelIntensity = int(
                0.299 * redChannel + 
                0.587 * greenChannel + 
                0.114 * blueChannel)

            image[x, y] = [pixelIntensity, pixelIntensity, pixelIntensity]
    return image

"""
BACKGROUND: This method returns a histogram with the intensity of each pixel in the grayscale image. It is repeated for every pixel in the image.
STEP 1: We first create a 'histogram' variable that stores a NumPy array that contains 256 floating point numbers.
  - We use 256, because in a standard grayscale image, a pixel can be an intensity from 0 to 255. 
   - Each INDEX in the 'histogram' array represents an intensity level.
   - Each value given to each INDEX at the beginning is 0.0, because we haven't gone through the image yet.

STEP 2: We use nested 'for' loops to loop through the whole image.
  - 'x' represents the row, 'y' represents the column.
   - 'x' loops from 0 to the number of rows in the image represented by 'image.shape[0]'
    - 'y' loops from 0 to the number of columns in the image represented by 'image.shape[1]'

STEP 3: We store the pixel intensity inside of the 'pixelIntensity' variable.
 - We use the 'x' & 'y' coordinates to locate the pixel, but we ONLY store the intensity.

STEP 4: The last thing we do is, once we have the pixel intensity, we want to add 1 to the counter in our 'histogram' NumPy array.
- (E.g. If we had a pixelIntensity = 220, that means at the index 220 in the NumPy array, add 1, meaning one pixel has a 220 intensity)
"""
def imageHistogramGrayscale(image):
    grayscaleHistogram = np.zeros(256)
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            pixelIntensity = image[x, y]
            grayscaleHistogram[(pixelIntensity)] += 1
    return grayscaleHistogram

"""
BACKGROUND: This method returns a histogram with the R, G & B intensities of each pixel in the RGB image. It is repeated for every pixel in the image.
STEP 1: This method is very similar to our grayscale one. We create 3 separate NumPy arrays, one for every RGB channel.
 - They all start at 0.0 for each index same as before.

STEP 2: We loop through the whole image same as before, but for each pixel, we retrieve the R, G and B intensity values and store them in separate values.
 - 'redIntensity' is an integer to be clear, and the same applies for G and B.

STEP 3: After we disect the R, G and B intensities of all the pixels and store them in their appropriate int variables
 we add them to its appropriate 'histogram' NumPy array

STEP 4: Since we have 3 separate histograms now, we return all of them, that we can then use in the main() method.
"""
def imageHistogramsRGB(image):
    redHistogram = np.zeros(256)
    greenHistogram = np.zeros(256)
    blueHistogram = np.zeros(256)

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            redIntensity = image[x, y, 0]
            greenIntensity = image[x, y, 1]
            blueIntensiy = image[x, y, 2]

            redHistogram[(redIntensity)] += 1
            greenHistogram[(greenIntensity)] += 1
            blueHistogram[(blueIntensiy)] += 1
    return redHistogram, greenHistogram, blueHistogram


#----- STEP 1 -----
#We use the 'NumPY' library to open the self-portrait image
baseImage = Image.open('selfPortrait.jpg')
plt.figure(figsize = (15, 6))
#We want to keep the OG image as is, so we will work on a copy of the image
# from now on using the '.copy()' method
baseImageCopy = baseImage.copy()
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(baseImageCopy)

#Next, we need to convert the base image into a NumPy array
numPyImage = np.array(baseImageCopy)

#----- STEP 1A - IMAGE RESIZING
#We resize our image to 512 by 512 pixels and store it in a variable called 'resizeBaseImage'
resizedBaseImage = resize(numPyImage, (512, 512))
#We then show it using the '.imshow()' method
plt.subplot(1, 3, 2)
plt.title("Resized Image")
plt.imshow(resizedBaseImage)

#STEP 1B - CONVERTING TO GREYSCALE
grayscaledImage = grayscaleImage(numPyImage) 
plt.subplot(1, 3, 3)
plt.title("Grayscaled Image")
plt.imshow(grayscaledImage)
plt.show()

#STEP 1C - DISPLAYING IMAGE HISTOGRAM for Grayscale IMAGE && for RGB IMAGES
plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.title("Grayscaled Image")
plt.imshow(grayscaledImage)

grayscaleHistogram = imageHistogramGrayscale(grayscaledImage)
plt.subplot(1, 2, 2)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixesls")
"""
To display the histogram we need to use the '.scatter()' method. We pass 4 parameters:
'range(256)'         - The range of values we are representing which is from 0 to 255.
'grayscaleHistogram' - The NumPy array that has the number of pixels at each intensity level.
'color = grey'       - The color of the dots representing the intensities on the histogram
's = 10'             - The size of each dot
"""
plt.scatter(range(256), grayscaleHistogram, color = 'grey', s = 10)
plt.show()

redHistogram, greenHistogram, blueHistogram = imageHistogramsRGB(numPyImage)

plt.figure(figsize = (15, 6))
plt.subplot(1, 4, 1)
plt.title("Original RGB Image")
plt.imshow(baseImageCopy)

plt.subplot(1, 4, 2)
plt.title("Red Histogram")
plt.fill_between(range(256), redHistogram, color = 'red', s = 10)

plt.subplot(1, 4, 3)
plt.title("Green Histogram")
plt.scatter(range(256), greenHistogram, color = 'green', s = 10)

plt.subplot(1, 4, 4)
plt.title("Blue Histogram")
plt.scatter(range(256), blueHistogram, color = 'blue', s = 10)
plt.show()
