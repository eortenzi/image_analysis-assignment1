import sklearn
import numpy as np
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
"""
BACKGROUND: This method transforms our RGB image, into a grayscaled image.
 - We use nested 'for' loops to loop through the whole image.
  - 'x' represents the row, 'y' represents the column.
   - 'x' loops from 0 to the number of rows in the image represented by 'image.shape[0]'
    - 'y' loops from 0 to the number of columns in the image represented by 'image.shape[1]'

STEP 1: The 'image' we pass as an argument is a NumPy array, and stores the dimensions of the array. 
It looks like this: [x, y, 3]
 - The x refers to the height of the image, so the number of rows.
  - The y refers to the width of the image, so the number of columns.
   - The 3, refers to the number of colour channels. In an RBG image it is 3, in a thresholded image it is 2.

STEP 2: Next, the integer 'redChannel' looks into the NumPy array, and stores the red intensity of the current pixel. (E.g. redChannel = 120)
 It repeats this process for all of the pixels on each iteration. Same applies for the 'greenChannel' and 'blueChannel', with '1 = GREEN' and '2 = BLUE' respectively.

STEP 3: Next, we create an int 'pixelIntensity' that stores the grayscale pixel intensity that will be assigned to the current pixel.
 - This is determined by fixed weights assigned to each colour channel. This is because the human eye is more sensitive to certain
  - colours, therefore, making the weights NOT an even 0.333 each.
   - We store the intensity in an 'int' because the result of our calculations could be a decimal number, but a standard 8-bit image
    only represents WHOLE intensity values!

STEP 4: Lastly, we need to set every pixel in the image to grayscale. We take the 'x' & 'y' coordinates of each pixel,
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

#----- STEP 1 -----
#We use the 'NumPY' library to open the self-portrait image
baseImage = Image.open('selfPortrait.jpg')
plt.figure(figsize = (15, 6))
#We want to keep the OG image as is, so we will work on a copy of the image
# from now on using the '.copy()' method
baseImageCopy = baseImage.copy()
plt.subplot(1, 3, 1)
plt.title(f"Original Image")
plt.imshow(baseImageCopy)

#Next, we need to convert the base image into a NumPy array
numPyImage = np.array(baseImageCopy)

#----- STEP 1A - IMAGE RESIZING
#We resize our image to 512 by 512 pixels and store it in a variable called 'resizeBaseImage'
resizedBaseImage = resize(numPyImage, (512, 512))
#We then show it using the '.imshow()' method
plt.subplot(1, 3, 2)
plt.title(f"Resized Image")
plt.imshow(resizedBaseImage)

#STEP 1B - CONVERTING TO GREYSCALE
greyscaledImage = grayscaleImage(numPyImage) 
plt.subplot(1, 3, 3)
plt.title(f"Grayscaled Image")
plt.imshow(greyscaledImage)
plt.show()
