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

"""
Images naturally have noise. Some images have more noise than others due to cicrumstances such as low light, electronic interference etc. If we use an image
 that is naturally relatively clean of noise, we introduce Gaussian noise. This means we purposefully make the image more noisy. Adding Gaussian noise to
  an image means we make random pixels lighter or brighter. More specifically, we add light intensity or decrease light intensity to each pixel in the image.
   This effect could make darker areas, darker. Lighter areas, lighter. Darker areas, lighter, and lighter areas, darker.
    (E.g. If a pixel in the image is 145 intensity. We add a Gaussian noise of +34, making the pixel now 179 intensity.)
     - The reason we add Gaussian noise is, we then want to determine how effective certain image processing techniques are at fixing noisy images.
      - This is so, when we have naturally noisy images, we can use these techniques to fix this noise, and make them cleaner.
       - CONCLSION: We create a problem by adding Gaussian noise to a relatively clean image, so that when we encounter naturally noisy images, we know how to handle them.

BACKGROUND: This method introduces Gaussian Noise to an RGB portrait image
'mean = 0.0': The mean is the average 'effect' of the Gaussian noise on the image. If we take all of the Gaussian noise values we applied to the image, and calculate their
 mean, it would equal Zero or close to Zero.

'variance = 0.02': The variance is how spread the Gaussian noise values are mathematically. In other words, we rather use the Standard Deviation
'standard deviation = 0.1414': The SD is how spread the Gaussian noise values are from the mean. That means that most noise values are +0.1414 or -0.1414 over/ lower the mean.

STEP 1: First, since the mean, variance and standard deviation are STANDARDIZED, meaning in a range between 0 and 1, we need to normalize our image too.
 - By normalizing it, that means the R, G and B values for each pixel, have to be a value between 0 and 1.
  - Before this method our 'image' is storing this [x, y, z] for EVERY PIXEL in the image as INT numbers. x being the RED channel, y being the BLUE channel, z being the GREEN channel.
   - After this method our 'image' is storing each of the RGB channels as FLOAT numbers. 
   
   '.astype()': Converts the values in the NumPy array into a dfferent data type
   'np.float32': Specifies the new data type, AKA 32-Bit Floating Point numbers
   '/ 255.00': Divides each RGB channel for each pixel by a 255 to give the normalized RGB values for each pixel. NOTE: Its 255.00 because it is a floating-point division

STEP 2: We calculate the Standard Deviation by square rooting the variance as we are spreading the noise using a Normal Guassian Distribution
 - We use the '.sqrt()' function on the variance, which gives us a Standard Deviation of around 0.1414.
  - As mentioned earlier, this means MOST noise will be values around 0.1414, both in the positive and negative range.

STEP 3: Next, we generate the Gaussian noise values. 
 - We start with using the '.random' function to generate random numbers
  - We then use the '.normal()' function because we want the noise to be spread following a Normal Gaussian distribution This means most of the 
   noise values will be close to the mean, in both the positive (+) and negative (-) direction.
    - Noise values that are far from Zero, can happen but rarely.
    
    'mean': Provide the mean as an argument as we need it for our Normal distribution
    'StandardDeviation': Provide the SD as we need it for the distribution
    'normalizedImage.shape()': This creates the shape of the noise array. It has the exact shape of the grayscale image, to match the dimensions.
     This NumPy noise array will have these dimensions [N° of Rows, N° of Columns, Number of RGB channels], which matches the images dimensions and RGB Channels.
      
STEP 4: Now, we ADD the Gaussian noise onto our original Grayscale image

STEP 5: Lastly, we normalize the image. What this means is, we ensure every pixel value in our image is between 0.0 and 1.0
 - This is because, after adding noise to our image, we might end up with pixel brightnesses that go below 0 or above 1. This
  could happen with pixels close to 0 or 1 to begin with, so close 0 or 255 in the 8-Bit scale.
   
  '.clip()': If we have a pixel value that is -0.15 for example, it will be set to 0, after the clipping.
             If a pixel value is 0.25 for example, it will remain 0.25, because it is in the 0 - 1 range.
             If a pixel value is 1.12 for example, it will be set to 1 after the clipping.
"""
def gaussianNoise(image, mean, variance):
    normalizedImage = image.astype(np.float32) / 255.00
    standardDeviation = np.sqrt(variance)

    gaussianNoise = np.random.normal(mean, standardDeviation, normalizedImage.shape)
    noisyImage = normalizedImage + gaussianNoise
    noisyImage = np.clip(noisyImage, 0.0, 1.0)
    return noisyImage

"""
BACKGROUND: 
"""
def saltAndPepperNoise(image, amount):
    normalizedImage = image.astype(np.float32) / 255.00

    saltAndPepperNoise = np.random(normalizedImage.shape)
    return image
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
grayscaledImage = grayscaleImage(numPyImage.copy()) 
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

plt.figure(figsize = (15, 10))
plt.subplot(2, 2, 1)
plt.title("Original RGB Image")
plt.imshow(baseImageCopy)

plt.subplot(2, 2, 2)
plt.title("Red Histogram")
plt.plot(range(256), redHistogram, color = 'red')
plt.fill_between(range(256), redHistogram, color = 'red')

plt.subplot(2, 2, 3)
plt.title("Green Histogram")
plt.plot(range(256), greenHistogram, color = 'green')
plt.fill_between(range(256), greenHistogram, color = 'green')

plt.subplot(2, 2, 4)
plt.title("Blue Histogram")
plt.plot(range(256), blueHistogram, color = 'blue')
plt.fill_between(range(256), blueHistogram, color = 'blue')
plt.show()

#STEP 2 - Applying Gaussian Noise && Salt and Pepper Noise to our Grayscale
plt.figure(figsize = (15, 6))
plt.subplot(1, 3, 1)
plt.title("Original Grayscaled Image")
plt.imshow(grayscaledImage)

gaussianNoiseImage = grayscaleImage(numPyImage.copy(), 0.0, 0.02)
plt.subplot(1, 3, 2)
plt.title("Gaussian Noise")
plt.imshow(gaussianNoiseImage)
plt.axis("off")

grayscaleSaltAndPepperImage = grayscaleImage(numPyImage.copy(), 0.09)
plt.subplot(1, 3, 3)
plt.title("Salt and Pepper Noise")
plt.axis("off")
plt.imshow(grayscaleSaltAndPepperImage)
plt.show()
