import sklearn
from PIL import Image

#------
#STEP 1
#------
#We use the 'Pillow' library to open the self-portrait image
ogImage = Image.open('selfPortrait.jpg')

#We want to keep the OG image as is, so we will work on a copy of the image
# from now on using the '.copy()' method
ogImageCopy = ogImage.copy()

#STEP 1A - IMAGE RESIZING
#We then resize our image to 512 by 512 pixels and store it in a variable.
resizedOriginalImage = ogImageCopy.resize((512, 512))
#We then show it using the '.show()' method
resizedOriginalImage.show()

#STEP 1B - CONVERTING TO GREYSCALE
