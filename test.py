import cv2
import imagehash
import os
from PIL import Image
#
# #Read Image
# img = cv2.imread('images/orig_small.png')
# #Display Image
# cv2.imshow('image',img)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# #Applying Grayscale filter to image
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# #Saving filtered image to new file
# mod_image_name = 'modified_image.png'
# if not os.path.exists('images/'+mod_image_name):
#     cv2.imwrite('images/graytest.jpg',gray)

im = Image.open('images/orig_small.png')
hash1 = imagehash.average_hash(img)
hash2 = imagehash.average_hash(gray)
print(hash1-hash2)

#
# im1 = Image.open('images/orig_small.png')
# im2 = Image.open('')
# hash2 = imagehash.average_hash(Image.open('orig_large.png'))
# diff = hash1 - hash2
# print(diff)




# from utils import unpickle
# file_name = 'data_batch_1'
# file_location = 'images/CIFAR-10_raw/cifar-10-batches-py/'
# dic = unpickle(file_location + file_name)
# pass
