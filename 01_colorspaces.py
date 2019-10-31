
# coding: utf-8

# # Assignment 1: Color Spaces, Morphological Operators

# ## Exercise 1.1

# For an image of your choice, implement the simple binarization method as shown in the lecture. We've put some example images in in /images.
# 
# Rough sketch:
# 
# 1. define the „positive“ subspace P in the RGB cube
# 2. iterate over all pixels in I and check if in P or ~P
# 3. write result to new image
# 4. play around with size and shape of P and display binary image (**RESULT**)
# 
# 

# In[2]:


from dataclasses import dataclass
from typing import Tuple, Union

from skimage import io, data, color
from skimage.util import img_as_ubyte
import numpy as np

image = io.imread('images/bottles2.png')
io.imshow(image)

print('shape =', image.shape)
# print(image)

if image.shape[2] == 4:
    image = img_as_ubyte(color.rgba2rgb(image))
    print('shape =', image.shape)
# print(image)


# def binarized(image, condition):
#     linear_shape = (1, image.shape[0]*image.shape[1], 3)
#     zeros = np.full(linear_shape, [0, 0, 0], dtype=np.uint8)
#     ones = np.where(image )
#     # zeros = zeros.reshape(zeros, (1, image.shape[0]*image.shape[1]))
#     print(zeros.shape)
#     binarized_linear = np.where([condition(px) for px in image.reshape(linear_shape)])
#     print(binarized_linear)
#     return binarized_linear.reshape(image.shape)


class RgbSubscriptable:
    def __getitem__(self, key):
        if key == 0:
            return self.r
        if key == 1:
            return self.g
        if key == 2:
            return self.b
        raise KeyError("Key must be an int and one of {0, 1, 2}.")


class Threshold(RgbSubscriptable):
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
    
    @classmethod
    def uniform(cls, n):
        return cls(n, n, n)


class Color(RgbSubscriptable):
    def __init__(self, r, g, b, threshold=None):
        self.r = r
        self.g = g
        self.b = b
        self.threshold = threshold 


@dataclass
class ColorSpace:
    colors: Tuple[Color] = None
    global_threshold: Threshold = Threshold.uniform(0)
    
    def contains(self, check_color):
        return any(self._is_similar(own_color, check_color) for own_color in self.colors)
    
    def _is_similar(self, own_color, check_color):
        local_threshold = own_color.threshold
        global_threshold = self.global_threshold
        
        def comp_similar(i):
            comp_own = own_color[i]
            comp_check = check_color[i]
            thresh = local_threshold[i] + global_threshold[i]
            return comp_own - thresh <= comp_check <= comp_own + thresh
    
        return all(
            comp_similar(i) 
            for i in range(0, 3)
        ) 


attempts_of_subspaces = (
    # 1st attempt: single color for each bottle
    (
        # bottle 1
        ColorSpace(
            colors=(
#                 Color(29, 177, 214, Threshold(10, 10, 100)),
#                 global_threshold=Threshold.uniform(50),
                Color(29, 177, 214),
                Color(30, 185, 214),
                Color(46, 192, 218),
                Color(84, 196, 219),
            ),
            global_threshold=Threshold.uniform(10),
        ),
#         [[29, 177, 214],],
        # bottle 2
        # [[234, 57, 147]],
        # bottle 3
        # [[215, 221, 59]],
    ),
)
# print(attempts_of_subspaces)
for subspaces in attempts_of_subspaces:
    for i, subspace in enumerate(subspaces):
        linear_shape = (image.shape[0]*image.shape[1], 3)
        linear_image = image.reshape(linear_shape)
        binarized_image = np.full(linear_shape, 0)
        ones = np.full(linear_shape[1:], 255)
        for i, pixel in enumerate(linear_image):
            if subspace.contains(pixel):
                binarized_image[i] = ones
        binarized_image = binarized_image.reshape(image.shape)
        io.imshow(binarized_image)


# ## Exercise 1.2

# * starting from the binary color detection image
# * erase noise with an erosion operation
# * dilate once to get original size of object
# * find connected components with the two-pass algorithm
# * extract bounding box on the fly
# * draw bounding box on original image (**RESULT**)

# ## Exercise 1.3

# * use your color detection and connected components algorithm
# * implement simplest tracking algorithm
# * draw history of all previous points on frame (**RESULT**)
# 
# (see images/racecar or images/taco for sample image sequences)
