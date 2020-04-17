import random
import numpy as np
from skimage import  transform, data


def rotate(img):
    rotate_degree = random.randint(1, 360)
    return  transform.rotate(img, rotate_degree, mode='reflect')

def HorizontallyFlip(img):
    img = np.flip(img,axis=2)
    return img

def VerticallyFlip(img):
    img = np.flip(img, axis=1)
    return img

