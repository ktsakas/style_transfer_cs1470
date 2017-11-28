import sys

import numpy as np
import scipy.misc
import imageio

from cnn import *

# Change this value to the path of wherever your VGG-19 CNN is
VGG_PATH = './imagenet-vgg-verydeep-19.mat'

def imread(path):
    return imageio.imread(path).astype(np.float)

def load_image(image_path):
    img = imread(image_path)
    img = img.astype("float32")
    return img


if __name__ == "__main__":
    content_img = load_image(sys.argv[1])
    style_img = load_image(sys.argv[2])


    vgg19_network = VGG19_CNN(VGG_PATH, content_img, style_img)


    vgg19_network.reconstruct_content_from_layer("relu4_2")
