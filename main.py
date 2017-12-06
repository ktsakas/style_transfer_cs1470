import sys

import numpy as np
import scipy.misc
import imageio

from cnn import *

# Change this value to the path of wherever your VGG-19 CNN is
VGG_PATH = './imagenet-vgg-verydeep-19.mat'

def load_image(path):
    return imageio.imread(path).astype("float32")

if __name__ == "__main__":
    contentImg = load_image(sys.argv[1])
    styleImg = load_image(sys.argv[2])


    styleTransferNetwork = VGG19_CNN(VGG_PATH, contentImg, styleImg)
    styleTransferNetwork.reconstruct_content_from_layer("relu4_2")
