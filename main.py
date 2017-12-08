import sys
import numpy as np
import imageio
from cnn import *
import skimage.transform

# Change this value to the path of wherever your VGG-19 CNN is
VGG_PATH = './imagenet-vgg-verydeep-19.mat'

def load_image(path):
    return imageio.imread(path).astype("float32")


if __name__ == "__main__":
    contentImg = load_image(sys.argv[1])
    styleImg = scipy.misc.imresize(load_image(sys.argv[2]), contentImg.shape).astype("float32")


    styleTransferNetwork = VGG19_CNN(VGG_PATH, contentImg, styleImg)
    styleTransferNetwork.reconstruct_content_from_layer("conv2_2")
