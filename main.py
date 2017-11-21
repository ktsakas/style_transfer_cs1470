import sys

import numpy as np
import scipy.misc
import imageio
# import scipy.io
# import tensorflow as tf

from cnn import *

VGG_PATH = '../sample_capstone/style-transfer-master/imagenet-vgg-verydeep-19.mat'

def imread(path):
    return imageio.imread(path).astype(np.float)

def load_image(image_path):
    img = imread(image_path)
    img = img.astype("float32")
    # img = np.ndarray.reshape(img, img.shape)
    return img


if __name__ == "__main__":
    content_img = load_image(sys.argv[1])
    style_img = load_image(sys.argv[2])


    vgg19_network = VGG19_CNN(VGG_PATH)


    # vgg19_network.extractFeatureFromLayer(content_img, "conv1_1")
    vgg19_network.reconstruct_content_from_layer(content_img, "conv1_1")

    # Reference code to convert to grayscale
    # g = tf.image.rgb_to_grayscale(content_img)
    # print g.get_shape()

    white_noise_image = tf.Variable(tf.random_normal([content_img.shape[0], content_img.shape[1]], stddev=.1))
