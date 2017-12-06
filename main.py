import sys
import numpy as np
import imageio
from cnn import *

# Change this value to the path of wherever your VGG-19 CNN is
VGG_PATH = './imagenet-vgg-verydeep-19.mat'

def load_image(path):
    return imageio.imread(path).astype("float32")

def save_image(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(path, img)

if __name__ == "__main__":
    contentImg = load_image(sys.argv[1])
    styleImg = load_image(sys.argv[2])


    styleTransferNetwork = VGG19_CNN(VGG_PATH, contentImg, styleImg)
    for img in styleTransferNetwork.reconstruct_content_from_layer("relu4_2"):
    	save_image("./output/live-training.jpg", img)

    save_image("./output/final.jpg", img)

