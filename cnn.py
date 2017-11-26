import tensorflow as tf
import scipy.io
import scipy.misc
import numpy as np
import png

class VGG19_CNN(object):

    architecture = [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool_1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool_2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "conv3_4",
        "relu3_4",
        "pool_3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "conv4_4",
        "relu4_4",
        "pool_4",
        "conv5_1",
        "relu5_1",
    ]

    def __init__(self, path):
        self.network = scipy.io.loadmat(path)
        self.layers = self.network["layers"][0]
        self.mean_pixel = np.mean(self.network['normalization'][0][0][0], axis=(0, 1))

    def preprocess(self, image):
        return (image - self.mean_pixel).astype(np.float32)

    def convertConvolutionalLayerToIndex(self, l):
        """
            Given the name of a convolutional layer in VGG-19, return the corresponding index
            in the pre-trained matconvnet CNN.
                @param l    Name of a convolutional layer in VGG-19.
                @return i   The corresponding index in the pre-trained matconvnet CNN.
        """
        if l == "conv1_1":
            return 0
        elif l == "conv1_2":
            return 2
        elif l == "conv2_1":
            return 5
        elif l == "conv2_2":
            return 7
        elif l == "conv3_1":
            return 10
        elif l == "conv3_2":
            return 12
        elif l == "conv3_3":
            return 14
        elif l == "conv3_4":
            return 16
        elif l == "conv4_1":
            return 19
        elif l == "conv4_2":
            return 21
        elif l == "conv4_3":
            return 23
        elif l == "conv4_4":
            return 25
        elif l == "conv5_1":
            return 28
        elif l == "conv5_2":
            return 30
        elif l == "conv5_3":
            return 32
        elif l == "conv5_4":
            return 34
        else:
            print("Error: No such convolutional layer named: " + l)
            return -1

    def getFiltersFromLayer(self, l):
        kernals, bias = self.layers[self.convertConvolutionalLayerToIndex(l)][0][0][0][0]
        return kernals


    def convolve(self, image):
        """
            Convolves an image through the VGG-19 CNN in order to extract
            the filter responses to that image at all layers in the network.
                @param image            An image (HxWx3) to convolve.
                @param l                A particular layer of the CNN
                @return feature_maps    Where feature_maps[l] contains the feature maps of layer l
        """

        # Converting the image to a Tensor
        if not isinstance(image, tf.Tensor) and not isinstance(image, tf.Variable):
            image = tf.convert_to_tensor(image)

        image_dimensions = image.get_shape().as_list()

        # Default data_format of tf.conv2d is [batchSz, height, width, channels (i.e. colors = 3)]
        image = tf.reshape(image, [1] + image.get_shape().as_list())

        feature_maps = dict()

        conv_input = image

        # Convolve image through all layers of the VGG-19 Neural Network
        for i, layer in enumerate(VGG19_CNN.architecture):
            if layer[0] == 'c':
                kernals, bias = self.layers[i][0][0][0][0]
                kernals = np.transpose(kernals, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                conv_input = tf.nn.conv2d(conv_input, tf.constant(kernals), strides=[1, 1, 1, 1], padding='SAME')
                conv_input = tf.nn.bias_add(conv_input, bias)
            elif layer[0] == 'r':
                conv_input = tf.nn.relu(conv_input)
            elif layer[0] == 'p':
                conv_input = tf.nn.avg_pool(conv_input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

            feature_maps[layer] = conv_input


        return feature_maps

    def reconstruct_content_from_layer(self, image, l):
        feature_maps_original = self.convolve(self.preprocess(image))

        # We are performing convolution on x and in Tensorflow
        # tf.nn.conv2d takes in input as [batchSz, height, width, num_channels]
        x = tf.Variable(tf.random_normal(image.shape, stddev=.1))
        feature_maps_noise = self.convolve(x)

        # P (NxM) contains the features of the random noise image in layer l
        P = feature_maps_original[l]

        # F (NxM) contains the features of the image in layer l
        # We use F to construct an image similar in content from a random noise image
        F = feature_maps_noise[l]

        # Content Loss
        content_loss = tf.nn.l2_loss(F-P)

        # Training Step
        train = tf.train.AdamOptimizer(10).minimize(content_loss)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                train.run()
                print(i)
                scipy.misc.imsave('subtractPixel.jpg', x.eval() + self.mean_pixel)


            scipy.misc.imsave('newfile2.jpg', x.eval())



        print("Finished")
