import tensorflow as tf
import scipy.io
import scipy.misc
import numpy as np
import png

class VGG19_CNN(object):

    def __init__(self, path):
        self.network = scipy.io.loadmat(path)
        self.layers = self.network["layers"][0]

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
            print "Error: No such convolutional layer named: " + l
            return -1

    def getFiltersFromLayer(self, l):
        kernals, bias = self.layers[self.convertConvolutionalLayerToIndex(l)][0][0][0][0]
        return kernals

    def extractFeaturesFromLayer(self, image, l):
        """
            Convolves an image through the VGG-19 CNN in order to extract
            the filter responses to that image in a particular layer.
            A layer with N_l distinct filters has N_l feature maps each of size M_l,
            where M_l is the height times the width of the feature map. So,
            the responses in a layer l can be stored in a (N_l x M_l) matrix F^l, where F_{i, j}^l
            is the activation of the ith filter at position j in layer l.
                @param image    An image (WxHx3) to convolve. The image is taken in as a Numpy array.
                @param l        A particular layer of the CNN
                @return F       The (N_l x M_l) matrix F^l holding the responses of layer l on the image
        """



        # Converting the image to a Tensor
        if not isinstance(image, tf.Tensor) and not isinstance(image, tf.Variable):
            image = tf.convert_to_tensor(image)

        image_dimensions = image.get_shape().as_list()

        # Default data_format of tf.conv2d is [batchSz, height, width, channels (i.e. colors = 3)]
        image = tf.reshape(image, [1] + image.get_shape().as_list())

        # Default data_format of filters in matconvnet is [width, height, in_channels, out_channels],
        # so we need to switch the 0th and 1st dimension.
        kernals, bias = self.layers[self.convertConvolutionalLayerToIndex(l)][0][0][0][0]
        kernals = np.transpose(kernals, (1, 0, 2, 3))

        # Remove the redundant dimension whose value is 1 anyways (i.e. convert to a vector)
        bias = bias.reshape(-1)

        image = tf.nn.conv2d(image, tf.constant(kernals), strides=[1, 1, 1, 1], padding='SAME')
        image = tf.nn.bias_add(image, bias)
        image = tf.reshape(image, [image_dimensions[0]*image_dimensions[1], 64])
        image = tf.transpose(image, [1, 0])


        return image


    def reconstruct_content_from_layer(self, image, l):
        # Number of filters in layer l
        N = self.getFiltersFromLayer(l).shape[3]

        # Size of each feature map in layer l
        M = image.shape[0] * image.shape[1]

        # We are performing convolution on x and in Tensorflow
        # tf.nn.conv2d takes in input as [batchSz, height, width, num_channels]
        x = tf.Variable(tf.random_normal(image.shape, stddev=.1))


        # F (NxM) contains the features of the image in layer l
        # We use F to construct an image similar in content from a random noise image
        F = self.extractFeaturesFromLayer(image, l)

        # P (NxM) contains the features of the random noise image in layer l
        P = self.extractFeaturesFromLayer(x, l)

        # Content Loss
        content_loss = tf.nn.l2_loss(F-P)

        # Training Step
        train = tf.train.AdamOptimizer(.001).minimize(content_loss)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            dim = x.get_shape().as_list()
            print dim
            for i in range(100):
                train.run()

                # Every time we train, create a new image to see the updates
                scipy.misc.imsave('newfile2.jpg', x.eval())

            # This will be the final image
            scipy.misc.imsave('newfile2.jpg', x.eval())


        print "Finished"
