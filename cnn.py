import tensorflow as tf
import scipy.io
import scipy.misc
import numpy as np
import imageio
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

CONTENT_W = 1
STYLE_W = 100
LEARNING_RATE = 10
ITERATIONS = 200
STYLING_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

class VGG19_CNN():

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
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "conv5_4",
        "relu5_4"
    ]

    def __init__(self, path, content, artwork):
        self.network = scipy.io.loadmat(path)
        self.layers = self.network["layers"][0]
        self.mean_pixel = np.mean(self.network['normalization'][0][0][0], axis=(0, 1))
        self.content = content
        self.artwork = artwork
        self.image = tf.Variable(tf.random_normal(self.content.shape))
        self.vgg_net = self.convolve(self.image)

    def preprocess(self, image):
        return (image - self.mean_pixel).astype(np.float32)

    def convolve(self, image):
        """
            Convolves an image through the VGG-19 CNN in order to extract
            the filter responses to that image at all layers in the network.
                @param image            An image (HxWx3) to convolve.
                @param l                A particular layer of the CNN
                @return feature_maps    Where feature_maps[l] contains the feature maps of layer l
        """

        # Default data_format of tf.conv2d is [batchSz, height, width, channels (i.e. colors = 3)]
        feature_maps = dict()
        conv_input = tf.reshape(image, [1] + image.shape.as_list())

        # Convolve image through all layers of the VGG-19 Neural Network
        for i, layer in enumerate(VGG19_CNN.architecture):
            if layer[0] == 'c':
                kernels, bias = self.layers[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                conv_input = tf.nn.conv2d(conv_input, tf.constant(kernels), strides=[1, 1, 1, 1], padding='SAME')
                conv_input = tf.nn.bias_add(conv_input, bias)
            elif layer[0] == 'r':
                conv_input = tf.nn.relu(conv_input)
            elif layer[0] == 'p':
                conv_input = tf.nn.avg_pool(conv_input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

            feature_maps[layer] = conv_input


        return feature_maps

    def style_loss(self):
        style_input = tf.placeholder('float', shape=self.artwork.shape)
        style_feature_maps = self.convolve(style_input)

        avg_w = 1 / len(STYLING_LAYERS)
        with tf.Session() as sess:
            loss = 0
            for layer in STYLING_LAYERS:
                style_feature = self.vgg_net[layer]
                _, height, width, number = style_feature.get_shape().as_list()
                size = height * width * number
                style_gram_matrix = self.compute_gram_matrix(style_feature_maps, style_input, layer)
                input_gram_matrix = self.compute_gram_matrix_for_layer(style_feature)

                loss += avg_w * (tf.nn.l2_loss(input_gram_matrix - style_gram_matrix) / (style_gram_matrix.size * (size ** 2)))

        return STYLE_W * loss

    def compute_gram_matrix(self, style_net, artwork_placeholder, layer):
        image_feature = style_net[layer].eval(feed_dict={artwork_placeholder: self.preprocess(self.artwork)})
        image_feature = np.reshape(image_feature, (-1, image_feature.shape[3]))
        return np.matmul(image_feature.T, image_feature)

    def compute_gram_matrix_for_layer(self, image_feature):
        number = image_feature.get_shape()[3]
        image_feature = tf.reshape(image_feature, (-1, number))
        return tf.matmul(tf.transpose(image_feature), image_feature)

    def reconstruct_content_from_layer(self, l):
        content_image = tf.placeholder('float', shape=self.content.shape)
        feature_maps_original = self.convolve(content_image)

        # We are performing convolution on x and in Tensorflow
        # tf.nn.conv2d takes in input as [batchSz, height, width, num_channels]
        feature_maps_noise = self.convolve(self.image)
        with tf.Session() as sess:
            # P (NxM) contains the features of the random noise image in layer l
            P = feature_maps_original[l].eval(feed_dict = {content_image: self.preprocess(self.content)})

            # F (NxM) contains the features of the image in layer l
            # We use F to construct an image similar in content from a random noise image
            F = feature_maps_noise[l]

            # Content Loss
            content_loss = CONTENT_W * (tf.nn.l2_loss(F-P) / P.size)

            style_loss = self.style_loss()

            total_loss = content_loss + style_loss

            # Training Step
            train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss)
            tf.summary.FileWriter('./train', sess.graph)

            sess.run(tf.global_variables_initializer())
            for i in range(ITERATIONS):
                train.run()
                print("Iteration: %d -- (style loss) %d + (content loss) %d = %d" % (i, style_loss.eval(), content_loss.eval(), total_loss.eval()))
                # print(x.eval().shape, self.mean_pixel.shape)
                t = self.image.eval() + self.mean_pixel
                yield t

