import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from urllib.request import urlretrieve
import depth_prediction.models as models
import utils

logger = utils.get_logger(name=__name__)

DEFAULT_MODEL_PATH = "NYU_ResNet-UpProj.npy"

"""
author: Adam Dziedzic ady@uchicago.edu

based on: https://github.com/iro-cp/FCRN-DepthPrediction
"""


def show_figure(img):
    """
    Plot the image as a figure.

    :param img: the input image
    """
    fig = plt.figure()

    input_img = plt.imshow(img, interpolation='nearest')
    fig.colorbar(input_img)

    plt.show()


class Predictor(object):
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        """
        Provides inference on arbitrary images.

        :param model_path: path to the model weights
        """
        super(Predictor, self).__init__()
        self.model_path = model_path
        # Default input sizes
        self.height = 228
        self.width = 304
        self.channels = 3
        self.batch_size = 1
        # Create a placeholder for the input image
        self.input_node = tf.placeholder(tf.float32,
                                         shape=(None, self.height, self.width,
                                                self.channels))
        # Construct the network
        self.net = models.ResNet50UpProj(inputs={'data': self.input_node},
                                         batch=self.batch_size,
                                         keep_prob=1, is_training=False)
        self.session = tf.Session()

        # Load the converted parameters
        logger.info('Loading the model')

        if model_path.endswith(".npy"):
            # load the pre-trained model from npy file
            if os.path.isfile(model_path) is False:
                model_url = "https://goo.gl/dt2geQ"
                logger.info("Downloading the model weights ... (the file is "
                            "about 240 MB big)")
                self.model_path = urlretrieve(model_url, model_path)
                logger.info("Model downloaded successfully")
            self.net.load(self.model_path, self.session)
        elif model_path.endswith(".ckpt"):
            # load the pre-trained model from ckpt file
            saver = tf.train.Saver()
            saver.restore(self.session, model_path)
        else:
            raise ValueError("Not known saved model weights' extension: "
                             "Expected: .npy or .ckpt, but given file: " +
                             model_path)

    def __del__(self):
        self.session.close()

    def predict(self, image_path):
        """
        For the provided image_path, returns map of the predicted depths.

        :param image_path: the path to the input image
        :return: map of the predicted depths and the re-sized input image
        """
        # Read image
        img = Image.open(image_path)
        img = img.resize([self.width, self.height], Image.ANTIALIAS)
        input_image = img
        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis=0)

        # Evaluate the network for the given image
        predicted_depth = self.session.run(self.net.get_output(),
                                           feed_dict={self.input_node: img})

        return predicted_depth[0, :, :, 0], input_image


def main():
    # Predict the images
    predictor = Predictor(args.model_path)
    predicted_depth, input_img = predictor.predict(args.image_path)

    # Plot result
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    input_img = plt.imshow(input_img, interpolation='nearest')
    fig.colorbar(input_img)

    fig.add_subplot(1, 2, 2)
    prediction_img = plt.imshow(predicted_depth,
                                interpolation='nearest')
    fig.colorbar(prediction_img)

    plt.show()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path',
                        help='Converted parameters for the model',
                        default=DEFAULT_MODEL_PATH)
    parser.add_argument('-i', '--image_path', help='Image path or directory of '
                                                   'images to predict')
    parser.add_argument("-l", "--log_file", default="deeplens.log",
                        help="The name of the log file.")
    args = parser.parse_args(sys.argv[1:])
    log_file = args.log_file
    utils.set_up_logging(log_file=log_file)
    logger = utils.get_logger(name=__name__)

    # predict the depth maps
    main()
