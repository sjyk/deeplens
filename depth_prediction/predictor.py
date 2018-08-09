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


def process_image(image_path, width, height):
    """
    Process the image (e.g. resize).

    :param image_path: the path to the image
    :param width: the width of the image expected by the model
    :param height: the height of the image expected by the model
    :return: the processed and resized images (the resized image can be used to
    plot is as an original input to the model)
    """
    img = Image.open(image_path)
    resized_img = img.resize([width, height], Image.ANTIALIAS)
    processed_img = np.array(resized_img).astype('float32')
    return processed_img, resized_img


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

        self._load_model_weights()

    def __del__(self):
        """
        Close the TensorFlow session.
        """
        self.session.close()

    def _load_model_weights(self):
        # Load the converted parameters
        logger.info('Loading pre-trained model weights')

        if self.model_path.endswith(".npy"):
            # load the pre-trained model from npy file
            if os.path.isfile(self.model_path) is False:
                model_url = "https://goo.gl/dt2geQ"
                logger.info("Downloading the model weights ... (the file is "
                            "about 243 MB big)")
                self.model_path = urlretrieve(model_url, self.model_path)
                logger.info("Model downloaded successfully")
            self.net.load(self.model_path, self.session)
        elif self.model_path.endswith(".ckpt"):
            # load the pre-trained model from ckpt file
            saver = tf.train.Saver()
            saver.restore(self.session, self.model_path)
        else:
            raise ValueError("Not known saved model weights' extension: "
                             "Expected: .npy or .ckpt, but given file: " +
                             self.model_path)

    def predict_image(self, image_path):
        """
        For the provided image_path, returns map of the predicted depths.

        :param image_path: the path to the input image
        :return: map of the predicted depths and the re-sized input image
        """
        processed_img, resized_img = process_image(
            image_path=image_path, width=self.width, height=self.height)
        img = np.expand_dims(np.asarray(processed_img), axis=0)

        # Evaluate the network for the given image
        predicted_depth = self.session.run(self.net.get_output(),
                                           feed_dict={self.input_node: img})

        return predicted_depth[0, :, :, 0], resized_img


def main():
    """
    Predict the images.
    """
    predictor = Predictor(args.model_path)
    predicted_depth, input_img = predictor.predict_image(args.image_path)

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

    logger.debug("current working directory: " + os.getcwd())

    # predict the depth maps
    main()
