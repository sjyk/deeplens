import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from urllib.request import urlretrieve
import depth_prediction.models as models
from main import utils
import glob
import logging

logger = utils.get_logger(name=__name__)

DEFAULT_MODEL_PATH = "../resources/models/depth_prediction/NYU_" \
                     "ResNet-UpProj.npy"
DEFAULT_IMAGES_PATH = "../resources/demo/image.jpg"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LOG_FILE = "deeplens_depth_maps.log"

IS_DEBUG = False

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


def process_images(images_path, width, height):
    """
    Process the image (e.g. resize).

    :param images_path: the path to the directory with images
    :param width: the width of the image expected by the model
    :param height: the height of the image expected by the model
    :return: the processed and resized images (the resized image can be used to
    plot is as an original input to the model)
    """
    processed_imgs = []
    resized_imgs = []
    for image_path in images_path:
        img = Image.open(image_path)
        resized_img = img.resize([width, height], Image.ANTIALIAS)
        resized_imgs.append(resized_img)
        processed_img = np.array(resized_img).astype('float32')
        processed_imgs.append(processed_img)
    if logger.level == logging.DEBUG:
        for image in resized_imgs:
            show_figure(image)
    return processed_imgs, resized_imgs


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
        """
        Load the pre-trained model parameters (weights).
        """
        logger.info('Loading pre-trained model weights')

        if self.model_path.endswith(".npy"):
            logger.debug("load the pre-trained model from npy file")
            if os.path.isfile(self.model_path) is False:
                # extract model folder by removing model file name from the end
                # of the path
                model_folder = self.model_path[:self.model_path.rfind("/")]
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                model_url = "https://goo.gl/dt2geQ"
                logger.info(
                    "Downloading the model weights to " + model_folder +
                    "... (the file is about 243 MB big)")
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
        logger.debug("model loaded")

    def predict_depth(self, images_path, batch_size=DEFAULT_BATCH_SIZE):
        """
        For each image return its depth map.

        For the provided paths to images (it can be a single image path) returns
        maps of the predicted depths.

        :param images_path: path to the directory with images (it can be also
        a single path to an image)
        :param batch_size: batch size for the image inference
        :return: map of the predicted depths and the re-sized input images
        """
        logger.debug("predict depth")
        if os.path.isfile(images_path):
            images_path = [images_path]
        else:
            # extract the whole content from the directory
            images_path = glob.glob(images_path + "/*")
            # retain only files
            images_path = [image_path for image_path in images_path if
                           os.path.isfile(image_path)]

        processed_imgs, resized_imgs = process_images(
            images_path=images_path, width=self.width, height=self.height)
        processed_imgs = np.array(processed_imgs)

        # adjust the batch size
        self.net.batch_size = min(len(images_path), batch_size)

        # Evaluate the network for the given images
        predicted_depths = self.session.run(
            self.net.get_output(), feed_dict={self.input_node: processed_imgs})

        return predicted_depths[..., 0], resized_imgs


def main(model_path=DEFAULT_MODEL_PATH, images_path=DEFAULT_IMAGES_PATH):
    """
    Main and self-contained method to predict the images.

    :param model_path: the path to the pre-trained model parameters
    :param images_path: the path to the directory with images or a single image
    path
    """
    predictor = Predictor(model_path)

    predicted_depths, resized_imgs = predictor.predict_depth(
        images_path=images_path)

    # Plot result
    fig = plt.figure()
    columns = 2
    rows = len(resized_imgs)
    logger.debug("number of processed images:" + str(rows))
    for counter in range(rows):
        first_index = 2 * counter + 1
        ax1 = fig.add_subplot(rows, columns, first_index)
        input_img = plt.imshow(resized_imgs[counter], interpolation='nearest')
        fig.colorbar(input_img)
        ax1.title.set_text("resized input image " + str(counter))

        second_index = 2 * counter + 2
        ax2 = fig.add_subplot(rows, columns, second_index)
        prediction_img = plt.imshow(predicted_depths[counter],
                                    interpolation='nearest')
        fig.colorbar(prediction_img)
        ax2.title.set_text("depth map " + str(counter))

    fig.tight_layout()
    img_folder = images_path[:images_path.rfind("/")]
    depth_folder = img_folder + "/image_depths/"
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)
    fig.savefig(depth_folder + "/image_depths.png")
    plt.show(block=True)
    plt.close(fig)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', default=DEFAULT_MODEL_PATH,
                        help='Converted parameters for the model')
    parser.add_argument('-i', '--image_path', default=DEFAULT_IMAGES_PATH,
                        help='Image path or directory of images to predict '
                             'their depth maps')
    parser.add_argument("-l", "--log_file", default=DEFAULT_LOG_FILE,
                        help="The name of the log file.")
    parser.add_argument("-b", "--batch_size", default=DEFAULT_BATCH_SIZE,
                        type=int, help="the batch size for inference")
    parser.add_argument("-g", "--is_debug", default=IS_DEBUG, type=bool,
                        help="is it the debug mode execution")

    args = parser.parse_args(args=sys.argv[1:])
    log_file = args.log_file
    IS_DEBUG = args.is_debug

    utils.set_up_logging(log_file=log_file, is_debug=args.is_debug)
    logger = utils.get_logger(name=__name__)
    if IS_DEBUG:
        logger.setLevel(logging.DEBUG)

    logger.debug("current working directory: " + os.getcwd())

    # predict the depth maps
    main(model_path=args.model_path, images_path=args.image_path)
