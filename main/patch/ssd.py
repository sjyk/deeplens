from main.patch.core import PatchGenerator
from main.io import Patch

from object_detection.tensorflow_detect.utils import label_map_util

import numpy as np

import tensorflow as tf


class SSDPatchGenerator(PatchGenerator):

    def __init__(self, model_file, label_file, num_classes, confidence=0.50):
        self.model_file = model_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.graph = self.loadGraph()
        self.category_index = self.getLabelMaps()
        self.confidence = confidence

    def loadGraph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(self.model_file + "/frozen_inference_graph.pb", 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def getLabelMaps(self):
        label_map = label_map_util.load_labelmap(self.label_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def run_inference_for_single_image(self, image):
      graph = self.graph
      with graph.as_default():
        with tf.Session() as sess:
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, self.confidence), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict

    def generatePatches(self, imgref, img):

        detection_dict = self.run_inference_for_single_image(img)
        height, width, channels = img.shape

        for i in range(0, detection_dict['num_detections']):
            if detection_dict['detection_scores'][i] > self.confidence:
                y0 = int(detection_dict['detection_boxes'][i][0]*height)
                y1 = int(detection_dict['detection_boxes'][i][2]*height)

                x0 = int(detection_dict['detection_boxes'][i][1]*width)
                x1 = int(detection_dict['detection_boxes'][i][3]*width)

                patch = img[y0:y1,x0:x1,:]
                metadata = {'tag': self.category_index[detection_dict['detection_classes'][i]]['name']}
                yield Patch(imgref, x0,y0, x1-x0, y1-y0, patch, metadata)




