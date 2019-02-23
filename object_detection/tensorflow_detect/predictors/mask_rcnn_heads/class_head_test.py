# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.predictors.mask_rcnn_heads.class_head."""
import tensorflow as tf

from google.protobuf import text_format
from object_detection.tensorflow_detect.builders import hyperparams_builder
from object_detection.tensorflow_detect.predictors.mask_rcnn_heads import class_head
from object_detection.tensorflow_detect.protos import hyperparams_pb2
from object_detection.tensorflow_detect.utils import test_case


class ClassHeadTest(test_case.TestCase):

  def _build_arg_scope_with_hyperparams(self,
                                        op_type=hyperparams_pb2.Hyperparams.FC):
    hyperparams = hyperparams_pb2.Hyperparams()
    hyperparams_text_proto = """
      activation: NONE
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Merge(hyperparams_text_proto, hyperparams)
    hyperparams.op = op_type
    return hyperparams_builder.build(hyperparams, is_training=True)

  def test_prediction_size(self):
    class_prediction_head = class_head.ClassHead(
        is_training=False,
        num_classes=20,
        fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(),
        use_dropout=True,
        dropout_keep_prob=0.5)
    roi_pooled_features = tf.random_uniform(
        [64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    prediction = class_prediction_head.predict(
        roi_pooled_features=roi_pooled_features)
    tf.logging.info(prediction.shape)
    self.assertAllEqual([64, 1, 21], prediction.get_shape().as_list())


if __name__ == '__main__':
  tf.test.main()
