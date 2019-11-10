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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import tensorflow as tf
import os

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)


  
  template = "{} (score={:0.5f})"
  
  for i in top_k:
    print(template.format(labels[i], results[i]))
    num=i
    health=labels[i]+" with probability "+str(results[i])
    break
  print("leaf status:",health)

  imagemat=cv2.imread("frame0.jpg",1)
  dim=(640,480)
  image=cv2.resize(imagemat,dim, interpolation = cv2.INTER_AREA)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(image,health,(10,60), font, 1,(255,255,255),2,cv2.LINE_AA)
  cv2.imshow("Captured Image",image)
  cv2.waitKey()
  '''
  filename="leafpad "+str(num)+".txt"
  os.system(filename)  
  '''
  if num==0:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==1:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==2:
    pass
  elif num==3:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==4:
    pass
  elif num==5:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==6:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==7:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==8:
    pass
  elif num==9:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==10:
    pass
  elif num==11:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==12:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==13:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==14:
    pass
  elif num==15:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==16:
    pass
  elif num==17:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==18:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==19:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==20:
    pass
  elif num==21:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)
  elif num==22:
    pass
  elif num==23:
    filename="leafpad "+str(num)+".txt"
    os.system(filename)

