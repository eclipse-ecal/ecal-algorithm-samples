# ========================= eCAL LICENSE =================================
#
# Copyright (C) 2016 - 2019 Continental Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ========================= eCAL LICENSE =================================

import argparse
import os
import sys
import time

import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from ecal.core.publisher  import ProtoPublisher

from foxglove.RawImage_pb2 import RawImage
from foxglove.ImageAnnotations_pb2 import ImageAnnotations
from foxglove.PointsAnnotation_pb2 import PointsAnnotation
from foxglove.Point2_pb2 import Point2
from foxglove.Color_pb2 import Color

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image
import io

import numpy as np

def create_bounding_box(detection, timestamp) -> PointsAnnotation:
  bbox = detection.bounding_box
  bounding_box = PointsAnnotation()
  bounding_box.type = PointsAnnotation.LINE_LOOP
  lower_left = Point2(x=bbox.origin_x, y=bbox.origin_y)
  lower_right = Point2(x=bbox.origin_x + bbox.width, y=bbox.origin_y)
  upper_right = Point2(x=bbox.origin_x + bbox.width, y=bbox.origin_y + bbox.height)
  upper_left = Point2(x=bbox.origin_x, y=bbox.origin_y + bbox.height)
  bounding_box.points.append(lower_left)
  bounding_box.points.append(lower_right)
  bounding_box.points.append(upper_right)
  bounding_box.points.append(upper_left)
  bounding_box.thickness = 4
  bounding_box.timestamp.FromMicroseconds(timestamp)
  bounding_box.outline_color.r = 1.0
  bounding_box.outline_color.g = 0.647
  bounding_box.outline_color.b = 0.0
  bounding_box.outline_color.a = 1.0
  return bounding_box

def create_annotations(detection_result, timestamp) -> ImageAnnotations:
  annotations = ImageAnnotations()
  for detection in detection_result.detections:
    if (detection.categories[0].category_name == "car"):
      bounding_box = create_bounding_box(detection, timestamp)
      annotations.points.append(bounding_box)
  return annotations

class ImageClassifier(object):
  
  def __init__(self):
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
      
    self.options = ObjectDetectorOptions(
      base_options=BaseOptions(model_asset_path='./efficientdet_lite0.tflite'),
      running_mode=VisionRunningMode.VIDEO,
      max_results=15
      #result_callback=print_result
    )
    self.detector = ObjectDetector.create_from_options(self.options)

  def reset(self):
    ObjectDetector = mp.tasks.vision.ObjectDetector
    self.detector = ObjectDetector.create_from_options(self.options)

def main(args):
  args = parse_arguments()
  
  # print eCAL version and date
  print("eCAL {} ({})\n".format(ecal_core.getversion(),ecal_core.getdate()))
  
  # initialize eCAL API
  ecal_core.initialize(sys.argv, "annotate images")
  
  # set process state
  ecal_core.set_process_state(1, 1, "I feel good")

  # create subscriber and connect callback
  sub = ProtoSubscriber(args.input, RawImage)
  #sub.set_callback(callback)
  pub = ProtoPublisher(args.output, ImageAnnotations)
  
  classifier = ImageClassifier()
  last_time = 0

  while ecal_core.ok():
    ret, image, time = sub.receive(500)

    if ret > 0:
      if time < last_time:
        classifier.reset()

      last_time = time

      np_array = np.frombuffer(image.data, dtype=np.uint8)
      np_array = np.reshape(np_array, newshape=(image.height, image.width, 3))

      mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np_array)
       
      detection_result = classifier.detector.detect_for_video(mp_image, time)

      annotations = create_annotations(detection_result, image.timestamp.ToMicroseconds())
      pub.send(annotations)

  ecal_core.finalize()
  
def parse_arguments():
  parser = argparse.ArgumentParser(description="Application to detect vehicles in an Image")
  parser.add_argument("--input",  default="camera/cam_front_left")
  parser.add_argument('--output', default="annotations/cam_front_left")
  args = parser.parse_args()     
  return args
 
if __name__ == "__main__":
  main()  
