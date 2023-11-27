import argparse
import os
import sys
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, BaseOptions

import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from ecal.core.publisher import ProtoPublisher

from ros.sensor_msgs.Image_pb2 import Image
from ros.std_msgs.Time_pb2 import Time

import numpy as np

import cv2
import math


def segment_image(category_mask, image):
    ros_output_image = Image()
    t = time.time()
    # assign basic data to ros image
    ros_output_image.header.stamp.sec = int(t)
    ros_output_image.header.stamp.nsec = 0
    ros_output_image.encoding = "bgr8"

    background_color = (192, 192, 192)  # gray
    mask_color = (255, 255, 255)  # white

    image_data = image.numpy_view()
    # create foreground image
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = mask_color
    # create background image
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = background_color

    # create condition from categorisized mask whether foreground or background is needed in a pixel
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
    output_image = np.where(condition, fg_image, bg_image)
    output_image_as_uint8 = output_image.astype(np.uint8)

    # ros image is in bytes, so foxglove can read it
    ros_output_image.data = output_image_as_uint8.tobytes()
    ros_output_image.width = image.width
    ros_output_image.height = image.height
    return ros_output_image


class ImageSegmenter(object):
    def __init__(self):
        base_options = mp.tasks.BaseOptions
        SelfieSegmenter = mp.tasks.vision.ImageSegmenter
        SelfieSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        model_path = os.path.join(os.path.dirname(__file__), 'selfie_segmenter_landscape.tflite')

        self.options = SelfieSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_category_mask=True
        )
        self.segmenter = SelfieSegmenter.create_from_options(self.options)

    # set Segmenter to basic settings
    def reset(self):
        SelfieSegmenter = mp.tasks.vision.ImageSegmenter
        self.segmenter = SelfieSegmenter.create_from_options(self.options)


def main():
    args = parse_arguments()
    
    # print eCAL version and date
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))

    # initialize eCAL API
    ecal_core.initialize(sys.argv, "segment images")

    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")

    # create subscriber and connect callback
    sub = ProtoSubscriber(args.input, Image)
    # sub.set_callback(callback)
    pub1 = ProtoPublisher(args.output, Image)

    segmenter = ImageSegmenter()
    last_time = 0

    while ecal_core.ok():
        ret, image, time_from_sub = sub.receive(500)
        if ret > 0:
            if time_from_sub < last_time:
                segmenter.reset()

            last_time = time_from_sub

            # create np array with datatype uint8 so it can be castet to bytes in segment_image function
            np_array = np.frombuffer(image.data, dtype=np.uint8)
            np_array = np.reshape(np_array, newshape=(image.height, image.width, 3))

            # create a mediapipe image, so the algorithm is able to check which class every pixel belongs to
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np_array)

            # category mask saves the segmented fields of the image by mediapipe
            category_mask = segmenter.segmenter.segment_for_video(mp_image, time_from_sub).category_mask
            segmentation = segment_image(category_mask, mp_image)
            # send the segmented image via Publisher
            pub1.send(segmentation)
    ecal_core.finalize()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Application to segment images.")
    parser.add_argument("--input",  default="camera/Webcam")
    parser.add_argument('--output', default="segments/Webcam")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
