import argparse
import os
import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, BaseOptions

import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from ecal.core.publisher import ProtoPublisher

from ros.sensor_msgs.Image_pb2 import Image
from ros.visualization_msgs.ImageMarker_pb2 import ImageMarker
from ros.geometry_msgs.Point_pb2 import Point
from ros.std_msgs.Time_pb2 import Time
from ros.std_msgs.ColorRGBA_pb2 import ColorRGBA
from gesture_recognition.RecognizedGesture_pb2 import Gesture


import numpy as np


def create_hand_gesture(detection: mp.tasks.vision.GestureRecognizerResult, timestamp: Time, height: int, width: int):
    handConnections = ImageMarker()
    recognizedGestureAndHand = Gesture()
    # create color for the lines of the hand, r(ed), b(lue), g(reen), a(lpha)-values assigned
    color_orange = ColorRGBA()
    color_orange.r = 1.0
    color_orange.g = 0.647
    color_orange.b = 0.0
    color_orange.a = 1.0

    handConnections.header.stamp.nsec = timestamp.nsec
    handConnections.header.stamp.sec = timestamp.sec
    handConnections.type = 2 # LINE_LIST, there are also other types, but they're not useful in this case
    handConnections.scale = 3 # thickness of lines

    # check if mediapipe detected a hand in image
    if detection.gestures:
        points = []
        hand = detection.hand_landmarks[0]
        gesture = detection.gestures[0][0]
        handedness = detection.handedness[0][0]
        recognizedGestureAndHand.recognizedGesture = recognizedGestureAndHand.Category.Value(gesture.category_name)
        recognizedGestureAndHand.recognizedHand = recognizedGestureAndHand.Hand.Value(handedness.category_name)

        # points are just in a relative size; they need to be scaled up by the size of image
        for point in hand:
            new_point = Point()
            new_point.x = point.x * width
            new_point.y = point.y * height
            points.append(new_point)

            handConnections.outline_colors.append(color_orange)

        # every two points create one line
        handConnPoints = [points[0], points[1], points[1], points[2], points[2], points[3], points[3], points[4],
                                  points[0], points[5], points[5], points[6], points[6], points[7], points[7], points[8],
                                  points[5], points[9], points[9], points[10], points[10], points[11], points[11], points[12],
                                  points[9], points[13], points[13], points[14], points[14], points[15], points[15], points[16],
                                  points[13], points[17], points[17], points[0], points[17], points[18], points[18], points[19], points[19], points[20]]

        for point in handConnPoints:
            handConnections.points.append(point)

    # if no hand was detected on the image
    else:
        # create one point with 0, 0 coordinates and one without coordinates so no line will be created
        new_point = Point()
        new_point2 = Point()
        new_point.x = 0.0
        new_point.y = 0.0
        handConnections.points.append(new_point)
        handConnections.points.append(new_point2)
        handConnections.outline_colors.append(color_orange)

    return handConnections, recognizedGestureAndHand


class GestureClassifier(object):
    def __init__(self):
        base_options = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        model_path = os.path.join(os.path.dirname(__file__), 'gesture_recognizer.task')

        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )
        self.detector = GestureRecognizer.create_from_options(self.options)

    # set Recognizer to basic settings
    def reset(self):
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.detector = GestureRecognizer.create_from_options(self.options)


def main():
    args = parse_arguments()
    
    # print eCAL version and date
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))

    # initialize eCAL API
    ecal_core.initialize(sys.argv, "recognize gestures")

    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")

    # create subscriber and connect callback
    sub = ProtoSubscriber(args.input, Image)
    pub1 = ProtoPublisher(args.output, ImageMarker)
    pub2 = ProtoPublisher(args.output_gesture, Gesture)

    classifier = GestureClassifier()
    last_time = 0

    while ecal_core.ok():
        ret, image, time = sub.receive(500)
        if ret > 0:
            if time < last_time:
                classifier.reset()

            last_time = time

            # create numpy array with imagedata in uint8 to make it usable for mediapipe
            np_array = np.frombuffer(image.data, dtype=np.uint8)
            np_array = np.reshape(np_array, newshape=(image.height, image.width, 3))

            # mediapipe image created out of numpy array data
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np_array)

            # result data of recognition from mediapipe is saved in recognition_result
            recognition_result = classifier.detector.recognize_for_video(mp_image, time)
            handConnections, recognizedGestureAndHand = create_hand_gesture(recognition_result, image.header.stamp, image.height, image.width)
            if handConnections.points:
                # send handconnections and the recognized hand/gesture via Publisher
                pub1.send(handConnections)
                pub2.send(recognizedGestureAndHand)
    ecal_core.finalize()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Application to recognize gestures in an image")
    parser.add_argument("--input",  default="camera/Webcam")
    parser.add_argument('--output', default="annotations/Webcam")
    parser.add_argument('--output_gesture', default="gestures/Webcam")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
