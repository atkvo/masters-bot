#!/usr/bin/env python

from std_msgs.msg import String
#import roslib
import sys

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2

import rospy
from autobot.msg import drive_param

from sensor_msgs.msg import Image

bridge = CvBridge() 

i = 0
bridge = CvBridge(); 
currentImage = 0
currentAngle = 0

def __init__(self):
    bridge = CvBridge(); 

def callback(temp):
    print("current image updated")
    global currentImage
    try:
        currentImage = bridge.imgmsg_to_cv2(temp, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)
    print("picture taken")
    global currentAngle
    global i
    filepath = "dataset/" + str(currentAngle) + "_" + str(i) + ".png"
    i+=1
    cv2.imwrite(filepath, currentImage) 

def takePicture(data):
    
    #define file path
    print("picture taken")
    global currentAngle
    currentAngle = data.angle
    global i
    filepath = "dataset/" + str(currentAngle) + "_" + str(i) + ".png"
    i+=1
    cv2.imwrite(filepath, currentImage) 
    #cv2.imshow('image', currentImage)
    #cv2.waitKey(0)

def listen():
    bridge = CvBridge();
    rospy.init_node('capture_image', anonymous=True)
    rospy.Subscriber("left/image_rect_color", Image, callback)  
    rospy.Subscriber("drive_parameters", drive_param, takePicture)
    rospy.spin()

if __name__ == '__main__':
    print("image capture initialized")
    listen()
