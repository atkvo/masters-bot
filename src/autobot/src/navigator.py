#!/usr/bin/env python

import rospy
from autobot.msg import drive_param
from autobot.msg import wall_dist
from autobot.msg import pathFinderState
from autobot.msg import detected_object
from autobot.srv import *
from sensor_msgs.msg import Image
from pathFinder import PathConfig
from obstruction import *

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

"""
This node is responsible for configuring the pathFinder node
when an object is detected.

TODO:
- [ ] Check other todos spread throughout code
- [ ] What are we avoiding via vision
        LiDAR will handle obvious obstructions like people, boxes
            backpacks, etc.
        Vision should be able to look out for things like chairs which
            can slip past LiDAR (legs of the chair are thin)
- [ ] Making decisions based on a simple "object is on the left/right"
        is primitive. Should decisions be made with a finer scale?
        See callback below for more notes
"""
PATH_STATE = PathConfig()
PUB_DRIVE = rospy.Publisher('drive_parameters', drive_param, queue_size=10)
OBJECT_MAP = ObstructionMap()


def srvTogglePathFinder(state):
    try:
        rospy.wait_for_service('togglePathFinder', timeout=0.2)
        srv = rospy.ServiceProxy('togglePathFinder', TogglePathFinder)
        srv(state)  # ignore ACK response
    except rospy.ROSException, e:
        # print "Service called failed: %s" % e
        pass


def stopCar():
    global PUB_DRIVE
    msg = drive_param()
    msg.velocity = 0
    msg.angle = 0
    PUB_DRIVE.publish(msg)
    srvTogglePathFinder(False)


def setWallDist(wall, dist):
    try:
        rospy.wait_for_service('adjustWallDist')
        adjustWall = rospy.ServiceProxy('adjustWallDist', AdjustWallDist)
        cmd = wall_dist()
        cmd.wall = wall
        cmd.dist = dist
        resp = adjustWall(cmd)
        return resp
    except rospy.ROSException, e:
        # print "Service called failed: %s" % e
        pass


def convertWallToString(wall):
    # WALL_LEFT=0
    # WALL_FRONT=1
    # WALL_RIGHT=2
    if (wall is wall_dist.WALL_LEFT):
        return "Left"
    elif (wall is wall_dist.WALL_RIGHT):
        return "Right"
    elif (wall is wall_dist.WALL_FRONT):
        return "Front"
    else:
        return "Unknown"


def pathFinderUpdated(status):
    global PATH_STATE
    PATH_STATE.velocity = status.velocity
    PATH_STATE.wallToWatch = status.hug.wall
    PATH_STATE.desiredTrajectory = status.hug.dist
    PATH_STATE.enabled = status.enabled


def getAverageColor(img):
    """Returns average color of img"""
    avgColorPerRow = np.average(img, axis=0)
    avgColor = np.average(avgColorPerRow, axis=0)
    return avgColor


def shadeToDepth(color):
    """Returns depth in meters from color b/w"""
    minDistance = 0.7
    maxDistance = 20
    maxColor = 255
    # depth = mx + b
    m = (minDistance - maxDistance)/maxColor
    x = color
    b = maxDistance
    return m * x + b


def hasObstruction(className, list):
    for o in list:
        if o.className == className:
            return (True, o)

    return (False, None)


def onDecisionInterval(event):
    """
    Makes pathing decision based on objects detected
    TODO:
    - [ ] When to prefer hugging the current wall vs moving
          to the apposite wall
          - Maybe when multiple objects are crowding the X side
    - [ ] Get list of how far/close to wall to get depending on class
    - [ ] May need a hierarchy of "priorities". E.g.
            if a CHAIR is in the view, stay clear of it even if there
            is a closed door coming up close
            if a stop sign is close, stop the car for a bit?
    """
    global OBJECT_MAP
    global PATH_STATE

    closest = OBJECT_MAP.getClosest()
    dangers = OBJECT_MAP.getHighPriorities()
    if dangers is None and closest is None:
        return

    hasIt, obstruction = hasObstruction('person', dangers)
    # TODO: make sure person is in a certain X position before stopping
    if hasIt and obstruction.distance < 2 and PATH_STATE.enabled:
        stopCar()
    else:
        setWallDist(0.5, PATH_STATE.wallToWatch)
        srvTogglePathFinder(True)

    # wallToHug = PATH_STATE.wallToWatch
    # if closest.position == ObstructionMap.LEFT:
    #     wallToHug = wall_dist.WALL_RIGHT
    # elif closest.position == ObstructionMap.RIGHT:
    #     wallToHug = wall_dist.WALL_LEFT
    # elif closest.position == ObstructionMap.CENTER:
    #     # Increase distance from the wall?
    #     setWallDist(wallToHug, PATH_STATE.desiredTrajectory + 0.25)

    # if closest.className == "DOOR":
    #     setWallDist(wallToHug, 2)
    # elif closest.className == "CHAIR":
    #     setWallDist(wallToHug, 0)
    #     pass

    OBJECT_MAP.clearMap()
    pass


def onObjectDetected(msg):
    """
    message type == detected_object.msg

    m.class: str
    m.depthImg: image
    m.box: bounding_box
    """
    bridge = CvBridge()
    try:
        depthMap = bridge.imgmsg_to_cv2(msg.depthImg,
                                        desired_encoding="passthrough")
        crop = depthMap[msg.box.origin_y: msg.box.origin_y + msg.box.height,
                        msg.box.origin_x: msg.box.origin_x + msg.box.width]
        avg = getAverageColor(crop)
        distance = shadeToDepth(avg)
        global OBJECT_MAP
        OBJECT_MAP.addToMap(msg.className,
                            msg.box.origin_x, msg.box.origin_y,
                            distance)
    except CvBridgeError as e:
        print(e)


if __name__ == '__main__':
    DECISION_RATE_SEC = 0.5
    rospy.Subscriber("pathFinderStatus", pathFinderState, pathFinderUpdated)
    rospy.Subscriber("drive_parameters", drive_param, driveParamsUpdated)
    rospy.Subscriber("object_detector", detected_object, onObjectDetected)
    rospy.Timer(rospy.Duration(DECISION_RATE_SEC), callback=onDecisionInterval)
    rospy.spin()
