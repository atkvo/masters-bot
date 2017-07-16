#!/usr/bin/env python

import rospy
from autobot.msg import drive_param
from autobot.msg import wall_dist
from autobot.msg import pathFinderState
from autobot.msg import detected_object
from autobot.srv import *
from sensor_msgs.msg import Image
from pathFinder import PathConfig

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2

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


class ObstructionInfo(object):

    def __init__(self):
        self.distance = 999
        self.position = 0
        self.className = ""


class ObstructionMap(object):
    """ TODO: Write down class information
    Store objects into a dict?
    Or a numpy type of array? Numpy can get things by distance

    Obstructions
    LEFT, RIGHT
    """
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3

    def __init__(self):
        self.obstructions = dict()

    def clearMap(self):
        self.obstructions.clear()

    def addToMap(self, className, x, y, distance):
        """NOTE
        if (x > threshold) gives the location of object in terms of left/right
        # Do objects closest to the car get priority?
        self.obstructions[k] = (className, distance)
        """
        side = LEFT
        obs = ObstructionInfo()
        obs.className = className
        obs.distance = distance
        obs.position = side
        if side in self.obstructions is False or
        distance < self.obstructions[side][1]:
            self.obstructions[side] = (className, distance)

    def getClosest(self):
        """NOTE
        Loop through obstruction list and make decision on which object
        is the most dangerous/closest

        Returns ObstructionInfo object
        """
        if len(self.obstructions) is 0:
            return None

        closestObject = ObstructionInfo()
        for side in self.obstructions:
            if this.obstructions[side].distance < closestObject.distance:
                closestObject = self.obstructions[side]

        return closestObject

    def getHighPriorities(self):
        """
        TODO: Return a list of high priority objects?
        High priorities include things like chairs (what else?).

        E.g. If currently hugging the right wall closely and a closed door is
        coming up, but there is a chair somewhat to the left of the car...
        do NOT attempt to drive away from the wall to avoid the door. Instead
        either stick to that wall or maybe swing the car all the way to the
        left wall to avoid the chair.
        """
        pass


PATH_STATE = PathConfig()
PUB_DRIVE = rospy.Publisher('drive_parameters', drive_param, queue_size=10)
DECISION_RATE_SEC = 0.5
OBJECT_MAP = ObjectMap()


def srvTogglePathFinder(state):
    try:
        rospy.wait_for_service('togglePathFinder', timeout=0.2)
        srv = rospy.ServiceProxy('togglePathFinder', TogglePathFinder)
        srv(state)  # ignore ACK response
    except rospy.ROSException, e:
        # print "Service called failed: %s" % e
        pass


def zeroVelocity():
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


def onDecisionInterval(event):
    """
    Makes pathing decision based on objects detected
    TODO:
    - [ ] When to prefer hugging the current wall vs moving
          to the opposite wall
    - [ ] Get list of how far/close to wall to get depending on class
    - [ ] May need a hierarchy of "priorities". E.g.
            if a CHAIR is in the view, stay clear of it even if there
            is a closed door coming up close
    """
    global OBJECT_MAP
    global PATH_STATE

    obstruct = OBJECT_MAP.getClosest()
    if obstruct is None:
        return

    wallToHug = PATH_STATE.wallToWatch
    if obstruct.position == ObstructionMap.LEFT:
        wallToHug = wall_dist.WALL_RIGHT
    elif obstruct.position == ObstructionMap.RIGHT:
        wallToHug = wall_dist.WALL_LEFT

    if obstruct.className == "DOOR":
        setWallDist(wallToHug, 2)
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
        # Step 1. map box onto depthImg to get distance map of object
        # Step 2. get the average distance of the object
        # Step 3. store these onto a list or array
        #   Organize by distance? Location?
    except CvBridgeError as e:
        print(e)


if __name__ == '__main__':
    global DECISION_RATE_SEC
    rospy.Subscriber("pathFinderStatus", pathFinderState, pathFinderUpdated)
    rospy.Subscriber("drive_parameters", drive_param, driveParamsUpdated)
    rospy.Subscriber("object_detector", detected_object, onObjectDetected)
    rospy.Timer(rospy.Duration(DECISION_RATE_SEC), callback=onDecisionInterval)
    rospy.spin()
