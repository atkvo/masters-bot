#!/usr/bin/env python
import rospy
import math
from autobot.msg import drive_param
from sensor_msgs.msg import LaserScan
from autobot.msg import pid_input
from autobot.msg import wall_dist
# from autobot.srv import AdjustWallDist
from autobot.srv import *

"""
TODO:
- [x] Decide if you want to hug right/left/or closest wall
    - Right wall hugged for now to simulate right side of road
    - Update: wall decision to be made by image processing node
- [x] Send error left to hug left wall
"""


class PathConfig(object):
    __slots__ = ('wallToWatch', 'desiredTrajectory', 'velocity', 'pubRate',
                 'minFrontDist')
    """
    wallToWatch: Set which wall to hug
    options: autobot.msg.wall_dist.WALL_LEFT
             autobot.msg.wall_dist.WALL_RIGHT
             autobot.msg.wall_dist.WALL_FRONT  #< probably won't be used
    """
    wallToWatch = autobot.msg.wall_dist.WALL_RIGHT
    desiredTrajectory = 0.5  # desired distance from the wall
    minFrontDist = 2.2       # minimum required distance in front of car
    velocity = 7.3           # velocity of drive
    pubRate = 0              # publish rate of node


errorPub = rospy.Publisher('error', pid_input, queue_size=10)
motorPub = rospy.Publisher('drive_parameters', drive_param, queue_size=10)


def HandleAdjustWallDist(req):
    """ Handler for adjusting wall hugging parameters

    Responds with wall_dist msg and a bool to verify that the
    service command has been accepted
    """
    global PathConfig

    print " wall {}".format(req.cmd.wall)
    print " dist {}".format(req.cmd.dist)

    resp = wall_dist()
    isValid = req.cmd.dist >= 0

    if isValid is True and req.cmd.wall != autobot.msg.wall_dist.WALL_FRONT:
        """ only accept WALL_LEFT or WALL_RIGHT
        Service client can send an invalid wall or distance
        query current settings
        """
        PathConfig.wallToWatch = req.cmd.wall
        PathConfig.desiredTrajectory = req.cmd.dist
    else:
        isValid = False

    resp.wall = PathConfig.wallToWatch
    resp.dist = PathConfig.desiredTrajectory
    return AdjustWallDistResponse(resp, isValid)


def getRange(data, theta):
    """ Find the index of the array that corresponds to angle theta.
    Return the lidar scan value at that index
    Do some error checking for NaN and absurd values
    data: the LidarScan data
    theta: the angle to return the distance for
    """
    car_theta = math.radians(theta) - math.pi / 2
    if car_theta > 3 * math.pi / 4:
        car_theta = 3 * math.pi / 4
    elif car_theta < -3 * math.pi / 4:
        car_theta = -3 * math.pi / 4

    float_index = (car_theta + 3 * math.pi / 4) / data.angle_increment
    index = int(float_index)
    return data.ranges[index]


def callback(data):
    global PathConfig

    frontDistance = getRange(data, 90)

    theta = 50  # PICK THIS ANGLE TO BE BETWEEN 0 AND 70 DEGREES

    thetaDistRight = getRange(data, theta)  # a
    rightDist = getRange(data, 0)  # b

    thetaDistLeft = getRange(data, 180-theta)  # aL
    leftDist = getRange(data, 180)  # bL

    if frontDistance < PathConfig.minFrontDist:
        # TURN
        print "Blocked!"
        driveParam = drive_param()
        if rightDist > leftDist:
            driveParam.angle = 90
            print "Turning Right"
        else:
            driveParam.angle = -90
            print "Turning Left"
        driveParam.velocity = PathConfig.velocity
        motorPub.publish(driveParam)
        return

    thetaRadsRight = math.radians(theta)  # aRads
    thetaRadsLeft = math.radians(130)     # bRads

    # alpha right
    carAngleRight = math.atan2(thetaDistRight * math.cos(thetaRadsRight) - rightDist,
                               thetaDistRight * math.sin(thetaRadsRight))

    # alpha left
    carAngleLeft = math.atan2(thetaDistLeft * math.cos(thetaRadsRight) - leftDist,
                              thetaDistLeft * math.sin(thetaRadsRight))

    carToWallRight = rightDist * math.cos(carAngleRight)  # AB
    carToWallLeft = leftDist * math.cos(carAngleLeft)     # ABL

    distTraveled = 1.0  # AC MAY NEED TO EDIT THIS VALUE

    # CD
    projectedDistRight = carToWallRight + distTraveled * math.sin(carAngleRight)

    # CDL
    projectedDistLeft = carToWallLeft + distTraveled * math.sin(carAngleLeft)

    """
    If too far from the wall:
        Turn right to get closer to it
        errorRight will be positive
    if too close to wall:
        Turn left to get further
        errorRight will be negative

    The error_ values are differences between projected future position
    and the distance to the wall
    """
    # ARE WE PROCESSING THIS ERROR CORRECTLY? GETS SENT TO PIDCONTROL.PY
    errorRight = projectedDistRight - PathConfig.desiredTrajectory
    errorLeft = projectedDistLeft - PathConfig.desiredTrajectory
    errorLeft *= -1

    print "carAngleRight {} carAngleLeft {}".format(carAngleRight,
                                                    carAngleLeft)
    print ("thetaDistRight {} thetaDistLeft {}\n"
           "rightDist {} leftDist {}").format(
        thetaDistRight,
        thetaDistLeft,
        rightDist,
        leftDist)

    print "carToWallRight {} carToWallLeft {}".format(
        carToWallRight,
        carToWallLeft)
    print "projectedDistRight {} projectedDistLeft {}".format(
        projectedDistRight,
        projectedDistLeft)

    print "errorRight {} errorLeft {}".format(errorRight, errorLeft)

    msg = pid_input()
    if PathConfig.wallToWatch == autobot.msg.wall_dist.WALL_LEFT:
        msg.pid_error = errorLeft
    else:
        msg.pid_error = errorRight

    msg.pid_vel = PathConfig.velocity

    PathConfig.pubRate += 1
    if (PathConfig.pubRate % 10 == 0):
        PathConfig.pubRate = 0
        errorPub.publish(msg)

if __name__ == '__main__':
    print("Path finding node started")
    rospy.Service('adjustWallDist', AdjustWallDist, HandleAdjustWallDist)
    rospy.init_node('pathFinder', anonymous=True)
    rospy.Subscriber("scan", LaserScan, callback)
    rospy.spin()
