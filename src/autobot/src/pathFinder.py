#!/usr/bin/env python

import rospy
import math
from autobot.msg import drive_param
from sensor_msgs.msg import LaserScan
from autobot.msg import pid_input

"""
TODO:
- [x] Decide if you want to hug right/left/or closest wall
    - Right wall hugged for now to simulate right side of road
- [ ] Send error left to hug left wall
"""
desired_trajectory = 0.5  # DESIRED DISTANCE FROM WALL
vel = 7.3   # DRIVE VELOCITY
rate = 0    # RATE TO PUBLISH TO ERROR TOPIC

pub = rospy.Publisher('error', pid_input, queue_size=10)
motorPub = rospy.Publisher('drive_parameters', drive_param, queue_size=10)


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

    frontDistance = getRange(data, 90)

    theta = 50  # PICK THIS ANGLE TO BE BETWEEN 0 AND 70 DEGREES

    thetaDistRight = getRange(data, theta)  # a
    rightDist = getRange(data, 0)  # b

    thetaDistLeft = getRange(data, 180-theta)  # aL
    leftDist = getRange(data, 180)  # bL

    if frontDistance < 2.2:
        # TURN
        print "Blocked!"
        driveParam = drive_param()
        if rightDist > leftDist:
            driveParam.angle = 90
            print "Turning Right"
        else:
            driveParam.angle = -90
            print "Turning Left"
        driveParam.velocity = 7.3
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
    errorRight = projectedDistRight - desired_trajectory
    errorLeft = projectedDistLeft - desired_trajectory
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
    msg.pid_error = errorRight
    # msg.pid_error = errorLeft
    msg.pid_vel = vel

    global rate
    rate += 1
    if (rate % 10 == 0):
        rate = 0
        pub.publish(msg)

if __name__ == '__main__':
    print("Laser node started")
    rospy.init_node('distFinder', anonymous=True)
    rospy.Subscriber("scan", LaserScan, callback)
    rospy.spin()
