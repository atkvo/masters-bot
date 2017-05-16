#!/usr/bin/env python

import rospy
import math
from autobot.msg import drive_param
from sensor_msgs.msg import LaserScan
from autobot.msg import pid_input

#TODO: NEED TO CHOOSE WHETHER WE WANT TO HUG LEFT OR HUG RIGHT OR HUG CLOSEST WALL.. LETS HUG RIGHT WALL FOR NOW (SIMULATE RIGHT SIDE OF ROAD)
#SEND ERROR LEFT TO HUG LEFT WALL
desired_trajectory = 0.5 #DESIRED DISTANCE FROM WALL
vel = 7.3 #DRIVE VELOCITY
rate = 0 #RATE TO PUBLISH TO ERROR TOPIC 

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

    frontDistance = getRange(data, 90); 

    theta = 50; #PICK THIS ANGLE TO BE BETWEEN 0 AND 70 DEGREES

    thetaDistanceRight = getRange(data, theta) #a
    rightDistance = getRange(data, 0) #b

    thetaDistanceLeft = getRange(data, 180-theta) #aL
    leftDistance = getRange(data, 180) #bL

    if frontDistance < 2:
        #TURN
        print "Blocked!"
        driveParam = drive_param()
        if rightDistance  > leftDistance:
            driveParam.angle = 90
            print "Turning Right"
        else:
            driveParam.angle = -90
            print "Turning Left"
        driveParam.velocity = 7.3
        motorPub.publish(driveParam)
        return

    thetaRadiansRight = math.radians(theta) #aRads
    thetaRadiansLeft = math.radians(130) #bRads

    carAngleRight = math.atan2( thetaDistanceRight * math.cos(thetaRadiansRight) - rightDistance , thetaDistanceRight * math.sin(thetaRadiansRight) ) #alpha
    carAngleLeft = math.atan2( thetaDistanceLeft * math.cos(thetaRadiansRight) - leftDistance , thetaDistanceLeft * math.sin(thetaRadiansRight) ) #alphaL

    carToWallRight = rightDistance * math.cos(carAngleRight) #AB
    carToWallLeft = leftDistance * math.cos(carAngleLeft) #ABL

    distanceTraveled = 1.0 #AC #MAY NEED TO EDIT THIS VALUE 
    projectedDistanceRight = carToWallRight + distanceTraveled * math.sin(carAngleRight) #CD
    projectedDistanceLeft = carToWallLeft + distanceTraveled * math.sin(carAngleLeft) #CDL

    #IF TOO FAR FROM WALL, TURN RIGHT TO GET CLOSER (ERRORRIGHT WILL BE POSITIVE)
    #IF TOO CLOSE TO WALL, TURN LEFT TO GET FARTHER (ERRORRIGHT WILL BE NEGATIVE)
    #THESE ERROR VALUES ARE DIFFERENCES BETWEEN YOUR PROJECTED FUTURE POSITION AND DISTANCE TO WALL
    #ARE WE PROCESSING THIS ERROR CORRECTLY? GETS SENT TO PIDCONTROL.PY
    errorRight = projectedDistanceRight - desired_trajectory  
    errorLeft = projectedDistanceLeft - desired_trajectory
    errorLeft *= -1

    print "carAngleRight {} carAngleLeft {}".format(carAngleRight, carAngleLeft)
    print "thetaDistanceRight {} thetaDistanceLeft {}\nrightDistance {} leftDistance {}".format(thetaDistanceRight, thetaDistanceLeft, rightDistance, leftDistance)
    print "carToWallRight {} carToWallLeft {}".format(carToWallRight, carToWallLeft)
    print "projectedDistanceRight {} projectedDistanceLeft {}".format(projectedDistanceRight, projectedDistanceLeft)
    print "errorRight {} errorLeft {}".format(errorRight, errorLeft)

    msg = pid_input()
    msg.pid_error = errorRight
    #msg.pid_error = errorLeft
    msg.pid_vel = vel

    global rate
    rate += 1
    if (rate % 10 == 0):
        rate = 0
        pub.publish(msg)

if __name__ == '__main__':
    print("Laser node started")
    rospy.init_node('distFinder',anonymous = True)
    rospy.Subscriber("scan",LaserScan,callback)
    rospy.spin()

