#!/usr/bin/env python

import rospy
import math
from sensor_msgs.msg import LaserScan
from autobot.msg import pid_input

desired_trajectory = 1.5
vel = 7.3
rate = 0

pub = rospy.Publisher('error', pid_input, queue_size=10)

def getRange(data, theta):
    """ Find the index of the arary that corresponds to angle theta.
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
    theta = 50;
    a = getRange(data, theta)
    b = getRange(data, 0)

    aL = getRange(data, 130)
    bL = getRange(data, 180)

    swing = math.radians(theta)
    swingL = math.radians(130)

    alpha = math.atan2( a * math.cos(swing) - b , a * math.sin(swing) )
#    alphaL = math.atan2( aL * math.cos(swingL) - bL , aL * math.sin(swingL) )
    alphaL = math.atan2( aL * math.cos(swing) - bL , aL * math.sin(swing) )
#   alphaL = alphaL - 0.69
    AB = b * math.cos(alpha)
    ABL = bL * math.cos(alphaL)

    AC = 1.0
    CD = AB + AC * math.sin(alpha)    
    CDL = ABL + AC * math.sin(alphaL)

    error = CD - desired_trajectory
    errorL = CDL - desired_trajectory
    errorL *= -1

    print "alpha {} alphaL {}".format(alpha, alphaL)
    print "a {} aL {}\nb {} bL {}".format(a, aL, b, bL)
    print "AB {} ABL {}".format(AB, ABL)
    print "CD {} CDL {}".format(CD, CDL)
    print "error {} errorL {}".format(error, errorL)

    msg = pid_input()
#    msg.pid_error = error
    msg.pid_error = errorL
    msg.pid_vel = vel

    global rate
    rate += 1
    if (rate % 10 == 0):
        rat = 0
        pub.publish(msg)

if __name__ == '__main__':
    print("Laser node started")
    rospy.init_node('distFinder',anonymous = True)
    rospy.Subscriber("scan",LaserScan,callback)
    rospy.spin()

