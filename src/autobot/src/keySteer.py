#!/usr/bin/env python

import rospy
from autobot.msg import drive_param
from autobot.msg import wall_dist
from autobot.srv import *
import curses

"""
TODO: 
- [ ] Update UI to show wall hug status
- [ ] Create menu boxes
"""

velocity = 0
velocityMultiplier = 10
steerAngle = 0
steerIncrement = 15
steerMultiplier = 1.5

stdscr = curses.initscr()
curses.cbreak()
curses.noecho()
stdscr.keypad(1)

stdscr.refresh()

stdscr.addstr(0, 5, "AUTOBOT CONTROL - \
Use WASD to steer (hold shift to increase ratio)")

stdscr.addstr(2, 14, "VELOCITY")
stdscr.addstr(3, 14, "ANGLE   ")


def modVelocity(incr):
    global velocity
    velocity = velocity + incr
    stdscr.addstr(2, 25, '%.2f' % velocity)


def zeroVelocity():
    global velocity
    velocity = 0
    stdscr.addstr(2, 25, '%.2f' % velocity)


def modAngle(incr):
    global steerAngle
    steerAngle = steerAngle + incr
    stdscr.addstr(3, 25, '%.2f' % steerAngle)


def zeroAngle():
    global steerAngle
    steerAngle = 0
    stdscr.addstr(3, 25, '%.2f' % steerAngle)


def setWallDist(wall, dist):
    rospy.wait_for_service('adjustWallDist')
    try:
        adjustWall = rospy.ServiceProxy('adjustWallDist', AdjustWallDist)
        cmd = wall_dist()
        cmd.wall = wall
        cmd.dist = dist
        resp = adjustWall(cmd)
        return resp
    except rospy.ServiceException, e:
        print "Service called failed: %s" % e


def main():
    rospy.init_node('keyboardSteer', anonymous=True)
    drivePub = rospy.Publisher('drive_parameters', drive_param, queue_size=10)
    key = ''
    while key != ord('q'):
        keyPressed = False
        key = stdscr.getch()
        stdscr.refresh()

        if key == curses.KEY_UP or key == ord('w'):
            modVelocity(0.1)
            keyPressed = True
        elif key == ord('W'):
            modVelocity(0.1 * velocityMultiplier)
            keyPressed = True
        elif key == curses.KEY_DOWN or key == ord('s'):
            modVelocity(-0.1)
            keyPressed = True
        elif key == ord('S'):
            modVelocity(-0.1 * velocityMultiplier)
            keyPressed = True

        if key == curses.KEY_LEFT or key == ord('a'):
            modAngle(-1 * steerIncrement)
            keyPressed = True
        elif key == ord('A'):
            modAngle(-1 * steerIncrement * steerMultiplier)
            keyPressed = True
        elif key == curses.KEY_RIGHT or key == ord('d'):
            modAngle(steerIncrement)
            keyPressed = True
        elif key == ord('D'):
            modAngle(steerIncrement * steerMultiplier)
            keyPressed = True
        elif key == ord(' '):
            zeroVelocity()
            keyPressed = True
        elif key == ord('c'):
            zeroAngle()
            keyPressed = True

        if keyPressed is True:
            msg = drive_param()
            msg.velocity = velocity
            msg.angle = steerAngle
            drivePub.publish(msg)

        if key == ord('r'):
            setWallDist(autobot.msg.wall_dist.WALL_RIGHT, 0.5)
        elif key == ord('l'):
            setWallDist(autobot.msg.wall_dist.WALL_LEFT, 0.5)

    curses.endwin()


if __name__ == '__main__':
    main()
