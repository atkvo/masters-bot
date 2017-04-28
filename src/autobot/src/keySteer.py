#!/usr/bin/env python

import rospy
from autobot.msg import drive_param
import curses

global velocity
global steerAngle
global multiplier

velocity = 0
steerAngle = 0
multiplier = 5

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


def main():
    rospy.init_node('keyboardSteer', anonymous=True)
    pub = rospy.Publisher('drive_parameters', drive_param, queue_size=10)
    key = ''
    while key != ord('q'):
        key = stdscr.getch()
        stdscr.refresh()

        if key == curses.KEY_UP or key == ord('w'):
            modVelocity(0.1)
        elif key == ord('W'):
            modVelocity(0.1 * multiplier)
        elif key == curses.KEY_DOWN or key == ord('s'):
            modVelocity(-0.1)
        elif key == ord('S'):
            modVelocity(-0.1 * multiplier)

        if key == curses.KEY_LEFT or key == ord('a'):
            modAngle(-0.1)
        elif key == ord('A'):
            modAngle(-0.1 * multiplier)
        elif key == curses.KEY_RIGHT or key == ord('d'):
            modAngle(0.1)
        elif key == ord('D'):
            modAngle(0.1 * multiplier)
        elif key == ord(' '):
            zeroVelocity()
        elif key == ord('c'):
            zeroAngle()

        msg = drive_param()
        msg.velocity = velocity
        msg.angle = steerAngle
        pub.publish(msg)

    curses.endwin()


if __name__ == '__main__':
    main()
