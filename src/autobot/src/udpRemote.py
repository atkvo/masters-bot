#!/usr/bin/env python

import rospy
import socket
from autobot.msg import drive_param

"""
Warning: This code has not been tested at all

Protocol could be comma delimited
Vaa.aa;Abb.bb

TODO
- [ ] Unit test
- [ ] Define protocol
- [ ] Use select() for non-blocking operation
- [ ] Use a timeout for setting drive/angle to '0' (safety)
"""


def parseCommand(cmd, pub, driveParam):
    """
    pass in Paa.aa where P is the ID of the command
    """
    val = 0.0

    # first test to see if able to parse the value from substring
    try:
        val = float(cmd.substring(1))
    except ValueError:
        return driveParam   # unable to parse, bail

    if cmd[0] == 'V':
        driveParam.velocity = val
    elif cmd[0] == 'A':
        driveParam.angle = val

    return driveParam       # valid drive parameter parsed


def parseMessage(msg, pub):
    """
        Attempts to parse a message for a proper command string
        If the command string is valid, a drive parameter will be
        published
    """
    driveParam = drive_param()
    if ";" in msg:
        arr = msg.split(";")
        for cmd in arr:
            driveParam = parseCommand(cmd, driveParam)
        pub.publish(driveParam)
    else:
        pass


def main():
    UDP_IP = "127.0.0.1"    # loopback
    UDP_PORT = 11156

    rospy.init_node("udpRemote", anonymous=True)
    pub = rospy.Publisher("drive_parameters", drive_param, queue_size=10)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    while True:
        data, addr = sock.recvfrom(1024)
        parseMessage(str(data, "utf-8"), pub)


if __name__ == "__main__":
    main()
