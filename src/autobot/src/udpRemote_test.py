#!/usr/bin/env python

import unittest
from udpRemote import parseCommand


class MockDriveParam:
    velocity = 0.0
    angle = 0.0


class UdpRemoteTest(unittest.TestCase):
    def testValidParse(self):
        p = MockDriveParam()
        p = parseCommand("V44.4", p)
        self.assertEqual(p.velocity, 44.4)
        self.assertEqual(p.angle, 0.0)

        p = parseCommand("A81.3", p)
        self.assertEqual(p.velocity, 44.4)
        self.assertEqual(p.angle, 81.3)

    def testInvalidParse(self):
        p = MockDriveParam()
        p = parseCommand("X44.4", p)
        self.assertEqual(p.velocity, 0.0)
        self.assertEqual(p.angle, 0.0)

        p = MockDriveParam()
        p = parseCommand("V0F.4", p)
        self.assertEqual(p.velocity, 0.0)
        self.assertEqual(p.angle, 0.0)
