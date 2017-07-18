#!/usr/bin/env python
import unittest
import mock
from autobot.msg import detected_object
from navigator import *


def fake_stopCar():
    return True


def fake_srvTogglePathFinder(state):
    return


def fake_setWallDist(dist, wall):
    return


class NavigatorTest(unittest.TestCase):
    @mock.patch('navigator.setWallDist',
                side_effect=fake_setWallDist)
    @mock.patch('navigator.srvTogglePathFinder',
                side_effect=fake_srvTogglePathFinder)
    @mock.patch('navigator.stopCar', side_effect=fake_stopCar)
    def testPersonInFront(self, fake_stopCar,
                          fake_srvTogglePathFinder,
                          fake_setWallDist):
        global OBJECT_MAP
        global PATH_STATE
        OBJECT_MAP.addToMap('person', 10, 50, 1.2)
        OBJECT_MAP.addToMap('cat', 10, 50, 60)
        OBJECT_MAP.addToMap('bat', 10, 50, 65)
        PATH_STATE.enabled = True
        onDecisionInterval(None)
        fake_setWallDist.assert_not_called()
        fake_stopCar.assert_called()
