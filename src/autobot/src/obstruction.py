#!/usr/bin/env python


class ObstructionInfo(object):

    def __init__(self):
        self.distance = 999
        self.position = 0
        self.className = ""


class ObstructionMap(object):
    """ TODO: Write down class information
    Store objects into a dict?
    Or a numpy type of array? Numpy can get things by distance
    This class could be our decision maker also. Probably more efficient
    to make decisions as things are added

    Obstructions
    LEFT, RIGHT
    """
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3
    HIGHPRIORITIES = ['chair', 'doorstop']

    def __init__(self):
        self.obstructions = dict()
        self.highprios = dict()

    def clearMap(self):
        self.obstructions.clear()
        self.highprios.clear()

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
        if (side in self.obstructions is False or
                distance < self.obstructions[side].distance):
            self.obstructions[side] = (className, distance)
        if obs.className in HIGHPRIORITIES:
            self.highprios.append(obj)

    def getClosest(self):
        """NOTE
        Loop through obstruction list and make decision on which object
        is the closest

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
