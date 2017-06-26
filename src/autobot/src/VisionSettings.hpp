
#ifndef _VISION_SETTINGS_HPP_
#define _VISION_SETTINGS_HPP_

#include "opencv2/core/cvstd.hpp"
#include <getopt.h>

using namespace cv;

extern struct option VisionOptions[];

struct VisionSettings
{
    String prototxt;
    String modelBin;
    String classNames;
    int cameraNumber;

    /**
    * Parses command line arguments to configure settings
    * Returns true if help option was invoked
    */
    bool importFromArgs(int argc, char **argv);

    /**
    * Displays command line usage information via stdout
    */
    void displayHelp();
};

#endif