#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include "VisionClassifier.hpp"
using namespace std;

const char * LEFT_WIN = "LEFT";
const char * RIGHT_WIN = "RIGHT";
const char * CENTER_WIN = "CENTER";

int main(int argc, char **argv)
{
    VisionClassifier v;
    VisionSettings settings;

    /* Load with default parameters */
    settings.prototxt = "model/bvlc_googlenet.prototxt";
    settings.modelBin = "model/bvlc_googlenet.caffemodel";
    settings.classNames = "model/synset_words.txt";
    settings.cameraNumber = 0;

    /* Extract settings from CLI if available */
    bool helpInvoked = settings.importFromArgs(argc, argv);

    if (helpInvoked)
    {
        exit(0);
    }

    if (!v.importModel(settings))
    {
        std::cerr << "Error importing model files" << std::endl;
        return false;
    }

    //! [Capture video images and classify]
    VideoCapture capture;
    if (!capture.open(settings.cameraNumber)) 
    {
        std::cerr << "Error opening camera #" << settings.cameraNumber << std::endl;
        exit(-1);
    }

    for (;;)
    {
        Mat frame;
        capture >> frame;
        if (frame.empty()) 
        {
            break;
        }

        int w = frame.cols;
        int h = frame.rows;

        Mat leftframe = cv::Mat(frame, cv::Rect(0, 0, w/3, h));
        Mat centerframe = cv::Mat(frame, cv::Rect(0 + w/3, 0, w/3, h));
        Mat rightframe  = cv::Mat(frame, cv::Rect(0 + (2*w)/3, 0, w/3, h));
        imshow(LEFT_WIN, leftframe);
        imshow(CENTER_WIN, centerframe);
        imshow(RIGHT_WIN, rightframe);
        v.classify(centerframe);
        waitKey(1);
    }
    return 0;
} //main
