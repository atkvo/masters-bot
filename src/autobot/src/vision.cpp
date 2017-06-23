#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
// using namespace cv::dnn;

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
    String modelTxt = "model/bvlc_googlenet.prototxt";
    String modelBin = "model/bvlc_googlenet.caffemodel";
    int cameraNumber = (argc > 1) ? atoi(argv[1]) : 0;

    CaffeModelFiles files;
    files.prototxt = modelTxt;
    files.modelBin = modelBin;
    files.classNames = "model/synset_words.txt";

    VisionClassifier v;
    if (!v.importModel(files))
    {
        std::cerr << "Error importing model files" << std::endl;
        return false;
    }

    //! [Capture video images and classify]
    VideoCapture capture;
    if (!capture.open(cameraNumber)) 
    {
        std::cerr << "Error opening camera #" << cameraNumber << std::endl;
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
