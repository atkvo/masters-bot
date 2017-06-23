#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

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
VisionClassifier v;

void processImage(const cv::Mat & frame);
void imageCallback(const sensor_msgs::ImageConstPtr& msg);

int main(int argc, char **argv)
{
    String modelTxt = "model/bvlc_googlenet.prototxt";
    String modelBin = "model/bvlc_googlenet.caffemodel";
    int cameraNumber = (argc > 1) ? atoi(argv[1]) : 0;

    CaffeModelFiles files;
    files.prototxt = modelTxt;
    files.modelBin = modelBin;
    files.classNames = "model/synset_words.txt";

    if (!v.importModel(files))
    {
        std::cerr << "Error importing model files" << std::endl;
        return false;
    }

    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);
    ros::spin();

    for (;;)
    {
    }
    return 0;
} //main

void imageCallback(const sensor_msgs::ImageConstPtr& msg) 
{
    try {
        // cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        processImage(img);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Couldn't convert image from '%s' to 'bgr8'", msg->encoding.c_str());
    }
}

void processImage(const cv::Mat & frame)
{
    if (frame.empty()) 
    {
        return;
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
