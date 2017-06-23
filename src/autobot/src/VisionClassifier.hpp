#ifndef _VISIONCLASSIFIER_HPP_
#define _VISIONCLASSIFIER_HPP_

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

#include <fstream>
#include <iostream>
#include <cstdlib>

struct CaffeModelFiles {
    String prototxt;
    String modelBin;
    String classNames;
};

class VisionClassifier
{
public:
    bool importModel(CaffeModelFiles &model);
    void classify(cv::Mat &img);
    VisionClassifier() : mImported(false) { } 
    ~VisionClassifier() { }
private:
    dnn::Net mNet;
    bool mImported;
    void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb);
    std::vector<String> mClassNames;
    bool readClassNames(const char *filename = "model/synset_words.txt");
};

#endif /* _VISIONCLASSIFIER_HPP_ */