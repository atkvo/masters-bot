#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
// using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

const char * LEFT_WIN = "LEFT";
const char * RIGHT_WIN = "RIGHT";
const char * CENTER_WIN = "CENTER";

void classify(dnn::Net &net, Mat &img);

/* Find best class for the blob (i. e. class with maximal probability) */
void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;

    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}

std::vector<String> readClassNames(const char *filename = "model/synset_words.txt")
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }

    fp.close();
    return classNames;
}

int main(int argc, char **argv)
{
    String modelTxt = "model/bvlc_googlenet.prototxt";
    String modelBin = "model/bvlc_googlenet.caffemodel";
    int cameraNumber = (argc > 1) ? atoi(argv[1]) : 0;

    //! [Create the importer of Caffe model]
    Ptr<dnn::Importer> importer;
    try                                     //Try to import Caffe GoogleNet model
    {
        importer = dnn::createCaffeImporter(modelTxt, modelBin);
    }
    catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
    {
        std::cerr << err.msg << std::endl;
    }
    //! [Create the importer of Caffe model]

    if (!importer)
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
        std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
        exit(-1);
    }


    //! [Initialize network]
    dnn::Net net;
    importer->populateNet(net);
    importer.release();                     //We don't need importer anymore
    //! [Initialize network]

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
        classify(net, centerframe);
        waitKey(1);
    }
    return 0;
} //main

void classify(dnn::Net &net, Mat &img)
{
    //! [Prepare blob]
    // Mat img = imread(imageFile);
    if (img.empty())
    {
        // std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        std::cerr << "Can't read image" << std::endl;
        exit(-1);
    }

    resize(img, img, Size(224, 224));       //GoogLeNet accepts only 224x224 RGB-images
    dnn::Blob inputBlob = dnn::Blob::fromImages(img);   //Convert Mat to dnn::Blob image batch
    //! [Prepare blob]

    //! [Set input blob]
    net.setBlob(".data", inputBlob);        //set the network input
    //! [Set input blob]

    //! [Make forward pass]

    std::cout << "Computing... ";
    net.forward();                          //compute output
    //! [Make forward pass]
    std::cout << " done." << std::endl;

    //! [Gather output]
    dnn::Blob prob = net.getBlob("prob");   //gather output of "prob" layer

    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);//find the best class
    //! [Gather output]

    //! [Print results]
    std::vector<String> classNames = readClassNames();
    std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    //! [Print results]
}
