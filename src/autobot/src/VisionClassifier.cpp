#include "VisionClassifier.hpp"

bool VisionClassifier::importModel(VisionSettings &settings)
{
    //! [Create the importer of Caffe model]
    Ptr<dnn::Importer> importer;
    try                                     //Try to import Caffe GoogleNet model
    {
        importer = dnn::createCaffeImporter(settings.prototxt, settings.modelBin);
    }
    catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
    {
        std::cerr << err.msg << std::endl;
    }
    //! [Create the importer of Caffe model]

    if (!importer)
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << settings.prototxt << std::endl;
        std::cerr << "caffemodel: " << settings.modelBin << std::endl;
        std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
        std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
        // exit(-1);
        return false;
    }

    importer->populateNet(mNet);
    importer.release();

    if (!readClassNames(settings.classNames.c_str())) {
        return false;
    }

    mImported = true;
    return true;
}

void VisionClassifier::classify(cv::Mat &img)
{
    if (!mImported)
    {
        return;
    }

    //! [Prepare blob]
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
    mNet.setBlob(".data", inputBlob);        //set the network input
    //! [Set input blob]

    //! [Make forward pass]

    std::cout << "Computing... ";
    mNet.forward();                          //compute output
    //! [Make forward pass]
    std::cout << " done." << std::endl;

    //! [Gather output]
    dnn::Blob prob = mNet.getBlob("prob");   //gather output of "prob" layer

    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);//find the best class
    //! [Gather output]

    //! [Print results]
    std::cout << "Best class: #" << classId << " '" << mClassNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    //! [Print results]

}

void VisionClassifier::getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;

    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}

bool VisionClassifier::readClassNames(const char *filename)
{
    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        return false;
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            mClassNames.push_back( name.substr(name.find(' ')+1) );
    }

    fp.close();
    return true;
}
