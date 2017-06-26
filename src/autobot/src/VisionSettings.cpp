#include "VisionSettings.hpp"
#include <iostream>

struct option VisionOptions[] = 
{
    { "prototxt", required_argument, 0, 'p' },
    { "model",    required_argument, 0, 'm' },
    { "names",    required_argument, 0, 'n' },
    { "camera",   required_argument, 0, 'c' },
    { "help",     optional_argument, 0, 'h' },
    { NULL, 0, 0, 0 },
};

bool VisionSettings::importFromArgs(int argc, char **argv)
{
    int c = 0;
    int optIndex = 0;
    while (c != -1) 
    {
        c = getopt_long(argc, argv, "p:m:n:c:h", VisionOptions, &optIndex);
        switch(c)
        {
            case 'p': /* --prototxt */
                std::cout << "  prototxt: " << optarg << std::endl;
                prototxt = optarg;
                break;
            case 'm': /* --modelbin */
                std::cout << "  modelbin: " << optarg << std::endl;
                modelBin = optarg;
                break;
            case 'n': /* --names */
                std::cout << "  classnames: " << optarg << std::endl;
                classNames = optarg;
                break;
            case 'c': /* --camera */
                std::cout << "  camera: " << optarg << std::endl;
                cameraNumber = atoi(optarg);
                break;
            case 'h':
                displayHelp();
                return true;
            default:
                break;
        }
    }

    return false;
}

void VisionSettings::displayHelp()
{
    using namespace std;
    cout << "Available options:" << endl
         << " --prototxt path/to/file.prototxt" << endl
         << " --model path/to/model.caffemodel" << endl
         << " --names path/to/classnames.txt" << endl
         << " --camera cameraIndex" << endl
         << " --help " << endl;
}