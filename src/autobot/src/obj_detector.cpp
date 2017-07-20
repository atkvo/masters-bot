/*
 * http://github.com/dusty-nv/jetson-inference
 */

// #include "gstCamera.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <autobot/compound_img.h>
#include <autobot/detected_img.h>
#include <autobot/bounding_box.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/make_shared.hpp>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <chrono>
#include <jetson-inference/cudaMappedMemory.h>
#include <jetson-inference/cudaNormalize.h>
#include <jetson-inference/cudaFont.h>
#include <jetson-inference/detectNet.h>
#include <sl/Camera.hpp>


#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)

using namespace std;

bool signal_recieved = false;
int rate = 15;

//void sig_handler(int signo)
//{
	//if( signo == SIGINT )
	//{
		//printf("received SIGINT\n");
		//signal_recieved = true;
	//}
//}

cv::Mat toCVMat(sl::Mat &mat) {
    if (mat.getMemoryType() == sl::MEM_GPU)
        mat.updateCPUfromGPU();

    int cvType;
    switch (mat.getDataType()) {
        case sl::MAT_TYPE_32F_C1:
            cvType = CV_32FC1;
            break;
        case sl::MAT_TYPE_32F_C2:
            cvType = CV_32FC2;
            break;
        case sl::MAT_TYPE_32F_C3:
            cvType = CV_32FC3;
            break;
        case sl::MAT_TYPE_32F_C4:
            cvType = CV_32FC4;
            break;
        case sl::MAT_TYPE_8U_C1:
            cvType = CV_8UC1;
            break;
        case sl::MAT_TYPE_8U_C2:
            cvType = CV_8UC2;
            break;
        case sl::MAT_TYPE_8U_C3:
            cvType = CV_8UC3;
            break;
        case sl::MAT_TYPE_8U_C4:
            cvType = CV_8UC4;
            break;
    }
    return cv::Mat((int) mat.getHeight(), (int) mat.getWidth(), cvType, mat.getPtr<sl::uchar1>(sl::MEM_CPU), mat.getStepBytes(sl::MEM_CPU));
}

static const std::string OPENCV_WINDOW = "Image post bridge conversion";
static const std::string OPENCV_WINDOW2 = "Image post bit depth conversion";
static const std::string OPENCV_WINDOW3 = "Image post color conversion";
static const std::string OPENCV_WINDOW4 = "Image window";

class ObjectDetector
{
	ros::NodeHandle nh_;
    ros::Publisher detect_img_pub;
    cv_bridge::CvImagePtr cv_ptr;
    std::chrono::steady_clock::time_point prev;

    // zed object
    sl::InitParameters param;
    sl::Camera zed;
    sl::ERROR_CODE err;


	float confidence = 0.0f;


	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;

	detectNet* net = NULL;
	uint32_t maxBoxes = 0;
	uint32_t classes = 0;

	float4* gpu_data = NULL;

	uint32_t imgWidth;
	uint32_t imgHeight;
	size_t imgSize;

public:
	ObjectDetector(int argc, char** argv )
	{
		cout << "start constructor" << endl;

        // Try to initialize the ZED
        param.camera_fps = rate;
        param.camera_resolution = sl::RESOLUTION_VGA;
        param.camera_linux_id = 0;


        param.coordinate_units = sl::UNIT_METER;
        param.coordinate_system = sl::COORDINATE_SYSTEM_IMAGE;
        param.depth_mode = static_cast<sl::DEPTH_MODE> (1);
        param.sdk_verbose = true;
        param.sdk_gpu_id = -1;

        err = zed.open(param);
        cout << errorCode2str(err) << endl;
        
        
        cout << "ZED OPENED" << endl;

        detect_img_pub = nh_.advertise<autobot::detected_img>("/detected_image", 2);

		cv::namedWindow(OPENCV_WINDOW);

		cout << "Named a window" << endl;

        prev = std::chrono::steady_clock::now();

		/*
		 * create detectNet
		 */
		net = detectNet::Create(argc, argv);
		cout << "Created DetectNet" << endl;

		if( !net )
		{
			printf("obj_detect:   failed to initialize imageNet\n");

		}
        
        maxBoxes = net->GetMaxBoundingBoxes();
		printf("maximum bounding boxes:  %u\n", maxBoxes);
		classes  = net->GetNumClasses();


		/*
		 * allocate memory for output bounding boxes and class confidence
		 */

		if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
			!cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
		{
			printf("detectnet-console:  failed to alloc output memory\n");

		}
		cout << "Allocated CUDA mem" << endl;


		maxBoxes = net->GetMaxBoundingBoxes();
		printf("maximum bounding boxes:  %u\n", maxBoxes);
		classes  = net->GetNumClasses();
		cout << "Constructor operations complete" << endl;

	}

	~ObjectDetector()
	{
		cv::destroyWindow(OPENCV_WINDOW);
	}

	void detect()
	{
        // Get the parameters of the ZED images
        int zed_width = zed.getResolution().width;
        int zed_height = zed.getResolution().height;
        
        err = zed.grab();
        
        
        
        sl::Mat leftZEDMat, depthZEDMat;

        cv::Size cvSize(zed_width, zed_height);
        //cv::Mat leftImage(cvSize, CV_8UC3);
        //cv::Mat depthImage(cvSize, CV_8UC3);
        zed.retrieveImage(leftZEDMat, sl::VIEW_LEFT);
        //cv::cvtColor(toCVMat(leftZEDMat), leftImage, CV_RGBA2RGB);
        cv::Mat leftImage = toCVMat(leftZEDMat);
        zed.retrieveMeasure(depthZEDMat, sl::MEASURE_DEPTH);
        cv::Mat depthImage = toCVMat(depthZEDMat);

        
        leftImage.convertTo(leftImage,CV_32FC3);
        // convert color
        cv::cvtColor(leftImage,leftImage,CV_BGR2RGBA);

        //cv::imshow("SL::CAMERA",leftImage);

        // allocate GPU data if necessary
        if(!gpu_data){
            ROS_INFO("first allocation");
            CUDA(cudaMalloc(&gpu_data, leftImage.rows*leftImage.cols * sizeof(float4)));
        }else if(imgHeight != leftImage.rows || imgWidth != leftImage.cols){
            ROS_INFO("re allocation");
            // reallocate for a new image size if necessary
            CUDA(cudaFree(gpu_data));
            CUDA(cudaMalloc(&gpu_data, leftImage.rows*leftImage.cols * sizeof(float4)));
        }

		imgHeight = leftImage.rows;
		imgWidth = leftImage.cols;
		imgSize = leftImage.rows*leftImage.cols * sizeof(float4);
		float4* cpu_data = (float4*)(leftImage.data);

		// copy to device
		CUDA(cudaMemcpy(gpu_data, cpu_data, imgSize, cudaMemcpyHostToDevice));

		void* imgCUDA = NULL;


		// find object with detectNet
		int numBoundingBoxes = maxBoxes;

		if( net->Detect((float*)gpu_data, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU))
		{
			//printf("%i bounding boxes detected\n", numBoundingBoxes);

			int lastClass = 0;
			int lastStart = 0;

			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);

				printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);
                cv::rectangle( leftImage, cv::Point( bb[0], bb[1] ), cv::Point( bb[2], bb[3]), cv::Scalar( 255, 55, 0 ), +1, 4 );

			}

            
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            float fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(now - prev).count() ;

            prev = now;
            
			char str[256];
			sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", fps);
			cv::setWindowTitle(OPENCV_WINDOW, str);

		}


		// update image back to original

        leftImage.convertTo(leftImage,CV_8UC3);
        cv::cvtColor(leftImage,leftImage,CV_RGBA2BGR);

		// Update GUI Window
        cv::imshow(OPENCV_WINDOW, leftImage);
        cv::imshow("depth", depthImage);
		cv::waitKey(1);

        for( int n=0; n < numBoundingBoxes; n++ ) {
            
            const int nc = confCPU[n*2+1];
            float* bb = bbCPU + (n * 4);
            
            sensor_msgs::ImagePtr msg;
            
            float crop_width = bb[2] - bb[0];
            float crop_height = bb[3] - bb[1];
            float origin_x = bb[0];
            float origin_y = bb[1];
            
            // placeholders for squared boxes to classify pics
            float sq_crop_width;
            float sq_crop_height;
            float sq_origin_x;
            float sq_origin_y;
            

            printf("imgWidth: %u imgHeight: %u\n",imgWidth,imgHeight);
            // printf("BEFORE: origin_x: %f origin_y: %f crop_width: %f crop_height: %f\n",origin_x, origin_y, crop_width, crop_height);
            
            if (crop_width < crop_height) {
                float diff = crop_height - crop_width;
                printf("diff: %f\n",diff);
                sq_origin_x = origin_x - (diff / 2.0);
                sq_crop_width = crop_width + (diff / 2.0);
                sq_crop_height = crop_height;
            } else if (crop_width > crop_height) {
                float diff = crop_width - crop_height;
                printf("diff: %f\n",diff);
                sq_origin_y = origin_y - (diff / 2.0);
                sq_crop_height = crop_height + (diff / 2.0);
                sq_crop_width = crop_width;
            } else {
                sq_origin_x = origin_x; 
                sq_crop_width = crop_width;
                sq_crop_height = crop_height;
                
            }
            // printf("MIDDLE: origin_x: %f origin_y: %f crop_width: %f crop_height: %f\n",origin_x, origin_y, crop_width, crop_height);

            
            if (sq_origin_x < 0.0) {
                sq_origin_x = 0.0;
            }
            
            if (sq_origin_y < 0.0) {
                sq_origin_y = 0.0;
            }
            
            if (origin_x + crop_width >= (float)imgWidth ) {
                sq_crop_width = (float)imgWidth - sq_origin_x - 1.0;
            }
            
            if (sq_origin_y + sq_crop_height >= (float)imgHeight) {
                sq_crop_height = (float)imgHeight - sq_origin_y - 1.0;
            }
            

            printf("AFTER : origin_x: %f origin_y: %f crop_width: %f crop_height: %f\n",origin_x, origin_y, crop_width, crop_height);
            
            // apply squared bounding box on 
            cout << "croppeing image" << endl;
            cv::Mat croppedImage = leftImage(cv::Rect(sq_origin_x, sq_origin_y, sq_crop_width, sq_crop_height));
            // take the cropped version of the original bounding box on the depth image
            cout << "cropping depth" << endl;
            cv::Mat croppedDepthImage = depthImage(cv::Rect(origin_x, origin_y, crop_width, crop_height));

            //cv::resize(croppedImage, croppedImage, cv::Size(224,224));
            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", croppedImage).toImageMsg();
            sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", croppedDepthImage).toImageMsg();
            boost::shared_ptr<autobot::detected_img> detected_img = boost::make_shared<autobot::detected_img>();
            boost::shared_ptr<autobot::bounding_box> bbox = boost::make_shared<autobot::bounding_box>();
            
            bbox->origin_x = origin_x;
            bbox->origin_y = origin_y;
            bbox->height = crop_height;
            bbox->width = crop_width;
            
            detected_img->img = *img_msg.get();
            detected_img->depthImg = *depth_msg.get();
            detected_img->box = *bbox.get();

    

            detect_img_pub.publish<autobot::detected_img>(detected_img);
            cv::imshow("crop depth", croppedDepthImage);
            cv::imshow("crop", croppedImage);

            cv::waitKey(1);
        }

	}
};

int main( int argc, char** argv ) {
	cout << "starting node" << endl;
	printf("obj_detect\n  args (%i):  ", argc);

	ros::init(argc, argv, "obj_detector");
    ros::NodeHandle nh;
    ros::Rate detect_rate(rate);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);

	printf("\n\n");

	ObjectDetector ic(argc, argv);
    
    while(1) {
        ic.detect();
        detect_rate.sleep();
    }
	ros::spin();

	return 0;
}

