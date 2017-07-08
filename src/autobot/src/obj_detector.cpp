/*
 * http://github.com/dusty-nv/jetson-inference
 */

// #include "gstCamera.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <jetson-inference/cudaMappedMemory.h>
#include <jetson-inference/cudaNormalize.h>
#include <jetson-inference/cudaFont.h>

#include <jetson-inference/detectNet.h>


#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)

using namespace std;

bool signal_recieved = false;

//void sig_handler(int signo)
//{
	//if( signo == SIGINT )
	//{
		//printf("received SIGINT\n");
		//signal_recieved = true;
	//}
//}

static const std::string OPENCV_WINDOW = "Image post bridge conversion";
static const std::string OPENCV_WINDOW2 = "Image post bit depth conversion";
static const std::string OPENCV_WINDOW3 = "Image post color conversion";
static const std::string OPENCV_WINDOW4 = "Image window";

class ImageConverter
{
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
    cv::Mat cv_im;
    cv_bridge::CvImagePtr cv_ptr;


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
	ImageConverter(int argc, char** argv ) : it_(nh_)
	{
		cout << "start constructor" << endl;
		// Subscrive to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/left/image_rect_color", 1,
		  &ImageConverter::imageCb, this);

		cv::namedWindow(OPENCV_WINDOW);
		cv::namedWindow(OPENCV_WINDOW2);
		cv::namedWindow(OPENCV_WINDOW3);
		cv::namedWindow(OPENCV_WINDOW4);
		cout << "Named a window" << endl;

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

	~ImageConverter()
	{
		cv::destroyWindow(OPENCV_WINDOW);
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{

		try
		{
			cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
			cv_im = cv_ptr->image;
            cv::imshow(OPENCV_WINDOW, cv_im);
			cv_im.convertTo(cv_im,CV_32FC3);
			//ROS_INFO("Image width %d height %d", cv_im.cols, cv_im.rows);
            cv::imshow(OPENCV_WINDOW2, cv_im);

			// convert color
			cv::cvtColor(cv_im,cv_im,CV_BGR2RGBA);
            cv::imshow(OPENCV_WINDOW3, cv_im);

		}
		catch (cv_bridge::Exception& e)
		{
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
		}


        // allocate GPU data if necessary
        if(!gpu_data){
            ROS_INFO("first allocation");
            CUDA(cudaMalloc(&gpu_data, cv_im.rows*cv_im.cols * sizeof(float4)));
        }else if(imgHeight != cv_im.rows || imgWidth != cv_im.cols){
            ROS_INFO("re allocation");
            // reallocate for a new image size if necessary
            CUDA(cudaFree(gpu_data));
            CUDA(cudaMalloc(&gpu_data, cv_im.rows*cv_im.cols * sizeof(float4)));
        }
        //ROS_INFO("allocation done");

		imgHeight = cv_im.rows;
		imgWidth = cv_im.cols;
		imgSize = cv_im.rows*cv_im.cols * sizeof(float4);
		float4* cpu_data = (float4*)(cv_im.data);

        //ROS_INFO("cuda memcpy begin");
		// copy to device
		CUDA(cudaMemcpy(gpu_data, cpu_data, imgSize, cudaMemcpyHostToDevice));
        //ROS_INFO("cuda memcpy end");

		//void* imgCPU  = NULL;
		void* imgCUDA = NULL;


		// classify image with detectNet
		int numBoundingBoxes = maxBoxes;

        //ROS_INFO("parameters gpu_data: %p imgWidth: %d imgHeight: %d bbCPU: %p numBB pointer: %p numBB: %d",gpu_data, imgWidth, imgHeight,bbCPU, &numBoundingBoxes, numBoundingBoxes);


		if( net->Detect((float*)gpu_data, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU))
		{
			printf("%i bounding boxes detected\n", numBoundingBoxes);

			int lastClass = 0;
			int lastStart = 0;

			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);

				printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);
				//cv::rectangle(cv_im, Rect rec, Scalar( rand()&255, rand()&255, rand()&255 ),1, LINE_8, 0 )


				//if( nc != lastClass || n == (numBoundingBoxes - 1) )
				//{
					//if( !net->DrawBoxes((float*)gpu_data, (float*)gpu_data, imgWidth, imgHeight,
						                        //bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
						//printf("detectnet-console:  failed to draw boxes\n");


					//lastClass = nc;
					//lastStart = n;

					////CUDA(cudaDeviceSynchronize());
				//}
			}

			/*if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));

				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 10, 10, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}*/


			char str[256];
			sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", -1.0);
			//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
			cv::setWindowTitle(OPENCV_WINDOW, str);

		}


		// update image back to original

        cv_im.convertTo(cv_im,CV_8UC3);
        cv::cvtColor(cv_im,cv_im,CV_RGBA2BGR);




		// Draw an example circle on the video stream
		if (cv_im.rows > 60 && cv_im.cols > 60)
		  cv::circle(cv_im, cv::Point(50, 50), 10, CV_RGB(255,0,0));

        // test image
        cv::String filepath = "/home/ubuntu/Desktop/dog.jpg";
        cv::Mat testpic = cv::imread(filepath);
		// Update GUI Window
        cv::imshow(OPENCV_WINDOW4, cv_im);
		//cv::imshow(OPENCV_WINDOW, cv_im);
		//cv::imshow(OPENCV_WINDOW, testpic);
		cv::waitKey(3);

	}
};

int main( int argc, char** argv ) {
	cout << "starting node" << endl;
	printf("obj_detect\n  args (%i):  ", argc);

	ros::init(argc, argv, "obj_detector");
    ros::NodeHandle nh;


	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);

	printf("\n\n");

	ImageConverter ic(argc, argv);


	/*
	 * parse network type from CLI arguments
	 */
	/*detectNet::NetworkType networkType = detectNet::PEDNET_MULTI;

	if( argc > 1 )
	{
		if( strcmp(argv[1], "multiped") == 0 || strcmp(argv[1], "pednet") == 0 || strcmp(argv[1], "multiped-500") == 0 )
			networkType = detectNet::PEDNET_MULTI;
		else if( strcmp(argv[1], "ped-100") == 0 )
			networkType = detectNet::PEDNET;
		else if( strcmp(argv[1], "facenet") == 0 || strcmp(argv[1], "facenet-120") == 0 || strcmp(argv[1], "face-120") == 0 )
			networkType = detectNet::FACENET;
	}*/

	//if( signal(SIGINT, sig_handler) == SIG_ERR )
		//printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	//gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);

	//if( !camera )
	//{
		//printf("\ndetectnet-camera:  failed to initialize video device\n");
		//return 0;
	//}

	//printf("\ndetectnet-camera:  successfully initialized video device\n");
	//printf("    width:  %u\n", camera->GetWidth());
	//printf("   height:  %u\n", camera->GetHeight());
	//printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());




	ros::spin();

	return 0;
}

