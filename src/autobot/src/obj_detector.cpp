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
		

bool signal_recieved = false;

//void sig_handler(int signo)
//{
	//if( signo == SIGINT )
	//{
		//printf("received SIGINT\n");
		//signal_recieved = true;
	//}
//}

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter
{
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;



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
	ImageConverter(int argc, char** argv ): it_(nh_)
	{
		// Subscrive to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/camera/image_raw", 1,
		  &ImageConverter::imageCb, this);

		cv::namedWindow(OPENCV_WINDOW);


		/*
		 * create detectNet
		 */
		net = detectNet::Create(argc, argv);

		if( !net )
		{
			printf("obj_detect:   failed to initialize imageNet\n");
			
		}


		/*
		 * allocate memory for output bounding boxes and class confidence
		 */

		if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
			!cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
		{
			printf("detectnet-console:  failed to alloc output memory\n");
			
		} 
		
		maxBoxes = net->GetMaxBoundingBoxes();		
		printf("maximum bounding boxes:  %u\n", maxBoxes);
		classes  = net->GetNumClasses();
	}

	~ImageConverter()
	{
		cv::destroyWindow(OPENCV_WINDOW);
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{
		cv::Mat cv_im;
		try
		{
			cv_im = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
			cv_im.convertTo(cv_im,CV_32FC3);
			// convert color
			cv::cvtColor(cv_im,cv_im,CV_BGR2RGBA);
		}
		catch (cv_bridge::Exception& e)
		{
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
		}
		
		
		imgHeight = cv_im.rows;
		imgWidth = cv_im.cols;
		imgSize = cv_im.rows*cv_im.cols * sizeof(float4);
		float4* cpu_data = (float4*)(cv_im.data);

		// copy to device
		CUDA(cudaMemcpy(gpu_data, cpu_data, imgSize, cudaMemcpyHostToDevice));

		//void* imgCPU  = NULL;
		void* imgCUDA = NULL;


		// classify image with detectNet
		int numBoundingBoxes = maxBoxes;
	
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


		// update display
		
	




		// Draw an example circle on the video stream
		if (cv_im.rows > 60 && cv_im.cols > 60)
		  cv::circle(cv_im, cv::Point(50, 50), 10, CV_RGB(255,0,0));


		// Update GUI Window
		cv::imshow(OPENCV_WINDOW, cv_im);
		cv::waitKey(3);

	}
};

int main( int argc, char** argv ) {
	printf("obj_detect\n  args (%i):  ", argc);

	ros::init(argc, argv, "image_converter");
	ImageConverter ic(argc, argv);


	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	

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

