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
#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "detectNet.h"


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
	image_transport::Publisher image_pub_;

	float confidence = 0.0f;
	

	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/image_raw", 1,
      &ImageConverter::imageCb, this);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\ndetectnet-camera:  failed to capture frame\n");

		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
			printf("detectnet-camera:  failed to convert from NV12 to RGBA\n");

		// classify image with detectNet
		int numBoundingBoxes = maxBoxes;
	
		if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
		{
			printf("%i bounding boxes detected\n", numBoundingBoxes);
		
			int lastClass = 0;
			int lastStart = 0;
			
			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);
				
				printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
				
				if( nc != lastClass || n == (numBoundingBoxes - 1) )
				{
					if( !net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), 
						                        bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
						printf("detectnet-console:  failed to draw boxes\n");
						
					lastClass = nc;
					lastStart = n;

					CUDA(cudaDeviceSynchronize());
				}
			}
		
			/*if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
				
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 10, 10, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}*/
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
		 						   camera->GetWidth(), camera->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}




    // Draw an example circle on the video stream
    if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
      cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));


    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);

  }
};

int main( int argc, char** argv )
{
	printf("obj_detect\n  args (%i):  ", argc);

	ros::init(argc, argv, "image_converter");
	ImageConverter ic;


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
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


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
	

	/*
	 * create detectNet
	 */
	detectNet* net = detectNet::Create(argc, argv);
	
	if( !net )
	{
		printf("obj_detect:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * allocate memory for output bounding boxes and class confidence
	 */
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	

	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}
	

	/*
	 * create openGL window
	 */
	//glDisplay* display = glDisplay::Create();
	//glTexture* texture = NULL;
	
	//if( !display ) {
		//printf("\ndetectnet-camera:  failed to create openGL display\n");
	//}
	//else
	//{
		//texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		//if( !texture )
			//printf("detectnet-camera:  failed to create openGL texture\n");
	//}
	
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\ndetectnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */

		
	
	ros::spin();

	
	printf("\ndetectnet-camera:  un-initializing video device\n");
	
	
	/*
	 * shutdown the camera device
	 */
	//if( camera != NULL )
	//{
		//delete camera;
		//camera = NULL;
	//}

	//if( display != NULL )
	//{
		//delete display;
		//display = NULL;
	//}
	
	//printf("detectnet-camera:  video device has been un-initialized.\n");
	//printf("detectnet-camera:  this concludes the test of the video device.\n");
	return 0;
}

