/* *************************************************************   **************************************   
  *                     Comments Add File   
  *             Copy Rights by Hisense @2010-2046   
  *   
  * FileName: sampleFaceDection.cpp   
  * Function: face dection testbench with the use of tensorrt under caffe mode
  * Author:  
  * Email:  
  * Date: 2018-11-05   
  *   
  *************************************************************   **************************************   */  
  
//TensorRT lib
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

//HC  lib
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


#include "TensorRtCaffeModel.h"
#include "pluginImpliment.h"


const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "detection_out";

#define MAX_FACE_NUM 20
cv::Mat gMean;

/********************************************************************
*
*name:storeEngineToFile
*function:store tensorRT engine to the txt file
*
*********************************************************************/
typedef struct  rect_box
{
	int		x;			
	int		y;
	int		width;
	int		height;
}rect_box;

typedef struct face_info
{
	int			faceNum;
	rect_box	faceRts[MAX_FACE_NUM];
	int			faceId[MAX_FACE_NUM];
	float		similarity[MAX_FACE_NUM];
}face_info;

int main(int argc, char** argv)
{
	face_info tmpFace;
	tmpFace.faceNum = 0;
	clock_t time_start, time_end;

	std::cout << "=========================" << std::endl;
	
	/*create TensorRT engine  from the caffe model and serialize it to a stream*/
	IHostMemory *gieModelStream{nullptr};
	PluginFactory parsePluginFactory;
	caffeToGIEModel("faceboxes_deploy01.prototxt", "faceboxes_iter_120000.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1,&parsePluginFactory, gieModelStream);
	parsePluginFactory.destroyPlugin();
	
	/* deserialize and    create   the runtime engine */
	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime!=nullptr);
	PluginFactory runPluginFactory;
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &runPluginFactory);
	assert(engine != nullptr);

	/*create contex*/
	IExecutionContext *context = engine->createExecutionContext();
	assert(context != nullptr);

	/*open    camera    or  video*/
	cv::VideoCapture cap(-1);
	if (!cap.isOpened()) {
		std::cout << "Failed to open video: " << std::endl;
		return 0;
	}
	
	/*create mean mat*/			
	std::vector<cv::Mat> channels;
	cv::Mat channel_B(1024, 1024, CV_32FC1,cv::Scalar(104.0f));
	cv::Mat channel_G(1024, 1024, CV_32FC1,cv::Scalar(117.0f));
	cv::Mat channel_R(1024, 1024, CV_32FC1,cv::Scalar(123.0f));
	channels.push_back(channel_B);
	channels.push_back(channel_G);
	channels.push_back(channel_R);

	cv::merge(channels, gMean);
	
	/*img data processing*/	
	cv::Mat img;
	while (true)
	{
		bool success = cap.read(img);

		if (!success) {
			std::cout << "Processing... " << std::endl;
			break;
		}
		
		/*restore the original img to draw the rect*/
		cv::Mat showIm = img.clone();
		
		/*data preprocess:convert img cv::Matdata to array data*/
		cv::Mat img_resize;
		cv::resize(img,img_resize,cv::Size(1024,1024));
		img_resize.convertTo(img_resize,CV_32F);

		cv::Mat img_mean;
		cv::subtract(img_resize, gMean , img_mean);
		img_mean = img_mean / 127.5;	
		
		int float_num = img_mean.cols * img_mean.rows  *img_mean.channels();
		float* img_data = new float[float_num];	
		memcpy(img_data, img_mean.data,sizeof(float)*float_num);

		time_start = clock();
		
		// run inference
		int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
		DimsCHW outputDims = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex));
		size_t outputSize = outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);
		float* img_out = new float[outputSize]();

		doInference(*context, img_data, img_out, 1);

		for(int i=0; i<10; i++)		
		{
			if(img_out[0] == -1)
			{
				img_out+=7;
				continue;
			}
			float tmpScore = img_out[2];
			cv::Rect tmpRt;
			if(tmpScore>0.5)
			{
				
				tmpRt.x = int(img_out[3]*img.cols);
				tmpRt.y = int(img_out[4]*img.rows);
				tmpRt.width = int(img_out[5]*img.cols)-tmpRt.x;
				tmpRt.height = int(img_out[6]*img.rows)-tmpRt.y;	
			}
			cv::rectangle(img,tmpRt,cv::Scalar(0, 255, 205), 2, 4, 0);
			img_out+=7;

		}
		cv::imshow("Face Boxes",img);
		cv::waitKey(10);
		
		/*print The time of detection per image */
		time_end = (double)(1000 * (clock() - time_start) / CLOCKS_PER_SEC);
		std::cout << "Face detection img number :" <<  tmpFace.faceNum << "consume  time:  " << time_end << "ms" << std::endl;

		//free memory
		delete[] img_out;
		delete[] img_data;
		img_data = NULL;
		img_out = NULL;

	}



	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	runPluginFactory.destroyPlugin();

	return 0;
}

