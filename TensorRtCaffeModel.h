/* *************************************************************   **************************************   
  *                     Comments Add File   
  *             Copy Rights by Hisense @2010-2046   
  *   
  * FileName: tensorrtmodel_caffe.h   
  *Function: the headfile of tensorrtmodel_caffe.cpp
  * Author:  
  * Email:  
  * Date: 2018-11-05   
  *   
  *************************************************************   **************************************   */  
#ifndef _TENSORRTCAFFEMODEL_H_
#define _TENSORRTCAFFEMODEL_H_

#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include "cuda_runtime_api.h"

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

static Logger gLogger;


extern void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 nvcaffeparser1::IPluginFactoryExt* pluginFactory,  //factory for plugin layers
					 IHostMemory *&gieModelStream)    ;// output buffer for the GIE model

extern void doInference(IExecutionContext& context, float* input, float* output, int batchSize);


#endif /* _TENSORRTMODEL_CAFFE_H_ */
