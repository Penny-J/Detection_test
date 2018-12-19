/* ***************************************************************************************************   
  *                     Comments Add File   
  *             Copy Rights by Hisense @2010-2046   
  *   
  * FileName: pluginImpliment.cpp  
  * Function: tensorRT plugin layer impliment
  * Author:  
  * Email:  
  * Date: 2018-11-05   
  *   
  ****************************************************************************************************/  
#include "pluginImpliment.h"
/****************************************************/
//pluginfactory
/****************************************************/
nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
	assert(PluginFactory::isPluginExt(layerName));
	std::cout << "====layer: "<< layerName <<" create start===" << std::endl;
	if(!strcmp(layerName,"inception3_priorbox"))
	{
		assert(inception3_priorbox_layer.get() == nullptr);
		HiPrioxBoxParameters params = {0};
		params.fixedSize[0] = 32;
		params.fixedSize[1] = 64;
		params.fixedSize[2] = 128;
		params.numFixedSize = 3;
		assert(params.numFixedSize <= PRIORBOX_MAX_SIZE);
		params.densitys[0] = 4;
		params.densitys[1] = 2;
		params.densitys[2] = 1;
		params.numDensity = 3;
		assert(params.numDensity <= PRIORBOX_MAX_SIZE);
		params.aspectRatios[0] = 1;
		params.numAspectRatios = 1;
		assert(params.numAspectRatios <= PRIORBOX_MAX_SIZE);
		params.clip = 0;
		params.flip = 0;
		params.variance[0] = 0.1;
		params.variance[1] = 0.1;
		params.variance[2] = 0.2;
		params.variance[3] = 0.2;
		params.stepH = 32;
		params.stepW = 32;
		params.offset = 0.5;
		inception3_priorbox_layer = std::unique_ptr<PriorBoxPlugin>(new PriorBoxPlugin(params));
		
		return (nvinfer1::IPlugin*)inception3_priorbox_layer.get();			
	}

	else if (!strcmp(layerName,"conv3_2_priorbox"))
	{
		assert(conv3_2_priorbox_layer.get() == nullptr);
		HiPrioxBoxParameters params = {0};
		params.fixedSize[0] = 256;
		params.numFixedSize = 1;
		assert(params.numFixedSize <= PRIORBOX_MAX_SIZE);
		params.densitys[0] = 1;
		params.numDensity = 1;
		assert(params.numDensity <= PRIORBOX_MAX_SIZE);
		params.aspectRatios[0] = 1;
		params.numAspectRatios = 1;
		assert(params.numAspectRatios <= PRIORBOX_MAX_SIZE);
		params.variance[0] = 0.1;
		params.variance[1] = 0.1;
		params.variance[2] = 0.2;
		params.variance[3] = 0.2;
		params.stepH = 64;
		params.stepW = 64;
		params.offset = 0.5;

		conv3_2_priorbox_layer = std::unique_ptr<PriorBoxPlugin>(new PriorBoxPlugin(params));
		return (nvinfer1::IPlugin*)conv3_2_priorbox_layer.get();	
	}

	else if (!strcmp(layerName,"conv4_2_priorbox"))
	{
		assert(conv4_2_priorbox_layer.get() == nullptr);
		HiPrioxBoxParameters params = {0};
		params.numFixedSize = 1;
		assert(params.numFixedSize <= PRIORBOX_MAX_SIZE);
		params.densitys[0] = 1;
		params.numDensity = 1;
		assert(params.numDensity <= PRIORBOX_MAX_SIZE);
		params.aspectRatios[0] = 1;
		params.numAspectRatios = 1;
		assert(params.numAspectRatios <= PRIORBOX_MAX_SIZE);
		params.variance[0] = 0.1;
		params.variance[1] = 0.1;
		params.variance[2] = 0.2;
		params.variance[3] = 0.2;
		params.stepH = 128;
		params.stepW = 128;
		params.offset = 0.5;

		conv4_2_priorbox_layer = std::unique_ptr<PriorBoxPlugin>(new PriorBoxPlugin(params));
		return  (nvinfer1::IPlugin*)conv4_2_priorbox_layer.get();	
	}

	else if (!strcmp(layerName,"detection_out"))
	{
		assert(mDetection_out.get() == nullptr);
		plugin::DetectionOutputParameters params;
		params.shareLocation = true;
		params.varianceEncodedInTarget = true;
		params.backgroundLabelId = 0;
		params.numClasses = 2;
		params.topK = 100;
		params.keepTopK = 100;
		params.confidenceThreshold = 0.1;
		params.nmsThreshold = 0.45;
		params.codeType = CodeTypeSSD::CENTER_SIZE;
		params.inputOrder[0] = 0;
		params.inputOrder[1] = 1;
		params.inputOrder[2] = 2;
		params.confSigmoid = true;
		params.isNormalized = true;

		mDetection_out = std::unique_ptr<nvinfer1::plugin::INvPlugin, decltype(pluginDeleter)>(plugin::createSSDDetectionOutputPlugin(params),pluginDeleter);
		std::cout << "====layer: "<< layerName <<" create succ===" << std::endl;
		return mDetection_out.get();
	}

	else
	{
		std::cout << "the layer is not exit:" << layerName << std::endl;
		assert(0);
		return nullptr;
	}

}

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName,const void* seriaData,size_t serialLength)
{
	assert(PluginFactory::isPluginExt(layerName));
	if(!strcmp(layerName,"inception3_priorbox"))
	{
		assert(inception3_priorbox_layer.get() == nullptr);
		inception3_priorbox_layer = std::unique_ptr<PriorBoxPlugin>(new PriorBoxPlugin(seriaData,serialLength));

		return (nvinfer1::IPlugin*)inception3_priorbox_layer.get();
	}
	else if(!strcmp(layerName,"conv3_2_priorbox"))
	{
		assert(conv3_2_priorbox_layer.get() == nullptr);
		conv3_2_priorbox_layer = std::unique_ptr<PriorBoxPlugin>(new PriorBoxPlugin(seriaData,serialLength));

		return (nvinfer1::IPlugin*)conv3_2_priorbox_layer.get();
	}
	else if(!strcmp(layerName,"conv4_2_priorbox"))
	{
		assert(conv4_2_priorbox_layer.get() == nullptr);
		conv4_2_priorbox_layer = std::unique_ptr<PriorBoxPlugin>(new PriorBoxPlugin(seriaData,serialLength));

		return (nvinfer1::IPlugin*)conv4_2_priorbox_layer.get();
	}
	else if(!strcmp(layerName,"detection_out"))
	{
		assert(mDetection_out.get() == nullptr);
		mDetection_out = std::unique_ptr<nvinfer1::plugin::INvPlugin, decltype(pluginDeleter)>(plugin::createSSDDetectionOutputPlugin(seriaData,serialLength),pluginDeleter);

		return mDetection_out.get();
	}
	
	else
	{
		std::cout << "the layer is not exit:" << layerName << std::endl;
		assert(0);
		return nullptr;
	}
}

bool PluginFactory::isPluginExt(const char* name)
{
	return (!strcmp(name,"inception3_priorbox")
			||!strcmp(name,"conv3_2_priorbox")
			||!strcmp(name,"conv4_2_priorbox")
			||!strcmp(name,"detection_out")
			);
}

void PluginFactory::destroyPlugin()
{
std::cout << "==========destroyPlugin=========:" << std::endl;
	inception3_priorbox_layer.release();
	inception3_priorbox_layer = nullptr;
	conv3_2_priorbox_layer.release();
	conv3_2_priorbox_layer = nullptr;
	conv4_2_priorbox_layer.release();
	conv4_2_priorbox_layer = nullptr;
	mDetection_out.release();
	mDetection_out = nullptr;
}
int PriorBoxPlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void* workspace,cudaStream_t stream)
{
	float step_h,step_w;
	assert(outputs);
	if (mPriorBoxParamters.stepH == 0 ||mPriorBoxParamters.stepW == 0)
	{
		step_w = mImgWidth/mLayerWidth;
		step_h = mImgHeight/mLayerHeight;
	}
	else
	{
		step_w = mPriorBoxParamters.stepW;
		step_h = mPriorBoxParamters.stepH;
	}

	int dim = mLayerWidth * mLayerHeight *mPriorBoxParamters.numPriors * 4;
	float top_data[dim];
	int idx = 0;
	std::cout << "enqueue::dim:" << dim <<" mLayerWidth:" << mLayerWidth << " mLayerHeight" << mLayerHeight<<std::endl;
	for (int h = 0;h < mLayerHeight;++h)
	{
		for(int w=0;w<mLayerWidth;++w)
		{
			float center_x = (w + mPriorBoxParamters.offset) * step_w;
	        float center_y = (h + mPriorBoxParamters.offset) * step_h;
	        float box_width, box_height;
			
			//ssd
			for (int s = 0;s<mPriorBoxParamters.numMinSize;++s)
			{
				std::cout << "enqueue::ssd" << std::endl;
				int min_size_ = mPriorBoxParamters.minSize[s];
				// first prior: aspect_ratio = 1, size = min_size
		        box_width = box_height = min_size_;
		        // xmin
		        top_data[idx++] = (center_x - box_width / 2.) / mImgWidth;
		        // ymin
		        top_data[idx++] = (center_y - box_height / 2.) / mImgHeight;
		        // xmax
		        top_data[idx++] = (center_x + box_width / 2.) / mImgWidth;
		        // ymax
		        top_data[idx++] = (center_y + box_height / 2.) / mImgHeight;

		        if (mPriorBoxParamters.numMaxSize > 0) 
				{
		          int max_size_ = mPriorBoxParamters.maxSize[s];
		          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
		          box_width = box_height = sqrt(min_size_ * max_size_);
		          // xmin
		          top_data[idx++] = (center_x - box_width / 2.) / mImgWidth;
		          // ymin
		          top_data[idx++] = (center_y - box_height / 2.) / mImgHeight;
		          // xmax
		          top_data[idx++] = (center_x + box_width / 2.) / mImgWidth;
		          // ymax
		          top_data[idx++] = (center_y + box_height / 2.) / mImgHeight;
		        }
				// rest of priors
		        for (int r = 0; r < mPriorBoxParamters.numAspectRatios; ++r) 
				{
		          float ar = mPriorBoxParamters.aspectRatios[r];
		          if (fabs(ar - 1.) < 1e-6)
				  {
		            continue;
		          }
		          box_width = min_size_ * sqrt(ar);
		          box_height = min_size_ / sqrt(ar);
		          // xmin
		          top_data[idx++] = (center_x - box_width / 2.) / mImgWidth;
		          // ymin
		          top_data[idx++] = (center_y - box_height / 2.) / mImgHeight;
		          // xmax
		          top_data[idx++] = (center_x + box_width / 2.) / mImgWidth;
		          // ymax
		          top_data[idx++] = (center_y + box_height / 2.) / mImgHeight;
		        }
							
			}

			
			//faceboxs
			for (int s=0; s<mPriorBoxParamters.numFixedSize;++s)
			{
				int fixed_size_ = mPriorBoxParamters.fixedSize[s];
				if (mPriorBoxParamters.numFixedRatios > 0)
				{
					 for (int r = 0; r < mPriorBoxParamters.numFixedRatios; ++r) 
					 {
						float ar = mPriorBoxParamters.fixedRatios[r];
						int density_ = mPriorBoxParamters.densitys[s]; 
						int shift = mPriorBoxParamters.fixedSize[s] / density_;
						float box_width_ratio = mPriorBoxParamters.fixedSize[s] * sqrt(ar);
						float box_height_ratio = mPriorBoxParamters.fixedSize[s] / sqrt(ar);
						for (int r = 0 ; r < density_ ; ++r){
						  for (int c = 0 ; c < density_ ; ++c){
							float center_x_temp = center_x - fixed_size_ / 2 + shift/2. + c*shift;
							float center_y_temp = center_y - fixed_size_ / 2 + shift/2. + r*shift;
							// xmin
							top_data[idx++] = (center_x_temp - box_width_ratio / 2.) / mImgWidth >=0 ? (center_x_temp - box_width_ratio / 2.) / mImgWidth : 0. ;
							// ymin
							top_data[idx++] = (center_y_temp - box_height_ratio / 2.) / mImgHeight >= 0 ? (center_y_temp - box_height_ratio / 2.) / mImgHeight : 0.;
							// xmax
							top_data[idx++] = (center_x_temp + box_width_ratio / 2.) / mImgWidth <= 1 ? (center_x_temp + box_width_ratio / 2.) / mImgWidth : 1.;
							// ymax
							top_data[idx++] = (center_y_temp + box_height_ratio / 2.) / mImgHeight <= 1 ? (center_y_temp + box_height_ratio / 2.) / mImgHeight : 1.;
						  }
						}
					  }
				}
				else
				{
					//this code added by gaozhihua for density anchor box
			          if (mPriorBoxParamters.numDensity > 0) 
					  {
				            int densitys_ = mPriorBoxParamters.densitys[s]; 
				            int shift = mPriorBoxParamters.fixedSize[s] / densitys_;
				            for (int r = 0 ; r < densitys_ ; ++r){
				              for (int c = 0 ; c < densitys_ ; ++c){
				                float center_x_temp = center_x - fixed_size_ / 2 + shift/2. + c*shift;
				                float center_y_temp = center_y - fixed_size_ / 2 + shift/2. + r*shift;
				                // xmin
				                top_data[idx++] = (center_x_temp - box_width / 2.) / mImgWidth >=0 ? (center_x_temp - box_width / 2.) / mImgWidth : 0. ;
				                // ymin
				                top_data[idx++] = (center_y_temp - box_height / 2.) / mImgHeight >= 0 ? (center_y_temp - box_height / 2.) / mImgHeight : 0.;
				                // xmax
				                top_data[idx++] = (center_x_temp + box_width / 2.) / mImgWidth <= 1 ? (center_x_temp + box_width / 2.) / mImgWidth : 1.;
				                // ymax
				                top_data[idx++] = (center_y_temp + box_height / 2.) / mImgHeight <= 1 ? (center_y_temp + box_height / 2.) / mImgHeight : 1.;
				              }
				            }
			          }
			          //rest of priors   aspect_ratios_=1.，这步不执行
			          for (int r = 0; r < mPriorBoxParamters.numAspectRatios ; ++r) 
					  {
			            	float ar = mPriorBoxParamters.aspectRatios[r];
				            if (fabs(ar - 1.) < 1e-6)
							{
				              continue;
				            }
				            int density_ = mPriorBoxParamters.densitys[s]; 
					        int shift = mPriorBoxParamters.fixedSize[s] / density_;
				            float box_width_ratio = mPriorBoxParamters.fixedSize[s] * sqrt(ar);
				            float box_height_ratio = mPriorBoxParamters.fixedSize[s] / sqrt(ar);
				            for (int r = 0 ; r < density_ ; ++r){
				              for (int c = 0 ; c < density_ ; ++c){
				                float center_x_temp = center_x - fixed_size_ / 2 + shift/2. + c*shift;
				                float center_y_temp = center_y - fixed_size_ / 2 + shift/2. + r*shift;
				                // xmin
				                top_data[idx++] = (center_x_temp - box_width_ratio / 2.) / mImgWidth >=0 ? (center_x_temp - box_width_ratio / 2.) / mImgWidth : 0. ;
				                // ymin
				                top_data[idx++] = (center_y_temp - box_height_ratio / 2.) / mImgHeight >= 0 ? (center_y_temp - box_height_ratio / 2.) / mImgHeight : 0.;
				                // xmax
				                top_data[idx++] = (center_x_temp + box_width_ratio / 2.) / mImgWidth <= 1 ? (center_x_temp + box_width_ratio / 2.) / mImgWidth : 1.;
				                // ymax
				                top_data[idx++] = (center_y_temp + box_height_ratio / 2.) / mImgHeight <= 1 ? (center_y_temp + box_height_ratio / 2.) / mImgHeight : 1.;
				              }
        					}
      					}
				}
			}
			

		}
	}

	// clip the prior's coordidate such that it is within [0, 1]
	if (mPriorBoxParamters.clip) 
	{
	    for (int d = 0; d < mPriorBoxParamters.numPriors; ++d)
		{
	      top_data[d] = std::min(std::max(top_data[d], 0.0f), 1.0f);
	    }
	}

	assert(dim == idx);
	// set the variance.
	float variance_data[dim];
	int count =0;
	for (int h = 0; h < mLayerHeight; ++h) {
		for (int w = 0; w < mLayerWidth; ++w) {
			for (int i = 0; i < mPriorBoxParamters.numPriors; ++i) {
				for (int j = 0; j < 4; ++j) {
					variance_data[count] = mPriorBoxParamters.variance[j];
					++count;
				}
			}
		}
	}

	cudaMemcpyAsync(outputs[0],top_data,(dim*sizeof(float)),cudaMemcpyDeviceToDevice,stream);
	cudaMemcpyAsync(outputs[1],variance_data,(dim*sizeof(float)),cudaMemcpyDeviceToDevice,stream);
	std::cout << "enqueue::end" <<std::endl;
	return 0;
}

