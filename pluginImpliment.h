/* ***************************************************************************************************   
  *                     Comments Add File   
  *             Copy Rights by Hisense @2010-2046   
  *   
  * FileName: pluginImpliment.h  
  * Function: tensorRT plugin layer impliment
  * Author:  
  * Email:  
  * Date: 2018-11-05   
  *   
  ****************************************************************************************************/  

#ifndef _PLUGIN_LAYER_H_
#define _PLUGIN_LAYER_H_

#include <cassert>
#include <iostream>
#include <cudnn.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <algorithm>
#include "NvCaffeParser.h"
#include <NvInferPlugin.h>
#include "NvInfer.h"
 
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

#define PRIORBOX_MAX_SIZE 4
/***********************************************************************************************************
*              struct  declaration
**********************************************************************************************************/
struct HiPrioxBoxParameters
{
 	float minSize[PRIORBOX_MAX_SIZE];
	float maxSize[PRIORBOX_MAX_SIZE];
	float aspectRatios[PRIORBOX_MAX_SIZE];
	float fixedSize[PRIORBOX_MAX_SIZE];
    float densitys[PRIORBOX_MAX_SIZE]; 
	float fixedRatios[PRIORBOX_MAX_SIZE];
 	int  numMinSize, numMaxSize, numAspectRatios, numFixedSize,numDensity,numFixedRatios;
 	bool flip;
 	bool clip;
	int numPriors;
 	float variance[4];
 	int imgH, imgW;
 	float stepH, stepW;
 	float offset;
};

/***********************************************************************************************************
*              class  declaration
**********************************************************************************************************/

//priorbox layer plugin
class PriorBoxPlugin : nvinfer1::IPluginExt
{
public:

	PriorBoxPlugin(HiPrioxBoxParameters params)
	{
		mPriorBoxParamters = params;

		//check Whether to carry out ratio transformation under fixed mode
		if (params.fixedRatios > 0 && params.aspectRatios > 0)
		{
			std::cout << "can not provide fixed_ratio and aspect_ratio simultaneously." << std::endl;
		}

		if (params.numAspectRatios == 0)
		{
			mPriorBoxParamters.numAspectRatios = 1;
		}

		if (params.numFixedSize > 0)
		{
			mPriorBoxParamters.numPriors = mPriorBoxParamters.numAspectRatios * mPriorBoxParamters.numFixedSize;
		}

		if (params.numDensity > 0)
		{
			for(int i=0;i<params.numDensity;i++)
			{
				assert(params.densitys[i] > 0);
				mPriorBoxParamters.densitys[i] = params.densitys[i];
				if (params.numFixedRatios > 0)
				{
					mPriorBoxParamters.numPriors += (mPriorBoxParamters.numFixedRatios *( pow(mPriorBoxParamters.densitys[i],2)-1)) ;
				}
				else
				{
					mPriorBoxParamters.numPriors += (mPriorBoxParamters.numAspectRatios *( pow(mPriorBoxParamters.densitys[i],2)-1)) ;
				}
			}
		}
		std::cout << "PriorBoxPlugin" << std::endl;
		//HiParamPrintf(mPriorBoxParamters);
	}

	PriorBoxPlugin(const void*data ,size_t length)
	{
		const char* d = static_cast<const char*>(data), *a = d;
		read(d,mDataType);
		read(d,mPriorBoxParamters);
		read(d,mLayerWidth);
		read(d,mLayerHeight);
		read(d,mImgWidth);
		read(d,mImgHeight);
		std::cout << "PriorBoxPlugin::deserialize" << std::endl;
		assert(d == a+length);
	}
	~PriorBoxPlugin(){}

	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims)override
	{
		assert(index == 0 && nbInputDims == 2 && inputs[0].nbDims==3  );
		//dim=layer_height * layer_width * num_priors_ * 4;		
		top_data_size = inputs[0].d[1] * inputs[0].d[2] *mPriorBoxParamters.numPriors *4;
		
		return DimsCHW(1,2, top_data_size);
	}

	bool supportsFormat(DataType type, PluginFormat format) const override
	{ 
		return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; 
	}


	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
	{
		assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
		assert(nbInputs == 2);
		mDataType = type;		
		
		mLayerWidth = inputDims[0].d[1];
		mLayerHeight = inputDims[0].d[2];
		if (mPriorBoxParamters.imgH == 0 || mPriorBoxParamters.imgW == 0)
		{
			mImgWidth = inputDims[1].d[1];
			mImgHeight =  inputDims[1].d[2];
		}
		else
		{
			mImgWidth = mPriorBoxParamters.imgW;
			mImgHeight = mPriorBoxParamters.imgH;

		}
	}

	int initialize() override
	{
		std::cout << "initialize" << std::endl;
		return 0;
	}

	virtual void terminate() override { ; }

	virtual size_t getWorkspaceSize(int batchSize) const override 
	{ 
		std::cout << "getWorkspaceSize:" << (top_data_size * batchSize) << std::endl;
		return top_data_size * batchSize * 2; 
	}
	virtual int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override;
	
	virtual size_t getSerializationSize() override
	{
		size_t size = sizeof(mDataType) + sizeof(mPriorBoxParamters)+sizeof(mLayerWidth)+sizeof(mLayerHeight)+sizeof(mImgWidth)+sizeof(mImgHeight);
		std::cout << "getSerializationSize" << size << std::endl;
		return (size*type2size(mDataType));
	}

	virtual void serialize(void *buffer)override
	{
		char* d = static_cast<char*>(buffer), *a = d;
		write(d,mDataType);
		write(d,mPriorBoxParamters);
		write(d,mLayerWidth);
		write(d,mLayerHeight);
		write(d,mImgWidth);
		write(d,mImgHeight);
		assert(d == a + getSerializationSize());
		std::cout << "serialize" << std::endl;
	}
	
private:

	size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
private:
	HiPrioxBoxParameters mPriorBoxParamters;
	int top_data_size,mImgHeight,mImgWidth,mLayerWidth,mLayerHeight;
	DataType mDataType{DataType::kFLOAT};
};



class PluginFactory : public nvinfer1::IPluginFactory,public nvcaffeparser1::IPluginFactoryExt
{
public:
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override;
	
	nvinfer1::IPlugin* createPlugin(const char* layerName,const void* seriaData,size_t seriaLength)override;
    // caffe parser plugin implementation
	bool isPlugin(const char* name) override { return isPluginExt(name); }
	
    bool isPluginExt(const char* name) override ;
    
    void destroyPlugin();

    void (*pluginDeleter)( nvinfer1::plugin::INvPlugin*) {[]( nvinfer1::plugin::INvPlugin* ptr) {ptr->destroy();}};
	//priorbox layer
    std::unique_ptr< PriorBoxPlugin> inception3_priorbox_layer{nullptr};
 	std::unique_ptr< PriorBoxPlugin> conv3_2_priorbox_layer{nullptr};
 	std::unique_ptr< PriorBoxPlugin> conv4_2_priorbox_layer{nullptr};
	//detection output layer
	std::unique_ptr< nvinfer1::plugin::INvPlugin, decltype(pluginDeleter)> 
 mDetection_out{nullptr, pluginDeleter};
	
};







#endif
