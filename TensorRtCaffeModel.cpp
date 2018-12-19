/* *************************************************************   **************************************   
  *                     Comments Add File   
  *             Copy Rights by Hisense @2010-2046   
  *   
  * FileName: tensorrtmodel_caffe.cpp   
  *Function:common interface about the use as follows:
  	(1)create TensorRT engine         from the caffe model and serialize it to a stream
  	(2)how to carry out the engine to inference 
  * Author:  
  * Email:  
  * Date: 2018-11-05   
  *   
  *************************************************************   **************************************   */  
#include "TensorRtCaffeModel.h"
#include "pluginImpliment.h"

extern const char* INPUT_BLOB_NAME;
extern const char* OUTPUT_BLOB_NAME;
const std::vector<std::string> directories{ "data/detection_test/" };


void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,// batch size - NB must be at least as large as the batch we want to run with)
					 nvcaffeparser1::IPluginFactoryExt* pluginFactory,  //factory for plugin layers
					 IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactoryExt(pluginFactory);
	std::cout << "=========parse caffe model start==========" << std::endl;
	//if platform support fp16 calculation,change the data tpye
	bool useFp16 = builder->platformHasFastFp16();
	//useFp16 = false;
	DataType modelDataType = useFp16?DataType::kHALF:DataType::kFLOAT;

	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile, directories).c_str(),locateFile(modelFile, directories).c_str(), *network, modelDataType);

	// specify which tensors are outputs
	for (auto& s : outputs)
	{
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	}
	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(2 << 20);
	builder->setFp16Mode(useFp16);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	std::cout << "=========buildCudaEngine end==========" << std::endl;
	//assert(engine);
	
	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	std::cout << "=========engine->serialize end==========" << std::endl;
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex  = engine.getBindingIndex(INPUT_BLOB_NAME);
	DimsCHW inputDims = static_cast<DimsCHW&&>(engine.getBindingDimensions(inputIndex));
	size_t inputSize = batchSize * inputDims.c() * inputDims.h()*inputDims.w() * sizeof(float);

	int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
	DimsCHW outputDims = static_cast<DimsCHW&&>(engine.getBindingDimensions(outputIndex));
	size_t outputSize = batchSize * outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);

	// allocate GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
	CHECK(cudaMalloc(&buffers[outputIndex], outputSize ));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

