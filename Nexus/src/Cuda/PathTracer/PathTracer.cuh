#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <glm/glm.hpp>
#include "OpenGL/PixelBuffer.h"
#include "Scene/Scene.h"

// Number of threads in a block
#define BLOCK_SIZE 64

#define WARP_SIZE 32	// Same size for all NVIDIA GPUs

#define PATH_MAX_LENGTH 100


// Size is the number of pixels on the screen
struct D_PathStateSAO
{
	float3* throughput;
	float3* radiance;
	float* lastPdf;
};

struct D_TraceRequestSAO
{
	D_RaySAO ray;
	D_IntersectionSAO intersection;
	uint32_t* pixelIdx;
};

struct D_ShadowTraceRequestSAO
{
	D_RaySAO ray;
	float* hitDistance;
	uint32_t* pixelIdx;
	float3* radiance;
};

struct D_MaterialRequestSAO
{
	float3* rayDirection;
	D_IntersectionSAO intersection;
	uint32_t* pixelIdx;
};

struct D_PixelQuery
{
	int32_t pixelIdx;
	int32_t instanceIdx;
};

// From Jan van Bergen: store all the queue sizes for different depths
// so that we only need to reset the sizes once after rendering
struct D_QueueSize
{
	int32_t traceSize[PATH_MAX_LENGTH];
	int32_t traceCount[PATH_MAX_LENGTH];

	int32_t traceShadowSize[PATH_MAX_LENGTH];
	int32_t traceShadowCount[PATH_MAX_LENGTH];

	int32_t diffuseSize[PATH_MAX_LENGTH];
	int32_t plasticSize[PATH_MAX_LENGTH];
	int32_t dielectricSize[PATH_MAX_LENGTH];
	int32_t conductorSize[PATH_MAX_LENGTH];
};


__global__ void GenerateKernel();
__global__ void LogicKernel();
__global__ void DiffuseMaterialKernel();
__global__ void PlasticMaterialKernel();
__global__ void DielectricMaterialKernel();
__global__ void ConductorMaterialKernel();
__global__ void TraceKernel();
__global__ void TraceShadowKernel();
__global__ void AccumulateKernel();

D_Scene* GetDeviceSceneAddress();
float3** GetDeviceAccumulationBufferAddress();
uint32_t** GetDeviceRenderBufferAddress();
uint32_t* GetDeviceFrameNumberAddress();
uint32_t* GetDeviceBounceAddress();
D_BVH8* GetDeviceTLASAddress();
D_BVH8** GetDeviceBVHAddress();
D_BVHInstance** GetDeviceBLASAddress();

D_PathStateSAO* GetDevicePathStateAddress();
D_ShadowTraceRequestSAO* GetDeviceShadowTraceRequestAddress();
D_TraceRequestSAO* GetDeviceTraceRequestAddress();
D_MaterialRequestSAO* GetDeviceDiffuseRequestAddress();
D_MaterialRequestSAO* GetDevicePlasticRequestAddress();
D_MaterialRequestSAO* GetDeviceDielectricRequestAddress();
D_MaterialRequestSAO* GetDeviceConductorRequestAddress();
D_QueueSize* GetDeviceQueueSizeAddress();
D_PixelQuery* GetDevicePixelQueryAddress();
