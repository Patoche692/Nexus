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
struct D_PathStateSOA
{
	float3* throughput;
	float3* radiance;

	// Unoccluded ray origin (used for MIS with semi-opaque materials)
	float3* rayOrigin;

	float* lastPdf;
	bool* allowMIS;
};

struct D_TraceRequestSOA
{
	D_RaySOA ray;
	D_IntersectionSOA intersection;
	uint32_t* pixelIdx;
};

struct D_ShadowTraceRequestSOA
{
	D_RaySOA ray;
	float* hitDistance;
	uint32_t* pixelIdx;
	float3* radiance;
};

struct D_MaterialRequestSOA
{
	float3* rayDirection;
	D_IntersectionSOA intersection;
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

D_PathStateSOA* GetDevicePathStateAddress();
D_ShadowTraceRequestSOA* GetDeviceShadowTraceRequestAddress();
D_TraceRequestSOA* GetDeviceTraceRequestAddress();
D_MaterialRequestSOA* GetDeviceDiffuseRequestAddress();
D_MaterialRequestSOA* GetDevicePlasticRequestAddress();
D_MaterialRequestSOA* GetDeviceDielectricRequestAddress();
D_MaterialRequestSOA* GetDeviceConductorRequestAddress();
D_QueueSize* GetDeviceQueueSizeAddress();
D_PixelQuery* GetDevicePixelQueryAddress();
