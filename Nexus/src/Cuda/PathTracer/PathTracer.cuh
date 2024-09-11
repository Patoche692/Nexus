#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <glm/glm.hpp>
#include "OpenGL/PixelBuffer.h"
#include "Scene/Scene.h"

// Number of threads in a block
#define BLOCK_SIZE 8

#define WARP_SIZE 32	// Same size for all NVIDIA GPUs

#define PATH_MAX_LENGTH 30



struct D_PathStateSAO
{
	D_RaySAO ray;
	D_IntersectionSAO intersection;

	uint32_t* pixelIdx;
	float3* throughput;
	float3* radiance;

	float* lastPdf;

	int32_t size;
};

struct D_ShadowRayStateSAO
{
	D_RaySAO ray;
	float* hitDistance;
	uint32_t* pixelIdx;
	float3* radiance;

	int32_t size;
};

struct D_MaterialRequestSAO
{
	float3* rayDirection;
	D_IntersectionSAO intersection;

	int32_t size;
};



//__global__ void TraceRay();

__global__ void GenerateKernel();
__global__ void LogicKernel();
__global__ void TraceKernel();
__global__ void TraceShadowKernel();
__global__ void DiffuseMaterialKernel();
__global__ void PlasticMaterialKernel();
__global__ void DielectricMaterialKernel();
__global__ void ConductorMaterialKernel();

D_Scene* GetDeviceSceneAddress();
float3** GetDeviceAccumulationBufferAddress();
uint32_t** GetDeviceRenderBufferAddress();
uint32_t* GetDeviceFrameNumberAddress();
D_BVH8* GetDeviceTLASAddress();
D_BVH8** GetDeviceBVHAddress();
D_BVHInstance** GetDeviceBLASAddress();

D_PathStateSAO* GetDevicePathStateAddress();
D_ShadowRayStateSAO* GetDeviceShadowRayStateAddress();
D_MaterialRequestSAO* GetDeviceDiffuseRequestAddress();
D_MaterialRequestSAO* GetDevicePlasticRequestAddress();
D_MaterialRequestSAO* GetDeviceDielectricRequestAddress();
D_MaterialRequestSAO* GetDeviceConductorRequestAddress();
