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


__global__ void TraceRay();

D_Scene* GetDeviceSceneAddress();
float3** GetDeviceAccumulationBufferAddress();
uint32_t** GetDeviceRenderBufferAddress();
uint32_t* GetDeviceFrameNumberAddress();
