#pragma once

#include <device_launch_parameters.h>
#include <iostream>
#include <glm.hpp>
#include "../OpenGL_API/PixelBuffer.h"
#include "../Camera.h"

struct CameraData
{
	float3 position;
	float3 forwardDirection;
	float3 lowerLeftCorner;
	float3 horizontal;
	float3 vertical;
	uint32_t viewportWidth;
	uint32_t viewportHeight;
};

__global__ void traceRay(void* bufferDevicePtr);
void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer);
void SendCameraDataToDevice(Camera *camera);
