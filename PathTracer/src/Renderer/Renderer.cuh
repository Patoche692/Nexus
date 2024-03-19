#pragma once

#include <device_launch_parameters.h>
#include <iostream>
#include <glm.hpp>
#include "../OpenGL_API/PixelBuffer.h"
#include "../Camera.h"

struct CameraData
{
	float verticalFOV;
	float imagePlaneHalfHeight;
	uint32_t viewportWidth;
	uint32_t viewportHeight;
	float3 position;
	float3 forwardDirection;
	float3 rightDirection;
	float3 upDirection;
};

__global__ void traceRay(void* bufferDevicePtr);
void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer);
void SendCameraDataToDevice(Camera *camera);
