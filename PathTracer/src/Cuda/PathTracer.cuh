#pragma once

#include <device_launch_parameters.h>
#include <iostream>
#include <glm/glm.hpp>
#include "OpenGL/PixelBuffer.h"
#include "Scene.h"
#include "AssetManager.cuh"

struct CameraData
{
	float3 position;
	float3 right;
	float3 up;
	float lensRadius;
	float3 lowerLeftCorner;
	float3 viewportX;
	float3 viewportY;
	uint2 resolution;
};

void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer, uint32_t frameNumber, float3* accumulationBuffer);
void SendCameraDataToDevice(Camera *camera);
