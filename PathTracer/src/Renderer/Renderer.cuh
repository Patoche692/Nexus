#pragma once

#include <device_launch_parameters.h>
#include <iostream>
#include <glm.hpp>
#include "../OpenGL/PixelBuffer.h"
#include "../Scene.h"

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

struct SceneData
{
	unsigned int nSpheres;
	Sphere spheres[MAX_SPHERES];
};

__global__ void traceRay(uint32_t* outBufferPtr);
void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer);
void SendCameraDataToDevice(Camera *camera);
void SendSceneDataToDevice(Scene* scene);
