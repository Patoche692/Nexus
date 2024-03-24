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
	uint2 resolution;
};

struct SceneData
{
	unsigned int nSpheres;
	Sphere spheres[MAX_SPHERES];
};

void InitGPUData(uint32_t width, uint32_t height);
void FreeGPUData();
void ResizeBuffers(uint32_t width, uint32_t height);
void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer, uint32_t frameNumber, float3* accumulationBuffer);
void SendCameraDataToDevice(Camera *camera);
void SendSceneDataToDevice(Scene* scene);
