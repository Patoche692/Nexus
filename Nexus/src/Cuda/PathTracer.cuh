#pragma once

#include <device_launch_parameters.h>
#include <iostream>
#include <glm/glm.hpp>
#include "OpenGL/PixelBuffer.h"
#include "Scene/Scene.h"

// Number of threads in a block
#define BLOCK_SIZE 8

#define WARP_SIZE 32	// Same size for all NVIDIA GPUs

#define LIMIT_BOUNCES 30

void RenderViewport(PixelBuffer& pixelBuffer, const D_Scene& scene, uint32_t frameNumber, float3* accumulationBuffer);
