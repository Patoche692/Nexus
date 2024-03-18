#pragma once

#include <device_launch_parameters.h>
#include <iostream>
#include <glm.hpp>
#include "../OpenGL_API/PixelBuffer.h"

__global__ void traceRay(void* device_ptr, uint32_t imageWidth, uint32_t imageHeight);
void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer);
