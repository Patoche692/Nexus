#pragma once

#include <device_launch_parameters.h>
#include <iostream>
#include <glm.hpp>

__global__ void traceRay(void* device_ptr, uint32_t imageWidth, uint32_t imageHeight);
void cudaRender(void* device_ptr, uint32_t imageWidth, uint32_t imageHeight);
