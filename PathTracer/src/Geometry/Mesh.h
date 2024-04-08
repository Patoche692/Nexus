#pragma once

#include <vector>
#include <cuda_runtime_api.h>

#include "Triangle.h"

struct Mesh
{
	__host__ __device__ Mesh() = default;

	Triangle* triangles;
	uint32_t materialId;
};
