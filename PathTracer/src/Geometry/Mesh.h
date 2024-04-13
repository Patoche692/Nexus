#pragma once

#include <vector>
#include <cuda_runtime_api.h>

#include "Triangle.h"

struct Mesh
{
	__host__ __device__ Mesh() = default;

	Triangle* triangles = nullptr;
	uint32_t nTriangles = 0;
	int materialId = -1;
};
