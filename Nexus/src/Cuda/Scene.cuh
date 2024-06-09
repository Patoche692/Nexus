#pragma once

#include <cuda_runtime_api.h>
#include "Cuda/BVH/TLAS.cuh"
#include "Material.cuh"

struct D_Scene
{
	bool hasHdrMap;
	cudaTextureObject_t hdrMap;

	cudaTextureObject_t* diffuseMaps;
	cudaTextureObject_t* emissiveMaps;

	D_TLAS tlas;
	D_Material* materials;
};