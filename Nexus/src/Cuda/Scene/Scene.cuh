#pragma once

#include <cuda_runtime_api.h>
#include "Cuda/BVH/TLAS.cuh"
#include "Material.cuh"
#include "Camera.cuh"

struct D_Scene
{
	bool hasHdrMap = false;
	cudaTextureObject_t hdrMap;

	cudaTextureObject_t* diffuseMaps;
	cudaTextureObject_t* emissiveMaps;

	D_TLAS tlas;
	D_Material* materials;
	D_Camera camera;
};