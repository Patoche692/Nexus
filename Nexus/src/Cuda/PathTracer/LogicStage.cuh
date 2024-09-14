#pragma once

#include "Cuda/BVH/TLASTraversal.cuh"

//__device__ void LogicStage(const D_Scene scene)
//{
//	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//	const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
//	const uint32_t PathIdx = i + j * scene.camera.resolution.y;
//
//	D_HitResult hitResult = ;
//}
