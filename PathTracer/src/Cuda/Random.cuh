#pragma once
#include <cuda_runtime_api.h>
#include "../Utils/cuda_math.h"

/**
 * Fast random number generator based on Ray Tracing Gems 2 book (cf page 169)
 */

class Random
{
public:
	__host__ __device__ inline static unsigned int InitRNG(uint2 pixel, uint2 resolution);
	__host__ __device__ inline static float Rand(unsigned int& rngState);
};

__host__ __device__ inline unsigned int jenkinsHash(unsigned int x)
{
	x += x << 10;
	x ^= x >> 6;
	x += x << 3;
	x ^= x >> 11;
	x += x << 15;
	return x;
}

__host__ __device__ inline unsigned int xorShift(unsigned int& rngState)
{
	rngState ^= rngState << 13;
	rngState ^= rngState >> 17;
	rngState ^= rngState << 5;
	return rngState;
}

__host__ __device__ inline float uintToFloat(unsigned int x)
{
	unsigned int a = 0x3f800000 | (x >> 9);
	float* b = (float*)&a;
	return *b - 1.0f;
}

__host__ __device__ inline unsigned int Random::InitRNG(uint2 pixel, uint2 resolution)
{
	unsigned int rngState = dot(pixel, make_uint2(1, resolution.x));
	return jenkinsHash(rngState);
}

__host__ __device__ inline float Random::Rand(unsigned int& rngState)
{
	return uintToFloat(xorShift(rngState));
}
