#pragma once
#include <cuda_runtime_api.h>
#include "../Utils/cuda_math.h"

/**
 * Fast random number generator based on Ray Tracing Gems 2 book (cf page 169)
 */

class Random
{
public:
	static inline __host__ __device__ unsigned int InitRNG(uint2 pixel, uint2 resolution, unsigned int frameNumber);
	static inline __host__ __device__ float Rand(unsigned int& rngState);
	static inline __host__ __device__ float3 RandomUnitVector(unsigned int& rngState);
	static inline __host__ __device__ float3 RandomInUnitSphere(unsigned int& rngState);
	static inline __host__ __device__ float3 RandomOnHemisphere(unsigned int& rngState, float3& normal);
};

inline __host__ __device__ unsigned int jenkinsHash(unsigned int x)
{
	x += x << 10;
	x ^= x >> 6;
	x += x << 3;
	x ^= x >> 11;
	x += x << 15;
	return x;
}

// PCG version

inline __host__ __device__ uint4 pcg4d(uint4 v)
{
	v = v * 1664525u + 1013904223u;

	v.x += v.y * v.w;
	v.y += v.z * v.x; 
	v.z += v.x * v.y; 
	v.w += v.y * v.z;

	v.x ^= v.x >> 16u;
	v.y ^= v.y >> 16u;
	v.z ^= v.z >> 16u;
	v.w ^= v.w >> 16u;

	v.x += v.y * v.w; 
	v.y += v.z * v.x; 
	v.z += v.x * v.y; 
	v.w += v.y * v.z;

	return v;
}

inline __host__ __device__ unsigned int xorShift(unsigned int& rngState)
{
	rngState ^= rngState << 13;
	rngState ^= rngState >> 17;
	rngState ^= rngState << 5;
	return rngState;
}

inline __host__ __device__ float uintToFloat(unsigned int x)
{
	unsigned int a = 0x3f800000 | (x >> 9);
	float* b = (float*)&a;
	return *b - 1.0f;
}

inline __host__ __device__ unsigned int Random::InitRNG(uint2 pixel, uint2 resolution, unsigned int frameNumber)
{
	unsigned int rngState = dot(pixel, make_uint2(1, resolution.x)) ^ jenkinsHash(frameNumber);
	if (rngState == 0)
		rngState = 1;
	return jenkinsHash(rngState);
}

inline __host__ __device__ float Random::Rand(unsigned int& rngState)
{
	return uintToFloat(xorShift(rngState));
}

inline __host__ __device__ float3 Random::RandomInUnitSphere(unsigned int& rngState)
{
	float3 p;
	do {
		p = 2.0f * (make_float3(Rand(rngState), Rand(rngState), Rand(rngState)) - 0.5f);
	} while (length(p) >= 1.0f);
	return p;
}

inline __host__ __device__ float3 Random::RandomUnitVector(unsigned int& rngState)
{
	return normalize(RandomInUnitSphere(rngState));
}


inline __host__ __device__ float3 Random::RandomOnHemisphere(unsigned int& rngState, float3& normal)
{
	float3 r = RandomUnitVector(rngState);
	if (dot(r, normal) > 0)
		return r;
	else
		return -r;
}
