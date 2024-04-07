#pragma once
#include "../Utils/cuda_math.h"

struct Triangle
{
	float3 pos0;
	float3 pos1;
	float3 pos2;

	float3 normal0;
	float3 normal1;
	float3 normal2;

	float3 texCoord0;
	float3 texCoord1;
	float3 texCoord2;

	__device__ __host__ Triangle() = default;

	__device__ __host__ Triangle(float3 p0, float3 p1, float3 p2)
		:pos0(p0), pos1(p1), pos2(p2), normal0(make_float3(0.0f)), normal1(make_float3(0.0f)),
		normal2(make_float3(0.0f)), texCoord0(make_float3(0.0f)), texCoord1(make_float3(0.0f)), texCoord2(make_float3(0.0f)) { }

	__device__ __host__ Triangle(float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2)
		:pos0(p0), pos1(p1), pos2(p2), normal0(n0), normal1(n1),
		normal2(n2), texCoord0(make_float3(0.0f)), texCoord1(make_float3(0.0f)), texCoord2(make_float3(0.0f)) { }

	__device__ __host__ Triangle(float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2, float3 t0, float3 t1, float3 t2)
		:pos0(p0), pos1(p1), pos2(p2), normal0(n0), normal1(n1),
		normal2(n2), texCoord0(t0), texCoord1(t1), texCoord2(t2) { }

};
